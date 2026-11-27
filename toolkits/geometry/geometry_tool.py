# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Complex geometry perception over caller-provided camera observations."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from rpent.session import EnvState
from rpent.tools import ToolResult, tool
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.logging import get_logger
from toolkits.geometry.geometry import FitOptions, ShapeFit
from toolkits.geometry.object_features import (
    FEATURES_BY_QUANTITY,
    QUANTITIES,
    GeometryFeature,
    GeometryQuantity,
    feature_result,
    fit_features,
    summarize_fits,
)
from toolkits.geometry.rgbd_parts import object_points
from toolkits.geometry.semantic_masks import ObjectSegmenter, select_object
from toolkits.observation import GeometryObservation, ObservationProvider

if TYPE_CHECKING:
    from toolkits.geometry.local_point_agent import VisualGrounder

logger = get_logger("geometry_tool")


class GeometryTool:
    """Combine grounding, segmentation and geometry in a complex perception tool.

    The observation provider owns camera selection, source steps, calibration
    and coordinate-frame identity. EnvState persists evidence at those existing
    steps. SAM3 and Molmo locate the object. Geometry fits the requested
    centers to the object's calibrated RGB-D samples.
    """

    def __init__(
        self,
        *,
        state: EnvState,
        observations: ObservationProvider,
        grounder: VisualGrounder,
        primary_camera: str | None = None,
        segmenter: ObjectSegmenter | None = None,
        fit_options: FitOptions | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> None:
        self._state = state
        self._observations = observations
        self._primary_camera = primary_camera
        self._segmenter = segmenter
        self._grounder = grounder
        self._fit_options = fit_options or FitOptions()
        self._check_cancelled = check_cancelled or (lambda: None)
        self._lock = threading.Lock()
        self._namespace = uuid.uuid4().hex[:12]
        self._index = 0
        self._cache_step: int | None = None
        self._cache: dict[tuple, str] = {}

    def reset_cache(self) -> None:
        """Discard a previous episode's measurements when its step numbers restart."""
        with self._lock:
            self._cache.clear()
            self._cache_step = None

    @tool(
        readonly=True,
        json_schema_extra={"additionalProperties": False},
    )
    def geometry(
        self,
        query: Annotated[str, Field(min_length=1, max_length=1024, pattern=r"\S")],
        camera: Annotated[str, Field(min_length=1, max_length=128)],
        step: Annotated[int, Field(ge=-1)] = -1,
        *,
        object_prompt: Annotated[
            str, Field(min_length=1, max_length=128, pattern=r"\S")
        ],
        required_geometry: Annotated[
            list[GeometryQuantity],
            Field(min_length=1, max_length=len(QUANTITIES)),
        ],
        feature: GeometryFeature,
    ) -> ToolResult:
        """Fit the requested centers in one calibrated RGB-D frame from the primary camera.

        A single SAM3 text mask is used directly. With several masks, a Molmo
        point on the original image selects among masks containing it. Overlapping
        hits use the highest score among those hits. A missing or unmatched point
        uses the highest score overall. Both fallbacks warn. With no text masks,
        SAM3 segments from the Molmo point. Each requested center that passes
        the fit checks is returned as a world-frame [x, y, z] in metres.
        Before using a center, confirm that the highlighted object in the overlay
        matches the query's appearance and spatial relations. If the match is
        wrong or unclear, discard the coordinates and locate the target again.

        Args:
            query: Describe the whole object and any distinguishing appearance or
                spatial relation to check in the overlay. Molmo uses this query when needed.
            object_prompt: Short noun phrase naming the whole object for SAM3. Use a common
                name and include shape or color when helpful. Put spatial relations in query.
            camera: Name of the primary camera used for this measurement.
            step: Recorded source step. Use -1 to pin the latest observation.
            required_geometry: Request the world coordinates of the centers on the
                selected feature.
            feature: For ring_center, select the visible inner or outer edge of the
                upper rim with upper_inner_ring or upper_outer_ring. For disk_center
                or rectangle_center, use upper_face for the highest visible
                near-horizontal face that fits the requested shape. For base_center, the
                local support surface must be visible. Choose supported_round_base
                for an upright round body with a cylindrical sidewall or a circular
                rim and concentric bowl or plate surface. The result is where the body's
                rotation axis meets its support.
        """
        if self._primary_camera is not None and camera != self._primary_camera:
            return ToolResult(
                data={
                    "status": "error",
                    "code": "unsupported_camera",
                    "error": f"Use the primary camera '{self._primary_camera}' for geometry.",
                }
            )
        query = query.strip()
        if not query:
            return ToolResult(
                data={
                    "status": "error",
                    "error": "Describe the visible target in query.",
                }
            )
        object_prompt = object_prompt.strip()
        if not object_prompt:
            return ToolResult(
                data={
                    "status": "error",
                    "error": "Name the whole object in object_prompt.",
                }
            )
        if not required_geometry or any(
            name not in FEATURES_BY_QUANTITY for name in required_geometry
        ):
            return ToolResult(
                data={
                    "status": "error",
                    "code": "unsupported_geometry_quantity",
                    "error": f"Choose a geometry quantity from {', '.join(QUANTITIES)}.",
                }
            )
        requested = list(dict.fromkeys(required_geometry))
        if any(feature not in FEATURES_BY_QUANTITY[name] for name in requested):
            return ToolResult(
                data={
                    "status": "error",
                    "code": "invalid_geometry_feature",
                    "error": (
                        "Use upper_inner_ring or upper_outer_ring for ring_center, "
                        "upper_face for disk_center or rectangle_center, "
                        "and supported_round_base for base_center."
                    ),
                }
            )
        with self._lock:
            self._check_cancelled()
            try:
                step = self._state.get(step).step_idx
            except (ValueError, LookupError) as exc:
                return ToolResult(
                    data={
                        "status": "error",
                        "code": "observation_unavailable",
                        "error": str(exc),
                    }
                )
            if self._cache_step != step:
                self._cache.clear()
                self._cache_step = step
            key = (
                camera,
                " ".join(query.casefold().split()),
                object_prompt,
                tuple(sorted(requested)),
                feature,
            )
            if key in self._cache:
                artifact = self._cache[key]
                try:
                    data = self._state.load(artifact, step=step)
                    return self._planner_result(data, artifact)
                except (OSError, LookupError, ValueError):
                    del self._cache[key]

            start = time.perf_counter()
            try:
                observation = self._observations.get_observation(
                    camera=camera, step=step
                )
                if (
                    not isinstance(observation, GeometryObservation)
                    or observation.camera != camera
                    or observation.step != step
                ):
                    raise ValueError("Provider returned a different camera or step.")
            except (ValueError, LookupError, OSError, RuntimeError) as exc:
                return ToolResult(
                    data={
                        "status": "error",
                        "code": "observation_unavailable",
                        "camera": camera,
                        "step": step,
                        "error": str(exc),
                    }
                )
            tag = f"{self._namespace}_{self._index:03d}"
            self._index += 1
            data = self._measure(
                observation,
                query,
                object_prompt,
                requested,
                feature,
                tag,
            )
            if data["status"] == "error":
                return ToolResult(data=_planner_measurement(data))
            data["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
            data["request"] = {
                "query": query,
                "required_geometry": required_geometry,
                "feature": feature,
                "object_prompt": object_prompt,
            }
            self._check_cancelled()
            try:
                artifact = self._save(f"geometry_{tag}.json", data, step)
                result = self._planner_result(data, artifact)
            except (OSError, RuntimeError) as exc:
                return ToolResult(
                    data={
                        "status": "error",
                        "code": "geometry_artifact_save_failed",
                        "error": str(exc),
                    }
                )
            self._cache[key] = artifact
            return result

    def _planner_result(self, data: dict, artifact: str) -> ToolResult:
        images = []
        if overlay := data.get("overlay_artifact"):
            images.append(self._state.load_bytes(overlay, step=data["step"]))
        return ToolResult(
            data=_planner_measurement({**data, "geometry_artifact": artifact}),
            images=images,
        )

    def _measure(
        self,
        observation: GeometryObservation,
        query: str,
        object_prompt: str,
        requested: list[GeometryQuantity],
        feature: GeometryFeature,
        tag: str,
    ) -> dict:
        data = {
            "agent": "GeometryTool",
            "query": query,
            "step": observation.step,
            "camera": observation.camera,
            "feature": feature,
            "required_geometry": requested,
            "identity_evidence": {},
            "world_frame_id": observation.world_frame_id,
            "units": "m",
            "status": "insufficient_evidence",
            "found": False,
            "definition": "object_mask_rgbd_geometry",
            "evidence_status": "unknown",
            "calls": {"grounding": 0, "segment": 0, "back_project": 0},
            "input_refs": dict(observation.source_refs),
            "segment_attempts": [],
            "quality": {},
        }
        try:
            self._check_cancelled()
            if observation.camera_to_world is None or observation.up_direction is None:
                obj, reason = None, "camera_pose_and_up_direction_required_for_geometry"
            elif self._segmenter is None:
                obj, reason = None, "semantic_segmentation_unavailable"
            else:
                obj, reason = select_object(
                    observation,
                    data,
                    self._segmenter,
                    self._grounder,
                    self._check_cancelled,
                    object_prompt,
                )
            if obj is not None:
                data["object_mask_artifact"] = self._save(
                    f"geometry_object_{tag}.npz", obj, observation.step
                )
                data["overlay_artifact"] = self._save(
                    f"geometry_overlay_{tag}.png",
                    observation.mask_overlay(obj, outline=True),
                    observation.step,
                )
                n_valid = int((obj & observation.valid_mask).sum())
                data["quality"].update(n_pixels=int(obj.sum()), n_valid=n_valid)
                if not n_valid:
                    reason = "no_valid_depth_in_object_mask"
            if reason is not None:
                fields = {
                    name: feature_result(name, observation, ShapeFit(reason=reason))
                    for name in requested
                }
                summarize_fits(data, fields)
                return data

            data["calls"]["back_project"] += 1
            self._fit_geometry(data, observation, obj)
        except ToolCancelled:
            raise
        except (ValueError, OSError, RuntimeError, LookupError) as exc:
            data.update(status="error", error=str(exc))
        except Exception as exc:
            logger.exception("geometry request failed at step %s", observation.step)
            data.update(status="error", error=str(exc))
        return data

    def _fit_geometry(self, data, observation, mask):
        requested = data["required_geometry"]
        artifact = data["object_mask_artifact"]
        self._check_cancelled()
        cloud = object_points(
            observation.world_xyz,
            observation.valid_mask,
            mask,
            observation.camera_to_world[:3, 2],
            keep_small_faces=data["feature"] == "upper_face",
        )
        data["quality"]["n_cleaned"] = len(cloud)
        fields = {}
        for name, fit in fit_features(
            requested,
            data["feature"],
            observation,
            mask,
            cloud,
            self._fit_options,
            self._check_cancelled,
        ):
            fields[name] = feature_result(name, observation, fit)
            fields[name]["source_refs"]["object_mask"] = artifact
        summarize_fits(data, fields)

    def _save(self, name: str, value: Any, step: int) -> str:
        saved = self._state.save(name, value, step=step)
        if saved is None:
            raise RuntimeError(f"failed to persist geometry evidence {name}")
        return saved


def _planner_measurement(data: dict) -> dict:
    """Send actionable evidence to the planner. The saved artifact retains the full fit."""
    keys = (
        "status",
        "found",
        "missing_geometry",
        "query",
        "feature",
        "step",
        "camera",
        "world_frame_id",
        "units",
        "definition",
        "evidence_status",
        "geometry_artifact",
        "reason",
        "warnings",
        "code",
        "error",
    )
    out = {k: data[k] for k in keys if data.get(k) is not None}
    for name, field in data.get("geometry", {}).items():
        summary = {
            k: field[k]
            for k in (
                "unit",
                "definition",
                "fit_accepted",
                "evidence_status",
                "source_step",
                "source_camera",
                "warnings",
                "reason",
            )
            if field.get(k) is not None
        }
        if (
            field.get("fit_accepted")
            and field.get("value") is not None
            and data.get("status") in {"usable", "partial", "warning"}
        ):
            summary["value"] = field["value"]
        out.setdefault("geometry", {})[name] = summary
    return out
