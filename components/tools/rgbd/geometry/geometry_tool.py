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
import uuid
from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from pydantic import Field

from components.tools.rgbd.geometry import object_features
from components.tools.rgbd.geometry.geometry import FitOptions
from components.tools.rgbd.geometry.multiview import observe_with_alternatives
from components.tools.rgbd.observation import (
    GeometryObservation,
    ObservationProvider,
)
from components.tools.rgbd.segment import Segmenter, SegmentTool
from rpent.session import EnvState
from rpent.tools import ToolResult, tool
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from components.tools.rgbd.geometry.local_point_agent import VisualGrounder

logger = get_logger("geometry_tool")

GeometryQuantity = Literal[
    "rim_center",
    "rim_radius",
    "bottom_center",
    "body_center",
    "axis",
    "radius",
    "height",
    "body_radii",
    "plane",
    "plane_normal",
    "inner_rim_center",
    "plate_center",
    "lid_center",
    "rectangle_center",
    "body_segment_center",
    "top_corners",
    "cuboid_corners",
    "top_size",
    "rim_extrema",
    "inner_rim_radius",
    "object_height",
    "supported_height",
    "height_above_support",
]


PART_DESCRIPTIONS = {
    "rim": "the visible circular rim boundary of",
    "body": "the visible curved outer body, excluding handle, rim and interior, of",
    "bottom": "the visible circular bottom face or its boundary of",
    "plane": "the visible planar surface of",
}


class GeometryTool:
    """Combine grounding, segmentation and geometry in a complex perception tool.

    The observation provider owns camera selection, source steps, calibration
    and coordinate-frame identity. EnvState persists evidence at those existing
    steps. Visual models select material pixels; the program handles metric
    coordinates, part dispatch, point clouds and geometric acceptance.
    Callers initialize and inject the grounding and segmentation clients.
    """

    def __init__(
        self,
        *,
        state: EnvState,
        observations: ObservationProvider,
        grounder: VisualGrounder,
        segmenter: Segmenter | None = None,
        fit_options: FitOptions | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> None:
        self._state = state
        self._observations = observations
        self._perception = SegmentTool(
            observations=observations, state=state, segmenter=segmenter
        )
        self._object_segmenter = segmenter
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
        measurement: Literal["surface_point", "visible_region"] = "surface_point",
        required_geometry: Annotated[
            list[GeometryQuantity], Field(min_length=1, max_length=10)
        ]
        | None = None,
    ) -> ToolResult:
        """Measure a named target in calibrated camera views of the same recorded step.

        The tool selects pixels internally and verifies the target before returning coordinates. A surface_point lies
        on visible material. A visible_region gives bounds and coordinate-wise
        medians. Request fitted parts or dimensions together in required_geometry.
        Use fit_accepted and warnings to interpret each field. On
        insufficient_evidence choose another task strategy.

        Args:
            query: Describe the visible target and its distinguishing appearance or relation in words.
            camera: A camera name supported by the injected observation provider.
            step: Recorded source step. Use -1 to pin the latest observation.
            measurement: Choose surface_point or visible_region. Bounds cover visible samples.
            required_geometry: Optional list of part and whole-object quantities for
                this target. Each is evaluated separately. Overrides measurement.
                Use height for the distance between fitted rim and bottom centers.
                Use height_above_support for the highest observed surface above
                a measured support plane. The saved geometry artifact also
                contains inspection-only estimates for object_height (visible
                extent), supported_height (assumed contact), body_segment_center
                (inferred endpoints), and cuboid_corners (projected base).
        """
        query = query.strip()
        if not query:
            return ToolResult(
                data={
                    "status": "error",
                    "error": "Describe the visible target in query.",
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
                measurement,
                tuple(sorted(required_geometry or ())),
            )
            if key in self._cache:
                artifact = self._cache[key]
                try:
                    data = self._state.load(artifact, step=step)
                except (OSError, LookupError, ValueError):
                    del self._cache[key]
                else:
                    return ToolResult(
                        data=_planner_measurement(
                            {**data, "geometry_artifact": artifact}
                        )
                    )

            tag = f"{self._namespace}_{self._index:03d}"
            result = observe_with_alternatives(
                self._observations,
                self._single_view,
                query,
                camera,
                step,
                measurement,
                required_geometry,
                self._check_cancelled,
            )
            data = result.data
            if data.get("status") != "error":
                data["request"] = {
                    "query": query,
                    "measurement": measurement,
                    "required_geometry": required_geometry,
                }
                try:
                    artifact = self._save(f"geometry_{tag}.json", data, step)
                except (OSError, RuntimeError) as exc:
                    return ToolResult(
                        data={
                            "status": "error",
                            "code": "geometry_artifact_save_failed",
                            "error": str(exc),
                        }
                    )
                data["geometry_artifact"] = artifact
                self._cache[key] = artifact
            return ToolResult(data=_planner_measurement(data))

    def _single_view(
        self,
        query,
        camera,
        step,
        measurement,
        required_geometry,
        cancel,
        *,
        observation,
        evidence,
        direction_camera=None,
        projected_rc=None,
    ) -> ToolResult:
        tag = f"{self._namespace}_{self._index:03d}"
        self._index += 1
        data = {
            "agent": "GeometryTool",
            "query": query,
            "step": step,
            "camera": camera,
            "measurement": measurement,
            "world_frame_id": observation.world_frame_id,
            "units": "m",
            "status": "need_observation",
            "found": False,
            "world_xyz": None,
            "definition": "visible_material_surface_point"
            if measurement == "surface_point"
            else "visible_region_coordinate_median",
            "evidence_status": "unknown",
            "calls": {"grounding": 0, "segment": 0, "back_project": 0},
            "input_refs": dict(observation.source_refs),
            "segment_attempts": [],
            "quality": {},
        }
        if observation.camera_to_world is None or observation.up_direction is None:
            data["reason"] = "camera_pose_and_up_direction_required_for_geometry"
            return ToolResult(data=data)
        try:
            cancel()
            requested = list(dict.fromkeys(required_geometry or ()))
            features = [name for name in requested if name in object_features.FEATURES]
            parts = [name for name in requested if name not in object_features.FEATURES]
            data["required_geometry"] = features

            def identify(target):
                return object_features.run(
                    observation,
                    target,
                    self._grounder,
                    self._object_segmenter,
                    cancel,
                    evidence=evidence,
                    direction_camera=direction_camera,
                    projected_rc=projected_rc,
                )

            if parts and not features:
                identity = {**data, "measurement": "visible_region"}
                region = identify(identity)
                if region is None:
                    return ToolResult(
                        data={
                            **identity,
                            "measurement": measurement,
                            "required_geometry": parts,
                            "missing_geometry": parts,
                        }
                    )
                data["identity_evidence"] = identity["identity_evidence"]
            else:
                region = identify(data)
                if parts and region is None:
                    identity = {
                        **data,
                        "measurement": "visible_region",
                        "required_geometry": [],
                    }
                    region = identify(identity)
                    if region is not None:
                        data["identity_evidence"] = identity["identity_evidence"]
            if parts:
                self._complete_parts(data, observation, tag, parts, region, cancel)
                data["required_geometry"] = requested
            elif (
                not requested and region is not None and measurement == "visible_region"
            ):
                data["mask_artifact"] = self._save(
                    f"geometry_mask_{tag}.npz", region, step
                )
                data["pointcloud_artifact"] = self._save(
                    f"geometry_points_{tag}.npz", observation.world_xyz[region], step
                )
                data["quality"]["n_valid"] = int(region.sum())
        except ToolCancelled:
            raise
        except (ValueError, OSError, RuntimeError, LookupError) as exc:
            data.update(status="error", error=str(exc))
        except Exception as exc:
            logger.exception("geometry request failed at step %s", step)
            data.update(status="error", error=str(exc))
        return ToolResult(data=data)

    def _complete_parts(self, data, observation, tag, names, region, cancel):
        """Fit remaining parts against one verified instance, keeping independent decisions."""
        extra = {}
        if region is not None and self._object_segmenter is not None:
            part = {**data, "required_geometry": names}
            self._fit_requested(part, observation, tag, cancel, region)
            if not data["required_geometry"]:
                # A part-only request exposes the fitter's definition and quality.
                data.update(part)
                return
            extra = part["geometry"]
            data["parts"] = part["parts"]
        else:
            for name in names:
                extra[name] = {
                    "value": None,
                    "fit_accepted": False,
                    "evidence_status": "unknown",
                    "reason": "target_region_unconfirmed"
                    if region is None
                    else "part_segmentation_unavailable",
                    "warnings": [],
                }
        fields = data.setdefault("geometry", {})
        fields.update(extra)
        missing = [name for name, field in fields.items() if field.get("value") is None]
        alerts = [
            {"quantity": name, "reason": reason}
            for name, field in fields.items()
            for reason in field.get("warnings", ())
        ]
        available = len(fields) - len(missing)
        data.update(
            found=bool(available),
            status=(
                "need_observation"
                if not available
                else "partial"
                if missing
                else "warning"
                if alerts
                else "usable"
            ),
            evidence_status="inferred" if available else "unknown",
            warnings=alerts,
        )
        if missing:
            data["reason"] = data.get("reason") or "requested_geometry_incomplete"
            data["missing_geometry"] = missing
        else:
            data.pop("missing_geometry", None)
            data.pop("reason", None)

    def _save(self, name: str, value: Any, step: int) -> str:
        saved = self._state.save(name, value, step=step)
        if saved is None:
            raise RuntimeError(f"failed to persist geometry evidence {name}")
        return saved

    def _segment_part(
        self,
        data: dict[str, Any],
        observation: GeometryObservation,
        tag: str,
        check_cancelled: Callable[[], None],
        verified_region: np.ndarray,
    ) -> None:
        if not data["query"]:
            raise ValueError("geometry requires a nonempty target description")
        nn = observation.step
        image = observation.png_bytes()
        height, width = observation.rgb.shape[:2]
        check_cancelled()
        data["calls"]["grounding"] += 1
        selected = self._grounder.ground(image, data["query"])
        if not selected.found or selected.point_xy is None:
            data["reason"] = "target_not_identified_in_source_image"
            return
        if selected.image_size != (width, height):
            raise ValueError(
                "grounding result does not match the pinned RGB resolution"
            )
        col, row = selected.point_xy
        if not np.isfinite([row, col]).all() or not (
            0 <= row < height and 0 <= col < width
        ):
            raise ValueError("grounding pixel outside source image")
        row, col = min(height - 1, round(row)), min(width - 1, round(col))
        if not verified_region[row, col]:
            data["reason"] = "part_point_outside_verified_object"
            return
        data["pixel_rc"] = [row, col]
        check_cancelled()
        data["calls"]["back_project"] += 1
        if observation.valid_mask[row, col]:
            data["surface_point_xyz"] = observation.world_xyz[row, col].tolist()
        for prompt in (
            {"prompt": data["query"]},
            {"point": [row, col]},
        ):
            check_cancelled()
            data["calls"]["segment"] += 1
            segmented = self._perception.segment_observation(
                observation, image, tag=tag, **prompt
            )
            result = segmented.data
            attempt = {
                "mode": "text" if "prompt" in prompt else "point",
                "found": result["found"],
                "sam_score": result["score"],
                "segment_artifact": result["segment_artifact"],
            }
            data["segment_attempts"].append(attempt)
            if not result["found"]:
                continue
            mask = self._state.load(result["mask_artifact"], step=nn)
            attempt["contains_target_pixel"] = bool(mask[row, col])
            if not mask[row, col]:
                continue
            mask = mask & verified_region
            result["mask_artifact"] = self._save(
                f"geometry_verified_mask_{tag}.npz", mask, nn
            )
            points = observation.world_xyz[mask & observation.valid_mask]
            data["quality"].update(
                n_pixels=int(mask.sum()),
                n_valid=len(points),
                valid_fraction=float(
                    (mask & observation.valid_mask).sum() / mask.sum()
                ),
                validity="provider_validity_and_finite_metric_coordinates",
            )
            data.update(
                mask_artifact=result["mask_artifact"],
                segment_artifact=result["segment_artifact"],
                overlay_artifact=result.get("overlay_artifact"),
                sam_score=result["score"],
            )
            if len(points) < 10:
                data["reason"] = "too_few_valid_region_points"
                return
            data.update(status="usable", found=True, evidence_status="measured")
            return
        data["reason"] = "no_segment_consistent_with_visual_target_within_budget"
        return

    def _fit_requested(
        self,
        data: dict[str, Any],
        observation: GeometryObservation,
        tag: str,
        check_cancelled: Callable[[], None],
        verified_region: np.ndarray,
    ) -> None:
        from scipy.ndimage import binary_dilation

        from components.tools.rgbd.geometry.geometry import (
            ShapeFit,
            clean_part_cloud,
            fit_part,
            geometry_fields,
            part_candidate_mask,
        )

        requested = data["required_geometry"]
        parts = set()
        for name in requested:
            if name in ("rim_center", "rim_radius", "radius", "body_center", "height"):
                parts.add("rim")
            if name in ("axis", "body_radii", "body_center", "height"):
                parts.add("body")
            if name in ("bottom_center", "body_center", "height"):
                parts.add("bottom")
            if name in ("plane", "plane_normal"):
                parts.add("plane")
        descriptions = PART_DESCRIPTIONS
        options = self._fit_options
        data.update(
            definition="part_fitted_geometry",
            measurement="part_geometry",
            parts={},
            budget={"max_grounding_calls": len(parts), "max_sam_calls": 2 * len(parts)},
        )
        fits = {}
        nn, camera = data["step"], data["camera"]
        for part in descriptions:
            if part not in parts:
                continue
            check_cancelled()
            part_tag = f"{tag}_{part}"
            sample = {
                "query": f"{descriptions[part]} {data['query']}",
                "step": nn,
                "camera": camera,
                "measurement": "visible_region",
                "status": "need_observation",
                "world_xyz": None,
                "definition": "visible_region_coordinate_median",
                "calls": {"grounding": 0, "segment": 0, "back_project": 0},
                "segment_attempts": [],
                "quality": {"calibrated_error_m": None},
            }
            sample["world_frame_id"] = observation.world_frame_id
            data["parts"][part] = sample
            try:
                self._segment_part(
                    sample, observation, part_tag, check_cancelled, verified_region
                )
            finally:
                for name, count in sample["calls"].items():
                    data["calls"][name] += count
                data["segment_attempts"].extend(
                    {**attempt, "part": part} for attempt in sample["segment_attempts"]
                )
            if sample["status"] != "usable":
                fits[part] = ShapeFit(
                    part, reason=sample.get("reason", "part_not_observed")
                )
                sample["fit"] = fits[part].summary()
                continue
            world = observation.world_xyz
            mask = self._state.load(sample["mask_artifact"], step=nn)
            image = observation.rgb
            camera_transform = observation.camera_to_world
            row, col = sample["pixel_rc"]
            anchor = sample.get("surface_point_xyz")
            anchor = None if anchor is None else np.asarray(anchor)
            valid = observation.valid_mask
            candidate_mask = part_candidate_mask(
                mask,
                world,
                image,
                part,
                valid_mask=valid,
                camera_transform=camera_transform,
                options=options,
            )
            if part != "bottom":
                candidate_mask[row, col] = True
            candidate_mask &= valid
            candidate_points = world[candidate_mask]
            context_mask = binary_dilation(mask, iterations=5) & ~mask & valid
            cleaned = clean_part_cloud(
                candidate_points,
                part=part,
                anchor=anchor if part != "bottom" else None,
                context_points=world[context_mask],
                options=options,
                up_direction=observation.up_direction,
            )
            camera_origin = (
                None if camera_transform is None else camera_transform[:3, 3]
            )
            fit = fit_part(
                cleaned.points,
                part,
                options=options,
                anchor=anchor,
                camera_origin=camera_origin,
            )
            check_cancelled()
            sample["cleaning"] = cleaned.quality
            sample["fit"] = fit.summary()
            fits[part] = fit
        fields, checks = geometry_fields(fits, requested, options)
        for item in fields.values():
            item["world_frame_id"] = data["world_frame_id"]
            item["source_step"] = nn
            item["source_refs"] = {
                part: {
                    key: value
                    for key, value in data["parts"][part].items()
                    if key.endswith("_artifact")
                }
                for part in item["source_parts"]
                if part in data["parts"]
            }
        available = [name for name, item in fields.items() if item["value"] is not None]
        warnings = [
            {"quantity": name, "reason": reason}
            for name, item in fields.items()
            for reason in item["warnings"]
        ]
        data.update(
            geometry=fields,
            found=bool(available),
            status="usable"
            if len(available) == len(requested) and not warnings
            else "warning"
            if len(available) == len(requested)
            else "partial"
            if available
            else "need_observation",
            evidence_status="inferred" if available else "unknown",
            warnings=warnings,
            quality={
                "calibrated_error_m": None,
                "fit_accepted": all(item["fit_accepted"] for item in fields.values()),
                "part_consistency": checks,
                "fit_options": asdict(options),
            },
        )
        missing = [name for name in requested if name not in available]
        if missing:
            data["reason"] = "requested_geometry_not_observable_or_fit_rejected"
            data["missing_geometry"] = missing


def _planner_measurement(data: dict) -> dict:
    """Send actionable evidence to the planner. The saved artifact retains the full fit."""
    keys = (
        "status",
        "query",
        "measurement",
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
    if data.get("status") == "usable" and data.get("world_xyz") is not None:
        for key in ("world_xyz", "pixel_rc", "visible_bounds_xyz"):
            if key in data:
                out[key] = data[key]
    if data.get("status") == "usable" and data.get("mask_artifact"):
        out["mask_artifact"] = data["mask_artifact"]
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
        summary.setdefault("source_camera", data.get("camera"))
        if field.get("fit_accepted") and data.get("status") in {
            "usable",
            "partial",
            "warning",
        }:
            summary["value"] = field.get("value")
        out.setdefault("geometry", {})[name] = summary
    views = data.get("view_evidence")
    if views:
        out["views_checked"] = [a["camera"] for a in views["attempts"]]
        if views["skipped"]:
            out["views_skipped"] = [
                {k: item[k] for k in ("camera", "reason") if k in item}
                for item in views["skipped"]
            ]
    if data.get("status") in {"partial", "warning", "insufficient_evidence"}:
        fields = out.get("geometry", {})
        missing = list(
            dict.fromkeys(
                [
                    *data.get("missing_geometry", ()),
                    *(name for name, field in fields.items() if "value" not in field),
                ]
            )
        )
        if missing:
            out["missing_geometry"] = missing
        if fields and not any("value" in field for field in fields.values()):
            out.update(
                status="insufficient_evidence",
                reason=data.get("reason") or "no_requested_fit_passed_quality_checks",
            )
    return out
