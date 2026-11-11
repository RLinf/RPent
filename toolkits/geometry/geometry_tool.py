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
from dataclasses import asdict
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field

from rpent.session import EnvState
from rpent.tools import ToolResult, tool
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.logging import get_logger
from toolkits.geometry import perception
from toolkits.geometry.geometry import FitOptions
from toolkits.geometry.observation import GeometryObservation, ObservationProvider
from toolkits.geometry.perception import Segmenter, VisualGrounder
from toolkits.geometry.prompts import PART_DESCRIPTIONS

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
]


class GeometryTool:
    """Combine grounding, segmentation and geometry in a complex perception tool.

    The observation provider owns camera selection, source steps, calibration
    and coordinate-frame identity. EnvState persists evidence at those existing
    steps. Visual models select material pixels; the program handles metric
    coordinates, part dispatch, point clouds and geometric acceptance.
    """

    def __init__(
        self,
        *,
        state: EnvState,
        observations: ObservationProvider,
        segmenter: Segmenter | None = None,
        grounder: VisualGrounder | None = None,
        model: str | None = None,
        base_url: str | None = None,
        fit_options: FitOptions | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> None:
        self._state = state
        self._observations = observations
        self._perception = perception.GeometryPerception(segmenter)
        self._grounder = grounder
        self._model = model
        if grounder is None and model:
            from rpent.robots.components.grounding_agent import GroundingAgent

            self._grounder = GroundingAgent(model=model, base_url=base_url)
        self._fit_options = fit_options or FitOptions()
        self._check_cancelled = check_cancelled or (lambda: None)
        self._lock = threading.Lock()
        self._namespace = uuid.uuid4().hex[:12]
        self._index = 0

    @tool(readonly=True, exclude=("check_cancelled",))
    def geometry(
        self,
        query: Annotated[str, Field(min_length=1, max_length=1024)],
        camera: Annotated[str, Field(min_length=1, max_length=128)],
        step: Annotated[int, Field(ge=-1)] = -1,
        measurement: Literal["surface_point", "visible_region"] = "surface_point",
        required_geometry: Annotated[
            list[GeometryQuantity], Field(min_length=1, max_length=10)
        ]
        | None = None,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> ToolResult:
        """Measure a described visible object or part without moving the robot.

        Describe appearance, the exact part and spatial relation. The tool
        selects pixels internally; calibration and coordinate frames come from
        the observation provider. surface_point returns a visible material point;
        visible_region returns coordinate-wise median and bounds of observed
        segmented samples, not a fitted center. required_geometry selects part
        fitting instead of measurement. Read geometry[field], its definition,
        evidence_status, source_refs, fit_accepted, warnings and quality. A finite
        estimate can be returned with status=warning when quality checks fail;
        its coordinates have not passed those checks. radius means rim radius.
        Full height and body_center require observed rim and bottom edges;
        inconsistent estimates carry warnings.
        Inspect partial/need_observation reasons and obtain missing evidence
        through the parent's normal observation or motion tools.

        Args:
            query: Target appearance, visible part and relation.
            camera: A camera name supported by the injected observation provider.
            step: Recorded source step; -1 pins the provider's latest observation.
            measurement: surface_point or visible_region; bounds cover visible samples.
            required_geometry: Optional fitted quantities, overriding measurement.
        """
        with self._lock:
            check_cancelled = check_cancelled or self._check_cancelled
            check_cancelled()
            try:
                observation = self._observations.get_observation(
                    camera=camera, step=step
                )
                if not isinstance(observation, GeometryObservation):
                    raise ValueError("provider must return a GeometryObservation")
                if observation.camera != camera or (
                    step != -1 and observation.step != step
                ):
                    raise ValueError(
                        "provider returned a different camera or source step"
                    )
                self._state.get(observation.step)
            except (LookupError, ValueError, OSError, RuntimeError) as exc:
                return ToolResult(
                    data={
                        "agent": "GeometryTool",
                        "status": "error",
                        "code": "observation_unavailable",
                        "error": str(exc),
                        "camera": camera,
                        "step": step,
                        "world_xyz": None,
                    }
                )
            nn = observation.step
            tag = f"{self._namespace}_{self._index:03d}"
            self._index += 1
            started = time.perf_counter()
            data: dict[str, Any] = {
                "agent": "GeometryTool",
                "query": query.strip(),
                "step": nn,
                "camera": camera,
                "measurement": measurement,
                "world_frame_id": observation.world_frame_id,
                "units": "m",
                "model": self._model,
                "status": "need_observation",
                "found": False,
                "world_xyz": None,
                "definition": "visible_material_surface_point"
                if measurement == "surface_point"
                else "visible_region_coordinate_median",
                "evidence_status": "unknown",
                "calls": {"grounding": 0, "segment": 0, "back_project": 0},
                "segment_attempts": [],
                "quality": {"calibrated_error_m": None},
                "input_refs": dict(observation.source_refs),
                "image_shape": list(observation.rgb.shape[:2]),
                "source_dtype": observation.source_dtype,
            }
            images: list[bytes] = []
            try:
                check_cancelled()
                for key, suffix, array in (
                    ("image_artifact", "rgb.png", observation.rgb),
                    ("world_artifact", "world.npz", observation.world_xyz),
                    ("valid_artifact", "valid.npz", observation.valid_mask),
                    (
                        "observation_artifact",
                        "observation.json",
                        observation.metadata(),
                    ),
                ):
                    data[key] = self._save(f"geometry_{tag}_{suffix}", array, nn)
                if required_geometry is None:
                    images = self._measure_surface(
                        data, observation, tag, check_cancelled
                    )
                else:
                    data["required_geometry"] = list(dict.fromkeys(required_geometry))
                    images = self._fit_requested(
                        data, observation, tag, check_cancelled
                    )
            except ToolCancelled:
                raise
            except (ValueError, OSError, RuntimeError, LookupError) as exc:
                data.update(status="error", error=str(exc))
            except Exception as exc:
                logger.exception("geometry request failed at step %s", nn)
                data.update(status="error", error=str(exc))
            data["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            try:
                data["geometry_artifact"] = self._save(f"geometry_{tag}.json", data, nn)
            except (OSError, RuntimeError) as exc:
                data.update(
                    status="error", error=str(exc), code="geometry_artifact_save_failed"
                )
            return ToolResult(data=data, images=images)

    def _save(self, name: str, value: Any, step: int) -> str:
        saved = self._state.save(name, value, step=step)
        if saved is None:
            raise RuntimeError(f"failed to persist geometry evidence {name}")
        return saved

    def _need_view(self, data: dict[str, Any], reason: str) -> None:
        data["reason"] = reason
        data["observation_request"] = {
            "kind": "additional_view",
            "target": data["query"],
            "reason": reason,
            "execution_owner": "parent_planner",
        }

    def _measure_surface(
        self,
        data: dict[str, Any],
        observation: GeometryObservation,
        tag: str,
        check_cancelled: Callable[[], None],
    ) -> list[bytes]:
        if not data["query"]:
            raise ValueError("geometry requires a nonempty target description")
        nn = observation.step
        image = self._state.load_bytes(data["image_artifact"], step=nn)
        height, width = observation.rgb.shape[:2]
        if self._grounder is None:
            raise ValueError("geometry requires an injected grounder or model")
        check_cancelled()
        data["calls"]["grounding"] += 1
        selected = self._grounder.ground(image, data["query"])
        if selected.image_size != (width, height):
            raise ValueError(
                "grounding result does not match the pinned RGB resolution"
            )
        if not selected.found or selected.point_xy is None:
            self._need_view(data, "target_not_identified_in_source_image")
            return []
        col, row = selected.point_xy
        if not np.isfinite([row, col]).all() or not (
            0 <= row < height and 0 <= col < width
        ):
            raise ValueError("grounding pixel outside source image")
        row, col = min(height - 1, round(row)), min(width - 1, round(col))
        data["pixel_rc"] = [row, col]
        check_cancelled()
        data["calls"]["back_project"] += 1
        projected = perception.back_project(observation, row, col).data
        data["surface_point_xyz"] = projected.get("world_xyz")
        if "error" in projected:
            data["surface_point_error"] = projected["error"]
        if data["measurement"] == "surface_point":
            if projected.get("world_xyz") is None:
                self._need_view(data, "selected_material_point_has_no_valid_depth")
            else:
                data.update(
                    status="usable",
                    found=True,
                    world_xyz=projected["world_xyz"],
                    evidence_status="measured",
                )
            return []
        for prompt in (
            {"prompt": data.get("segmentation_query", data["query"])},
            {"point": [row, col]},
        ):
            check_cancelled()
            data["calls"]["segment"] += 1
            segmented = self._perception.segment(
                observation, image, state=self._state, tag=tag, **prompt
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
                self._need_view(data, "too_few_valid_region_points")
                return segmented.images
            cloud_name = self._save(f"geometry_points_{tag}.npz", points, nn)
            data.update(
                status="usable",
                found=True,
                evidence_status="measured",
                world_xyz=np.median(points, axis=0).tolist(),
                visible_bounds_xyz=[
                    points.min(axis=0).tolist(),
                    points.max(axis=0).tolist(),
                ],
                pointcloud_artifact=cloud_name,
            )
            return segmented.images
        self._need_view(data, "no_segment_consistent_with_visual_target_within_budget")
        return []

    def _fit_requested(
        self,
        data: dict[str, Any],
        observation: GeometryObservation,
        tag: str,
        check_cancelled: Callable[[], None],
    ) -> list[bytes]:
        from scipy.ndimage import binary_dilation

        from toolkits.geometry.geometry import (
            ShapeFit,
            clean_part_cloud,
            fit_part,
            geometry_fields,
            part_candidate_mask,
        )
        from toolkits.geometry.part_geometry import (
            PART_QUANTITIES,
            fit_part_geometry,
            inner_rim_candidates,
            part_geometry_fields,
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
            if name in PART_QUANTITIES:
                parts.add(PART_QUANTITIES[name])
        descriptions = PART_DESCRIPTIONS
        options = self._fit_options
        data.update(
            definition="part_fitted_geometry",
            measurement="part_geometry",
            parts={},
            budget={"max_grounding_calls": len(parts), "max_sam_calls": 2 * len(parts)},
        )
        fits = {}
        images = []
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
            if part == "bottle_segment":
                sample["segmentation_query"] = (
                    f"the entire {data['query']}, including its main barrel, shoulder, neck and base"
                )
            for key in (
                "image_artifact",
                "world_artifact",
                "valid_artifact",
                "observation_artifact",
                "source_dtype",
            ):
                sample[key] = data[key]
            sample["world_frame_id"] = observation.world_frame_id
            data["parts"][part] = sample
            try:
                part_images = self._measure_surface(
                    sample, observation, part_tag, check_cancelled
                )
            finally:
                for name, count in sample["calls"].items():
                    data["calls"][name] += count
                data["segment_attempts"].extend(
                    {**attempt, "part": part} for attempt in sample["segment_attempts"]
                )
            images.extend(part_images[-1:])
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
            z = observation.optical_depth()
            if part == "inner_rim":
                candidate_mask, extraction = inner_rim_candidates(
                    mask & valid,
                    world,
                    anchor,
                    optical_depth=z,
                    options=options,
                    image=image,
                )
                sample["candidate_extraction"] = extraction
            elif part in PART_QUANTITIES.values():
                candidate_mask = mask.copy()
            else:
                candidate_mask = part_candidate_mask(
                    mask,
                    world,
                    image,
                    part,
                    camera_transform=camera_transform,
                    options=options,
                )
                if part != "bottom":
                    candidate_mask[row, col] = True
            candidate_mask &= valid
            candidate_points = world[candidate_mask]
            context_mask = binary_dilation(mask, iterations=5) & ~mask & valid
            cleaning_part = {
                "inner_rim": "rim",
                "plate_surface": "plane",
                "lid": "plane",
                "rectangular_top": "plane",
                "bottle_segment": "body",
            }.get(part, part)
            cleaned = clean_part_cloud(
                candidate_points,
                part=cleaning_part,
                anchor=anchor if part != "bottom" else None,
                context_points=world[context_mask],
                options=options,
                require_anchor_connection=part != "inner_rim",
                up_direction=observation.up_direction,
            )
            camera_origin = (
                None if camera_transform is None else camera_transform[:3, 3]
            )
            part_fit = None
            if part in PART_QUANTITIES.values():
                part_fit = fit_part_geometry(
                    cleaned.points,
                    part,
                    world,
                    mask & valid,
                    anchor,
                    camera_origin,
                    options,
                )
                fit = part_fit.fit
                if part == "inner_rim" and not candidate_mask.any():
                    fit.reason = sample["candidate_extraction"]["reason"]
            else:
                fit = fit_part(
                    cleaned.points,
                    part,
                    options=options,
                    anchor=anchor,
                    camera_origin=camera_origin,
                )
            check_cancelled()
            arrays = {
                "candidate_mask_artifact": (
                    f"geometry_candidates_mask_{part_tag}.npz",
                    candidate_mask,
                ),
                "candidate_points_artifact": (
                    f"geometry_candidates_{part_tag}.npz",
                    candidate_points,
                ),
                "cleaned_points_artifact": (
                    f"geometry_cleaned_{part_tag}.npz",
                    cleaned.points,
                ),
                "cleaned_indices_artifact": (
                    f"geometry_cleaned_indices_{part_tag}.npz",
                    cleaned.indices,
                ),
                "fit_inlier_indices_artifact": (
                    f"geometry_inliers_{part_tag}.npz",
                    fit.inlier_indices,
                ),
            }
            if part_fit is not None:
                sample["part_extraction"] = part_fit.extraction
                arrays.update(
                    {
                        "fit_points_artifact": (
                            f"geometry_fit_points_{part_tag}.npz",
                            part_fit.points,
                        ),
                        "observed_support_artifact": (
                            f"geometry_observed_support_{part_tag}.npz",
                            part_fit.observed_points,
                        ),
                        "part_boundary_artifact": (
                            f"geometry_part_boundary_{part_tag}.npz",
                            part_fit.boundary_mask,
                        ),
                    }
                )
            else:
                sample["fit_points_artifact"] = arrays["cleaned_points_artifact"][0]
            for key, (name, array) in arrays.items():
                if self._state.save(name, array, step=nn) is None:
                    raise RuntimeError(f"failed to persist geometry evidence {name}")
                sample[key] = name
            sample["cleaning"] = cleaned.quality
            sample["fit"] = fit.summary()
            fits[part] = fit
        fields, checks = geometry_fields(
            fits, [name for name in requested if name not in PART_QUANTITIES], options
        )
        fields.update(
            part_geometry_fields(
                fits, [name for name in requested if name in PART_QUANTITIES]
            )
        )
        fields = {name: fields[name] for name in requested}
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
            self._need_view(data, "requested_geometry_not_observable_or_fit_rejected")
            data["observation_request"]["missing_geometry"] = missing
            data["observation_request"]["missing_parts"] = [
                part
                for part in fits
                if any(part in fields[name]["source_parts"] for name in missing)
            ]
        return images
