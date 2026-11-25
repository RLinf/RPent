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

"""Propose geometry-only parallel-jaw grasps on recorded visible surfaces."""

from typing import Annotated, Any

import numpy as np
from pydantic import Field, FiniteFloat

from rpent.session import EnvState
from rpent.tools import ToolResult, tool
from toolkits.observation import GeometryObservation, ObservationProvider

_MIN_POINTS = 20
_MAX_POINTS = 4096
_YAW_SAMPLES = 12


@tool(readonly=True, exclude=("observations", "state"))
def estimate_grasp_pose(
    camera: Annotated[str, Field(min_length=1, max_length=128)],
    segment_artifact: Annotated[
        str, Field(pattern=r"^segment_[A-Za-z0-9_-]+\.json$", max_length=128)
    ],
    step: Annotated[int, Field(ge=-1)] = -1,
    top_k: Annotated[int, Field(strict=True, ge=1, le=6)] = 3,
    approach_xyz: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
    | None = None,
    max_opening_m: Annotated[FiniteFloat, Field(gt=0)] | None = None,
    jaw_margin_m: Annotated[FiniteFloat, Field(ge=0)] = 0.002,
    *,
    observations: ObservationProvider,
    state: EnvState,
) -> ToolResult:
    """Estimate parallel-jaw pose candidates from a recorded segmentation mask.

    All positions and axes use the recorded world frame. Rotation matrix columns
    are closing, lateral and approach axes, with approach pointing toward the
    target. Each centre moves along its closing axis toward the midpoint of the
    observed limits. If that shift exceeds the inlier scale, the visible-surface
    centre is retained and the opening is increased to cover all observed
    points. Evaluate contact, reachability and collision separately.

    Args:
        camera: Camera used to create the segmentation.
        segment_artifact: Public segment record returned at the requested step.
        step: Recorded segmentation step, or -1 for the latest record.
        top_k: Maximum number of candidates to return, 3 by default.
        approach_xyz: World-frame direction of approach toward the target.
            Defaults to the negative up direction from the observation.
        max_opening_m: Optional gripper opening limit in metres.
        jaw_margin_m: Added opening on each side of the visible span in metres.
        observations: Provider for recorded camera observations.
        state: Store for the segment record and mask.
    """
    try:
        view = observations.get_observation(camera=camera, step=step)
        if view.camera != camera or (step >= 0 and view.step != step):
            raise ValueError("Provider returned a different camera or step.")
        mask = _load_segment_mask(view, segment_artifact, state)
        data: dict[str, Any] = {
            "camera": view.camera,
            "step": view.step,
            "world_frame_id": view.world_frame_id,
            "segment_artifact": segment_artifact,
            "units": "m",
        }
        if mask is None:
            return ToolResult(data={**data, **_uncertain("missing_segment_mask")})
        points = view.world_xyz[mask & view.valid_mask]
        data["n_valid"] = len(points)
        if len(points) < _MIN_POINTS:
            return ToolResult(data={**data, **_uncertain("insufficient_valid_points")})
        if approach_xyz is None and view.up_direction is None:
            return ToolResult(data={**data, **_uncertain("missing_approach_direction")})
        approach = -view.up_direction if approach_xyz is None else _unit(approach_xyz)
        data.update(
            _sample_grasps(points, approach, top_k, max_opening_m, jaw_margin_m)
        )
        return ToolResult(data=data)
    except (
        ValueError,
        LookupError,
        OSError,
        RuntimeError,
        np.linalg.LinAlgError,
    ) as exc:
        return ToolResult(data={"status": "error", "error": str(exc)})


def _load_segment_mask(
    view: GeometryObservation, name: str, state: EnvState
) -> np.ndarray | None:
    artifacts = state.get(view.step).artifacts
    if (
        name not in artifacts
        or not name.startswith("segment_")
        or not name.endswith(".json")
    ):
        raise ValueError(
            f"Expected a public segment artifact at recorded step {view.step}."
        )
    record = state.load(name, step=view.step)
    if not isinstance(record, dict) or (
        record.get("camera"),
        record.get("source_step"),
        record.get("world_frame_id"),
        record.get("image_shape"),
    ) != (view.camera, view.step, view.world_frame_id, list(view.rgb.shape[:2])):
        raise ValueError(
            f"Segment artifact {name!r} does not match the recorded observation."
        )
    if record.get("found") is False:
        return None
    mask_name = (
        "segment_mask_" + name.removeprefix("segment_").removesuffix(".json") + ".npz"
    )
    if record.get("found") is not True or record.get("mask_artifact") != mask_name:
        raise ValueError(
            f"Expected the saved mask reference for segment artifact {name!r}."
        )
    if mask_name not in artifacts:
        raise ValueError(f"Expected the saved mask {mask_name!r} at step {view.step}.")
    mask = state.load(mask_name, step=view.step)
    if mask.dtype != np.bool_ or mask.shape != view.rgb.shape[:2]:
        raise ValueError(f"Expected a boolean mask matching the image for {name!r}.")
    return mask


def _unit(axis: list[float]) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    scale = np.abs(vector).max()
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Expected a finite, nonzero approach_xyz vector.")
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def _uncertain(reason: str) -> dict[str, Any]:
    return {"status": "need_observation", "reason": reason, "candidates": []}


def _sample_grasps(
    points: np.ndarray,
    approach: np.ndarray,
    top_k: int,
    max_opening: float | None,
    margin: float,
) -> dict[str, Any]:
    sample = points
    if len(points) > _MAX_POINTS:
        indices = np.random.default_rng(0).choice(
            len(points), _MAX_POINTS, replace=False
        )
        sample = points[indices]
    center = np.median(sample, axis=0)
    cloud = sample - center
    radii = np.linalg.norm(cloud, axis=1)
    q1, q3 = np.quantile(radii, [0.25, 0.75])
    inlier_radius = q3 + 3 * (q3 - q1)
    cloud = cloud[radii <= inlier_radius]
    data: dict[str, Any] = {"n_used": len(cloud)}
    if len(cloud) < _MIN_POINTS:
        return {**data, **_uncertain("insufficient_inliers")}

    shift = np.median(cloud, axis=0)
    center = center + shift
    cloud = cloud - shift
    axis = np.eye(3)[np.argmin(np.abs(approach))]
    first = axis - (axis @ approach) * approach
    first /= np.linalg.norm(first)
    basis = np.column_stack((first, np.cross(approach, first)))
    projected = cloud @ basis
    variance, axes = np.linalg.eigh(np.cov(projected, rowvar=False))
    if variance[0] <= max(1e-14, variance[1] * 1e-6):
        return {**data, **_uncertain("degenerate_projected_cloud")}

    # Break the PCA sign ambiguity and use a fixed yaw grid for symmetric clouds.
    symmetric = bool(variance[0] >= 0.9 * variance[1])
    seed = first if symmetric else basis @ axes[:, 0]
    if seed[np.argmax(np.abs(seed))] < 0:
        seed = -seed
    across = np.cross(approach, seed)
    data["pca_yaw_ambiguous"] = symmetric
    spans = []
    for angle in np.arange(_YAW_SAMPLES) * np.pi / _YAW_SAMPLES:
        closing = np.cos(angle) * seed + np.sin(angle) * across
        low, high = np.quantile(cloud @ closing, [0.02, 0.98])
        width = float(high - low)
        if width > 0:
            spans.append((width, closing))
    if not spans:
        return {**data, **_uncertain("degenerate_projected_cloud")}

    spans.sort(key=lambda pair: pair[0])
    narrowest = spans[0][0]
    candidates = []
    for width, closing in spans:
        # Rare real protrusions can disappear during PCA sampling or outlier trimming.
        full_projection = np.einsum("ij,j->i", points, closing)
        low, high = float(full_projection.min()), float(full_projection.max())
        span = high - low
        center_projection = float(center @ closing)
        closing_offset = low + span / 2 - center_projection
        if abs(closing_offset) > inlier_radius:
            # A distant return can put the midpoint outside the visible object.
            closing_offset = 0.0
            span = 2 * max(center_projection - low, high - center_projection)
        opening = span + 2 * margin
        if not np.isfinite(opening):
            raise ValueError("Expected a finite gripper opening including jaw margins.")
        if max_opening is not None and opening > max_opening:
            continue
        candidate_center = center + closing_offset * closing
        rotation = np.column_stack((closing, np.cross(approach, closing), approach))
        candidates.append(
            {
                "center_xyz": candidate_center.tolist(),
                "rotation_matrix": rotation.tolist(),
                "estimated_width_m": width,
                "suggested_opening_m": opening,
                "score": narrowest / width,
            }
        )
        if len(candidates) == top_k:
            break
    if not candidates:
        data.update(status="no_candidate", reason="opening_limit", candidates=[])
        return data
    return {**data, "status": "estimated", "candidates": candidates}
