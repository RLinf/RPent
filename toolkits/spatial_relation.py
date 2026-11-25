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

"""Measure how a target is positioned relative to a reference in metric 3D."""

import math
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, TypeAdapter

from rpent.session import EnvState
from rpent.tools import ToolResult, tool
from toolkits.observation import GeometryObservation, ObservationProvider
from toolkits.segment import MIN_MASK_METRIC_SAMPLES

Vector3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
FrameId = Annotated[str, Field(min_length=1)]


class SpatialPoint(BaseModel):
    """A point with the metric coordinate frame it belongs to."""

    model_config = ConfigDict(extra="forbid")
    xyz: Vector3
    world_frame_id: FrameId


class SegmentReference(BaseModel):
    """A public segmentation record at the requested observation step."""

    model_config = ConfigDict(extra="forbid")
    segment_artifact: Annotated[
        str, Field(pattern=r"^segment_[A-Za-z0-9_-]+\.json$", max_length=128)
    ]


class _SegmentRecord(BaseModel):
    found: bool
    camera: Annotated[str, Field(min_length=1)]
    source_step: int
    world_frame_id: FrameId
    image_shape: Annotated[list[int], Field(min_length=2, max_length=2)]
    mask_artifact: str | None


Endpoint = SpatialPoint | SegmentReference
_ENDPOINT = TypeAdapter(Endpoint)


@tool(readonly=True, exclude=("observations", "state"))
def spatial_relation(
    camera: Annotated[str, Field(min_length=1, max_length=128)],
    reference: Endpoint,
    target: Endpoint,
    forward_xyz: Vector3,
    step: Annotated[int, Field(ge=-1)] = -1,
    tolerance_m: Annotated[FiniteFloat, Field(ge=0)] = 0.001,
    *,
    observations: ObservationProvider,
    state: EnvState,
) -> ToolResult:
    """Measure target minus reference using a recorded world frame and its up axis.

    Returns Euclidean distance and signed forward, right and height offsets in
    metres. An offset within tolerance_m is labeled aligned. Mask endpoints are
    medians of valid visible surface samples. Check that each mask covers its target.

    Args:
        camera: Recorded camera that supplies the world frame and up direction.
        reference: Point with xyz and world_frame_id, or a segment_artifact from
            the specified step.
        target: Point or segmentation record to locate relative to the reference.
        forward_xyz: Desired forward axis in world coordinates. Its horizontal
            projection defines front. Right is forward cross up.
        step: Recorded step for the axes and mask inputs, or -1 for the latest.
        tolerance_m: Maximum absolute offset labeled aligned, 0.001 m by default.
        observations: Provider for recorded camera observations.
        state: Store for segmentation records and their masks.
    """
    try:
        view = _get_observation(observations, camera, step)
        data: dict[str, Any] = {
            "camera": view.camera,
            "step": view.step,
            "world_frame_id": view.world_frame_id,
            "units": "m",
        }
        up = view.up_direction
        if up is None:
            data.update(status="need_observation", reason="missing_up_direction")
            return ToolResult(data=data)
        forward, right = _horizontal_axes(forward_xyz, up)
        start = _resolve(
            _ENDPOINT.validate_python(reference), view, observations, state
        )
        end = _resolve(_ENDPOINT.validate_python(target), view, observations, state)
        data.update(reference=start, target=end)
        if start["xyz"] is None or end["xyz"] is None:
            data.update(
                status="need_observation", reason="insufficient_valid_mask_points"
            )
            return ToolResult(data=data)

        delta = [b - a for a, b in zip(start["xyz"], end["xyz"], strict=True)]
        distance = math.hypot(*delta)
        if not math.isfinite(distance):
            raise ValueError("Expected a finite distance between the supplied points.")
        forward_m, right_m, height_m = (
            float(axis @ delta) for axis in (forward, right, up)
        )
        data.update(
            status="measured",
            distance_m=distance,
            height_delta_m=height_m,
            forward_delta_m=forward_m,
            right_delta_m=right_m,
            forward_xyz=forward.tolist(),
            right_xyz=right.tolist(),
            up_xyz=up.tolist(),
            directions={
                "front_back": _direction(forward_m, tolerance_m, "back", "front"),
                "left_right": _direction(right_m, tolerance_m, "left", "right"),
                "above_below": _direction(height_m, tolerance_m, "below", "above"),
            },
        )
        return ToolResult(data=data)
    except (ValueError, LookupError, OSError, RuntimeError) as exc:
        return ToolResult(data={"status": "error", "error": str(exc)})


def _get_observation(
    observations: ObservationProvider, camera: str, step: int
) -> GeometryObservation:
    view = observations.get_observation(camera=camera, step=step)
    if view.camera != camera or (step >= 0 and view.step != step):
        raise ValueError("Provider returned a different camera or step.")
    return view


def _horizontal_axes(
    axis: list[float], up: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    forward = np.asarray(axis, dtype=float)
    scale = np.abs(forward).max()
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(
            "forward_xyz must be finite with a nonzero horizontal component."
        )
    right = np.cross(forward / scale, up)
    length = np.linalg.norm(right)
    if length < 1e-8:
        raise ValueError("forward_xyz must have a nonzero horizontal component.")
    right /= length
    return np.cross(up, right), right


def _direction(value: float, tolerance: float, negative: str, positive: str) -> str:
    if abs(value) <= tolerance:
        return "aligned"
    return positive if value > 0 else negative


def _resolve(
    endpoint: Endpoint,
    view: GeometryObservation,
    observations: ObservationProvider,
    state: EnvState,
) -> dict[str, Any]:
    if isinstance(endpoint, SpatialPoint):
        _require_frame(endpoint.world_frame_id, view.world_frame_id)
        return {"xyz": endpoint.xyz, "definition": "supplied_point"}

    name = endpoint.segment_artifact
    artifacts = state.get(view.step).artifacts
    if name not in artifacts:
        raise ValueError(
            f"Expected segment artifact {name!r} at recorded step {view.step}."
        )
    record = _SegmentRecord.model_validate(state.load(name, step=view.step))
    _require_frame(record.world_frame_id, view.world_frame_id)
    if record.source_step != view.step:
        raise ValueError(
            f"Segment artifact {name!r} refers to a different source step."
        )
    source = (
        view
        if record.camera == view.camera
        else _get_observation(observations, record.camera, view.step)
    )
    _require_frame(source.world_frame_id, view.world_frame_id)
    if record.image_shape != list(source.rgb.shape[:2]):
        raise ValueError(
            f"Segment artifact {name!r} does not match its recorded image shape."
        )
    point: dict[str, Any] = {
        "xyz": None,
        "camera": source.camera,
        "definition": "visible_mask_coordinate_median",
        "segment_artifact": name,
        "n_valid": 0,
    }
    if not record.found:
        return point
    mask_name = (
        "segment_mask_" + name.removeprefix("segment_").removesuffix(".json") + ".npz"
    )
    if record.mask_artifact != mask_name or mask_name not in artifacts:
        raise ValueError(f"Expected the saved mask for segment artifact {name!r}.")
    mask = state.load(mask_name, step=view.step)
    if mask.dtype != np.bool_ or mask.shape != source.rgb.shape[:2]:
        raise ValueError(f"Expected a boolean mask matching the image for {name!r}.")
    valid = mask & source.valid_mask
    point["n_valid"] = int(valid.sum())
    if point["n_valid"] >= MIN_MASK_METRIC_SAMPLES:
        xyz = np.median(source.world_xyz[valid], axis=0)
        if not np.isfinite(xyz).all():
            raise ValueError(f"Expected a finite visible surface median for {name!r}.")
        point["xyz"] = xyz.tolist()
    return point


def _require_frame(actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"Expected world frame {expected!r}, got {actual!r}.")
