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

"""Compare RGB and depth between two recorded views from a fixed camera."""

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat

from rpent.tools import ToolResult, tool
from toolkits.observation import GeometryObservation, ObservationProvider


@tool(readonly=True, exclude=("observations",))
def image_diff(
    camera: Annotated[str, Field(min_length=1, max_length=128)],
    before_step: Annotated[int, Field(ge=0)],
    after_step: Annotated[int, Field(ge=-1)] = -1,
    mode: Literal["rgb", "depth", "both"] = "both",
    rgb_threshold: Annotated[int, Field(strict=True, ge=1, le=255)] = 20,
    depth_threshold_m: Annotated[FiniteFloat, Field(gt=0)] = 0.01,
    min_depth_coverage: Annotated[FiniteFloat, Field(ge=0, le=1)] = 0.1,
    *,
    observations: ObservationProvider,
) -> ToolResult:
    """Measure visible changes between two recorded frames from a fixed camera.

    Each compared modality reports changed_ratio and a tight bbox_rc
    [top,left,bottom,right] with exclusive bottom and right, or null if no
    pixels changed. RGB compares every pixel. Depth compares the pixels with
    valid depth in both frames and reports their fraction as valid_ratio. If only
    RGB is usable in both mode, status is partial. A missing or changed camera
    pose prevents comparison. Confirm the task outcome separately.

    Args:
        camera: Name of the recorded camera, preferably a fixed view.
        before_step: Explicit recorded step before the action.
        after_step: A later recorded step, or -1 for the latest record.
        mode: Compare RGB, optical-axis depth, or both (default).
        rgb_threshold: Minimum absolute change in any 8-bit color channel,
            20 by default.
        depth_threshold_m: Minimum absolute depth change in metres,
            0.01 by default.
        min_depth_coverage: Minimum fraction of the full image with valid
            depth in both frames, 0.1 by default.
        observations: Provider for recorded camera observations.
    """
    try:
        before = observations.get_observation(camera=camera, step=before_step)
        after = observations.get_observation(camera=camera, step=after_step)
        if (
            before.camera != camera
            or after.camera != camera
            or before.step != before_step
            or (after_step >= 0 and after.step != after_step)
        ):
            raise ValueError("Provider returned a different camera or step.")
        if after.step <= before.step:
            raise ValueError(
                f"Expected a later after_step than {before.step}, got {after.step}."
            )
    except (ValueError, LookupError, OSError, RuntimeError) as exc:
        return ToolResult(data={"status": "error", "error": str(exc)})

    result: dict[str, Any] = {
        "status": "compared",
        "camera": camera,
        "before_step": before.step,
        "after_step": after.step,
        "image_shape": list(before.rgb.shape[:2]),
    }
    if (issue := _alignment_issue(before, after)) is not None:
        result.update(status="need_observation", reason=issue)
        return ToolResult(data=result)

    if mode in ("rgb", "both"):
        delta = np.subtract(after.rgb, before.rgb, dtype=np.int16)
        changed = (np.abs(delta) >= rgb_threshold).any(axis=-1)
        result["rgb"] = _summarize_change(changed, changed.size)

    if mode in ("depth", "both"):
        depth = _compare_depth(before, after, depth_threshold_m, min_depth_coverage)
        result["depth"] = depth
        if depth["status"] == "need_observation":
            result["status"] = "partial" if mode == "both" else "need_observation"
            result["reason"] = depth["reason"]
    return ToolResult(data=result)


def _alignment_issue(
    before: GeometryObservation, after: GeometryObservation
) -> str | None:
    if before.rgb.shape != after.rgb.shape:
        return "different_image_shape"
    if before.world_frame_id != after.world_frame_id:
        return "different_world_frame"
    if before.camera_to_world is None or after.camera_to_world is None:
        return "missing_camera_pose"
    # Ignore numeric noise in stored poses while checking for a fixed view.
    if not np.allclose(
        before.camera_to_world, after.camera_to_world, rtol=0, atol=1e-5
    ):
        return "camera_pose_changed"
    if before.intrinsics is None and after.intrinsics is None:
        return None
    if before.intrinsics is None or after.intrinsics is None:
        return "incomplete_camera_intrinsics"
    if not np.allclose(before.intrinsics, after.intrinsics, rtol=0, atol=1e-5):
        return "camera_intrinsics_changed"
    return None


def _compare_depth(
    before: GeometryObservation,
    after: GeometryObservation,
    threshold_m: float,
    min_coverage: float,
) -> dict[str, Any]:
    valid = before.valid_mask & after.valid_mask
    count = int(np.count_nonzero(valid))
    coverage = count / valid.size
    if count == 0 or coverage < min_coverage:
        return {
            "status": "need_observation",
            "reason": "insufficient_common_depth",
            "compared_pixels": count,
            "valid_ratio": coverage,
        }

    before_depth, after_depth = before.optical_depth(), after.optical_depth()
    assert before_depth is not None and after_depth is not None, (
        "Expected optical depth from both camera poses."
    )
    changed = np.zeros(valid.shape, dtype=bool)
    earlier, later = before_depth[valid], after_depth[valid]
    delta = np.abs(later - earlier)
    # An inclusive threshold must account for rounding at the scale of the depths.
    roundoff = np.finfo(delta.dtype).eps * np.maximum(earlier, later)
    changed[valid] = (delta > 0) & (delta >= threshold_m - roundoff)
    return {**_summarize_change(changed, count), "valid_ratio": coverage}


def _summarize_change(changed: np.ndarray, compared_pixels: int) -> dict[str, Any]:
    count = int(np.count_nonzero(changed))
    bbox = None
    if count:
        rows = np.flatnonzero(changed.any(axis=1))
        cols = np.flatnonzero(changed.any(axis=0))
        bbox = [int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1]
    return {
        "status": "compared",
        "compared_pixels": compared_pixels,
        "changed_pixels": count,
        "changed_ratio": count / compared_pixels,
        "bbox_rc": bbox,
    }
