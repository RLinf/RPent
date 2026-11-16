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

"""Read measured surfaces from a provider's calibrated image."""

from typing import Annotated, Any

import numpy as np
from pydantic import Field

from rpent.tools import ToolResult, tool
from toolkits.observation import (
    GeometryObservation,
    ImageSelection,
    ObservationProvider,
)


@tool(readonly=True, exclude=("observations",))
def back_project(
    camera: Annotated[str, Field(min_length=1, max_length=128)],
    selection: ImageSelection,
    step: Annotated[int, Field(ge=-1)] = -1,
    *,
    observations: ObservationProvider,
) -> ToolResult:
    """Read metric surface coordinates at known pixels without model inference.

    Use the labeled full-frame image_shape [height,width] and either points_rc
    (up to 16 row,col pairs in order) or a half-open box_rc [top,left,bottom,right].
    Coordinates lie inside the declared image. The exclusive bottom and right
    may equal its height and width. For either selection, height_m [low,high]
    filters the world coordinate along the unit up axis in metres from the
    frame origin. A filtered or invalid point keeps its slot as null.
    Returned pixel indices and image_shape refer to the native image.
    The result describes sampled surfaces. Confirm the intended object in the
    current images before using its coordinates.
    """
    try:
        selection = ImageSelection.model_validate(selection)
        observation = observations.get_observation(camera=camera, step=step)
        if observation.camera != camera or (step >= 0 and observation.step != step):
            raise ValueError("Provider returned a different camera or step.")
        data = measure_selection(observation, selection)
    except (ValueError, LookupError, OSError, RuntimeError) as exc:
        return ToolResult(data={"status": "error", "error": str(exc)})
    return ToolResult(data=data)


def _region_summary(points: np.ndarray) -> dict:
    if len(points) < 8:
        return {
            "status": "need_observation",
            "reason": "insufficient_valid_surface_samples",
        }
    # Report a real observed point. A midpoint can land inside an empty cavity.
    center = np.median(points, axis=0)
    surface = points[np.argmin(np.sum((points - center) ** 2, axis=1))]
    return {
        "status": "sampled",
        "surface_xyz": surface.tolist(),
        "visible_bounds_xyz": [
            points.min(axis=0).tolist(),
            points.max(axis=0).tolist(),
        ],
        "n_valid": len(points),
    }


def measure_selection(obs: GeometryObservation, selection: ImageSelection) -> dict:
    """Read depth samples at the selected pixels of the pinned camera."""
    points, box = selection.native(obs)
    result: dict[str, Any] = {
        "definition": "selected_surface_only",
        "evidence_status": "identity_unchecked",
        "image_shape": list(obs.rgb.shape[:2]),
        "selection_image_shape": selection.image_shape,
        "world_frame_id": obs.world_frame_id,
        "step": obs.step,
        "camera": obs.camera,
        "units": "m",
    }
    if box is not None:
        r0, c0, r1, c1 = box
        indices = (slice(r0, r1), slice(c0, c1))
    else:
        rows, cols = np.asarray(points).T
        indices = (rows, cols)
    samples = obs.world_xyz[indices]
    valid = obs.valid_mask[indices]
    if selection.height_m is not None:
        if obs.up_direction is None:
            raise ValueError("A height interval requires the provider's up direction.")
        low, high = selection.height_m
        heights = samples[valid] @ obs.up_direction
        keep = (heights >= (-np.inf if low is None else low)) & (
            heights <= (np.inf if high is None else high)
        )
        result.update(height_m=selection.height_m, n_height_filtered=int((~keep).sum()))
        # Preserve the positions of invalid and filtered pixels in point batches.
        within_height = np.zeros_like(valid)
        within_height[valid] = keep
        valid = within_height
    if box is not None:
        result.update(_region_summary(samples[valid]), box_rc=list(box))
    else:
        result["points_rc"] = [list(p) for p in points]
        result["samples_xyz"] = [
            sample.tolist() if ok else None for sample, ok in zip(samples, valid)
        ]
        if valid.any():
            result["status"] = "sampled"
        else:
            result.update(
                status="need_observation",
                reason="no_selected_points_in_height_interval"
                if result.get("n_height_filtered")
                else "no_valid_3d_at_selected_pixels",
            )
    return result
