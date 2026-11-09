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

"""Segmentation and metric coordinate reads for explicit observation snapshots."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Protocol

import numpy as np

from rpent.session import EnvState
from rpent.tools import ToolResult
from toolkits.geometry.observation import GeometryObservation

if TYPE_CHECKING:
    from rpent.robots.components.molmo_client import MolmoResult
    from rpent.robots.components.sam3_client import Sam3Result


class VisualGrounder(Protocol):
    """Select an original-image column/row point from RGB and a description."""

    def ground(self, image: bytes, query: str) -> MolmoResult:
        """Return a visible point or an absent target."""
        ...


class Segmenter(Protocol):
    """Produce masks using text or a positive row/column point prompt."""

    def segment(
        self,
        image: bytes,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> Sam3Result:
        """Return a mask at the supplied RGB resolution, or no candidate."""
        ...


def back_project(observation: GeometryObservation, row: int, col: int) -> ToolResult:
    """Read an aligned metric point without origin, height or depth-range heuristics."""
    height, width = observation.rgb.shape[:2]
    if not (0 <= row < height and 0 <= col < width):
        return ToolResult(data={"error": "selected pixel is outside the observation"})
    if not observation.valid_mask[row, col]:
        return ToolResult(data={"error": "selected pixel has no valid 3-D observation"})
    return ToolResult(
        data={
            "camera": observation.camera,
            "step": observation.step,
            "world_frame_id": observation.world_frame_id,
            "units": "m",
            "pixel_rc": [row, col],
            "world_xyz": observation.world_xyz[row, col].tolist(),
        }
    )


class GeometryPerception:
    """Own an injected segmenter and its evidence sequence; never move the robot."""

    def __init__(self, segmenter: Segmenter | None) -> None:
        self._segmenter = segmenter
        self._lock = threading.Lock()
        self._index = 0

    def segment(
        self,
        observation: GeometryObservation,
        image: bytes,
        *,
        state: EnvState,
        tag: str,
        prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> ToolResult:
        """Segment a pinned snapshot and save the mask with its source identity."""
        if bool(prompt and prompt.strip()) == (point is not None):
            raise ValueError("segment needs exactly one text or point prompt")
        if self._segmenter is None:
            raise ValueError("region geometry requires an injected segmenter")
        result = self._segmenter.segment(
            image, text_prompt=prompt, point=point, min_score=min_score
        )
        with self._lock:
            index = self._index
            self._index += 1
        step = observation.step
        data = {
            "found": result.found,
            "camera": observation.camera,
            "source_step": step,
            "world_frame_id": observation.world_frame_id,
            "mode": "text" if prompt else "point",
            "prompt": prompt,
            "point": point,
            "min_score": min_score,
            "score": None if result.score is None else float(result.score),
            "mask_artifact": None,
        }
        images = []
        if result.found:
            mask = np.asarray(result.mask)
            if mask.dtype != np.bool_ or mask.shape != observation.rgb.shape[:2]:
                raise ValueError("segmentation mask must be boolean and match RGB")
            mask_name = f"geometry_mask_{tag}_{index:03d}.npz"
            data["mask_artifact"] = state.save(mask_name, mask, step=step)
            if data["mask_artifact"] is None:
                raise RuntimeError("failed to persist segmentation mask")
            overlay = observation.rgb.copy()
            red = np.zeros_like(overlay)
            red[..., 0] = 255
            overlay[mask] = (0.55 * overlay[mask] + 0.45 * red[mask]).astype(np.uint8)
            overlay_name = f"geometry_overlay_{tag}_{index:03d}.png"
            data["overlay_artifact"] = state.save(overlay_name, overlay, step=step)
            if data["overlay_artifact"] is None:
                raise RuntimeError("failed to persist segmentation overlay")
            images.append(state.load_bytes(overlay_name, step=step))
        else:
            data["reason"] = result.reason or "segmenter found no mask"
        name = f"geometry_segment_{tag}_{index:03d}.json"
        if state.save(name, data, step=step) is None:
            raise RuntimeError("failed to persist segmentation evidence")
        data["segment_artifact"] = name
        return ToolResult(data=data, images=images)
