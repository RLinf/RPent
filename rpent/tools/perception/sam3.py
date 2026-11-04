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

"""SAM3 segmentation and world localization of recorded camera images."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rpent.tools.perception.perception_base import PerceptionToolBase

if TYPE_CHECKING:
    from rpent.robots.components.sam3_client import Sam3Client, Sam3Result
    from rpent.session import EnvState


class Sam3Tools(PerceptionToolBase):
    """Segment recorded camera images with SAM3 and localize the masks."""

    _client: Sam3Client

    def segment(
        self,
        state: EnvState,
        *,
        step: int,
        image: str,
        prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> tuple[Sam3Result, np.ndarray | None]:
        """Segment the ``image`` artifact recorded at ``step``.

        Provide exactly one of ``prompt`` or a ``[row, col]`` ``point``. Errors
        from the image read or the segmentation service propagate.

        Returns:
            The top-ranked SAM3 result, and the image with the mask tinted red,
            or ``None`` without a mask or when their shapes differ.
        """
        result = self._client.segment(
            state.load_bytes(image, step=step),
            text_prompt=prompt,
            point=point,
            min_score=min_score,
        )
        mask = result.mask
        if not (result.found and isinstance(mask, np.ndarray)):
            return result, None
        rgb = state.load(image, step=step)
        if rgb.ndim != 3 or rgb.shape[:2] != mask.shape:
            return result, None
        overlay = rgb.copy()
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        overlay[mask] = (
            0.55 * overlay[mask].astype(np.float32)
            + 0.45 * red[mask].astype(np.float32)
        ).astype(np.uint8)
        return result, overlay

    def mask_to_world(
        self, state: EnvState, *, step: int, world_map: str, mask: np.ndarray
    ) -> dict[str, Any]:
        """Return the median world XYZ of ``mask`` from a recorded world map.

        ``world_map`` names an HxWx3 world-XYZ artifact in meters, recorded at
        ``step`` and pixel-aligned with the segmented image. ``world_xyz`` is
        ``None`` with a ``world_error`` when the map is unavailable or invalid,
        or fewer than 10 mask pixels have finite, nonzero coordinates.
        """
        try:
            world = state.load(world_map, step=step)
        except Exception as exc:
            return {
                "world_xyz": None,
                "world_error": f"world map artifact not available: {exc}",
                "expected_world_artifact": world_map,
            }
        if world.ndim != 3 or world.shape[2] < 3:
            return {
                "world_xyz": None,
                "world_error": f"invalid world map shape: {tuple(world.shape)}",
                "n_pixels": int(mask.sum()),
                "n_valid": 0,
                "mask_resized_to_world_shape": False,
                "world_artifact": world_map,
            }
        if mask.shape != world.shape[:2]:
            return {
                "world_xyz": None,
                "world_error": (
                    f"mask/world shape mismatch: mask={tuple(mask.shape)}, "
                    f"world={tuple(world.shape[:2])}"
                ),
                "n_pixels": int(mask.sum()),
                "n_valid": 0,
                "mask_resized_to_world_shape": False,
                "world_artifact": world_map,
            }

        ys, xs = np.where(mask)
        if ys.size == 0:
            return {
                "world_xyz": None,
                "world_error": "empty mask",
                "world_artifact": world_map,
            }
        pts = world[ys, xs].astype(np.float64)
        valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-6)
        pts = pts[valid]
        result = {
            "centroid_pixel": [
                int(round(float(np.median(xs)))),
                int(round(float(np.median(ys)))),
            ],
            "n_pixels": int(mask.sum()),
            "n_valid": int(pts.shape[0]),
            "mask_resized_to_world_shape": False,
        }
        if pts.shape[0] < 10:
            result["world_xyz"] = None
            result["world_error"] = f"too few valid depth pixels ({int(pts.shape[0])})"
        else:
            result["world_xyz"] = [
                round(float(np.median(pts[:, 0])), 4),
                round(float(np.median(pts[:, 1])), 4),
                round(float(np.median(pts[:, 2])), 4),
            ]
        result["world_artifact"] = world_map
        return result
