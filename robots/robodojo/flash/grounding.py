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

"""Shared recording/replay anchor derivation from a SAM3 binary mask."""

import numpy as np

ANCHOR_METHOD = "sam3_mask_centroid_floor_v1"


def mask_geometry(mask) -> dict:
    """Record sufficient mask moments to verify the integer centroid pixel."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Anchor requires a nonempty two-dimensional mask")
    rows, cols = np.nonzero(mask)
    count = int(rows.size)
    moments = [count, int(rows.sum()), int(cols.sum())]
    return {
        "anchor_method": ANCHOR_METHOD,
        "mask_shape": list(mask.shape),
        "mask_moments": moments,
        "centroid_rc": [moments[1] // count, moments[2] // count],
    }


def centroid_pixel(result: dict) -> tuple[int, int]:
    """Validate recorded/live mask moments and reject a changed centroid."""
    if result.get("anchor_method") != ANCHOR_METHOD:
        raise ValueError("Unsupported anchor derivation; record mask centroids again")
    shape, moments = result.get("mask_shape"), result.get("mask_moments")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(type(v) is int and v > 0 for v in shape)
        or not isinstance(moments, list)
        or len(moments) != 3
        or not all(type(v) is int and v >= 0 for v in moments)
    ):
        raise ValueError("Invalid mask moments")
    count, row_sum, col_sum = moments
    if not 1 <= count <= shape[0] * shape[1]:
        raise ValueError("Invalid mask area")
    if row_sum > count * (shape[0] - 1) or col_sum > count * (shape[1] - 1):
        raise ValueError("Mask moments outside image")
    pixel = [row_sum // count, col_sum // count]
    if result.get("centroid_rc") != pixel:
        raise ValueError("Mask centroid does not match mask moments")
    return pixel[0], pixel[1]
