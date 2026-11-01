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

"""Offline tests for the reconstruction metadata contract."""

from __future__ import annotations

import numpy as np

from rpent.robots.components.lingbotmap_server import _build_gct_metadata
from rpent.tools.ground_map import PreprocessSpec

FRAMES = 2


def _metadata(
    source_size: tuple[int, int], input_size: tuple[int, int]
) -> dict[str, object]:
    """Build metadata for a reconstruction whose model grid is ``input_size``."""
    return _build_gct_metadata(
        map_name="scene",
        extrinsic=np.tile(np.eye(3, 4), (FRAMES, 1, 1)),
        intrinsic=np.tile(np.eye(3), (FRAMES, 1, 1)),
        source_sizes=[source_size] * FRAMES,
        timestamps=[float(index) for index in range(FRAMES)],
        poses=[None] * FRAMES,
        keyframe_interval=1,
        points=None,
        input_size=input_size,
        alignment=None,
        alignment_reason="no base poses",
    )


def test_landscape_map_records_the_depth_grid() -> None:
    source = (480, 640)
    depth_grid = (392, 518)
    metadata = _metadata(source, depth_grid)

    assert metadata["preprocess"]["input_size"] == [392, 518]

    spec = PreprocessSpec.from_metadata(metadata)
    assert spec is not None
    mask = np.zeros(source, dtype=bool)
    mask[200:280, 200:440] = True
    warped = spec.source_mask_to_model(mask, source)
    assert warped.shape == depth_grid
    assert warped.any()


def test_tall_source_records_the_square_grid() -> None:
    metadata = _metadata((960, 540), (518, 518))

    assert metadata["preprocess"]["input_size"] == [518, 518]
