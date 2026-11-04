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

"""Offline contracts for shared perception helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from rpent.tools.perception import Sam3Tools

MASK = np.array([[True, False], [False, False]])


def _state(tmp_path):
    state = EnvState(tmp_path / "state")
    with state.record_step(state={}):
        state.save("front.png", np.full((2, 2, 3), 100, dtype=np.uint8))
    return state


def test_requires_a_segmentation_client():
    with pytest.raises(ValueError):
        Sam3Tools(None)


def test_segments_the_recorded_image_and_tints_the_mask(tmp_path):
    state = _state(tmp_path)
    calls = []

    def segment(image, **kwargs):
        calls.append((image, kwargs))
        return Sam3Result(found=True, mask=MASK, score=0.9)

    result, overlay = Sam3Tools(SimpleNamespace(segment=segment)).segment(
        state, step=0, image="front.png", point=[0, 0], min_score=0.5
    )

    assert calls == [
        (
            state.load_bytes("front.png", step=0),
            {"text_prompt": None, "point": [0, 0], "min_score": 0.5},
        )
    ]
    assert result.score == 0.9
    assert overlay[0, 0].tolist() == [169, 55, 55]
    assert overlay[1, 1].tolist() == [100, 100, 100]


@pytest.mark.parametrize(
    "result",
    [
        Sam3Result(found=False, reason="no mask"),
        Sam3Result(found=True, mask=np.ones((3, 3), dtype=bool)),
    ],
)
def test_returns_no_overlay_without_a_matching_mask(tmp_path, result):
    sam3 = Sam3Tools(SimpleNamespace(segment=lambda image, **kwargs: result))

    assert sam3.segment(_state(tmp_path), step=0, image="front.png", prompt="cup") == (
        result,
        None,
    )


def test_localizes_a_mask_through_the_recorded_world_map(tmp_path):
    state = _state(tmp_path)
    world = np.zeros((4, 4, 3))
    world[:3, :4] = [0.1, 0.2, 0.3]
    world[0, :2] = [9.0, 9.0, 9.0]
    state.save("front_world.npz", world, step=0)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:3] = True
    sam3 = Sam3Tools(SimpleNamespace())

    located = sam3.mask_to_world(state, step=0, world_map="front_world.npz", mask=mask)
    missing = sam3.mask_to_world(state, step=0, world_map="side_world.npz", mask=mask)

    assert located["world_xyz"] == [0.1, 0.2, 0.3]
    assert located["n_valid"] == 12
    assert located["world_artifact"] == "front_world.npz"
    assert missing["world_xyz"] is None
    assert missing["expected_world_artifact"] == "side_world.npz"
