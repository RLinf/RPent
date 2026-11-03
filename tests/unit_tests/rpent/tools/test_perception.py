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

"""Offline contracts for shared perception tools."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from rpent.tools import Toolkit, iter_tools
from rpent.tools.common import CommonTools
from rpent.tools.perception import CameraView, Sam3Tools, with_cameras

FRONT_HIGH = CameraView("front_high.png", "front_world_high.npz")
FRONT = CameraView("front.png", "front_world.npz")
SIDE = CameraView("side.png", "side_world.npz")
CAMERAS = {"front": (FRONT_HIGH, FRONT), "side": (SIDE,)}
FALLBACK = "Use the robot's manual localization."


def _found(image, **kwargs):
    return Sam3Result(found=True, mask=np.ones((4, 4), dtype=bool), score=0.9)


def _toolkit(tmp_path, state):
    return Toolkit(
        dashboard_events=NullDashboardEventSink(),
        state=state,
        memory=MemoryManager(tmp_path / "memory"),
    )


def _sam3(state, segment=_found, cameras=CAMERAS):
    return Sam3Tools(
        SimpleNamespace(segment=segment),
        state=state,
        cameras=cameras,
        fallback=FALLBACK,
    )


def _sam3_toolkit(tmp_path, state, segment=_found, cameras=CAMERAS):
    toolkit = _toolkit(tmp_path, state)
    for definition in iter_tools(_sam3(state, segment, cameras)):
        toolkit.add_tool(with_cameras(definition, cameras))
    return toolkit


def _state(tmp_path, *views):
    state = EnvState(tmp_path / "state")
    with state.record_step(state={}):
        for view in views:
            state.save(view.image, np.zeros((4, 4, 3), dtype=np.uint8))
            state.save(view.world_map, np.full((4, 4, 3), 0.5))
    return state


def test_toolkit_has_no_segment_tool_until_a_robot_opts_in(tmp_path, monkeypatch):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    state = _state(tmp_path, FRONT)
    names = {item.name for item in _toolkit(tmp_path, state).list_tools()}

    assert "segment" not in names
    assert "segment" in {
        item.name for item in _sam3_toolkit(tmp_path, state).list_tools()
    }


def test_with_cameras_publishes_robot_cameras_without_runtime_resources():
    assert [item.name for item in iter_tools(Sam3Tools)] == ["segment"]
    declaration = with_cameras(Sam3Tools.segment, ("front", "side"))
    camera = declaration.input_schema["properties"]["camera"]

    assert declaration.readonly
    assert camera["enum"] == ["front", "side"]
    assert camera["description"] == "Artifact camera to use (default front)."


@pytest.mark.parametrize(
    ("declaration", "cameras"),
    [(Sam3Tools.segment, ()), (CommonTools.finish, ("front",))],
)
def test_with_cameras_rejects_unusable_inputs(declaration, cameras):
    with pytest.raises(ValueError):
        with_cameras(declaration, cameras)


@pytest.mark.parametrize(
    ("segmenter", "cameras"),
    [
        (None, CAMERAS),
        (SimpleNamespace(), {}),
        (SimpleNamespace(), {"front": ()}),
        (SimpleNamespace(), {"front": (("front.png", "front_world.npz"),)}),
    ],
)
def test_rejects_unusable_configuration_at_construction(tmp_path, segmenter, cameras):
    with pytest.raises(ValueError):
        Sam3Tools(
            segmenter,
            state=EnvState(tmp_path / "state"),
            cameras=cameras,
            fallback=FALLBACK,
        )


def test_selects_the_first_complete_view_and_localizes_the_mask(tmp_path):
    state = _state(tmp_path, FRONT, FRONT_HIGH)
    result = _sam3_toolkit(tmp_path, state).execute_tool("segment", {"prompt": "cup"})

    assert result.data["image_artifact"] == "front_high.png"
    assert result.data["world_xyz"] == [0.5, 0.5, 0.5]
    assert result.data["found"] is True
    artifact = state.load(result.data["segment_artifact"], step=0)
    assert artifact["world_artifact"] == "front_world_high.npz"
    assert artifact["camera"] == "front"
    assert result.images == [state.load_bytes(result.data["overlay_artifact"], step=0)]


def test_falls_back_to_the_next_view_when_the_preferred_one_is_incomplete(tmp_path):
    state = _state(tmp_path, FRONT)
    result = _sam3_toolkit(tmp_path, state).execute_tool("segment", {"prompt": "cup"})

    assert result.data["image_artifact"] == "front.png"


def test_localizes_other_artifact_layouts(tmp_path):
    head = CameraView("head_rgb.png", "head_world_xyz.npy")
    state = _state(tmp_path, head)
    result = _sam3_toolkit(tmp_path, state, cameras={"head": (head,)}).execute_tool(
        "segment", {"prompt": "cup"}
    )

    assert result.data["camera"] == "head"
    assert result.data["world_xyz"] == [0.5, 0.5, 0.5]


def test_missing_artifacts_report_every_checked_name(tmp_path):
    state = _state(tmp_path, FRONT)
    result = _sam3_toolkit(tmp_path, state).execute_tool(
        "segment", {"prompt": "cup", "camera": "side"}
    )

    assert result.data["error"] == "complete segment artifacts not found"
    assert result.data["checked_artifacts"] == ["side.png", "side_world.npz"]


def test_rejects_cameras_the_robot_did_not_configure(tmp_path):
    state = _state(tmp_path, FRONT)
    result = _sam3_toolkit(tmp_path, state).execute_tool(
        "segment", {"prompt": "cup", "camera": "wrist"}
    )

    assert result.data["error"] == "bad arguments for segment"


def test_direct_calls_report_unconfigured_cameras(tmp_path):
    result = _sam3(_state(tmp_path, FRONT)).segment(prompt="cup", camera="wrist")

    assert result.data["error"] == "unknown camera 'wrist'"
    assert result.data["cameras"] == ["front", "side"]


def test_service_failure_returns_the_robot_fallback(tmp_path):
    def unavailable(image, **kwargs):
        raise RuntimeError("connection refused")

    state = _state(tmp_path, FRONT)
    result = _sam3_toolkit(tmp_path, state, unavailable).execute_tool(
        "segment", {"prompt": "cup"}
    )

    assert "connection refused" in result.data["error"]
    assert result.data["fallback"] == FALLBACK


def test_instances_sharing_a_state_never_reuse_artifact_names(tmp_path):
    state = _state(tmp_path, FRONT)
    first = _sam3_toolkit(tmp_path, state).execute_tool("segment", {"prompt": "cup"})
    second = _sam3_toolkit(tmp_path, state).execute_tool("segment", {"prompt": "bowl"})

    assert first.data["segment_artifact"] == "segment_00.json"
    assert second.data["segment_artifact"] == "segment_01.json"
