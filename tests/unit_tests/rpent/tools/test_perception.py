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

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from rpent.tools import Toolkit, iter_tools
from rpent.tools.perception import Sam3Tools

FRONT_HIGH = ("front_high.png", "front_world_high.npz")
FRONT = ("front.png", "front_world.npz")
SIDE = ("side.png", "side_world.npz")
CAMERAS = {"front": (FRONT_HIGH, FRONT), "side": (SIDE,)}


def _found(image, **kwargs):
    return Sam3Result(found=True, mask=np.ones((4, 4), dtype=bool), score=0.9)


def _toolkit(tmp_path, state):
    return Toolkit(
        dashboard_events=NullDashboardEventSink(),
        state=state,
        memory=MemoryManager(tmp_path / "memory"),
    )


def _sam3_toolkit(tmp_path, state, segment=_found):
    toolkit = _toolkit(tmp_path, state)
    toolkit.add_tools(
        iter_tools(
            Sam3Tools(
                SimpleNamespace(segment=segment),
                state=state,
                cameras=CAMERAS,
                fallback="Use the robot's manual localization.",
            )
        )
    )
    return toolkit


def _state(tmp_path, *pairs):
    state = EnvState(tmp_path / "state")
    with state.record_step(state={}):
        for image_name, world_name in pairs:
            state.save(image_name, np.zeros((4, 4, 3), dtype=np.uint8))
            state.save(world_name, np.full((4, 4, 3), 0.5))
    return state


def test_toolkit_has_no_segment_tool_until_a_robot_opts_in(tmp_path, monkeypatch):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    state = _state(tmp_path, FRONT)
    names = {item.name for item in _toolkit(tmp_path, state).list_tools()}

    assert "segment" not in names
    assert "segment" in {
        item.name for item in _sam3_toolkit(tmp_path, state).list_tools()
    }


def test_schema_publishes_the_robot_cameras_with_the_first_as_default(tmp_path):
    state = _state(tmp_path, FRONT)
    declaration = Sam3Tools(
        SimpleNamespace(), state=state, cameras=CAMERAS, fallback=""
    ).segment
    camera = declaration.input_schema["properties"]["camera"]

    assert declaration.name == "segment"
    assert declaration.readonly
    assert camera["enum"] == ["front", "side"]
    assert camera["description"] == "Artifact camera to use (default front)."


def test_selects_the_first_complete_pair_and_localizes_the_mask(tmp_path):
    state = _state(tmp_path, FRONT, FRONT_HIGH)
    result = _sam3_toolkit(tmp_path, state).execute_tool("segment", {"prompt": "cup"})

    assert result.data["image_artifact"] == "front_high.png"
    assert result.data["world_xyz"] == [0.5, 0.5, 0.5]
    assert result.data["found"] is True
    artifact = state.load(result.data["segment_artifact"], step=0)
    assert artifact["world_artifact"] == "front_world_high.npz"
    assert artifact["camera"] == "front"
    assert result.images == [state.load_bytes(result.data["overlay_artifact"], step=0)]


def test_falls_back_to_the_next_pair_when_the_preferred_one_is_incomplete(tmp_path):
    state = _state(tmp_path, FRONT)
    result = _sam3_toolkit(tmp_path, state).execute_tool("segment", {"prompt": "cup"})

    assert result.data["image_artifact"] == "front.png"


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


def test_service_failure_returns_the_robot_fallback(tmp_path):
    def unavailable(image, **kwargs):
        raise RuntimeError("connection refused")

    state = _state(tmp_path, FRONT)
    result = _sam3_toolkit(tmp_path, state, unavailable).execute_tool(
        "segment", {"prompt": "cup"}
    )

    assert "connection refused" in result.data["error"]
    assert result.data["fallback"] == "Use the robot's manual localization."


def test_each_instance_numbers_its_own_artifacts(tmp_path):
    state = _state(tmp_path, FRONT)
    first = _sam3_toolkit(tmp_path, state).execute_tool("segment", {"prompt": "cup"})
    second = _sam3_toolkit(tmp_path, state).execute_tool("segment", {"prompt": "bowl"})

    assert first.data["segment_artifact"] == "segment_00.json"
    assert second.data["segment_artifact"] == "segment_00.json"
