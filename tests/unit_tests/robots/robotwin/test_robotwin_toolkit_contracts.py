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

"""Offline contracts for the RoboTwin toolkit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robots.robotwin import toolkit
from robots.robotwin.primitives import RoboTwinPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.tools.toolkit import Toolkit, _is_readonly, readonly
from rpent.utils import templates

COMMON_TOOLS = {"read_text_file", "write_text_file", "list_dir", "finish"}

EXPECTED_TOOLS = COMMON_TOOLS | {
    "view_env_state",
    "render",
    "sample_world_xyz",
    "query_world_map",
    "lingbot_act",
    "move_to",
    "rotate_wrist",
    "set_gripper",
    "release",
}

PRIMITIVE_METHODS = {
    "start_recording",
    "recorded_frame_count",
    "frame_slice",
    "stop_recording",
    "status",
    "finish",
    "lingbot_act",
    "move_to",
    "rotate_wrist",
    "set_gripper",
    "release",
}


class FakeRoboTwinPrimitives:
    instances: list[FakeRoboTwinPrimitives] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.status_calls = 0
        self.recording_started = False
        self.env = SimpleNamespace(last_reset_info={"actual_seed": 7})
        type(self).instances.append(self)

    def start_recording(self) -> None:
        self.recording_started = True

    def recorded_frame_count(self) -> int:
        return 0

    def frame_slice(self, start: int) -> list[Any]:
        del start
        return []

    def stop_recording(self) -> list[Any]:
        return []

    def status(self) -> dict[str, Any]:
        self.status_calls += 1
        return {
            "eval_success": False,
            "take_action_cnt": 0,
            "step_lim": 100,
            "actual_seed": 7,
        }

    def finish(self, *, status: str, summary: str) -> dict[str, Any]:
        return {"_finish": True, "status": status, "summary": summary}

    @staticmethod
    def _operation(name: str, **kwargs: Any) -> dict[str, Any]:
        return {"operation": name, "arguments": kwargs}

    def lingbot_act(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("lingbot_act", **kwargs)

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_to", **kwargs)

    def rotate_wrist(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rotate_wrist", **kwargs)

    def set_gripper(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("set_gripper", **kwargs)

    def release(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("release", **kwargs)


def _record(step_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(step_idx=step_idx, terminated=False)


def _tool_names(robot_toolkit: Toolkit) -> set[str]:
    return {spec["name"] for spec in robot_toolkit.get_tools_spec()}


def _readonly_names(robot_toolkit: Toolkit) -> set[str]:
    return {
        name
        for name, (_, handler) in robot_toolkit._tools.items()
        if _is_readonly(handler)
    }


def test_fake_and_real_implement_toolkit_primitive_protocol() -> None:
    for primitive_type in (RoboTwinPrimitives, FakeRoboTwinPrimitives):
        missing = {
            name
            for name in PRIMITIVE_METHODS
            if not callable(getattr(primitive_type, name, None))
        }
        assert missing == set(), (
            f"{primitive_type.__name__} is missing toolkit methods: {sorted(missing)}"
        )


def test_toolkit_constructs_and_captures_an_initial_observation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakeRoboTwinPrimitives.instances.clear()
    dumped: list[dict[str, Any]] = []
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )
    monkeypatch.setattr(toolkit, "RoboTwinPrimitives", FakeRoboTwinPrimitives)
    monkeypatch.setattr(toolkit, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        toolkit.RoboTwinToolkit,
        "_capture_full_observation",
        lambda self: {"views": {}, "robot_state": {}, "task_language": "offline"},
    )
    monkeypatch.setattr(
        toolkit.tools,
        "dump_observation",
        lambda observation, env_state, status, log: (
            dumped.append({"observation": observation, "status": status, "log": log})
            or _record()
        ),
    )
    monkeypatch.setattr(
        toolkit.tools,
        "view_env_state",
        readonly(lambda step=-1, *, state: {"step": step}),
    )

    robot_toolkit = toolkit.RoboTwinToolkit(
        primitives_kwargs={"env": object(), "model": object(), "seed": 7},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )

    assert _tool_names(robot_toolkit) == EXPECTED_TOOLS
    assert _readonly_names(robot_toolkit) == COMMON_TOOLS | {
        "view_env_state",
        "sample_world_xyz",
        "query_world_map",
    }
    assert len(dumped) == 1
    assert dumped[0]["log"] == {
        "command": {"action": "reset"},
        "result": {"actual_seed": 7, "success": True},
        "elapsed_s": 0.0,
    }
    primitive = FakeRoboTwinPrimitives.instances[0]
    assert primitive.status_calls == 1
    assert primitive.recording_started is True
    assert callable(primitive.kwargs["check_cancelled"])

    robot_toolkit.get_env_state = lambda *, command, result, elapsed_s: dict(result)
    render = robot_toolkit.execute_tool("render", {})
    assert render.result == {"success": True}
    finish = robot_toolkit.execute_tool(
        "finish", {"status": "failure", "summary": "offline"}
    )
    assert finish.is_finish is True
