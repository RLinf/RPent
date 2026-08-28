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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robots.libero import toolkit as libero_toolkit
from robots.robocasa import toolkit as robocasa_toolkit
from robots.robotwin import toolkit as robotwin_toolkit
from robots.robotwin.primitives import RoboTwinPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.tools.toolkit import Toolkit, _is_readonly, readonly
from rpent.utils import templates

COMMON_TOOLS = {"read_text_file", "write_text_file", "list_dir", "finish"}

LIBERO_EVALUATION_TOOLS = COMMON_TOOLS | {
    "view_env_state",
    "move_to",
    "pi0_pick",
    "pi0_doubled",
    "release",
    "set_gripper",
    "rotate_wrist",
    "rotate_pitch",
    "move_pose",
    "view_camera_meta",
    "segment",
    "back_project",
}

ROBOCASA_TOOLS = COMMON_TOOLS | {
    "move_to",
    "move_delta",
    "rotate_pitch",
    "set_gripper",
    "release",
    "scripted_grasp",
    "rldx_skill",
    "rldx_arm",
    "navigate_to",
    "move_base",
    "reset",
    "view_env_state",
    "view_camera_meta",
    "back_project",
    "back_project_batch",
    "query_world_map",
}

ROBOTWIN_TOOLS = COMMON_TOOLS | {
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

ROBOTWIN_TOOLKIT_PRIMITIVE_METHODS = {
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


class FakeSingleArmPrimitives:
    """No-runtime primitive surface shared by the LIBERO/RoboCasa fakes."""

    instances: list["FakeSingleArmPrimitives"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.reset_calls = 0
        self.recording_started = False
        type(self).instances.append(self)

    def reset(self) -> dict[str, Any]:
        self.reset_calls += 1
        return {"success": True}

    def reset_episode(self, reason: str) -> dict[str, Any]:
        return {"success": True, "reason": reason}

    def start_recording(self) -> None:
        self.recording_started = True

    def recorded_frame_count(self) -> int:
        return 0

    def frame_slice(self, start: int) -> list[Any]:
        return []

    def stop_recording(self) -> list[Any]:
        return []

    def dump_success_criteria(self) -> str:
        return "offline success criteria"

    @readonly
    def segment(self, **kwargs: Any) -> dict[str, Any]:
        return {"segment": kwargs}

    @staticmethod
    def _operation(name: str, **kwargs: Any) -> dict[str, Any]:
        return {"operation": name, "arguments": kwargs}

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_to", **kwargs)

    def pi0_pick(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("pi0_pick", **kwargs)

    def pi0_doubled(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("pi0_doubled", **kwargs)

    def release(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("release", **kwargs)

    def set_gripper(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("set_gripper", **kwargs)

    def rotate_wrist(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rotate_wrist", **kwargs)

    def rotate_pitch(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rotate_pitch", **kwargs)

    def move_pose(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_pose", **kwargs)

    def move_delta(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_delta", **kwargs)

    def scripted_grasp(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("scripted_grasp", **kwargs)

    def rldx_skill(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rldx_skill", **kwargs)

    def rldx_arm(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rldx_arm", **kwargs)

    def navigate_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("navigate_to", **kwargs)

    def move_base(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_base", **kwargs)


class FakeRoboTwinPrimitives:
    instances: list["FakeRoboTwinPrimitives"] = []

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


def _tool_names(toolkit: Toolkit) -> set[str]:
    return {spec["name"] for spec in toolkit.get_tools_spec()}


def _readonly_names(toolkit: Toolkit) -> set[str]:
    return {
        name for name, (_, handler) in toolkit._tools.items() if _is_readonly(handler)
    }


def test_robotwin_fake_and_real_implement_toolkit_primitive_protocol() -> None:
    for primitive_type in (RoboTwinPrimitives, FakeRoboTwinPrimitives):
        missing = {
            name
            for name in ROBOTWIN_TOOLKIT_PRIMITIVE_METHODS
            if not callable(getattr(primitive_type, name, None))
        }
        assert missing == set(), (
            f"{primitive_type.__name__} is missing toolkit methods: {sorted(missing)}"
        )


@pytest.fixture(autouse=True)
def _offline_template_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        templates,
        "default_variables",
        lambda: {"output_dir": "/offline/output"},
    )


def test_libero_toolkit_modes_construct_with_fake_primitives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakeSingleArmPrimitives.instances.clear()
    dumped: list[FakeSingleArmPrimitives] = []

    monkeypatch.setattr(
        libero_toolkit.libero_tools,
        "LiberoPrimitives",
        FakeSingleArmPrimitives,
    )
    monkeypatch.setattr(
        libero_toolkit.libero_tools,
        "dump_state",
        lambda primitives, state, log: dumped.append(primitives) or _record(),
    )

    evaluation = libero_toolkit.LiberoToolkit(
        primitives_kwargs={"env_client": object()},
        dashboard_events=NullDashboardEventSink(),
        mode="evaluation",
        state_output_dir=tmp_path / "evaluation",
    )
    exploration = libero_toolkit.LiberoToolkit(
        primitives_kwargs={"env_client": object()},
        dashboard_events=NullDashboardEventSink(),
        mode="exploration",
        attempts_per_session=3,
        state_output_dir=tmp_path / "exploration",
    )

    assert _tool_names(evaluation) == LIBERO_EVALUATION_TOOLS
    assert _tool_names(exploration) == LIBERO_EVALUATION_TOOLS | {"reset"}
    assert _readonly_names(evaluation) == COMMON_TOOLS | {
        "view_env_state",
        "view_camera_meta",
        "segment",
        "back_project",
    }
    assert _readonly_names(exploration) == _readonly_names(evaluation)
    assert len(dumped) == 2
    assert all(
        instance.reset_calls == 1 for instance in FakeSingleArmPrimitives.instances
    )
    assert all(
        instance.recording_started for instance in FakeSingleArmPrimitives.instances
    )
    assert all(
        callable(instance.kwargs["check_cancelled"])
        for instance in FakeSingleArmPrimitives.instances
    )

    refused = exploration.execute_tool(
        "finish", {"status": "failure", "summary": "first attempt"}
    )
    assert refused.result["error"] == "finish refused"
    assert refused.is_finish is False

    exploration.get_env_state = lambda *, command, result, elapsed_s: dict(result)
    assert (
        exploration.execute_tool("reset", {"reason": "new approach"}).result["attempt"]
        == 2
    )
    assert (
        exploration.execute_tool("reset", {"reason": "third approach"}).result[
            "attempt"
        ]
        == 3
    )
    allowed = exploration.execute_tool(
        "finish", {"status": "failure", "summary": "budget spent"}
    )
    assert allowed.is_finish is True


def test_robocasa_toolkit_constructs_and_classifies_tools_with_a_fake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RLDX_MAX_CHUNKS", raising=False)
    monkeypatch.delenv("RLDX_SETTLE_PATIENCE", raising=False)
    import robots.robocasa.primitives as primitives_module

    FakeSingleArmPrimitives.instances.clear()
    dumped: list[FakeSingleArmPrimitives] = []
    monkeypatch.setattr(
        primitives_module,
        "RoboCasaPrimitives",
        FakeSingleArmPrimitives,
    )
    monkeypatch.setattr(robocasa_toolkit, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        robocasa_toolkit.robocasa_tools,
        "dump_state",
        lambda primitives, state, log: dumped.append(primitives) or _record(),
    )

    toolkit = robocasa_toolkit.RoboCasaToolkit(
        primitives_kwargs={"env_client": object(), "vla_client": object()},
        dashboard_events=NullDashboardEventSink(),
    )

    assert _tool_names(toolkit) == ROBOCASA_TOOLS
    assert _readonly_names(toolkit) == COMMON_TOOLS | {
        "view_env_state",
        "view_camera_meta",
        "back_project",
        "back_project_batch",
        "query_world_map",
    }
    assert dumped == FakeSingleArmPrimitives.instances
    primitive = FakeSingleArmPrimitives.instances[0]
    assert primitive.reset_calls == 1
    assert primitive.recording_started is True
    assert callable(primitive.kwargs["check_cancelled"])
    assert (tmp_path / "success_criteria.md").read_text() == "offline success criteria"


def test_robotwin_toolkit_constructs_and_captures_an_initial_observation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakeRoboTwinPrimitives.instances.clear()
    dumped: list[dict[str, Any]] = []
    monkeypatch.setattr(
        robotwin_toolkit,
        "RoboTwinPrimitives",
        FakeRoboTwinPrimitives,
    )
    monkeypatch.setattr(robotwin_toolkit, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        robotwin_toolkit.RoboTwinToolkit,
        "_capture_full_observation",
        lambda self: {"views": {}, "robot_state": {}, "task_language": "offline"},
    )
    monkeypatch.setattr(
        robotwin_toolkit.tools,
        "dump_observation",
        lambda observation, env_state, status, log: (
            dumped.append({"observation": observation, "status": status, "log": log})
            or _record()
        ),
    )
    monkeypatch.setattr(
        robotwin_toolkit.tools,
        "view_env_state",
        readonly(lambda step=-1, *, state: {"step": step}),
    )

    toolkit = robotwin_toolkit.RoboTwinToolkit(
        primitives_kwargs={"env": object(), "model": object(), "seed": 7},
        dashboard_events=NullDashboardEventSink(),
    )

    assert _tool_names(toolkit) == ROBOTWIN_TOOLS
    assert _readonly_names(toolkit) == COMMON_TOOLS | {
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

    toolkit.get_env_state = lambda *, command, result, elapsed_s: dict(result)
    render = toolkit.execute_tool("render", {})
    assert render.result == {"success": True}
    finish = toolkit.execute_tool("finish", {"status": "failure", "summary": "offline"})
    assert finish.is_finish is True
