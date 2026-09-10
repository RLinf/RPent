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
        self.reset_calls = 0
        self.recording_started = False
        self.env = SimpleNamespace(last_reset_info={"actual_seed": 7})
        type(self).instances.append(self)

    def start_recording(self) -> None:
        self.recording_started = True

    def reset(self) -> dict[str, Any]:
        self.reset_calls += 1
        return {"success": True}

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
    assert primitive.reset_calls == 0
    assert "reset" not in _tool_names(robot_toolkit)
    assert primitive.recording_started is True
    assert callable(primitive.kwargs["check_cancelled"])

    robot_toolkit.get_env_state = lambda *, command, result, elapsed_s: dict(result)
    render = robot_toolkit.execute_tool("render", {})
    assert render.result == {"success": True}
    finish = robot_toolkit.execute_tool(
        "finish", {"status": "failure", "summary": "offline"}
    )
    assert finish.is_finish is True


@pytest.fixture
def explore_factory(monkeypatch, tmp_path):
    class EpisodePrimitives(FakeRoboTwinPrimitives):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.env = kwargs["env"]
            self.policy_actions = 99
            self.native_actions = 99

        def reset(self):
            return RoboTwinPrimitives.reset(self)

        def status(self):
            return self.env.last_info["episode_status"]

    class Env:
        def __init__(self):
            self.seeds = []
            self.last_info = {"episode_status": {"eval_success": False}}
            self.last_reset_info = {}
            self.terminated = True
            self.truncated = True

        def reset(self):
            self.seeds.append(7)
            self.terminated = self.truncated = False
            self.last_info = {
                "episode_status": {
                    "eval_success": False,
                    "actual_seed": 7,
                    "take_action_cnt": 0,
                    "step_lim": 100,
                }
            }
            self.last_reset_info = self.last_info
            return {}, self.last_info

    env = Env()
    monkeypatch.setattr(toolkit, "RoboTwinPrimitives", EpisodePrimitives)
    monkeypatch.setattr(toolkit, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path)}
    )
    monkeypatch.setattr(toolkit.RoboTwinToolkit, "_publish_step", lambda *a: None)

    def capture(self, *, command, result, elapsed_s):
        status = self._primitives.status()
        with self._state.record_step(
            state={"episode_status": dict(status)},
            terminated=status["eval_success"],
            truncated=False,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        ):
            pass
        return dict(result)

    monkeypatch.setattr(toolkit.RoboTwinToolkit, "get_env_state", capture)

    def make(budget=3, session=1, mode="exploration"):
        return toolkit.RoboTwinToolkit(
            primitives_kwargs={"env": env, "seed": 7},
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
            mode=mode,
            attempts_per_session=budget,
            state_output_dir=tmp_path / "sessions" / f"session_{session:03d}",
            recipe_output_dir=tmp_path,
        )

    return make, env


@pytest.mark.parametrize("budget", [1, 3, 0])
def test_exploration_budget_and_native_success(explore_factory, budget):
    make, env = explore_factory
    robot = make(budget)
    assert _tool_names(robot) == EXPECTED_TOOLS | {"reset"}
    reset_description = robot._SPECS["reset"]["description"]
    assert "configured exact seed" in reset_description
    assert "layout determinism has not been verified" in reset_description
    assert robot.solved() is False  # successful reset is not task success

    def finish():
        return robot.execute_tool("finish", {"status": "success", "summary": "claim"})

    assert finish().is_finish is (budget in (0, 1))
    for attempt in range(2, (budget or 5) + 1):
        result = robot.execute_tool("reset", {"reason": "different strategy"})
        assert result.result["attempt"] == attempt
        assert (
            "Episode reinitialized with the configured exact seed"
            in (result.result["notice"])
        )
        assert "layout determinism has not been verified" in result.result["notice"]
        assert not robot.solved()
    assert finish().is_finish
    if budget:
        resets = len(env.seeds)
        assert robot.execute_tool("reset", {"reason": "over budget"}).result["error"]
        assert len(env.seeds) == resets
    env.last_info["episode_status"]["eval_success"] = True
    assert robot.solved()
    assert finish().is_finish
    assert robot.execute_tool("reset", {"reason": "already solved"}).result["error"]


def test_new_session_resets_episode_and_counts(explore_factory):
    make, env = explore_factory
    first = make()
    first._primitives.policy_actions = 20
    first._primitives.native_actions = 30
    first.execute_tool("reset", {"reason": "retry"})
    assert first._primitives.policy_actions == first._primitives.native_actions == 0
    env.terminated = env.truncated = True
    env.last_info["episode_status"]["take_action_cnt"] = 100
    second = make(session=2)
    assert env.seeds == [7, 7, 7]
    assert not env.terminated and not env.truncated
    assert second._primitives.policy_actions == second._primitives.native_actions == 0
    assert env.last_info["episode_status"]["take_action_cnt"] == 0
    assert second._exploration.current_attempt == 1
    assert len(first._state.records()) == 2
    records = second._state.records()
    assert len(records) == 1
    assert records[0].step_idx == 0
    assert records[0].command == {"action": "reset"}
    assert records[0].state["episode_status"]["actual_seed"] == 7
    assert records[0].terminated is False


def test_recipe_contains_only_successful_attempt(explore_factory, tmp_path):
    import json

    make, env = explore_factory
    robot = make()
    robot.execute_tool("move_to", {"label": "failed attempt"})
    assert robot.write_recipe("cell") == ""
    robot.execute_tool("reset", {"reason": "retry"})
    robot.execute_tool("render", {})
    robot.get_env_state(
        command={"action": "move_to", "label": "error"},
        result={"error": "failed"},
        elapsed_s=0,
    )
    robot.get_env_state(
        command={"action": "release", "label": "unsuccessful"},
        result={"success": False},
        elapsed_s=0,
    )
    env.last_info["episode_status"]["eval_success"] = True
    robot.execute_tool("move_to", {"label": "winning attempt"})
    path = Path(robot.write_recipe("cell"))
    assert path == tmp_path / "cell_recipe.jsonl"
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"action": "move_to", "label": "winning attempt"}
    ]


def test_exact_seed_client_rejects_changed_actual_seed():
    from robots.robotwin.env_client import RoboTwinEnvClient

    class Rpc:
        actual_seed = 7

        def call(self, method, **kwargs):
            if method == "env.get_env_meta":
                return {"seed": 7}
            assert method == "env.reset"
            return {}, {
                "requested_seed": 7,
                "instruction": "task",
                "episode_status": {
                    "actual_seed": self.actual_seed,
                    "eval_success": False,
                    "take_action_cnt": 0,
                    "step_lim": 100,
                },
            }

    rpc = Rpc()
    client = RoboTwinEnvClient(rpc, expected_meta={"seed": 7})
    client.terminated = client.truncated = True
    client.reset()
    assert not client.terminated and not client.truncated
    assert client.last_info["episode_status"]["actual_seed"] == 7
    rpc.actual_seed = 8
    with pytest.raises(ValueError, match="requested seed"):
        client.reset()


def test_evaluation_recipe_still_exports_without_native_success(
    explore_factory, tmp_path
):
    import json

    make, env = explore_factory
    robot = make(mode="evaluation")
    robot.execute_tool("move_to", {"label": "existing evaluation behavior"})
    assert not robot.solved()
    path = Path(robot.write_recipe("eval"))
    assert path.parent == tmp_path / "sessions" / "session_001"
    assert json.loads(path.read_text()) == {
        "action": "move_to",
        "label": "existing evaluation behavior",
    }


def test_robotwin_reset_exception_consumes_attempt(explore_factory) -> None:
    make, env = explore_factory
    robot = make()
    seeds_before = list(env.seeds)
    action_calls = []

    def fail() -> dict[str, Any]:
        raise RuntimeError("native reset failed")

    robot._primitives.reset = fail
    robot._primitives.move_to = lambda **kwargs: action_calls.append(kwargs)
    result = robot.execute_tool("reset", {"reason": "retry"})

    assert "native reset failed" in result.result["error"]
    assert robot._exploration.current_attempt == 2
    assert robot._exploration.reset_failed
    assert env.seeds == seeds_before
    blocked = robot.execute_tool("move_to", {"label": "must not execute"})
    assert blocked.result == {
        "error": "Reset failed; reset successfully before further actions."
    }
    assert action_calls == []


def test_successful_reset_boundary_survives_state_capture_failure(
    explore_factory, tmp_path
) -> None:
    import json

    make, env = explore_factory
    robot = make()
    robot.execute_tool("move_to", {"label": "failed attempt"})
    pre_reset_step = robot._state.latest_record().step_idx
    capture = robot.get_env_state

    def fail_reset_capture(*, command, result, elapsed_s):
        if command["action"] == "reset":
            raise RuntimeError("post-reset capture failed")
        return capture(command=command, result=result, elapsed_s=elapsed_s)

    robot.get_env_state = fail_reset_capture
    reset = robot.execute_tool("reset", {"reason": "retry"})

    assert reset.result["state_capture_error"] == "post-reset capture failed"
    assert robot._exploration.latest_successful_reset_step == pre_reset_step
    env.last_info["episode_status"]["eval_success"] = True
    robot.execute_tool("move_to", {"label": "winning attempt"})

    path = Path(robot.write_recipe("capture_failure"))
    assert path == tmp_path / "capture_failure_recipe.jsonl"
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"action": "move_to", "label": "winning attempt"}
    ]


def test_robotwin_solved_tracks_current_native_status(explore_factory) -> None:
    make, env = explore_factory
    robot = make()

    env.last_info["episode_status"]["eval_success"] = True
    assert robot.solved() is True
    env.last_info["episode_status"]["eval_success"] = False
    assert robot.solved() is False
