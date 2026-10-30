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

import importlib
import json
import sys
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from threading import Lock
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.robotwin import toolkit
from robots.robotwin.env_client import RoboTwinEnvClient
from robots.robotwin.primitives import RoboTwinPrimitives
from robots.robotwin.robot_spec import env_runtime_contract
from robots.robotwin.toolkit import _capture_robotwin_call
from rpent.dashboard.events import NullDashboardEventSink
from rpent.flywheel.reader import read_array
from rpent.memory import MemoryManager
from rpent.session import EnvState, Payload
from rpent.session.env_capture import CapturingEnvClient, ToolCallCapture
from rpent.tools import Tool, Toolkit, ToolResult, tool
from rpent.tools.direct_action import direct_action_tool
from rpent.utils import templates
from rpent.utils.rpc import RpcError

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
        return ToolResult(data={"_finish": True, "status": status, "summary": summary})

    @staticmethod
    def _operation(name: str, **kwargs: Any) -> dict[str, Any]:
        return ToolResult(data={"operation": name, "arguments": kwargs})

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


for _attribute, _definition in vars(RoboTwinPrimitives).items():
    if isinstance(_definition, Tool):
        setattr(
            FakeRoboTwinPrimitives,
            _attribute,
            _definition.with_handler(getattr(FakeRoboTwinPrimitives, _attribute)),
        )


def _record(step_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(step_idx=step_idx, terminated=False)


def _tool_names(robot_toolkit: Toolkit) -> set[str]:
    return {definition.name for definition in robot_toolkit.list_tools()}


def _readonly_names(robot_toolkit: Toolkit) -> set[str]:
    return {
        definition.name
        for definition in robot_toolkit.list_tools()
        if definition.readonly
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
        toolkit.tools.view_env_state.with_handler(
            lambda step=-1, *, state: ToolResult(data={"step": step})
        ),
    )

    runtime_kwargs = {"env": object(), "model": object(), "seed": 7}
    robot_toolkit = toolkit.RoboTwinToolkit(
        runtime_kwargs=runtime_kwargs,
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
    assert primitive.kwargs["env"].wrapped is runtime_kwargs["env"]
    assert primitive.kwargs["env"].state is robot_toolkit.state
    assert primitive.status_calls == 1
    assert primitive.recording_started is True
    assert callable(primitive.kwargs["check_cancelled"])

    robot_toolkit.get_env_state = lambda *, command, result, elapsed_s: ToolResult(
        data=dict(result)
    )
    render = robot_toolkit.execute_tool("render", {})
    assert render.data == {"success": True}
    finish = robot_toolkit.execute_tool(
        "finish", {"status": "failure", "summary": "offline"}
    )
    assert finish.data.get("_finish", False) is True


CAMERAS = {
    "head": "head_camera",
    "left_wrist": "left_camera",
    "right_wrist": "right_camera",
}


class _Task:
    def __init__(self):
        self.take_action_cnt = 0
        self.step_lim = 200
        self.ep_num = 7
        self.eval_success = False
        self.success_at = None
        self.fail_at = None
        self.executed = []
        self.observation_reads = 0
        self.reset_count = 0
        self.lock = Lock()
        self.obs = {
            "observation": {
                name: {
                    "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                    "intrinsic_cv": np.eye(3),
                }
                for name in CAMERAS.values()
            }
        }
        self.robot = SimpleNamespace()
        for arm in ("left", "right"):
            for kind in ("arm_jointState", "arm_real_jointState"):
                setattr(
                    self.robot,
                    f"get_{arm}_{kind}",
                    lambda: [self.take_action_cnt] * 6 + [0.5],
                )
            for kind in ("ee_pose", "tcp_pose"):
                setattr(
                    self.robot,
                    f"get_{arm}_{kind}",
                    lambda: [self.take_action_cnt, 0, 0, 1, 0, 0, 0],
                )
            setattr(self.robot, f"get_{arm}_gripper_val", lambda: 0.5)

    def get_instruction(self):
        return "native instruction"

    def get_obs(self):
        assert self.lock.locked()
        self.observation_reads += 1
        return self.obs

    def take_action(self, action, *, action_type):
        assert self.lock.locked()
        self.executed.append((action.copy(), action_type))
        self.take_action_cnt += 1
        for camera in self.obs["observation"].values():
            camera["rgb"].fill(self.take_action_cnt)
        self.eval_success = self.take_action_cnt == self.success_at
        if self.take_action_cnt == self.fail_at:
            raise RuntimeError("native action failed")


@pytest.fixture
def facade(monkeypatch):
    pytest.importorskip("torch")
    for name, attributes in (
        ("rlinf.envs.sim.robotwin.robotwin_env", {"RoboTwinEnv": object}),
        ("robotwin.assets", {"validate_root": lambda *a: None}),
        ("robotwin.config", {"load_task_config": lambda *a: {}}),
    ):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            stub = ModuleType(name)
            stub.__dict__.update(attributes)
            monkeypatch.setitem(sys.modules, name, stub)
    for name in ("robots.robotwin.rlinf_env", "robots.robotwin.env_server"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("robots.robotwin.env_server")
    task = _Task()
    env = object.__new__(module.RoboTwinAgentEnv)
    slot = SimpleNamespace(task=task, lock=task.lock)
    env.venv = SimpleNamespace(envs=[slot])
    env.record_metrics = False
    env.center_crop = False
    env.center_and_crop = lambda image, **kw: np.array(image)
    env._extract_obs_image = lambda obs: obs
    env._get_camera = lambda *a: None

    def get_obs():
        assert not task.lock.locked()
        with task.lock:
            native = task.get_obs()
            return {
                "main_images": native["observation"]["head_camera"]["rgb"][None].copy()
            }

    def reset(**kwargs):
        task.reset_count += 1
        observation = get_obs()
        with task.lock:
            info = {
                "robot_state": env._robot_state(slot),
                "episode_status": env._episode_status(slot),
            }
        return observation, info

    env.venv.get_obs = get_obs
    env.reset = reset
    yield module.RoboTwinEnvFacade(
        env, metadata={"seed": 7, "execution": {"chunk_step_all_frames": True}}
    )
    sys.modules.pop("robots.robotwin.env_server", None)
    sys.modules.pop("robots.robotwin.rlinf_env", None)


@pytest.mark.parametrize(
    ("transport", "all_frames", "stop"),
    [
        ("socket", True, "success"),
        ("http", True, "budget"),
        ("socket", False, "budget"),
        ("http", False, "success"),
    ],
)
def test_chunk_returns_only_executed_steps_with_distinct_observations(
    facade, make_server_and_client, transport, all_frames, stop
):
    task = facade._env.venv.envs[0].task
    if stop == "success":
        task.success_at = 3
    else:
        task.step_lim = 3
    commands = np.arange(80, dtype=np.float64).reshape(5, 16)
    with make_server_and_client(facade, transport) as rpc:
        client = RoboTwinEnvClient(rpc, expected_meta=facade.get_env_meta())
        first = client.step(np.zeros(14))
        assert "before_observation" not in first[4]
        assert task.executed[0][1] == "qpos"
        np.testing.assert_array_equal(task.executed[0][0], np.zeros(14))
        reads = task.observation_reads
        obs, rewards, terminated, truncated, info = client.chunk_step(
            commands, action_type="ee", return_all_frames=all_frames
        )
        assert info["executed_actions"] == 2
        assert info["requested_actions"] == 5
        assert "capture" not in info
        assert ("before_observation" in info) is all_frames
        np.testing.assert_array_equal(rewards, [0, float(stop == "success")])
        np.testing.assert_array_equal(terminated, [False, stop == "success"])
        np.testing.assert_array_equal(truncated, [False, stop == "budget"])
        assert client.terminated is (stop == "success")
        assert client.truncated is (stop == "budget")
        np.testing.assert_array_equal(
            client.last_obs["main_images"], np.full((2, 2, 3), 3)
        )
        np.testing.assert_array_equal(
            client.last_info["robot_state"]["qpos_target14"],
            [3] * 6 + [0.5] + [3] * 6 + [0.5],
        )
        assert task.observation_reads - reads == 1 + (
            info["executed_actions"] + 1 if all_frames else 0
        )
        if all_frames:
            assert len(obs["frames"]) == 2
    assert task.reset_count == 1
    assert len(task.executed) == 3
    for index, (action, action_type) in enumerate(task.executed[1:]):
        assert action_type == "ee"
        np.testing.assert_array_equal(action, commands[index])
        assert info["per_step"][index]["episode_status"]["take_action_cnt"] == index + 2
    if not all_frames:
        return
    observations = [info["before_observation"], *obs["frames"]]
    for count, observation in enumerate(observations, 1):
        assert observation["task_language"] == "native instruction"
        assert observation["robot_state"]["left_eef_pose"][0] == count
        for camera in observation["observation"].values():
            np.testing.assert_array_equal(camera["rgb"], np.full((2, 2, 3), count))
            np.testing.assert_array_equal(camera["intrinsic_cv"], np.eye(3))


def test_old_frame_format_is_rejected_before_reset(
    facade, make_server_and_client, transport
):
    expected_meta = env_runtime_contract(
        task_name="beat_block_hammer", task_config="demo_clean", seed=7
    )
    facade._metadata = deepcopy(expected_meta)
    facade._metadata["execution"].pop("chunk_step_frame_format")
    with make_server_and_client(facade, transport) as rpc:
        with pytest.raises(AssertionError, match="chunk_step_frame_format"):
            RoboTwinEnvClient(rpc, expected_meta=expected_meta)
    task = facade._env.venv.envs[0].task
    assert task.reset_count == 0
    assert task.executed == []


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("step", {}),
        ("chunk_step", {"return_all_frames": False}),
        ("chunk_step", {"return_all_frames": True}),
    ],
)
def test_captured_call_preserves_return_shape_and_cache(
    facade, make_server_and_client, transport, tmp_path, method, kwargs
):
    state = EnvState(tmp_path)
    state.start_task(instruction="native instruction", config=Payload("json", {}))
    planner = state.register_planner(backend="api")
    tool_id = state.register_tool_call(
        planner_id=planner, name="move", arguments=Payload("json", {})
    )
    with make_server_and_client(facade, transport) as rpc:
        client = RoboTwinEnvClient(rpc, expected_meta=facade.get_env_meta())
        env = CapturingEnvClient(
            client, state, capture_call=partial(_capture_robotwin_call, client)
        )
        action_count = 1 if method == "step" else 2
        actions = np.arange(action_count * 14, dtype=np.float32).reshape(-1, 14)
        with ToolCallCapture(state, tool_id).bind():
            if method == "step":
                with pytest.raises(ValueError, match="must be a single action"):
                    env.step(np.zeros((2, 14)))
                assert facade._env.venv.envs[0].task.executed == []
            argument = "action" if method == "step" else "actions"
            result = getattr(env, method)(**{argument: actions}, **kwargs)
        assert isinstance(result, tuple) and len(result) == 5
        assert "before_observation" not in result[4]
        assert "before_observation" not in client.last_info
        assert (
            result[4]["executed_actions"]
            == len(state.action_records())
            == len(facade._env.venv.envs[0].task.executed)
            == action_count
        )
        if method == "step":
            assert isinstance(result[1], float)
            assert isinstance(result[2], bool)
            assert isinstance(result[3], bool)
        else:
            for signal in result[1:4]:
                assert signal.shape == (action_count,)
        if kwargs.get("return_all_frames"):
            assert len(result[0]["frames"]) == action_count
            assert result[0]["final"] is client.last_obs
        else:
            assert result[0] is client.last_obs
        for recorded, action in zip(state.action_records(), actions, strict=True):
            assert recorded.action.data.dtype == np.float64
            np.testing.assert_array_equal(recorded.action.data, action)


@pytest.mark.parametrize(
    ("transport", "recording"), [("socket", True), ("http", False)]
)
def test_lingbot_predictions_and_action_sources_reach_disk(
    facade, make_server_and_client, transport, tmp_path, recording
):
    state = EnvState(tmp_path)
    task = facade._env.venv.envs[0].task
    task.success_at = 104
    inputs = []
    commands = np.arange(53 * 16, dtype=np.float64).reshape(53, 16) / 1000

    def infer(observation):
        inputs.append(deepcopy(observation))
        return commands

    if recording:
        state.start_task(instruction="native instruction", config=Payload("json", {}))
        planner = state.register_planner(backend="api")
    with make_server_and_client(facade, transport) as rpc:
        client = RoboTwinEnvClient(rpc, expected_meta=facade.get_env_meta())
        env = CapturingEnvClient(
            client, state, capture_call=partial(_capture_robotwin_call, client)
        )
        primitives = RoboTwinPrimitives(
            env=env,
            model=SimpleNamespace(infer=infer),
            seed=7,
            check_cancelled=lambda: None,
        )
        primitives.start_recording()
        toolkit = Toolkit(
            state=state,
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
        )
        toolkit.get_env_state = lambda **kw: ToolResult(data={})

        @tool
        def computed():
            reads = task.observation_reads
            result = env.step(np.zeros(14))
            assert "capture" not in result[4]
            assert task.observation_reads - reads == 1 + 2 * int(recording)
            return ToolResult(data={})

        toolkit.add_tool(computed)
        toolkit.add_tool(primitives.lingbot_act)
        toolkit.add_tool(
            direct_action_tool(env.action_specs, primitives.execute_action)
        )
        for name, arguments in (
            ("execute_action", {"values": [0.0] * 14, "action_type": "qpos"}),
            ("lingbot_act", {"chunks": 1, "prompt": "ignored instruction"}),
            ("computed", {}),
            ("lingbot_act", {"chunks": 2}),
        ):
            if recording:
                tool_id = state.register_tool_call(
                    planner_id=planner, name=name, arguments=Payload("json", arguments)
                )
            with ToolCallCapture(state, tool_id).bind() if recording else nullcontext():
                result = toolkit.execute_tool(name, arguments)
            assert not result.is_error, result.data
            assert "capture" not in client.last_info
        assert client.terminated
        assert len(task.executed) == 104
        assert len(inputs) == 3
        assert primitives.policy_actions == 102
        assert primitives.recorded_frame_count() == 103
        for frame, count in zip(
            primitives.frame_slice(0), [*range(1, 52), *range(53, 105)], strict=True
        ):
            assert frame.dtype == np.uint8
            np.testing.assert_array_equal(frame, np.full((2, 2, 3), count))
    assert task.reset_count == 1
    if not recording:
        assert state.action_records() == state.prediction_records() == []
        assert not (tmp_path / "trajectories").exists()
        return
    state.end_task(status="completed", stop_reason="finish")
    root = state.save_planner_trajectory()
    assert json.loads((root / "task.json").read_text())["schema_version"] == 5
    actions = [
        json.loads(line) for line in (root / "actions.jsonl").read_text().splitlines()
    ]
    predictions = [
        json.loads(line)
        for line in (root / "predictions.jsonl").read_text().splitlines()
    ]
    assert [a["source"] for a in actions] == [
        "direct",
        *(["vla"] * 50),
        "tool_computed",
        *(["vla"] * 52),
    ]
    for prediction, start, length, expected in zip(
        predictions, (2, 53, 103), (50, 50, 2), inputs, strict=True
    ):
        assert prediction["action_ids"] == [
            f"A{i}" for i in range(start, start + length)
        ]
        assert (
            prediction["instruction"]
            == expected["task_language"]
            == "native instruction"
        )
        assert "input_observation" not in prediction
        assert "input_observation_id" not in prediction
        payload = actions[start - 1]["before_observation"]["payload"]
        assert payload["format"] == "robotwin.observation"
        for name, camera in expected["views"].items():
            np.testing.assert_array_equal(
                read_array(root, payload["data"]["observation"][CAMERAS[name]]["rgb"]),
                camera["rgb"],
            )
        np.testing.assert_array_equal(
            read_array(root, payload["data"]["robot_state"]["left_eef_pose"]),
            expected["robot_state"]["left_eef_pose"],
        )
    assert predictions[1]["tool_call_id"] == predictions[2]["tool_call_id"]
    for index, (action, (native, action_type)) in enumerate(
        zip(actions, task.executed, strict=True)
    ):
        assert action["action"]["format"] == f"robotwin.action.{action_type}"
        np.testing.assert_array_equal(
            read_array(root, action["action"]["data"]), native
        )
        assert action["before_observation_id"] == f"O{index + 1}"
        assert action["after_observation_id"] == f"O{index + 2}"
        for side, count in (("before", index), ("after", index + 1)):
            data = action[f"{side}_observation"]["payload"]["data"]
            np.testing.assert_array_equal(
                read_array(root, data["robot_state"]["left_eef_pose"]),
                [count, 0, 0, 1, 0, 0, 0],
            )
            for camera in data["observation"].values():
                np.testing.assert_array_equal(
                    read_array(root, camera["rgb"]), np.full((2, 2, 3), count)
                )


def test_failed_chunk_discards_its_steps_but_keeps_prior_calls(
    facade, make_server_and_client, transport, tmp_path
):
    task = facade._env.venv.envs[0].task
    task.fail_at = 3
    state = EnvState(tmp_path)
    state.start_task(instruction="native instruction", config=Payload("json", {}))
    planner = state.register_planner(backend="api")
    tool_id = state.register_tool_call(
        planner_id=planner, name="move", arguments=Payload("json", {})
    )
    with make_server_and_client(facade, transport) as rpc:
        client = RoboTwinEnvClient(rpc, expected_meta=facade.get_env_meta())
        env = CapturingEnvClient(
            client, state, capture_call=partial(_capture_robotwin_call, client)
        )
        with ToolCallCapture(state, tool_id).bind():
            env.step(np.zeros(14))
            with env.prediction("native instruction"):
                with pytest.raises(RpcError, match="native action failed"):
                    env.chunk_step(np.zeros((5, 16)), action_type="ee")
    assert len(task.executed) == 3
    assert len(state.action_records()) == 1
    assert state.prediction_records()[0].action_ids == []
