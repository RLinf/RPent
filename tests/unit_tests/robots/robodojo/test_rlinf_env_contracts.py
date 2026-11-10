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

"""Agent-only observation, policy shape and CLI configuration contracts."""

import json
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo import env_server, tools
from robots.robodojo.awomo05_client import Awomo05Client
from robots.robodojo.env_client import RoboDojoEnvClient
from robots.robodojo.env_facade import RoboDojoEnvFacade
from robots.robodojo.robot_spec import FLYWHEEL_CONFIG
from robots.robodojo.toolkit import RoboDojoToolkit, _capture_robodojo_call
from robots.robodojo.vpp2_client import VPP2Client
from rpent.dashboard.events import NullDashboardEventSink
from rpent.flywheel.flywheel import read_frame, resolve_action_config
from rpent.flywheel.reader import read_array
from rpent.memory import MemoryManager
from rpent.session import Payload
from rpent.session.env_capture import ToolCallCapture


def test_server_arguments_reach_runtime_config():
    args = env_server.build_parser().parse_args(
        [
            "--task",
            "fill_pen_holder",
            "--layout",
            "8",
            "--random",
            "--env-cfg-type",
            "arx_x5",
            "--source-root",
            "/data/code",
            "--save-dir",
            "/data/output",
            "--cuda-device",
            "2",
            "--headless",
            "--max-episode-steps",
            "1200",
        ]
    )
    cfg = env_server.build_config(args)
    assert cfg.seed == 8 and not cfg.auto_reset
    assert cfg.task_config.task_name == "fill_pen_holder"
    assert cfg.task_config.source_root == "/data/code"
    assert cfg.task_config.save_dir == "/data/output"
    assert cfg.task_config.cuda_device == 2
    assert cfg.task_config.headless and cfg.task_config.random
    assert cfg.task_config.max_episode_steps == 1200
    assert (
        env_server.build_config(
            env_server.build_parser().parse_args([])
        ).task_config.source_root
        is None
    )
    with pytest.raises(ValueError, match="one environment"):
        env_server.build_config(
            env_server.build_parser().parse_args(["--num-envs", "2"])
        )


def test_policy_vectors_map_to_native_keys(agent_module):
    action = agent_module._policy_action(np.arange(14))
    assert list(action) == [
        "left_arm_joint_state",
        "right_arm_joint_state",
        "left_ee_joint_state",
        "right_ee_joint_state",
    ]
    assert [v.tolist() for v in action.values()] == [
        list(range(6)),
        list(range(6, 12)),
        [12],
        [13],
    ]
    assert agent_module._policy_action(action) is action


@pytest.mark.parametrize(
    "action",
    [
        np.zeros((1, 14)),
        np.zeros(13),
        np.zeros((50, 14)),
        np.full(14, np.nan),
    ],
)
def test_policy_shape_fails_before_motion(agent_module, make_agent, action):
    agent = make_agent(SimpleNamespace())
    with pytest.raises(ValueError, match="shape"):
        agent.step_native(action)


def test_camera_and_language_preserve_rpc_contract(make_agent):
    frames = []
    camera = {
        "color": np.zeros((2, 3, 3), dtype=np.uint8),
        "depth": np.ones((2, 3)),
        "intrinsic_matrix": np.eye(3),
        "extrinsic_matrix": np.eye(4),
    }
    env = SimpleNamespace(
        get_obs=lambda env_idx: {"vision": {"cam_head": camera}, "state": {}},
        obs_manager=SimpleNamespace(
            desc_manager=SimpleNamespace(
                get_one_description=lambda: ["Pick the object."],
            )
        ),
    )
    agent = make_agent(env, recorder=SimpleNamespace(record=frames.append))
    facade = env_server.RoboDojoEnvFacade(agent)
    meta = facade.get_camera_meta("cam_head")
    assert (meta["height"], meta["width"]) == (2, 3)
    np.testing.assert_array_equal(meta["intrinsic_matrix"], np.eye(3))
    np.testing.assert_array_equal(meta["extrinsic_matrix"], np.eye(4))
    rgb, depth = facade.render_camera("cam_head", depth=True)
    assert rgb.shape == (2, 3, 3) and depth.shape == (2, 3)
    assert (
        facade.get_obs()["instruction"]
        == facade.get_task_language()
        == "Pick the object."
    )
    assert len(frames) == 1
    with pytest.raises(IndexError):
        agent._sub_env(1)
    with pytest.raises(ValueError, match="unknown camera"):
        facade.get_camera_meta("missing")


class _Scene:
    def __init__(self):
        self.take_action_cnt = [0]
        self.step_lim = 99
        self.end_flag = [False]
        self.robot_manager = SimpleNamespace(
            robot_list=[
                SimpleNamespace(type="target", arm_name=f"{arm}_arm")
                for arm in ("left", "right")
            ]
        )
        self.executed = []
        self.frames = []
        self.reset_count = 0
        self.obs = {"state": {}, "vision": {}, "score": 100}
        for arm in ("left", "right"):
            self.obs["state"].update(
                {
                    f"{arm}_ee_pose": np.array([0, 0, 1.0, 1, 0, 0, 0]),
                    f"{arm}_ee_joint_state": np.array([0.2]),
                    f"{arm}_arm_joint_state": np.zeros(6),
                }
            )
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            self.obs["vision"][camera] = {
                "color": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth": np.ones((2, 2)),
                "intrinsic_matrix": np.eye(3),
                "extrinsic_matrix": np.eye(4),
            }
        self.obs_manager = SimpleNamespace(
            desc_manager=SimpleNamespace(get_one_description=lambda: ["pick bottle"])
        )

    def reset(self, **kwargs):
        self.reset_count += 1

    def get_obs(self, env_idx=0):
        return self.obs

    def is_success(self, env_idx=0):
        return False

    def get_motion_state(self):
        state = self.obs["state"]
        return {
            "targets": {
                name: value.tolist()
                for name, value in state.items()
                if name.endswith("_joint_state")
            },
            **{
                f"{arm}_gripper": float(state[f"{arm}_ee_joint_state"][0])
                for arm in ("left", "right")
            },
        }

    def apply_action(self, action, action_type, *, eval_fair):
        self.executed.append((deepcopy(action), action_type))
        self.take_action_cnt[0] += 1
        count = self.take_action_cnt[0]
        z = [1, 1, 0.9, 0.96, 0.96][count - 1]
        for arm in ("left", "right"):
            self.obs["state"][f"{arm}_ee_pose"][2] = z
        for camera in self.obs["vision"].values():
            camera["color"].fill(count)


@pytest.fixture
def scene(make_agent, agent_module, monkeypatch):
    scene = _Scene()
    scene.agent = make_agent(
        scene,
        meta={"mode": "dev", "layout": 0},
        slot=SimpleNamespace(env=scene, apply_action=scene.apply_action),
        recorder=SimpleNamespace(record=lambda obs: scene.frames.append(deepcopy(obs))),
    )
    scene.agent.venv.reset = scene.reset
    scene.agent.get_motion_state = scene.get_motion_state
    monkeypatch.setattr(agent_module._SafetyMonitor, "check", lambda self, env: [])
    monkeypatch.setattr(agent_module, "_reward_details", lambda *a: {"reward": 0.5})
    monkeypatch.setattr(
        agent_module,
        "_status",
        lambda env, monitor: {
            "step": env.take_action_cnt[0],
            "step_limit": env.step_lim,
        },
    )
    return scene


@pytest.mark.parametrize(
    ("capture", "eval_fair", "action_type"),
    [
        (True, False, "joint"),
        (True, True, "ee"),
        (False, True, "joint"),
        (False, False, "ee"),
    ],
)
def test_native_capture_preserves_execution_observations_and_video(
    scene, capture, eval_fair, action_type
):
    scene.agent.eval_fair = eval_fair
    facade = RoboDojoEnvFacade(scene.agent)
    command = (
        np.linspace(0, 1, 14)
        if action_type == "joint"
        else {
            f"{arm}_ee_pose": np.array([0, 0, 0.9, 1, 0, 0, 0])
            for arm in ("left", "right")
        }
    )
    response = facade.step(command, capture=capture)
    result = response["result"] if capture else response
    obs, reward, done, info = result
    assert reward == (0.0 if eval_fair else 0.5)
    assert done is False
    assert info["status"] == {"step": 1, "step_limit": 99}
    assert ("score" in obs) is not eval_fair
    assert len(scene.frames) == len(scene.executed) == 1
    if not capture:
        return
    native_action, native_type = scene.executed[0]
    assert response["action_type"] == native_type == action_type
    for key, value in native_action.items():
        np.testing.assert_array_equal(response["action"][key], value)
    for data, value in (
        (response["before_observation"], 0),
        (obs, 1),
    ):
        assert data["instruction"] == "pick bottle"
        assert ("score" in data) is not eval_fair
        for camera in data["vision"].values():
            np.testing.assert_array_equal(camera["color"], np.full((2, 2, 3), value))
            np.testing.assert_array_equal(camera["intrinsic_matrix"], np.eye(3))


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("eval_fair", [False, True])
def test_camera_sync_is_limited_to_captured_steps(
    scene, monkeypatch, capture, eval_fair
):
    scene.agent.eval_fair = eval_fair
    previous_vision = deepcopy(scene.obs["vision"])
    for camera in previous_vision.values():
        camera["color"].fill(255)
    reads = 0

    def get_obs(env_idx=0):
        nonlocal previous_vision, reads
        reads += 1
        obs = deepcopy(scene.obs)
        obs["vision"], previous_vision = previous_vision, obs["vision"]
        return obs

    monkeypatch.setattr(scene, "get_obs", get_obs)
    response = scene.agent.step_native(np.zeros(14), capture=capture)
    assert len(scene.executed) == len(scene.frames) == 1
    if capture:
        before = response["before_observation"]
        after = response["result"][0]
        for obs, value in ((before, 0), (after, 1), (scene.frames[0], 1)):
            for camera in obs["vision"].values():
                np.testing.assert_array_equal(
                    camera["color"], np.full((2, 2, 3), value)
                )
    else:
        assert reads == 1

    reads = 0
    scene.agent.get_obs()
    assert reads == 1


@pytest.mark.parametrize(
    ("eval_fair", "action_kind"),
    [(False, "vector"), (False, "joint"), (True, "vector"), (True, "ee")],
)
def test_capture_call_preserves_native_action_and_observations(
    scene, eval_fair, action_kind, transport, make_server_and_client
):
    scene.agent.eval_fair = eval_fair
    scene.agent.meta["mode"] = "eval-fair" if eval_fair else "dev"
    facade = RoboDojoEnvFacade(scene.agent)

    if action_kind == "vector":
        command = np.arange(14)
    elif action_kind == "joint":
        command = {
            "left_arm_joint_state": np.arange(6, dtype=np.float32),
            "right_ee_joint_state": np.array([0.4], dtype=np.float64),
        }
    else:
        command = {"left_ee_pose": np.array([0, 0, 0.9, 1, 0, 0, 0])}
    action_type = "ee" if action_kind == "ee" else "joint"
    with make_server_and_client(facade, transport) as rpc:
        client = RoboDojoEnvClient(rpc, expected_meta=scene.agent.get_env_meta())
        if eval_fair:
            assert scene.reset_count == 0
            assert client.last_obs["instruction"] == "pick bottle"
        frame_count = len(scene.frames)
        response = _capture_robodojo_call(client, "step", command)
    obs, reward, done, info = response["result"]
    assert reward == (0.0 if eval_fair else 0.5)
    assert done is False
    assert info["status"] == {"step": 1, "step_limit": 99}
    np.testing.assert_equal(client.last_obs, obs)
    assert len(scene.frames) == frame_count + 1
    ((native_action, native_type),) = scene.executed
    (step,) = response["steps"]
    assert native_type == action_type
    assert step["action"]["format"] == f"robodojo.action.{action_type}"
    np.testing.assert_equal(step["action"]["data"], native_action)
    for key, value in native_action.items():
        assert step["action"]["data"][key].dtype == value.dtype
    assert step["result"]["format"] == "robodojo.step_result"
    assert step["result"]["data"] == {"reward": reward, "done": done, "info": info}
    np.testing.assert_equal(step["after_observation"]["data"], obs)
    for payload, value in (
        (response["before_observation"], 0),
        (step["after_observation"], 1),
    ):
        assert payload["format"] == "robodojo.observation"
        assert ("score" in payload["data"]) is not eval_fair
        for camera in payload["data"]["vision"].values():
            assert camera["color"].dtype == np.uint8
            np.testing.assert_array_equal(camera["color"], np.full((2, 2, 3), value))


def _register_legacy_pick(toolkit):
    # Exercise a legacy policy tool that is no longer registered by default.
    toolkit.add_tool(
        tools.pi0_pick.with_handler(
            partial(tools.pi0_pick, primitives=toolkit._primitives, state=toolkit.state)
        )
    )


@pytest.fixture
def scene_rpc(scene):
    facade = RoboDojoEnvFacade(scene.agent)

    def call(method, *, args=(), kwargs=None, **options):
        return deepcopy(facade._rpc[method](*args, **(kwargs or {})))

    return SimpleNamespace(call=call)


@pytest.mark.parametrize("recording", [False, True])
def test_toolkit_saves_only_executed_predictions_and_restores_source(
    scene, scene_rpc, tmp_path, monkeypatch, recording
):
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    client = RoboDojoEnvClient(scene_rpc, expected_meta=scene.agent.get_env_meta())
    inputs = []
    commands = np.linspace(0, 1, 42).reshape(3, 14)

    def predict(obs):
        inputs.append(deepcopy(obs))
        return commands

    toolkit = RoboDojoToolkit(
        primitives_kwargs={
            "env": client,
            "vla_client": SimpleNamespace(predict=predict),
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    _register_legacy_pick(toolkit)
    state = toolkit.state
    if recording:
        state.start_task(instruction="pick bottle", config=Payload("json", {}))
        planner = state.register_planner(backend="api")
    for name, arguments in (
        ("pi0_pick", {}),
        ("set_gripper", {"arm": "left", "val": 1.0, "steps": 1}),
    ):
        if recording:
            tool = state.register_tool_call(
                planner_id=planner, name=name, arguments=Payload("json", arguments)
            )
        with ToolCallCapture(state, tool).bind() if recording else nullcontext():
            result = toolkit.execute_tool(name, arguments)
        assert not result.is_error, result.data
    assert scene.reset_count == 1
    assert len(inputs) == 2
    assert len(scene.executed) == 5
    assert len(scene.frames) == 11
    assert [action_type for _, action_type in scene.executed] == ["joint"] * 5
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
    assert [a["source"] for a in actions] == ["vla"] * 4 + ["tool_computed"]
    assert [p["action_ids"] for p in predictions] == [["A1", "A2", "A3"], ["A4"]]
    for index, action in enumerate(actions):
        native, action_type = scene.executed[index]
        assert action["action"]["format"] == f"robodojo.action.{action_type}"
        for key, value in native.items():
            saved = action["action"]["data"][key]
            if isinstance(value, np.ndarray):
                saved = read_array(root, saved)
                assert saved.dtype == value.dtype
            np.testing.assert_array_equal(saved, value)
        assert action["before_observation_id"] == f"O{index + 1}"
        assert action["after_observation_id"] == f"O{index + 2}"
        for side, count in (("before", index), ("after", index + 1)):
            obs = action[f"{side}_observation"]["payload"]["data"]
            for camera in obs["vision"].values():
                image = read_array(root, camera["color"])
                assert image.dtype == np.uint8
                np.testing.assert_array_equal(image, np.full((2, 2, 3), count))
        if action_type == "joint":
            config = resolve_action_config(FLYWHEEL_CONFIG, action["action"]["format"])
            frame = read_frame(root, action, "pick bottle", config)
            expected_action = (
                commands[index % 3]
                if index < 4
                else np.array([*np.zeros(12), 1.0, 0.2], dtype=np.float32)
            )
            np.testing.assert_array_equal(frame["action"], expected_action)
            assert frame["action"].dtype == expected_action.dtype
            np.testing.assert_array_equal(
                frame["observation.state"], np.r_[np.zeros(12), 0.2, 0.2]
            )
            assert frame["prompt"] == "pick bottle"
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                np.testing.assert_array_equal(
                    frame[f"observation.images.{camera}"], np.full((2, 2, 3), index)
                )
    actions_by_id = {action["action_id"]: action for action in actions}
    for prediction, expected in zip(predictions, inputs, strict=True):
        assert prediction["instruction"] == expected["instruction"] == "pick bottle"
        assert "input_observation" not in prediction
        assert "input_observation_id" not in prediction
        first = actions_by_id[prediction["action_ids"][0]]
        obs = first["before_observation"]["payload"]["data"]
        for key, value in expected["state"].items():
            np.testing.assert_array_equal(read_array(root, obs["state"][key]), value)
        for name, camera in expected["vision"].items():
            for key, value in camera.items():
                np.testing.assert_array_equal(
                    read_array(root, obs["vision"][name][key]), value
                )


@pytest.mark.parametrize(
    ("backend", "fault"),
    [
        ("vpp2", None),
        ("awomo05", None),
        ("awomo05", "delivery"),
        ("vpp2", "missing_obs"),
    ],
)
def test_policy_capture_keeps_confirmed_actions(
    scene, scene_rpc, tmp_path, monkeypatch, backend, fault
):
    scene.step_lim = 3
    is_vpp2 = backend == "vpp2"
    instruction = "lift the bottle" if is_vpp2 else "pick bottle"
    observed_steps = []
    if fault == "missing_obs":
        step_native = scene.agent.step_native

        def drop_second_observation(*args, **kwargs):
            response = step_native(*args, **kwargs)
            if scene.take_action_cnt[0] == 2:
                response["result"] = (None, *response["result"][1:])
            return response

        monkeypatch.setattr(scene.agent, "step_native", drop_second_observation)

    def model_call(method, *, args=(), **options):
        if method == "vla.predict":
            assert args[0]["instruction"] == instruction
            return np.zeros((32, 16), dtype=np.float32)
        if method == "vla.get_action":
            return [scene.get_motion_state()["targets"]] * 8
        if method == "vla.update_obs":
            obs, step = args
            assert obs["instruction"] == instruction
            observed_steps.append(step)
            if fault == "delivery" and step == 2:
                raise RuntimeError("model feedback failed")

    client = RoboDojoEnvClient(
        scene_rpc, expected_meta=scene.agent.get_env_meta(), record_vpp2_history=is_vpp2
    )
    policy = (VPP2Client if is_vpp2 else Awomo05Client)(
        SimpleNamespace(call=model_call)
    )
    toolkit = RoboDojoToolkit(
        primitives_kwargs={"env": client, "vla_client": policy, "vla_backend": backend},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path,
    )
    name = f"{backend}_act"
    arguments = {"instruction": instruction} if is_vpp2 else {}
    state = toolkit.state
    state.start_task(instruction="pick bottle", config=Payload("json", {}))
    tool_id = state.register_tool_call(
        planner_id=state.register_planner(backend="api"),
        name=name,
        arguments=Payload("json", arguments),
    )
    with ToolCallCapture(state, tool_id).bind():
        feedback = toolkit.execute_tool(name, arguments)
    toolkit.close()

    assert feedback.is_error is bool(fault)
    call = state.get_tool_call(tool_id)
    result = call.raw_result.data["data"]
    actions = state.action_records()
    (prediction,) = state.prediction_records()
    assert prediction.instruction == instruction
    assert prediction.action_ids == [action.action_id for action in actions]
    assert (
        result["executed_actions"]
        == len(scene.executed)
        == len(actions)
        == (2 if fault else 3)
    )
    for action, (native, kind) in zip(actions, scene.executed, strict=True):
        assert action.source == "vla" and action.tool_call_id == tool_id
        assert action.action.format == f"robodojo.action.{kind}"
        np.testing.assert_equal(action.action.data, native)
    assert call.after_observation_id == actions[-1].after_observation_id
    assert observed_steps == ([] if is_vpp2 else [0, 1, 2])
    if fault:
        assert result["stop_reason"] == "synchronization_lost"
        assert result["execution_uncertain"] is False
        if fault == "missing_obs":
            payload = state.get_observation(call.after_observation_id).payload
            assert (
                payload.data is None and payload.missing[""].reason == "capture_failed"
            )
            assert "no observation" in result["error"]
        else:
            assert "model feedback failed" in result["error"]
    else:
        assert result["stop_reason"] == "budget_exhausted"


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [("pi0_pick", {}), ("set_gripper", {"arm": "left", "val": 1.0, "steps": 3})],
)
def test_failed_step_keeps_only_confirmed_actions(
    scene,
    tmp_path,
    monkeypatch,
    transport,
    make_server_and_client,
    tool_name,
    arguments,
):
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    apply_action = scene.agent._sub_env().apply_action

    def fail_after_second_action(action, action_type, *, eval_fair):
        apply_action(action, action_type, eval_fair=eval_fair)
        if scene.take_action_cnt[0] == 2:
            raise RuntimeError("native step failed")

    monkeypatch.setattr(
        scene.agent._sub_env(), "apply_action", fail_after_second_action
    )
    with make_server_and_client(RoboDojoEnvFacade(scene.agent), transport) as rpc:
        client = RoboDojoEnvClient(rpc, expected_meta=scene.agent.get_env_meta())
        toolkit = RoboDojoToolkit(
            primitives_kwargs={
                "env": client,
                "vla_client": SimpleNamespace(predict=lambda obs: np.zeros((3, 14))),
            },
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
        )
        if tool_name == "pi0_pick":
            _register_legacy_pick(toolkit)
        state = toolkit.state
        state.start_task(instruction="pick bottle", config=Payload("json", {}))
        tool_id = state.register_tool_call(
            planner_id=state.register_planner(backend="api"),
            name=tool_name,
            arguments=Payload("json", arguments),
        )
        with ToolCallCapture(state, tool_id).bind():
            result = toolkit.execute_tool(tool_name, arguments)

    assert result.is_error
    assert len(scene.executed) == 2
    (action,) = state.action_records()
    assert action.tool_call_id == tool_id
    if tool_name == "pi0_pick":
        (prediction,) = state.prediction_records()
        assert action.source == "vla"
        assert prediction.action_ids == [action.action_id]
        assert prediction.tool_call_id == tool_id
    else:
        assert action.source == "tool_computed"
        assert state.prediction_records() == []
    call = state.get_tool_call(tool_id)
    assert call.missing["after_observation_id"].reason == "capture_failed"
    assert "native step failed" in call.missing["after_observation_id"].detail


@pytest.mark.parametrize(
    ("method", "stopped"),
    [
        ("step", "native_noop"),
        ("step", "budget"),
        ("step", "ended"),
        ("step_joint_update", "native_noop"),
        ("step_joint_update", "budget"),
        ("step_joint_update", "ended"),
    ],
)
def test_unexecuted_actions_are_not_recorded(
    scene, tmp_path, monkeypatch, transport, make_server_and_client, method, stopped
):
    scene.agent.eval_fair = True
    scene.agent.meta["mode"] = "eval-fair"
    if stopped == "native_noop":
        monkeypatch.setattr(
            scene.agent._sub_env(), "apply_action", lambda *a, **kw: None
        )

    with make_server_and_client(RoboDojoEnvFacade(scene.agent), transport) as rpc:
        client = RoboDojoEnvClient(rpc, expected_meta=scene.agent.get_env_meta())
        toolkit = RoboDojoToolkit(
            primitives_kwargs={"env": client},
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
            state_output_dir=tmp_path / "state",
            eval_fair=True,
        )
        if stopped == "budget":
            scene.step_lim = scene.take_action_cnt[0]
        elif stopped == "ended":
            scene.end_flag[0] = True
        state = toolkit.state
        state.start_task(instruction="pick bottle", config=Payload("json", {}))
        tool_id = state.register_tool_call(
            planner_id=state.register_planner(backend="api"),
            name="set_gripper",
            arguments=Payload("json", {"arm": "left", "val": 1.0}),
        )
        with ToolCallCapture(state, tool_id).bind():
            if method == "step":
                result = toolkit._primitives.env.step(np.zeros(14))
                assert result[3]["executed_actions"] == 0
            else:
                result = toolkit._primitives.env.step_joint_update("left", gripper=1.0)
                assert result["executed_actions"] == 0

    assert scene.executed == []
    assert state.action_records() == []
    call = state.get_tool_call(tool_id)
    assert call.before_observation_id is None
    assert call.after_observation_id is None


def action():
    return [0.1] * 12 + [0.25, 0.75]


@pytest.mark.parametrize("eval_fair", [False, True])
@pytest.mark.parametrize("expected_step", [None, 0])
def test_server_terminal_guard_never_updates_controller_targets(
    boundary, monkeypatch, eval_fair, expected_step
):
    b = boundary
    b.agent.eval_fair = eval_fair
    if eval_fair:
        monkeypatch.setattr(
            b.agent, "get_status", type(b.agent).get_status.__get__(b.agent)
        )
        b.native.is_success = lambda **kw: pytest.fail("success queried in eval-fair")
    monkeypatch.setattr(
        b.agent, "get_motion_state", lambda: pytest.fail("terminal targets read")
    )
    b.native.end_flag[0] = True
    for operation in (
        lambda: b.facade.step(action(), expected_step=expected_step),
        lambda: b.facade.step_joint_update(
            "right", gripper=0.5, expected_step=expected_step
        ),
    ):
        receipt = operation()
        info = receipt[3] if isinstance(receipt, tuple) else receipt
        assert info["executed_actions"] == 0 and info["truncated"]
        assert not info["terminated"]
        if isinstance(receipt, tuple):
            assert "status" in info and "episode_status" not in info
        else:
            assert "episode_status" in info and "status" not in info
        if eval_fair:
            assert "success" not in info.get("status", info.get("episode_status"))
    assert not b.applied
