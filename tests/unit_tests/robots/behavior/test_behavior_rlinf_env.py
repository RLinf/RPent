# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Native simulator boundary contracts; these CPU fakes do not validate Isaac."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.behavior.motion import R1ProMotionModel
from robots.behavior.rlinf_env import PublicBehaviorEnv, RlinfBehaviorEnv
from robots.behavior.tools import extract_public_proprio, prepare_behavior_observation


class FakeSimulator:
    """Mutable buffers and actual per-step accounting at the public RLinf API."""

    num_envs = 1
    robot_name = "robot_r1"
    cameras = {
        "head": "robot_r1/zed_link:Camera:0",
        "left_wrist": "robot_r1/left_realsense_link:Camera:0",
        "right_wrist": "robot_r1/right_realsense_link:Camera:0",
    }

    def __init__(self, frequency=30):
        self.native_env = SimpleNamespace(env_config={"action_frequency": frequency})
        self.reset_calls = 0
        self.actions = []
        self.physics_steps = 0
        self.episode_steps = 0
        self.terminate_at = None
        self.truncate_at = None
        self.fail_at = None
        self.after_step = None
        self.close_calls = 0
        self._make_buffers()

    def _make_buffers(self):
        self.names = (
            [f"base_joint{i}" for i in range(6)]
            + [f"torso_joint{i}" for i in range(1, 5)]
            + [f"left_arm_joint{i}" for i in range(1, 8)]
            + [f"left_gripper_finger_joint{i}" for i in range(1, 3)]
            + [f"right_arm_joint{i}" for i in range(1, 8)]
            + [f"right_gripper_finger_joint{i}" for i in range(1, 3)]
        )
        self.q = np.arange(28, dtype=np.float64) / 100
        self.dq = np.zeros(28)
        self.robot = {"proprio": np.arange(256, dtype=np.float32)}
        for camera in self.cameras.values():
            self.robot[camera] = {
                "rgb": np.zeros((4, 6, 3), np.uint8),
                "depth_linear": np.full((4, 6, 1), 2.0, np.float32),
                "seg_semantic": np.full((4, 6), 7, np.int32),
            }
        self.observation = {self.robot_name: self.robot, "task": {"oracle": 99}}

    def reset(self):
        self.reset_calls += 1
        self.episode_steps = 0
        self._make_buffers()
        return [self.observation], [{"instance_id": 301, "episode_length": 0}]

    def step(self, actions, *, render=True):
        assert render is True
        actions = np.asarray(actions)
        assert actions.shape == (1, 23)
        self.actions.append(actions.copy())
        self.physics_steps += 1
        if self.fail_at == self.physics_steps:
            # Native execution can advance before failing; do not invent an ack.
            self.q += 0.25
            raise RuntimeError("native failure after partial physics")
        self.episode_steps += 1
        self.q[10] += 0.001
        for camera in self.cameras.values():
            self.robot[camera]["rgb"].fill(self.episode_steps)
        terminated = self.episode_steps == self.terminate_at
        truncated = self.episode_steps == self.truncate_at
        info = {
            **self.get_metrics()[0],
            "success": terminated,
            "image_freshness": "rendered",
        }
        if self.after_step is not None:
            self.after_step()
        return (
            [self.observation],
            np.array([0.25]),
            np.array([terminated]),
            np.array([truncated]),
            [info],
        )

    def get_robot_state(self):
        return {
            "robot_name": self.robot_name,
            "joint_names": self.names,
            "joint_positions": self.q,
            "joint_velocities": self.dq,
            "robot_frame_base_velocity": np.array([0.1, -0.2, 0.3]),
            "raw_virtual_joint_velocity": np.array([8.0, 9.0, 10.0]),
            "world_pose": {
                "position": np.array([2.0, 3.0, 0.0]),
                "orientation_xyzw": Rotation.from_euler(
                    "z", 90, degrees=True
                ).as_quat(),
            },
            "eef_poses": {
                arm: {
                    "robot_base": {
                        "position": np.array([0.5, sign * 0.25, 0.8]),
                        "orientation_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
                    }
                }
                for arm, sign in (("left", 1), ("right", -1))
            },
        }

    def get_camera_metadata(self):
        return {
            role: {
                "sensor_name": name.replace("/", ":", 1),
                "intrinsic_matrix": np.array(
                    [[10 + i, 0, 2], [0, 20 + i, 1], [0, 0, 1]]
                ),
                "height": 4,
                "width": 6,
                "world_pose": {
                    "position": np.array([2.0, 4.0 + i, 0.5]),
                    "orientation_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
                },
            }
            for i, (role, name) in enumerate(self.cameras.items())
        }

    def get_metrics(self):
        return [
            {
                "official_metrics": {}
                if not self.episode_steps
                else {
                    "q_score": {"final": 0.4},
                    "time": {"simulator_steps": self.episode_steps},
                },
                "q_score": 0.4 if self.episode_steps else None,
                "instance_id": 301,
                "episode_length": self.episode_steps,
            }
        ]

    def close(self):
        self.close_calls += 1


@pytest.fixture
def native():
    simulator = FakeSimulator()
    env = RlinfBehaviorEnv(simulator, task_id=40, task_language="make popcorn")
    return env, simulator


def test_native_reset_maps_measured_state_and_actual_camera_calibration(native):
    env, simulator = native
    assert simulator.reset_calls == 0
    raw, info = env.reset()
    assert simulator.reset_calls == 1 and simulator.physics_steps == 0
    public = prepare_behavior_observation(raw, task_id=40, task_language="make popcorn")
    state = extract_public_proprio(raw)
    np.testing.assert_allclose(state[:3], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(state[3:10], simulator.q[10:17])
    np.testing.assert_allclose(state[28:35], simulator.q[19:26])
    np.testing.assert_allclose(state[53:57], simulator.q[6:10])
    np.testing.assert_allclose(state[24:26], simulator.q[17:19])
    np.testing.assert_allclose(state[49:51], simulator.q[26:28])
    np.testing.assert_array_equal(
        public["policy_observation"]["robot_r1::proprio"], simulator.robot["proprio"]
    )
    assert public["policy_observation"]["robot_r1::proprio"].shape == (256,)
    assert public["proprio"].shape == (61,)
    np.testing.assert_array_equal(
        public["policy_observation"]["robot_r1::proprio"][-3:], [253, 254, 255]
    )
    assert "task" not in public["policy_observation"]
    for i, role in enumerate(simulator.cameras):
        view = public["views"][role]
        assert view["rgb"].shape == (4, 6, 3)
        np.testing.assert_array_equal(view["depth"], np.full((4, 6), 2))
        metadata = view["camera_meta"]
        assert metadata["calibration_available"] is True
        np.testing.assert_array_equal(
            metadata["intrinsic_K"], [[10 + i, 0, 2], [0, 20 + i, 1], [0, 0, 1]]
        )
        # Robot is at (2,3), yaw=+90 degrees: world +Y is robot +X.
        np.testing.assert_allclose(
            np.asarray(metadata["camera_to_base_gl"])[:3, 3],
            [1 + i, 0, 0.5],
            atol=1e-12,
        )
    assert info["episode_length"] == 0 and info["q_score"] is None


def test_reads_are_owned_and_do_not_create_control_frames(native):
    env, simulator = native
    env.reset()
    before = env.get_motion_snapshot()
    copy = env.get_observation()
    copy["robot_r1::proprio"][:] = -1
    copy["robot_r1::robot_r1/zed_link:Camera:0::rgb"][:] = 255
    for _ in range(3):
        env.get_observation()
        env.session_status()
    after = env.get_motion_snapshot()
    assert before["observation_token"] == after["observation_token"]
    np.testing.assert_array_equal(
        before["motion_context"]["current_proprio"],
        after["motion_context"]["current_proprio"],
    )
    assert simulator.physics_steps == 0
    assert np.all(
        after["observation"]["robot_r1::robot_r1/zed_link:Camera:0::rgb"] == 0
    )


def test_completed_chunk_keeps_adjacent_history_and_actual_frames(native):
    env, simulator = native
    env.reset()
    initial = env.get_motion_snapshot()
    commands = np.arange(4 * 23, dtype=np.float64).reshape(4, 23) / 100
    frames, reward, terminated, truncated, info = env.chunk_step(
        commands, return_all_frames=True
    )
    assert len(frames) == simulator.physics_steps == 4
    for i, frame in enumerate(frames, 1):
        assert np.all(frame["robot_r1::robot_r1/zed_link:Camera:0::rgb"] == i)
    np.testing.assert_array_equal(np.concatenate(simulator.actions), commands)
    assert (
        info["issued_actions"]
        == info["confirmed_actions"]
        == info["executed_actions"]
        == 4
    )
    assert info["unconfirmed_actions"] == 0
    context = info["motion_context"]
    assert context["current_token"]["observation_index"] == 4
    assert context["previous_token"]["observation_index"] == 3
    assert initial["observation_token"]["observation_index"] == 0
    assert context["pair_confirmed"] is True
    assert context["control_dt_s"] == pytest.approx(1 / 30)
    assert not terminated and not truncated
    assert env.session_status()["q_score"] == 0.4


@pytest.mark.parametrize("kind", ["terminate_at", "truncate_at"])
def test_native_terminal_condition_stops_chunk_at_actual_last_step(native, kind):
    env, simulator = native
    setattr(simulator, kind, 2)
    env.reset()
    frames, _, term, trunc, info = env.chunk_step(
        np.zeros((5, 23)), return_all_frames=True
    )
    assert len(frames) == simulator.physics_steps == 2
    assert bool(term) is (kind == "terminate_at")
    assert bool(trunc) is (kind == "truncate_at")
    assert info["executed_actions"] == info["confirmed_actions"] == 2
    assert env.session_status()["ended"] is True
    # Rejected/empty terminal calls may return feedback; neither may run physics.
    try:
        env.chunk_step(np.zeros((1, 23)))
    except RuntimeError:
        pass
    assert simulator.physics_steps == 2


def test_failed_native_step_keeps_prefix_and_requires_explicit_reset(native):
    env, simulator = native
    env.reset()
    simulator.fail_at = 2
    frames, _, _, _, info = env.chunk_step(np.zeros((4, 23)), return_all_frames=True)
    assert len(frames) == 1 and simulator.physics_steps == 2
    assert info["issued_actions"] == 2
    assert info["confirmed_actions"] == info["executed_actions"] == 1
    assert info["unconfirmed_actions"] == 1
    assert info["observation_after_last_issued_action"] is False
    assert "unknown" in env.session_status()["outcome"]
    try:
        env.chunk_step(np.ones((2, 23)))
    except RuntimeError:
        pass
    assert len(simulator.actions) == 2
    old_token = deepcopy(info["observation_token"])
    simulator.fail_at = None
    env.reset()
    snapshot = env.get_motion_snapshot()
    assert snapshot["observation_token"]["generation"] == old_token["generation"] + 1
    assert snapshot["observation_token"]["observation_index"] == 0
    result = env.chunk_step(np.ones((1, 23)))
    assert result[4]["executed_actions"] == 1


def test_stale_planning_frame_is_rejected_before_any_action(native):
    env, simulator = native
    env.reset()
    token = env.get_motion_snapshot()["observation_token"]
    env.chunk_step(np.zeros((1, 23)))
    with pytest.raises(RuntimeError, match="stale_observation"):
        env.chunk_step(np.ones((2, 23)), expected_observation_token=token)
    assert simulator.physics_steps == 1


def test_stop_after_one_step_does_not_claim_remaining_actions_or_close_sdk(native):
    env, simulator = native
    env.reset()
    simulator.after_step = env.stop
    result = env.chunk_step(np.zeros((5, 23)), return_all_frames=True)
    assert simulator.physics_steps == len(result[0]) == 1
    assert result[4]["executed_actions"] == result[4]["issued_actions"] == 1
    assert result[4]["unconfirmed_actions"] == 0
    assert simulator.close_calls == 0


def test_motion_does_not_assume_30hz_for_a_60hz_environment():
    env = RlinfBehaviorEnv(
        FakeSimulator(frequency=60), task_id=40, task_language="make popcorn"
    )
    env.reset()
    snapshot = env.get_motion_snapshot()
    assert snapshot["motion_context"]["control_dt_s"] == pytest.approx(1 / 60)
    contract = R1ProMotionModel.planning_state_contract(
        extract_public_proprio(snapshot["observation"]), snapshot["motion_context"]
    )
    assert contract["valid"] is False


def test_native_access_cannot_move_to_a_transport_thread(native):
    from concurrent.futures import ThreadPoolExecutor

    env, simulator = native
    env.reset()
    with ThreadPoolExecutor(max_workers=1) as worker:
        for operation in (env.get_observation, env.get_motion_snapshot, env.reset):
            with pytest.raises(RuntimeError, match="owning thread"):
                worker.submit(operation).result(timeout=2)
        worker.submit(env.stop).result(timeout=2)
    assert simulator.physics_steps == 0 and simulator.reset_calls == 1


@pytest.mark.parametrize(
    "actions", [np.zeros((0, 23)), np.zeros((2, 22)), np.full((1, 23), np.nan)]
)
def test_invalid_native_actions_never_reach_the_simulator(native, actions):
    env, simulator = native
    env.reset()
    with pytest.raises(ValueError):
        env.chunk_step(actions)
    assert simulator.physics_steps == 0 and simulator.actions == []


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("width", 224, "dimensions"),
        ("depth_type", "distance_to_camera", "depth annotator"),
    ],
)
def test_tools_reject_native_calibration_for_a_different_image(
    native, field, value, reason
):
    env, _ = native
    raw, _ = env.reset()
    raw["camera_metadata"]["head"][field] = value
    with pytest.raises(ValueError, match=reason):
        prepare_behavior_observation(raw, task_id=40, task_language="make popcorn")


def test_public_state_uses_joint_names_even_when_native_order_changes(
    native, monkeypatch
):
    env, simulator = native
    original = simulator.get_robot_state
    order = np.arange(28)[::-1]

    def reordered_state():
        state = original()
        state["joint_names"] = [state["joint_names"][i] for i in order]
        state["joint_positions"] = state["joint_positions"][order]
        state["joint_velocities"] = state["joint_velocities"][order]
        return state

    monkeypatch.setattr(simulator, "get_robot_state", reordered_state)
    raw, _ = env.reset()
    state = extract_public_proprio(raw)
    np.testing.assert_allclose(state[3:10], simulator.q[10:17])
    np.testing.assert_allclose(state[28:35], simulator.q[19:26])
    np.testing.assert_allclose(state[53:57], simulator.q[6:10])


def test_camera_rpc_uses_same_confirmed_frame_without_an_extra_step(native):
    from robots.behavior.env_client import BehaviorEnvClient
    from robots.behavior.env_server import BehaviorEnvFacade
    from tests.unit_tests.robots.behavior.conftest import LocalRpc

    env, simulator = native
    env.reset()
    client = BehaviorEnvClient(LocalRpc(BehaviorEnvFacade(env, task_id=40)), task_id=40)
    metadata = client.get_camera_meta("head")
    image = client.render_camera("head")
    assert metadata["width"] == 6 and image.shape == (4, 6, 3)
    image[:] = 255
    metadata["intrinsic_K"][0][0] = 999
    assert not client.render_camera("head").any()
    assert client.get_camera_meta("head")["intrinsic_K"][0][0] == 10
    assert simulator.physics_steps == 0


def test_adaptation_failure_after_native_execution_is_not_reported_as_unexecuted(
    native, monkeypatch
):
    env, simulator = native
    env.reset()
    metadata = simulator.get_camera_metadata

    def invalid_metadata():
        result = metadata()
        if simulator.physics_steps == 2:
            result["head"]["width"] = 99
        return result

    monkeypatch.setattr(simulator, "get_camera_metadata", invalid_metadata)
    result = env.chunk_step(np.zeros((5, 23)), return_all_frames=True)
    info = result[4]
    assert simulator.physics_steps == 2 and len(result[0]) == 1
    assert info["code"] == "observation_adapter_failed"
    assert info["native_step_returned"] is True
    assert info["issued_actions"] == 2 and info["confirmed_actions"] == 1
    assert info["unconfirmed_actions"] == 1
    assert result[2:4] == (None, None)
    env.chunk_step(np.zeros((1, 23)))
    assert simulator.physics_steps == 2


@pytest.mark.parametrize("phase", ["native_step", "observation_adapter"])
@pytest.mark.parametrize("prime_observation", [False, True])
def test_first_step_failure_preserves_unknown_counts_through_the_client(
    native, monkeypatch, phase, prime_observation
):
    from robots.behavior.env_client import BehaviorEnvClient
    from robots.behavior.env_server import BehaviorEnvFacade
    from tests.unit_tests.robots.behavior.conftest import LocalRpc

    env, simulator = native
    env.reset()
    client = BehaviorEnvClient(LocalRpc(BehaviorEnvFacade(env, task_id=40)), task_id=40)
    if prime_observation:
        client.get_observation()
    previous = client.last_obs
    token = env.get_motion_snapshot()["observation_token"]
    if phase == "native_step":
        simulator.fail_at = 1
    else:
        original = simulator.get_camera_metadata

        def invalid_metadata():
            result = original()
            result["head"]["width"] = 99
            return result

        monkeypatch.setattr(simulator, "get_camera_metadata", invalid_metadata)
    result = client.chunk_step(
        np.zeros((3, 23)), return_all_frames=True, expected_observation_token=token
    )
    assert result[0] == [] and result[2:4] == (None, None)
    assert result[4]["code"] == f"{phase}_failed"
    assert result[4]["issued_actions"] == result[4]["unconfirmed_actions"] == 1
    assert result[4]["confirmed_actions"] == result[4]["executed_actions"] == 0
    assert client.last_obs is previous
    assert client.last_info == result[4]
    assert simulator.physics_steps == 1
    blocked = client.chunk_step(np.zeros((1, 23)), return_all_frames=True)
    assert blocked[0] == [] and blocked[4]["issued_actions"] == 0
    assert simulator.physics_steps == 1


class FakePublicEnv:
    """Single-slot tensor layout returned by BehaviorEnv.reset and chunk_step."""

    num_envs = 1

    def __init__(self, lengths, *, end=None):
        from omegaconf import OmegaConf

        self.cfg = OmegaConf.create({"omni_config": {"env": {"action_frequency": 30}}})
        self.lengths = lengths
        self.end = end
        self.actions = []

    def observation(self):
        tick = len(self.actions)
        state = np.zeros((1, 61), dtype=np.float32)
        state[0, [23, 48]] = 1  # unit xyzw end-effector quaternions
        state[0, 3] = tick / 10
        wrists = np.stack([np.full((2, 4, 3), k + tick, np.uint8) for k in (20, 30)])
        return {
            "states": state,
            "main_images": np.full((1, 3, 5, 3), 10 + tick, np.uint8),
            "wrist_images": wrists[None],
            "main_depths": np.full((1, 3, 5), 1.5, np.float32),
            "extra_view_depths": np.full((1, 2, 2, 4), 2.5, np.float32),
            "task_descriptions": ["make popcorn"],
        }

    @staticmethod
    def info(length):
        return {"episode": {"episode_length": np.array([length]), "success": [False]}}

    def reset(self):
        self.actions.clear()
        return self.observation(), self.info(0)

    def chunk_step(self, actions):
        assert np.shape(actions) == (1, 1, 23)
        self.actions.append(np.asarray(actions)[0, 0].copy())
        tick = len(self.actions)
        last = tick == len(self.lengths)
        return (
            [self.observation()],
            np.array([[0.25]]),
            np.array([[last and self.end == "terminated"]]),
            np.array([[last and self.end == "truncated"]]),
            [self.info(self.lengths[tick - 1])],
        )


@pytest.mark.parametrize(("first", "end"), [(0, "terminated"), (1, "truncated")])
def test_public_env_reads_three_views_and_stops_at_confirmed_terminal(first, end):
    from robots.behavior.env_client import BehaviorEnvClient
    from robots.behavior.env_server import BehaviorEnvFacade
    from tests.unit_tests.robots.behavior.conftest import LocalRpc

    simulator = FakePublicEnv([first, first + 1], end=end)
    options = {"task_id": 40, "task_language": "make popcorn"}
    env = PublicBehaviorEnv(simulator, **options)
    env.reset()
    client = BehaviorEnvClient(LocalRpc(BehaviorEnvFacade(env, **options)), task_id=40)
    initial = prepare_behavior_observation(client.get_observation(), **options)
    np.testing.assert_array_equal(
        initial["proprio"], simulator.observation()["states"][0]
    )
    for role, color, depth in (
        ("head", 10, 1.5),
        ("left_wrist", 20, 2.5),
        ("right_wrist", 30, 2.5),
    ):
        view = initial["views"][role]
        assert np.all(view["rgb"] == color) and np.all(view["depth"] == depth)
        assert "camera_meta" not in view
    assert client.get_camera_meta("head") == {}

    actions = np.zeros((4, 23), dtype=np.float32)
    actions[:, 0] = [0.1, 0.2, 0.3, 0.4]
    frames, reward, term, trunc, info = client.chunk_step(
        actions, return_all_frames=True
    )
    np.testing.assert_array_equal(simulator.actions, actions[:2])
    assert len(frames) == info["confirmed_actions"] == info["executed_actions"] == 2
    assert info["requested_actions"] == 4 and info["unconfirmed_actions"] == 0
    assert reward == 0.5 and (term, trunc) == (end == "terminated", end == "truncated")
    for frame, tick in zip(frames, (1, 2), strict=True):
        assert np.all(
            prepare_behavior_observation(frame, **options)["views"]["head"]["rgb"]
            == 10 + tick
        )
    status = client.get_session_status()
    assert status["episode_length"] == first + 1
    assert "q_score" not in status and "official_metrics" not in status
    assert len(simulator.actions) == 2


def test_public_vla_uses_native_inputs_and_records_its_returned_frame(tmp_path):
    from unittest.mock import Mock

    from robots.behavior.env_client import BehaviorEnvClient
    from robots.behavior.env_server import BehaviorEnvFacade
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient
    from tests.unit_tests.robots.behavior.conftest import LocalRpc, _template_toolkit

    simulator = FakePublicEnv([1, 2])
    options = {"task_id": 40, "task_language": "make popcorn"}
    env = PublicBehaviorEnv(simulator, **options)
    initial, _ = env.reset()
    assert initial.keys() == simulator.observation().keys() | {"task_id"}
    rpc = LocalRpc(BehaviorEnvFacade(env, **options))
    client = BehaviorEnvClient(rpc, task_id=40)
    prediction = np.arange(46, dtype=np.float32).reshape(1, 2, 23) / 100
    model_rpc = Mock()
    model_rpc.call.return_value = prediction
    with _template_toolkit(
        tmp_path,
        ["view_env_state", "vla_act", "finish"],
        env=client,
        model=Pi05VLAClient(model_rpc, embodiment="behavior"),
        instruction_mode="override",
    ) as toolkit:
        rpc.calls.clear()
        result = toolkit.execute_tool("vla_act", {"prompt": "move the left hand"})
        assert not result.is_error, result.data
        model_rpc.call.assert_called_once()
        wire, _ = model_rpc.call.call_args.kwargs["args"]
        assert set(wire) == {
            "main_images",
            "wrist_images",
            "states",
            "task_descriptions",
        }
        for key in ("main_images", "wrist_images", "states"):
            np.testing.assert_array_equal(wire[key], initial[key])
        assert wire["task_descriptions"] == ["move the left hand"]
        np.testing.assert_array_equal(simulator.actions, prediction[0])
        assert rpc.calls.count("env.get_observation") == 1
        assert rpc.calls.index("env.get_observation") < rpc.calls.index(
            "env.chunk_step"
        )
        final = simulator.observation()
        for name, rgb in zip(
            ("head", "left_wrist", "right_wrist"),
            (final["main_images"][0], *final["wrist_images"][0]),
            strict=True,
        ):
            np.testing.assert_array_equal(
                toolkit.state.load(f"{name}_rgb.png", step=1), rgb
            )


def test_public_env_repeated_source_counter_preserves_last_confirmed_frame():
    simulator = FakePublicEnv([1, 1, 2])
    env = PublicBehaviorEnv(simulator, task_id=40, task_language="make popcorn")
    env.reset()
    frames, reward, term, trunc, info = env.chunk_step(
        np.zeros((3, 23)), return_all_frames=True
    )
    assert len(simulator.actions) == info["issued_actions"] == 2
    assert len(frames) == info["confirmed_actions"] == info["unconfirmed_actions"] == 1
    assert reward == 0.25 and term is None and trunc is None
    assert info["stop_reason"] == "execution_outcome_unknown"
    assert info["native_step_returned"] is True
    assert extract_public_proprio(env.get_observation())[3] == pytest.approx(0.1)
    assert np.all(env.render_camera("head") == 11)
