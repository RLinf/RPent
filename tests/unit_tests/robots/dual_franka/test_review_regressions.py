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

"""Offline regression coverage for PR review fixes; never import hardware drivers."""

import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.dual_franka import perception, robot_spec
from robots.dual_franka.tasks import CLEAN_DESK_VLA_PROMPT, DUAL_FRANKA_TASKS
from robots.franka import runtime_config


@pytest.fixture
def worker_classes(monkeypatch):
    modules = {
        "rlinf.scheduler": {"Worker": object},
        "rlinf.envs.real.env": {"RealWorldEnv": object},
        "rlinf.robotics.parts.cameras": {
            "Camera": object,
            "CameraInfo": object,
        },
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    from robots.dual_franka.env_server import _create_worker_class as dual
    from robots.franka.env_server import _create_worker_class as single

    return single(), dual()


def _robot_with_parts(parts):
    return SimpleNamespace(
        child=lambda side: SimpleNamespace(child=parts[side].__getitem__)
    )


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("tracking", [True, False])
@pytest.mark.parametrize("timeout", [False, True])
def test_move_completion_is_independent_of_accuracy(
    worker_classes, monkeypatch, arm, tracking, timeout
):
    from robots.dual_franka import env_server

    worker = worker_classes[1].__new__(worker_classes[1])
    worker.per_arm_dim = 10
    worker.controller = {
        "move_max_step_m": 0.03,
        "move_timeout_s": 1,
        "move_tolerance_m": 0.006,
        "min_iterations": 2,
        "iteration_multiplier": 1,
    }
    clock = [0.0]
    monkeypatch.setattr(env_server, "time", SimpleNamespace(time=lambda: clock[0]))
    poses = np.array([[0.5, 0, 0.4, 0, 0, 0, 1.0]] * 2)
    index = ["left", "right"].index(arm)
    worker._refresh_robot_state = lambda: None
    worker._arm_poses = lambda: (poses[0].copy(), poses[1].copy())
    worker._pose_to_world = lambda arm, p: p.copy()
    worker._pose_from_world = lambda arm, p: p.copy()

    def step(action, *, auto_reset):
        assert not auto_reset
        if tracking:
            poses[index, :3] = action[0, index * 10 : index * 10 + 3]
        if timeout:
            clock[0] = 2.0

    worker.env = SimpleNamespace(step=step)
    result = worker.move_delta(arm, [0.02, 0, 0])
    assert result["ok"]
    assert result["status"] == "completed"
    assert result["message"] == "Action completed."
    assert result["target_reached"] == tracking
    assert result["final_error_m"] == pytest.approx(0 if tracking else 0.02, abs=1e-6)

    def failing_step(*args, **kwargs):
        raise RuntimeError("controller failed")

    worker.env.step = failing_step
    with pytest.raises(RuntimeError, match="controller failed"):
        worker.move_delta(arm, [0.02, 0, 0])


def test_health_scores_preserve_native_robot_errors(worker_classes):
    cls = worker_classes[1]
    worker = cls.__new__(cls)
    worker._refresh_robot_state = lambda: None
    state = {
        "arm_joint_position": [0.0] * 7,
        "has_errors": True,
        "current_errors": {"joint_position_limits_violation": True},
        "joint_collision": [True] + [False] * 6,
    }
    worker._arm_states = lambda: (dict(state), dict(state))
    worker.action_dim = 20
    worker.per_arm_dim = 10
    worker.action_scale = np.ones(3)
    worker.controller = {"joint_health_thresholds": {"left": {}, "right": {}}}
    worker._raw_rlinf_env = lambda: SimpleNamespace(
        config=SimpleNamespace(joint_reset_qpos=[[0.0] * 7, [0.0] * 7], is_dummy=True)
    )
    result = worker.get_robot_state()
    assert result["joint_health"]["left"]["status"] == "critical"
    assert "joint_collision" in result["joint_health"]["left"]["reasons"]
    for arm in ("left_arm", "right_arm"):
        assert result[arm]["has_errors"] is True
        assert result[arm]["current_errors"] == state["current_errors"]
        assert result[arm]["joint_collision"] == state["joint_collision"]


def test_missing_native_diagnostics_are_unknown(worker_classes):
    native = worker_classes[1]._native_joint_health({})
    assert native["critical_reasons"] == []
    assert native["warning_reasons"] == []
    assert native["summary"]["available"] is False
    assert native["summary"]["status"] == "unknown"
    for key in (
        "has_errors",
        "current_errors",
        "last_motion_errors",
        "joint_contact",
        "cartesian_contact",
        "joint_collision",
        "cartesian_collision",
    ):
        assert native["summary"][key] is None


def test_present_empty_native_diagnostics_are_not_missing(worker_classes):
    state = {
        "robot_mode": "Move",
        "has_errors": False,
        "current_errors": {},
        "last_motion_errors": {},
        "joint_contact": [False] * 7,
        "cartesian_contact": [False] * 6,
        "joint_collision": [False] * 7,
        "cartesian_collision": [False] * 6,
    }
    summary = worker_classes[1]._native_joint_health(state)["summary"]
    assert summary["available"] is True
    assert summary["status"] == "ok"
    assert summary["current_errors"] == []
    assert summary["joint_collision"] is False


def test_joint_health_respects_configured_margin_thresholds(worker_classes):
    cls = worker_classes[1]
    worker = cls.__new__(cls)
    worker.controller = {
        "joint_health_thresholds": {
            "left": {"warning_min_joint_margin": 0.2, "critical_min_joint_margin": 0.1}
        }
    }
    q = [0.0, 0.0, 0.0, -1.5, 0.0, 2.0, 0.0]
    worker._raw_rlinf_env = lambda: SimpleNamespace(
        config=SimpleNamespace(joint_reset_qpos=[q.copy(), q.copy()])
    )
    state = {"arm_joint_position": q}
    assert worker._joint_health_for_arm("left", state)["status"] == "ok"
    q[-1] = 2.8973 - 0.15
    assert worker._joint_health_for_arm("left", state)["status"] == "warning"
    q[-1] = 2.8973 - 0.05
    assert worker._joint_health_for_arm("left", state)["status"] == "critical"


@pytest.mark.parametrize("dual", [False, True])
def test_last_rotation_step_recomputes_error(worker_classes, dual):
    cls = worker_classes[int(dual)]
    worker = cls.__new__(cls)
    pose = np.array([0.5, 0, 0.4, 0, 0, 0, 1.0])
    target = Rotation.from_euler("xyz", [0.1, 0, 0]).as_quat()
    worker.controller = {
        "rotate_timeout_s": 10,
        "rotate_tolerance_rad": 0.001,
        "rotate_max_step_rad": 1,
        "rotate_position_tolerance_m": 0.006,
        "rotate_max_drift_m": 0.03,
        "rotate_integral_gain_per_s": 0.0,
        "rotate_integral_limit_rad": 0.12,
        "rotate_position_integral_gain_per_s": 0.0,
        "rotate_position_integral_limit_m": 0.015,
        "rotate_settle_s": 0,
        "min_iterations": 1,
        "iteration_multiplier": 1,
    }
    worker.action_scale = [1, 1, 1]
    if dual:
        worker.per_arm_dim = 10
        worker._refresh_robot_state = lambda: None
        worker._arm_poses = lambda: (pose.copy(), pose.copy())
        worker._pose_to_world = lambda arm, p: p.copy()
        worker._pose_from_world = lambda arm, p: p.copy()
        worker._hold_action = lambda *args: np.zeros(20)

        def step(*args, **kwargs):
            pose[3:] = target

        worker.env = SimpleNamespace(step=step)
        result = worker.rotate_delta("left", [0.1, 0, 0])
    else:
        worker._raw_tcp_pose = lambda: pose.copy()

        def step(*args, **kwargs):
            pose[3:] = target

        worker._step_delta = step
        result = worker.rotate_delta([0.1, 0, 0])
    assert result["steps_used"] == 1
    assert result["ok"]
    assert result["final_error_rad"] < 1e-6


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize(
    "drift,iterations,expected",
    [
        (0.012, 4, "reached"),
        (0.012, 1, "iteration_limit"),
        (0.04, 4, "position_drift"),
        (0.04, 1, "position_drift"),
    ],
)
def test_rotation_holds_initial_position_and_detects_drift(
    worker_classes, arm, drift, iterations, expected
):
    worker = worker_classes[1].__new__(worker_classes[1])
    worker.per_arm_dim = 10
    worker.controller = {
        "rotate_timeout_s": 10,
        "rotate_tolerance_rad": 0.001,
        "rotate_max_step_rad": 1,
        "rotate_position_tolerance_m": 0.006,
        "rotate_max_drift_m": 0.03,
        "rotate_integral_gain_per_s": 0.0,
        "rotate_integral_limit_rad": 0.12,
        "rotate_position_integral_gain_per_s": 0.0,
        "rotate_position_integral_limit_m": 0.015,
        "rotate_settle_s": 0,
        "min_iterations": iterations,
        "iteration_multiplier": 1,
    }
    poses = np.array([[0.5, -0.2, 0.4, 0, 0, 0, 1.0], [0.4, 0.1, 0.5, 0, 0, 0, 1.0]])
    initial = poses.copy()
    index = ["left", "right"].index(arm)
    frame_rotation = Rotation.from_euler("z", 0.4)
    offset = np.array([0.01, 0.69, 0.0])

    def to_world(name, pose):
        p = pose.copy()
        if name == "left":
            p[:3] = frame_rotation.apply(p[:3]) + offset
            p[3:] = (frame_rotation * Rotation.from_quat(p[3:])).as_quat()
        return p

    def from_world(name, pose):
        p = pose.copy()
        if name == "left":
            p[:3] = frame_rotation.inv().apply(p[:3] - offset)
            p[3:] = (frame_rotation.inv() * Rotation.from_quat(p[3:])).as_quat()
        return p

    worker._pose_to_world = to_world
    worker._pose_from_world = from_world
    worker._refresh_robot_state = lambda: None
    worker._arm_poses = lambda: (poses[0].copy(), poses[1].copy())
    initial_action = worker._hold_action(*initial)
    start = to_world(arm, initial[index])
    target = start.copy()
    target[3:] = (
        Rotation.from_euler("z", 0.6) * Rotation.from_quat(start[3:])
    ).as_quat()
    target_local = from_world(arm, target)
    commands = []

    def step(action, *, auto_reset):
        assert not auto_reset
        command = action[0]
        commands.append(command.copy())
        base = index * 10
        np.testing.assert_allclose(
            command[base : base + 3], initial[index, :3], atol=1e-6
        )
        np.testing.assert_array_equal(command[[9, 19]], [0, 0])
        other = (1 - index) * 10
        np.testing.assert_array_equal(
            command[other : other + 10], initial_action[other : other + 10]
        )
        poses[index] = target_local
        if len(commands) == 1:
            poses[index, 1] += drift

    worker.env = SimpleNamespace(step=step)
    result = worker.rotate_delta(arm, [0, 0, 0.6])
    assert result["exit_reason"] == expected
    assert result["ok"] == (expected != "position_drift")
    assert result["target_reached"] == (expected == "reached")
    assert result["status"] == (
        "aborted" if expected == "position_drift" else "completed"
    )
    assert result["rotation_reached"]
    assert result["max_position_error_m"] == pytest.approx(drift)
    assert len(commands) == (2 if expected == "reached" else 1)
    if expected == "reached":
        assert result["final_position_error_m"] < 1e-6
    else:
        assert not result["position_reached"]


@pytest.mark.parametrize("gain", [0.0, 0.5])
@pytest.mark.parametrize("angle", [0.1, 0.6])
@pytest.mark.parametrize("tracking_bias", [0.06, 0.12, 0.3])
@pytest.mark.parametrize("position_bias", [0.0, 0.01])
def test_rotation_compensates_static_tracking_error(
    worker_classes, monkeypatch, gain, angle, tracking_bias, position_bias
):
    from robots.dual_franka import env_server

    worker = worker_classes[1].__new__(worker_classes[1])
    worker.per_arm_dim = 10
    worker.controller = {
        "rotate_timeout_s": 20,
        "rotate_tolerance_rad": 0.04,
        "rotate_max_step_rad": 0.1,
        "rotate_position_tolerance_m": 0.006,
        "rotate_max_drift_m": 0.03,
        "rotate_integral_gain_per_s": gain,
        "rotate_position_integral_gain_per_s": gain,
        "rotate_position_integral_limit_m": 0.015,
        "rotate_settle_s": 0.5,
        "rotate_integral_limit_rad": 0.12,
        "min_iterations": 200,
        "iteration_multiplier": 1,
    }
    ticks = iter(np.arange(0, 100, 0.1))
    monkeypatch.setattr(
        env_server,
        "time",
        SimpleNamespace(time=lambda: 0.0, monotonic=lambda: next(ticks)),
    )
    pose = np.array([0.5, 0, 0.4, 0, 0, 0, 1.0])
    worker._refresh_robot_state = lambda: None
    worker._arm_poses = lambda: (pose.copy(), pose.copy())
    worker._pose_to_world = lambda arm, p: p.copy()
    worker._pose_from_world = lambda arm, p: p.copy()
    commands = []

    def step(action, **kwargs):
        r6 = action[0, 3:9]
        matrix = np.column_stack([r6[:3], r6[3:], np.cross(r6[:3], r6[3:])])
        reference = Rotation.from_matrix(matrix)
        commands.append(reference)
        pose[:3] = action[0, :3] + [0, position_bias, 0]
        pose[3:] = (Rotation.from_euler("z", -tracking_bias) * reference).as_quat()

    worker.env = SimpleNamespace(step=step)
    result = worker.rotate_delta("left", [0, 0, angle])
    expected_success = bool(gain) and tracking_bias < 0.15
    assert result["ok"]
    assert result["status"] == "completed"
    assert result["target_reached"] == expected_success
    assert np.linalg.norm(result["position_integral"]) <= 0.015001
    assert np.linalg.norm(result["integral_rotvec"]) <= 0.120001
    if angle == 0.6:
        assert commands[1].as_rotvec()[2] == pytest.approx(0.2)
    for previous, current in zip(commands, commands[1:]):
        assert (current * previous.inv()).magnitude() <= 0.100001
    if expected_success:
        assert result["final_error_rad"] <= 0.04
        assert result["final_position_error_m"] <= 0.006
    else:
        assert result["exit_reason"] == "iteration_limit"


@pytest.mark.parametrize("opened,expected", [(True, 1), (False, -1)])
def test_single_arm_motion_preserves_gripper(worker_classes, opened, expected):
    worker = worker_classes[0].__new__(worker_classes[0])
    worker.action_dim = 7
    worker.action_scale = [1, 1, 1]
    worker.use_relative_frame = False
    worker._raw_state = lambda: SimpleNamespace(gripper_open=opened)
    actions = []

    def step(action):
        actions.append(action)
        return ({"states": np.zeros((1, 7))},)

    worker.env = SimpleNamespace(step=step)
    worker._step_delta(np.zeros(3), np.zeros(3), frame="base")
    assert actions[0][0, -1] == expected


def test_selected_perception_config(monkeypatch, tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text("perception:\n  base_frames:\n    marker: custom\n")
    monkeypatch.setattr(runtime_config, "_robot_config_path", path)
    assert perception._load_perception_config()["base_frames"]["marker"] == "custom"


def test_joint_reset_waits_for_both_controller_results(worker_classes):
    worker = worker_classes[1].__new__(worker_classes[1])
    waited = []

    def controller(arm):
        def reset_joint(qpos):
            waited.append((arm, qpos))
            return None

        return SimpleNamespace(reset_joint=reset_joint)

    raw = SimpleNamespace(
        robot=_robot_with_parts(
            {arm: {"arm": controller(arm)} for arm in ("left", "right")}
        )
    )
    worker._raw_rlinf_env = lambda: raw
    assert worker._reset_both_joints_no_gripper([[1], [2]]) == {
        "left": None,
        "right": None,
    }
    assert sorted(waited) == [("left", [1]), ("right", [2])]


def test_missing_hand_state_is_not_silently_defaulted(worker_classes):
    worker = worker_classes[1].__new__(worker_classes[1])
    raw = SimpleNamespace(
        config=SimpleNamespace(is_dummy=False),
        robot=_robot_with_parts(
            {arm: {"end_effector": SimpleNamespace()} for arm in ("left", "right")}
        ),
    )
    worker._raw_rlinf_env = lambda: raw
    with pytest.raises(AttributeError, match="is_open"):
        worker._gripper_states()


def test_refresh_uses_official_tcp_accessor(worker_classes):
    worker = worker_classes[1].__new__(worker_classes[1])
    calls = []
    raw = SimpleNamespace(
        config=SimpleNamespace(is_dummy=False),
        get_tcp_pose=lambda: calls.append("read"),
    )
    worker._raw_rlinf_env = lambda: raw
    worker._refresh_robot_state()
    assert calls == ["read"]
    raw.config.is_dummy = True
    worker._refresh_robot_state()
    assert calls == ["read"]


def test_robot_state_reads_hands_without_mutating_arm_snapshots(worker_classes):
    worker = worker_classes[1].__new__(worker_classes[1])
    arm_state = {
        "arm_joint_position": [0.0] * 7,
        "gripper_open": False,
        "gripper_position": 0.0,
    }
    hands = {
        "left": SimpleNamespace(is_open=True, position=0.08),
        "right": SimpleNamespace(is_open=False, position=0.025),
    }
    raw = SimpleNamespace(
        config=SimpleNamespace(is_dummy=False, joint_reset_qpos=[[0.0] * 7, [0.0] * 7]),
        get_tcp_pose=lambda: None,
        robot=_robot_with_parts(
            {arm: {"end_effector": hand} for arm, hand in hands.items()}
        ),
    )
    worker._raw_rlinf_env = lambda: raw
    worker._arm_states = lambda: (arm_state, arm_state)
    worker.controller = {"joint_health_thresholds": {"left": {}, "right": {}}}
    worker.action_dim = 20
    worker.per_arm_dim = 10
    worker.action_scale = np.ones(3)

    result = worker.get_robot_state()
    assert result["left_arm"]["gripper_open"] is True
    assert result["left_arm"]["gripper_position"] == 0.08
    assert result["right_arm"]["gripper_open"] is False
    assert result["right_arm"]["gripper_position"] == 0.025
    assert arm_state["gripper_open"] is False
    assert arm_state["gripper_position"] == 0.0
    hands["left"].is_open = False
    hands["left"].position = 0.01
    assert worker.get_robot_state()["left_arm"]["gripper_open"] is False
    assert worker.get_robot_state()["left_arm"]["gripper_position"] == 0.01


def test_hold_action_never_reissues_gripper_state(worker_classes):
    worker = worker_classes[1].__new__(worker_classes[1])
    left = np.array([0.5, 0.2, 0.4, 0, 0, 0, 1.0])
    right = np.array([0.4, -0.2, 0.3, 0, 0, 0, 1.0])
    action = worker._hold_action(left, right)
    assert action.shape == (20,)
    np.testing.assert_array_equal(action[[9, 19]], [0, 0])
    np.testing.assert_allclose(action[:3], left[:3])
    np.testing.assert_allclose(action[10:13], right[:3])
    explicit = worker._hold_action(left, right, left_grip=-1, right_grip=1)
    np.testing.assert_array_equal(explicit[[9, 19]], [-1, 1])


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("opened", [False, True])
def test_set_gripper_commands_only_selected_hand(worker_classes, arm, opened):
    worker = worker_classes[1].__new__(worker_classes[1])
    worker.per_arm_dim = 10
    worker.gripper_idx = 9
    worker.controller = {
        "gripper_timeout_s": 1,
        "gripper_max_iterations": 2,
        "gripper_settle_s": 0,
    }
    hands = {
        name: SimpleNamespace(is_open=not opened, position=0.04)
        for name in ("left", "right")
    }
    raw = SimpleNamespace(
        config=SimpleNamespace(is_dummy=False),
        get_tcp_pose=lambda: None,
        robot=_robot_with_parts(
            {name: {"end_effector": hand} for name, hand in hands.items()}
        ),
    )
    worker._raw_rlinf_env = lambda: raw
    pose = np.array([0.5, 0.2, 0.4, 0, 0, 0, 1.0])
    worker._arm_poses = lambda: (pose.copy(), pose.copy())
    worker.get_robot_state = worker._gripper_states
    actions = []

    def step(action, *, auto_reset):
        assert auto_reset is False
        actions.append(action)
        hands[arm].is_open = opened

    worker.env = SimpleNamespace(step=step)
    result = worker.set_gripper(arm.upper(), open=opened)
    assert result["ok"] is True
    assert result["steps_used"] == 1
    selected = ["left", "right"].index(arm)
    assert actions[0][0, selected * 10 + 9] == (1 if opened else -1)
    assert actions[0][0, (1 - selected) * 10 + 9] == 0
    other = "right" if arm == "left" else "left"
    assert hands[other].is_open == (not opened)


@pytest.mark.parametrize("threshold", [0.0, -0.5, float("nan"), float("inf"), 0.5])
def test_worker_requires_a_gripper_noop_band(worker_classes, monkeypatch, threshold):
    from robots.dual_franka import env_server

    events = []
    module = types.ModuleType("robots.dual_franka.rpent_env")
    module.register_rpent_dual_franka_env = lambda: events.append("register")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runtime_config, "set_robot_config_path", lambda path: None)
    monkeypatch.setattr(runtime_config, "validate_calibration_sources", lambda: None)
    env_config = SimpleNamespace(
        binary_gripper_threshold=threshold, action_scale=np.ones(3)
    )
    attrs = {
        "PER_ARM_ACTION_DIM": 10,
        "GRIPPER_IDX_IN_ARM": 9,
        "config": env_config,
    }

    def make_env(*args, **kwargs):
        events.append("construct")
        return SimpleNamespace(
            action_space=SimpleNamespace(shape=(20,)),
            env=SimpleNamespace(call=lambda method, name: [attrs[name]]),
            close=lambda: events.append("close"),
        )

    monkeypatch.setattr(sys.modules["rlinf.envs.real.env"], "RealWorldEnv", make_env)
    cls = env_server._create_worker_class()
    cls.worker_info = None
    cls._open_perception_cameras = lambda self: None
    cfg = SimpleNamespace(env=SimpleNamespace(eval={}))
    if threshold == 0.5:
        cls(cfg, {})
        assert events == ["register", "construct"]
    else:
        with pytest.raises(
            ValueError, match="binary_gripper_threshold must be positive"
        ):
            cls(cfg, {})
        assert events == ["register", "construct", "close"]


@pytest.mark.parametrize("left_open", [False, True])
def test_joint_recovery_preserves_independent_hands(
    worker_classes, monkeypatch, left_open
):
    from robots.dual_franka import env_server

    monkeypatch.setattr(env_server.time, "sleep", lambda seconds: None)
    worker = worker_classes[1].__new__(worker_classes[1])
    expected = {"left": left_open, "right": not left_open}
    calls = []
    hands = {}
    parts = {}
    poses = np.array([[0.5, 0.0, 0.4, 0, 0, 0, 1.0]] * 2)
    for arm in ("left", "right"):
        hand = SimpleNamespace(is_open=expected[arm], position=0.02)

        def command(arm, opened):
            calls.append((arm, "open" if opened else "close"))
            hands[arm].is_open = opened

        def reset_joint(qpos, arm=arm):
            calls.append((arm, "reset_joint"))
            assert hands[arm].is_open == expected[arm]

        hand.open = lambda arm=arm: command(arm, True)
        hand.close = lambda arm=arm: command(arm, False)
        hands[arm] = hand
        parts[arm] = {
            "arm": SimpleNamespace(reset_joint=reset_joint),
            "end_effector": hand,
        }
    raw = SimpleNamespace(
        config=SimpleNamespace(is_dummy=False, joint_reset_qpos=[[0.0] * 7, [0.0] * 7]),
        robot=_robot_with_parts(parts),
        get_tcp_pose=lambda: None,
    )
    worker._raw_rlinf_env = lambda: raw
    worker._arm_states = lambda: (
        {
            "tcp_pose": poses[0],
            "arm_joint_position": [0.0] * 7,
            "gripper_open": False,
        },
        {
            "tcp_pose": poses[1],
            "arm_joint_position": [0.0] * 7,
            "gripper_open": False,
        },
    )
    worker._arm_poses = lambda: (poses[0].copy(), poses[1].copy())
    worker._pose_to_world = lambda arm, pose: pose.copy()
    worker.action_dim = 20
    worker.per_arm_dim = 10
    worker.action_scale = np.ones(3)
    worker.controller = {
        "joint_health_thresholds": {"left": {}, "right": {}},
        "gripper_settle_s": 0,
        "move_tolerance_m": 0.01,
        "rotate_tolerance_rad": 0.1,
    }
    result = worker.recover_joint_posture(return_to_start=False)
    assert result["ok"] is True
    assert result["start"]["gripper_open"] == expected
    assert result["final"]["gripper_open"] == expected
    assert result["final"]["gripper_preserved"] is True
    for arm in ("left", "right"):
        commands = [command for side, command in calls if side == arm]
        hold = "open" if expected[arm] else "close"
        assert commands == [hold, "reset_joint", hold, hold]


def test_exploration_candidate_keeps_policy_instruction():
    for task_id in (1, 3, 4, 5):
        assert DUAL_FRANKA_TASKS[task_id].vla_instruction == CLEAN_DESK_VLA_PROMPT
    assert robot_spec.get_robot_spec().is_real_robot is True


@pytest.mark.parametrize("task_id", [None, 4])
def test_runtime_starts_vla_for_dashboard_and_exploration(
    monkeypatch, tmp_path, task_id
):
    started = []
    monkeypatch.setattr(
        robot_spec,
        "try_spawn_server",
        lambda owned, events, name, fn: started.append(name) or (None, object()),
    )
    monkeypatch.setattr(robot_spec, "try_wait_server", lambda *args, **kwargs: {})
    args = SimpleNamespace(
        task_id=task_id,
        vla_endpoint=None,
        sam3_endpoint=None,
    )
    robot_spec._init_runtime(
        args, tmp_path, SimpleNamespace(emit=lambda event: None), {"vla"}
    )
    assert started == ["vla"]


def test_observation_refreshes_without_reset(worker_classes):
    worker = worker_classes[1].__new__(worker_classes[1])
    events = []
    generation = [0]

    def read():
        events.append("camera")
        generation[0] += 1
        return {"states": np.array([generation[0]])}

    worker._refresh_robot_state = lambda: events.append("state")
    worker._raw_rlinf_env = lambda: SimpleNamespace(_get_observation=read)
    worker.env = SimpleNamespace(_wrap_obs=lambda obs: obs)
    worker._observation_payload = lambda obs: obs
    first = worker.get_observation()
    second = worker.get_observation()
    assert events == ["state", "camera", "state", "camera"]
    assert np.asarray(first["states"]).item() == 1
    assert np.asarray(second["states"]).item() == 2
    assert not hasattr(worker, "last_obs")


def test_chunk_uses_step_observation_without_second_acquisition(worker_classes):
    worker = worker_classes[1].__new__(worker_classes[1])
    seen = []

    def step(action, *, auto_reset):
        assert auto_reset is False
        return {"states": action.copy()}, None, False, False, {}

    def payload(obs):
        seen.append(obs)
        return obs

    worker.env = SimpleNamespace(step=step)
    worker._observation_payload = payload
    result = worker.chunk_step([[1], [2]], return_all_frames=True)
    assert len(seen) == 2
    assert np.asarray(result["observation"][1]["states"]).item() == 2
    assert not hasattr(worker, "last_obs")


@pytest.mark.parametrize("dedicated", [False, True])
@pytest.mark.parametrize("deployment_configured", [False, True])
def test_live_environment_does_not_inherit_coding_profile(
    tmp_path, dedicated, deployment_configured
):
    (tmp_path / "repo").mkdir()
    script = (
        Path(__file__).resolve().parents[4] / "robots/dual_franka/rpent_live_env.sh"
    )
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    env.update(
        {
            "RPENT_REPO_ROOT": str(tmp_path / "repo"),
            "CODEX_HOME": str(tmp_path / "coding"),
            "CODEX_API_KEY": "coding-sentinel",
            "CODEX_BASE_URL": "https://coding.invalid",
            "OPENAI_API_KEY": "coding-sentinel",
            "OPENAI_BASE_URL": "https://coding.invalid",
        }
    )
    if dedicated:
        env.update(
            {
                "RPENT_CODEX_HOME": str(tmp_path / "robot"),
                "RPENT_CODEX_API_KEY": "robot-sentinel",
                "RPENT_CODEX_BASE_URL": "https://robot.invalid",
            }
        )
    if deployment_configured:
        env.update(
            {
                "RPENT_CODEX_MODEL": "experiment-model",
                "RPENT_REASONING_EFFORT": "ultra",
                "PI05_CHECKPOINT_PATH": str(tmp_path / "checkpoint"),
                "DUAL_FRANKA_REPO_ID": "experiment-stats",
                "RPENT_ROBOT_CONFIG": str(tmp_path / "robot.yaml"),
            }
        )
    command = """source "$1"
[[ "$CODEX_HOME" != "$HOME/coding" ]]
[[ -z "${OPENAI_API_KEY:-}" && -z "${OPENAI_BASE_URL:-}" ]]
[[ "${CODEX_API_KEY:-}" == "${RPENT_CODEX_API_KEY:-}" ]]
[[ "${CODEX_BASE_URL:-}" == "${RPENT_CODEX_BASE_URL:-}" ]]
[[ "$CODEX_SERVICE_TIER" == fast ]]
[[ "$CODEX_MODEL" == "$RPENT_CODEX_MODEL" ]]
"""
    if deployment_configured:
        command += """
[[ "$RPENT_CODEX_MODEL" == experiment-model ]]
[[ "$RPENT_REASONING_EFFORT" == ultra ]]
[[ "$PI05_CHECKPOINT_PATH" == "$HOME/checkpoint" ]]
[[ "$DUAL_FRANKA_REPO_ID" == experiment-stats ]]
[[ "$RPENT_ROBOT_CONFIG" == "$HOME/robot.yaml" ]]
"""
    else:
        command += """
[[ "$RPENT_CODEX_MODEL" == gpt-5.5 ]]
[[ "$RPENT_REASONING_EFFORT" == medium ]]
[[ -z "${PI05_CHECKPOINT_PATH+x}" && -z "${DUAL_FRANKA_REPO_ID+x}" ]]
"""
    subprocess.run(
        ["bash", "-eu", "-c", command, "test", str(script)], env=env, check=True
    )
