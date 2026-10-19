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

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.dual_franka import perception, robot_spec
from robots.dual_franka.runtime_config import CONTROL
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


@pytest.mark.parametrize("dual,delta", [(False, 0.1), (True, 0.1), (True, 0)])
def test_last_rotation_step_recomputes_error(worker_classes, dual, delta):
    cls = worker_classes[int(dual)]
    worker = cls.__new__(cls)
    pose = np.array([0.5, 0, 0.4, 0, 0, 0, 1.0])
    target = Rotation.from_euler("x", delta).as_quat()
    worker.controller = {
        "move_tolerance_m": 0.006,
        "rotate_timeout_s": 10,
        "rotate_tolerance_rad": 0.001,
        "rotate_max_step_rad": 1,
        "rotate_max_drift_m": 0.03,
        "min_iterations": 1,
        "iteration_multiplier": 1,
    }
    worker.action_scale = [1, 1, 1]
    if dual:
        worker.per_arm_dim = 10
        worker._refresh_robot_state = lambda: None
        readings = iter(
            [pose.copy(), np.r_[pose[:3], Rotation.from_euler("x", 0.1).as_quat()]]
        )
        worker._arm_poses = lambda: (next(readings, pose.copy()), pose.copy())
        worker._pose_to_world = lambda arm, p: p.copy()
        worker._pose_from_world = lambda arm, p: p.copy()
        worker._hold_action = lambda *args: np.zeros(20)

        def step(*args, **kwargs):
            pose[3:] = target

        worker.env = SimpleNamespace(step=step)
        result = worker.rotate_delta("left", [delta, 0, 0])
    else:
        worker._raw_tcp_pose = lambda: pose.copy()

        def step(*args, **kwargs):
            pose[3:] = target

        worker._step_delta = step
        result = worker.rotate_delta([delta, 0, 0])
    assert result["steps_used"] == 1
    assert result["ok"]
    assert result["final_error_rad"] < 1e-6


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


def test_joint_reset_dispatches_both_arm_resets(worker_classes):
    worker = worker_classes[1].__new__(worker_classes[1])
    waited = []

    def controller(arm):
        return SimpleNamespace(reset_joint=lambda qpos: waited.append((arm, qpos)))

    parts = {arm: {"arm": controller(arm)} for arm in ("left", "right")}
    raw = SimpleNamespace(robot=_robot_with_parts(parts))
    worker._raw_rlinf_env = lambda: raw
    assert worker._reset_both_joints_no_gripper([[1], [2]]) == {
        "left": None,
        "right": None,
    }
    assert sorted(waited) == [("left", [1]), ("right", [2])]


def _robot_with_parts(parts):
    return SimpleNamespace(
        child=lambda side: SimpleNamespace(child=parts[side].__getitem__)
    )


@pytest.mark.parametrize(
    "outcome",
    [
        "reached",
        "lagged",
        "stalled",
        "loop_drift",
        "final_drift",
        "corrected",
        "position_failure",
        "move_failure",
        "orientation_failure",
    ],
)
def test_rotation_reference_holds_position_and_checks_drift(worker_classes, outcome):
    worker = object.__new__(worker_classes[1])
    worker.controller = {
        **runtime_config.flatten_control(CONTROL),
        "rotate_max_drift_m": 0.03,
        "min_iterations": 4,
        "iteration_multiplier": 1,
    }
    poses = np.array([[0.5, 0, 0.4, 0, 0, 0, 1.0]] * 2)
    worker._refresh_robot_state = lambda: None
    worker._arm_poses = lambda: tuple(p.copy() for p in poses)
    worker._pose_to_world = lambda arm, pose: pose.copy()
    worker._pose_from_world = lambda arm, pose: pose.copy()
    worker.per_arm_dim = 10
    commands = []
    corrections = []
    needs_correction = outcome in {
        "corrected",
        "position_failure",
        "move_failure",
        "orientation_failure",
    }

    def move_delta(arm, delta):
        assert arm == "left"
        np.testing.assert_allclose(delta, [0, -0.02, 0])
        corrections.append(delta.copy())
        if outcome != "position_failure":
            poses[0, :3] += delta
        if outcome == "orientation_failure":
            poses[0, 3:] = Rotation.identity().as_quat()
        return {"ok": outcome != "move_failure", "steps_used": 2}

    worker.move_delta = move_delta

    def step(action, *, auto_reset):
        assert auto_reset is False
        arms = action.reshape(2, 10)
        np.testing.assert_allclose(arms[0, :3], [0.5, 0, 0.4])
        np.testing.assert_allclose(
            arms[1, :9], np.r_[poses[1, :3], worker._arm_rot6d(poses[1])]
        )
        np.testing.assert_array_equal(arms[:, 9], 0)
        commands.append(float(np.arctan2(arms[0, 8], arms[0, 7])))
        if (
            needs_correction
            or outcome == "reached"
            or (outcome == "lagged" and len(commands) == 4)
        ):
            poses[0, 3:] = Rotation.from_euler("x", 0.33).as_quat()
        drift = (outcome == "loop_drift" and len(commands) == 1) or (
            outcome == "final_drift" and len(commands) == 4
        )
        poses[0, 1] = 0.04 if drift else (0.02 if needs_correction else 0)

    worker.env = SimpleNamespace(step=step)
    if outcome.endswith("drift"):
        with pytest.raises(RuntimeError, match="position drift"):
            worker.rotate_delta("left", [0.33, 0, 0])
    else:
        result = worker.rotate_delta("left", [0.33, 0, 0])
        assert result["ok"] is (outcome in {"reached", "lagged", "corrected"})
        assert result["steps_used"] == 4 + 2 * needs_correction
        np.testing.assert_allclose(result["final_tcp_pose"], poses[0])
        np.testing.assert_allclose(result["start_tcp_pose"], [0.5, 0, 0.4, 0, 0, 0, 1])
        assert result["final_error_rad"] == pytest.approx(
            0.33 if outcome in {"stalled", "orientation_failure"} else 0, abs=1e-6
        )
    assert len(corrections) == int(needs_correction)
    expected = [0.1] if outcome == "loop_drift" else [0.1, 0.2, 0.3, 0.33]
    assert commands == pytest.approx(expected, abs=1e-6)


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


@pytest.mark.parametrize("task_id", [None, 2])
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
        _flash_options=None,
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
