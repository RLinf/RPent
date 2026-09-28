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
            def wait():
                waited.append((arm, qpos))
                return [None]

            return SimpleNamespace(wait=wait)

        return SimpleNamespace(reset_joint=reset_joint)

    raw = SimpleNamespace(
        _left_ctrl=controller("left"), _right_ctrl=controller("right")
    )
    worker._raw_rlinf_env = lambda: raw
    assert worker._reset_both_joints_no_gripper([[1], [2]]) == {
        "left": [None],
        "right": [None],
    }
    assert sorted(waited) == [("left", [1]), ("right", [2])]


def test_missing_gripper_state_is_not_silently_defaulted(worker_classes):
    worker = worker_classes[1].__new__(worker_classes[1])
    worker._arm_states = lambda: (SimpleNamespace(), SimpleNamespace())
    with pytest.raises(AttributeError, match="gripper_open"):
        worker._current_gripper_commands()


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
def test_live_environment_does_not_inherit_coding_profile(tmp_path, dedicated):
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
    command = """source "$1"
[[ "$CODEX_HOME" != "$HOME/coding" ]]
[[ -z "${OPENAI_API_KEY:-}" && -z "${OPENAI_BASE_URL:-}" ]]
[[ "${CODEX_API_KEY:-}" == "${RPENT_CODEX_API_KEY:-}" ]]
[[ "${CODEX_BASE_URL:-}" == "${RPENT_CODEX_BASE_URL:-}" ]]
[[ "$CODEX_SERVICE_TIER" == fast && "$RPENT_REASONING_EFFORT" == medium ]]
"""
    subprocess.run(
        ["bash", "-eu", "-c", command, "test", str(script)], env=env, check=True
    )
