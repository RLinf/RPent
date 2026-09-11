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

"""Offline tests for dual-Franka environment discovery and config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from robots.dual_franka import get_robot_spec
from robots.dual_franka.runtime_config import load_runtime_config
from robots.dual_franka.tasks import CLEAN_DESK_VLA_PROMPT, DUAL_FRANKA_TASKS
from rpent.robots.base import enumerate_robots
from rpent.robots.base import get_robot_spec as resolve_robot_spec


def test_dual_franka_extension_is_discoverable():
    assert "dual_franka" in enumerate_robots()
    spec = resolve_robot_spec("dual_franka")
    assert spec.name == "dual_franka"
    assert get_robot_spec().name == spec.name
    runtime_components = {item["name"] for item in spec.dashboard["runtime_components"]}
    assert "sam3" in runtime_components
    frame_channels = {item["name"] for item in spec.dashboard["frame_channels"]}
    assert "d455" in frame_channels


def test_dual_franka_uses_rpent_owned_robot_config():
    pytest.importorskip("rlinf.envs.realworld.franka.franka_env")
    config_path = Path(__file__).parents[4] / "robots/dual_franka/config/example.yaml"
    runtime = load_runtime_config(None, task_description="test task")
    cfg = runtime.rlinf

    assert config_path.is_file()
    assert cfg.env.eval.init_params.id == "DualFrankaTCPEnv-v1"
    assert cfg.env.eval.override_cfg.task_description == "test task"
    assert runtime.controller["move_tolerance_m"] == 0.006
    assert runtime.controller["rotate_tolerance_rad"] == 0.04
    assert runtime.controller["iteration_multiplier"] == 50
    assert runtime.controller["min_iterations"] == 200
    assert runtime.controller["gripper_settle_s"] == 1.5
    assert runtime.controller["joint_health_thresholds"]["left"]["warning_q1"] == 1.5


def test_clean_desk_task_registers_named_vla_skills_and_fixed_prompt():
    task = DUAL_FRANKA_TASKS[1]

    assert task.name == "clean_desk_dual_franka_agent_vla"
    assert "bowls, plates, cup, chopsticks, then spoon" in task.instruction
    assert "vla_right_grasp" in "\n".join(task.constraints)
    assert "segment" in "\n".join(task.constraints)
    assert "recover_joint_posture" in "\n".join(task.constraints)
    assert CLEAN_DESK_VLA_PROMPT.startswith(
        "I am currently performing a desk organizing task."
    )
