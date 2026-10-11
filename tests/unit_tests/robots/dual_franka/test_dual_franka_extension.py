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

from robots.dual_franka import get_robot_spec
from robots.dual_franka.prompt_bundle import system_prompt, user_prompt
from robots.dual_franka.runtime_config import load_runtime_config
from robots.dual_franka.tasks import DUAL_FRANKA_TASKS
from rpent.prompt.utils import format_prompt
from rpent.robots.base import enumerate_robots
from rpent.robots.base import get_robot_spec as resolve_robot_spec


def test_dual_franka_extension_is_discoverable():
    assert "dual_franka" in enumerate_robots()
    spec = resolve_robot_spec("dual_franka")
    assert spec.name == "dual_franka"
    assert spec.supports_exploration is True
    assert get_robot_spec().name == spec.name
    runtime_components = {item["name"] for item in spec.dashboard["runtime_components"]}
    assert "sam3" in runtime_components


def test_franka_extensions_declare_real_robot():
    single = resolve_robot_spec("franka")
    dual = resolve_robot_spec("dual_franka")
    assert single.is_real_robot and dual.is_real_robot


def test_dual_franka_uses_rpent_owned_robot_config(fake_rlinf_realworld_modules):
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
    assert runtime.controller["joint_health_thresholds"]["left"]["warning_q1"] == 1.7


def test_dirty_clean_exploration_candidate_reuses_deployed_task_prompt():
    task = DUAL_FRANKA_TASKS[3]
    candidate = DUAL_FRANKA_TASKS[4]

    assert candidate.instruction == task.instruction
    assert candidate.setup == task.setup
    assert candidate.success_criteria == task.success_criteria
    assert candidate.constraints != task.constraints


def test_dual_franka_exploration_prompt_is_opt_in():
    eval_vars = {"mode": "eval"}
    explore_vars = {
        "mode": "explore",
        "task_id": 4,
        "session_number": 1,
        "session_max": 1,
        "recipe_tag": "dual_franka_t4",
        "memory_dir": "/tmp/memory",
        "task_name": "offline",
        "instruction": "test instruction",
        "setup": "test setup",
        "success_criteria": "test success",
        "constraints": "1. test constraint",
        "output_dir": "/tmp/run",
        "memory_inbox": "/tmp/memory/_internal/inbox/dual_franka_t4",
    }
    eval_prompt = format_prompt(system_prompt(eval_vars), variables=eval_vars)
    explore_prompt = format_prompt(
        system_prompt(explore_vars),
        variables=explore_vars,
    )
    explore_user_prompt = format_prompt(
        user_prompt(explore_vars),
        variables=explore_vars,
    )

    assert "request_scene_reset" not in eval_prompt
    assert "request_scene_reset" in explore_prompt
    assert "request_operator_verdict" in explore_prompt
    assert "/tmp/memory/_internal/inbox/dual_franka_t4" in explore_user_prompt
