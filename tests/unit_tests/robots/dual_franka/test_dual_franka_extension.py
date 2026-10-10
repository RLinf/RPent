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
from robots.dual_franka.prompt_bundle import system_prompt, user_prompt
from robots.dual_franka.runtime_config import load_mapping, load_runtime_config
from robots.dual_franka.tasks import CLEAN_DESK_VLA_PROMPT, DUAL_FRANKA_TASKS
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


def test_manual_schemas_preserve_native_tool_declarations():
    from robots.dual_franka.dual_franka_manual_call import (
        _registered_tool_names,
        _schema_payload,
    )
    from robots.dual_franka.toolkit import DualFrankaToolkit

    tools = DualFrankaToolkit.declared_tools()
    assert _registered_tool_names() == {tool.name for tool in tools}
    for tool in tools:
        assert _schema_payload(tool.name) == {
            "primitive": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
    assert _schema_payload("reset")["description"].startswith("Manual-only")


def test_dual_franka_uses_rpent_owned_robot_config(fake_rlinf_realworld_modules):
    config_path = Path(__file__).parents[4] / "robots/dual_franka/config/example.yaml"
    runtime = load_runtime_config(None, task_description="test task")
    cfg = runtime.rlinf

    assert config_path.is_file()
    assert cfg.env.eval.init_params.id == "RPentDualFrankaTCPEnv-v1"
    assert cfg.env.eval.override_cfg.task_description == "test task"
    assert runtime.controller["move_tolerance_m"] == 0.006
    assert runtime.controller["rotate_tolerance_rad"] == 0.04
    assert runtime.controller["rotate_position_tolerance_m"] == 0.006
    assert runtime.controller["rotate_max_drift_m"] == 0.03
    assert runtime.controller["rotate_integral_gain_per_s"] == 0.5
    assert runtime.controller["rotate_integral_limit_rad"] == 0.12
    assert runtime.controller["rotate_position_integral_gain_per_s"] == 0.5
    assert runtime.controller["rotate_position_integral_limit_m"] == 0.015
    assert runtime.controller["iteration_multiplier"] == 50
    assert runtime.controller["min_iterations"] == 200
    assert runtime.controller["gripper_settle_s"] == 1.5
    thresholds = runtime.controller["joint_health_thresholds"]
    assert thresholds["left"] == thresholds["right"]
    assert thresholds == load_mapping(config_path)["joint_health"]["thresholds"]


def test_clean_desk_task_registers_named_vla_skills_and_fixed_prompt():
    task = DUAL_FRANKA_TASKS[1]

    assert task.name == "clean_desk_dual_franka_agent_vla"
    assert "bowls, plates, cup, chopsticks, then spoon" in task.instruction
    assert "vla_right_grasp" in "\n".join(task.constraints)
    assert "segment" in "\n".join(task.constraints)
    assert "joint_health" not in "\n".join(task.constraints)
    assert CLEAN_DESK_VLA_PROMPT.startswith(
        "I am currently performing a desk organizing task."
    )


def test_dirty_clean_exploration_candidate_reuses_deployed_task_prompt():
    task = DUAL_FRANKA_TASKS[3]
    candidate = DUAL_FRANKA_TASKS[4]

    assert candidate.name.endswith("_explore_candidate")
    assert candidate.instruction == task.instruction
    assert candidate.setup == task.setup
    assert candidate.success_criteria == task.success_criteria
    assert candidate.constraints != task.constraints
    candidate_constraints = "\n".join(candidate.constraints)
    assert "Build a brief D455 localization and sorting table" not in (
        candidate_constraints
    )
    assert "Use short phrases" not in candidate_constraints
    assert "For bowls, do not use rule-based move_delta" not in candidate_constraints
    assert "10cm above" not in candidate_constraints
    assert "projected center/rim x/y can be far from the right TCP" not in (
        candidate_constraints
    )
    assert "keep the current left TCP z" not in candidate_constraints
    assert "dirty bowls/plates to the metal wire basket/frame" in candidate_constraints


def test_two_cup_sorting_task_extends_exploration_candidate():
    from robots.dual_franka.tasks import get_dual_franka_task

    task = get_dual_franka_task(6)
    original = DUAL_FRANKA_TASKS[4]
    prompt = "\n".join(
        (task.instruction, task.setup, task.success_criteria, *task.constraints)
    )
    assert task.name.endswith("two_cups_explore_candidate")
    assert task.vla_instruction == original.vla_instruction
    assert "exactly two cups" in task.instruction
    assert "egg-tart foil tray" in task.setup
    assert "Chicken nuggets inside a bowl" in task.setup
    assert "bowl marker is not an egg tart or foil tray" in task.setup
    assert "Chicken nuggets inside a bowl" in "\n".join(task.constraints)
    assert "dirty bowls, plates, and cups go to the metal basket" in prompt
    assert "clean bowls, plates, and cups go to the cardboard box" in prompt
    assert "Process bowls -> plates -> both cups." in prompt
    assert "There are no chopsticks or spoons to process" in task.instruction
    assert "chopstick" not in "\n".join(task.constraints)
    assert "spoon" not in task.success_criteria
    assert "If no clean cup is available" not in prompt
    assert "Both cups are accounted for separately" in task.success_criteria
    assert "the cup still goes" not in prompt.lower()
    assert "clean bowls/plates and the cup" not in prompt
    assert "10cm above" not in prompt
    for constraint in (*original.constraints[:2], original.constraints[19]):
        assert constraint in task.constraints
    for index in (8, 10, 12, 14, 18, 20):
        assert original.constraints[index] not in task.constraints
    assert "Contact is not restricted to VLA" in prompt
    assert "does not require stopping for operator feedback" in prompt
    assert "arm/object motion can improve visibility" in prompt
    assert "The cup still goes into the cardboard box" in original.instruction


def test_metal_basket_task_registers_non_sorting_prompt():
    task = DUAL_FRANKA_TASKS[5]
    constraints = "\n".join(task.constraints)

    assert task.name == "clean_desk_all_objects_to_metal_basket_agent_vla"
    assert "without dirty/clean classification" in task.instruction
    assert "fixed category order" in task.instruction
    assert "Every object must go through one right-to-left handoff" in (
        task.instruction
    )
    assert "The spoon must be inserted into the cup" in task.instruction
    assert "no other destination is acceptable" in task.instruction
    assert "outside the cup does not satisfy the task" in task.success_criteria
    assert "Do not perform dirty/clean classification" in constraints
    assert "The metal wire basket/frame is the only valid placement container" in (
        constraints
    )
    assert (
        "Insert the spoon into the cup; no other destination is acceptable"
        in constraints
    )
    assert "a failed grasp or drop is recoverable" in constraints
    assert "Treat the two chopsticks as two separate objects" in task.instruction
    assert "inspect the right_wrist artifact as primary evidence" in constraints
    assert "bowls -> plates -> cup -> chopsticks -> spoon" in constraints
    assert "green object before the blue object" in constraints
    assert "inside_metal_basket without in_cup is a wrong destination" in constraints
    assert "Track the spoon as complete only when inserted into the cup" in constraints


def test_task5_placement_does_not_prescribe_alignment():
    task = DUAL_FRANKA_TASKS[5]
    prompt = format_prompt(system_prompt())
    combined = prompt + "\n" + "\n".join(task.constraints)
    assert "Put bowls and plates into the metal basket" in combined
    for directive in (
        "delta_z=0",
        "strict staging gate",
        "latest staging move",
        "free-space x/y alignment",
    ):
        assert directive not in combined
    for task_id in (1, 3):
        assert "keep the current left TCP z" in "\n".join(
            DUAL_FRANKA_TASKS[task_id].constraints
        )


def test_task6_treats_residual_error_as_diagnostic():
    from robots.dual_franka.prompts import explore, system

    rules = "\n".join(system.RULES)
    assert "target_reached=false alone is not a controller" in rules
    assert "protective motion aborts" in rules
    assert explore.base.RULES is system.RULES
    constraints = "\n".join(DUAL_FRANKA_TASKS[6].constraints)
    assert "proceed even when target_reached is false" in constraints
    assert "Do not repeatedly micro-adjust" in constraints
    assert "no fixed correction size or mandatory alignment" in constraints
    assert "reports a safe/effective target" not in constraints
    assert "does not reach the target, do not call" not in constraints


def test_shared_prompt_does_not_prescribe_task_strategy():
    for mode in ("eval", "explore"):
        prompt = format_prompt(
            system_prompt({"mode": mode}),
            variables={
                "output_dir": "/tmp/test-output",
                "memory_inbox": "/tmp/test-memory",
                "memory_dir": "/tmp/test-memory",
                "session_number": 1,
                "session_max": 1,
                "recipe_tag": "dual_franka_t5",
                "task_id": 5,
            },
        )
        for phrase in (
            "D455",
            "vla_right_grasp",
            "vla_handoff",
            "vla_left_place",
            "current category",
            "delta_z=0",
            "metal basket",
            "active-vision",
            "Use the task's VLA segment tools for contact-rich motion.",
            "before manual pixel projection",
        ):
            assert phrase not in prompt


@pytest.mark.parametrize("task_id", [0, 1, 3, 4, 5, 6])
@pytest.mark.parametrize("explore", [False, True])
def test_rendered_tasks_retain_their_own_operating_instructions(task_id, explore):
    from rpent.cli.main import _build_argparser

    spec = get_robot_spec()
    parser = _build_argparser()
    spec.add_cli_args(parser, use_dashboard=False)
    argv = ["--robot", "dual_franka", "--task-id", str(task_id)]
    if explore:
        argv.append("--explore")
    config = spec.parse_config(parser.parse_args(argv))
    variables = {**config.prompt_vars, "output_dir": str(config.output_dir)}
    task = DUAL_FRANKA_TASKS[task_id]
    system = format_prompt(spec.prompts.system(variables), variables=variables)
    user = format_prompt(spec.prompts.user(variables), variables=variables)
    for phrase in ("joint_health", "joint-health", "recover_joint_posture"):
        assert phrase not in system + user
    for text in (
        task.instruction,
        task.setup,
        task.success_criteria,
        *task.constraints,
    ):
        assert text in user
    if task_id == 0:
        assert "Do not call a VLA skill" in user
        assert "vla_handoff" not in system + user
        return
    for text in ("closure plus lift", "release delay", "left gripper", "only mean"):
        assert text in user
    assert "vla_handoff" in user and "vla_left_place" in user
    if task_id == 5:
        assert (
            "this task imposes no small-motion requirement or fixed adjustment sequence"
            in user
        )
        assert "Missing depth or ambiguous placement alone is not a reason" in user
        assert "stop for operator feedback instead of opening" not in user
    elif task_id != 6:
        assert "Do not move" in user and "camera visibility" in user
    if task_id in (1, 3, 5):
        assert "localization table" in user or "localization and sorting table" in user
        assert "source/table height" in user
    if task_id in (1, 3):
        assert "delta_z=0" in user
    else:
        assert "delta_z=0" not in user + system
    if task_id == 4:
        assert "localization and sorting table" not in user
        assert "source/table height" not in user
    if task_id == 5:
        assert "left gripper actually holds the intended object" in user
        assert "latest staging move" not in user + system
        assert (
            "For spoon placement, the cup interior is the required projection target"
            in user
        )
        assert (
            "Insert the spoon into the cup; no other destination is acceptable" in user
        )
        assert "Put the chopsticks into the metal basket" in user
        assert "bowls -> plates -> cup -> chopsticks -> spoon" in user
        assert "Choose available camera views and perception tools as useful" in user
        assert "RGB-based visual control is allowed" in user
        assert "does not require a successful depth projection first" in user
        assert (
            "Do not present RGB estimates or invalid depth as calibrated metric measurements"
            in user
        )
        assert "Invalid depth is not a valid metric measurement" in user
        for stale_rule in (
            "the cup as a utensil destination",
            "use segment or back_project again instead of",
            "before falling back to a manual pixel",
            "If it is unclear whether the object has lifted or remains on the table, call back_project",
            "the marker is not on the intended target, retry or stop",
        ):
            assert stale_rule not in user


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
