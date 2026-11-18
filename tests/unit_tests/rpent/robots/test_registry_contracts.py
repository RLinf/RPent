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

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from dataclasses import FrozenInstanceError
from pathlib import Path
from string import Formatter

import pytest
import yaml

from robots.robotwin.robot_spec import (
    MODEL_SPEC,
    ROBOTWIN_CAMERA_NAMES,
    env_runtime_contract,
    vla_runtime_contract,
)
from rpent.robots import enumerate_robots, get_robot_spec
from rpent.robots.robot_spec import RobotSpec, RunConfig

EXPECTED_ROBOTS = (
    "dual_franka",
    "franka",
    "libero",
    "robocasa",
    "robodojo",
    "robotwin",
    "yam",
)

PROMPT_VARIABLES = {
    "yam": {
        "task_id": 1,
        "seed": 0,
        "recipe_tag": "yam_t1_s0",
        "instruction": "Move the object to the target",
        "mode": "eval",
        "memory_dir": "/memory",
    },
    "libero": {
        "suite": "libero_object_task",
        "task": 2,
        "seed": 3,
        "recipe_tag": "object_task_t2_s3",
        "mode": "eval",
        "memory_profile": "hf",
        "memory_dir": "/memory",
        "reference_tag": "object_task_t2_s0",
        "memory_inbox": "/memory/_internal/inbox/object_task_t2_s3",
        "session_number": 1,
        "session_max": 1,
        "output_dir": Path("/output"),
    },
    "robocasa": {
        "task_name": "OpenDrawer",
        "split": "target",
        "seed": 3,
        "recipe_tag": "OpenDrawer_target_s3",
        "mode": "eval",
        "memory_profile": "hf",
        "reference_tag": "OpenDrawer_s0",
        "memory_dir": "/memory",
        "output_dir": Path("/output"),
    },
    "robodojo": {
        "task": "put_bottles_into_dustbin",
        "seed": 0,
        "layout": 0,
        "env_cfg_type": "arx_x5",
        "action_type": "joint",
        "recipe_tag": "put_bottles_into_dustbin_l0",
        "reference_tag": "put_bottles_into_dustbin_s0_l0",
        "task_summary": {"task": "put_bottles_into_dustbin", "rigid": ["bottle"]},
        "memory_dir": "/memory",
        "output_dir": Path("/output"),
    },
    "robotwin": {
        "task_name": "block_hammer_beat",
        "seed": 3,
        "task_config": "demo_randomized",
        "instruction": "pick up the hammer",
        "memory_dir": "/memory",
        "mode": "eval",
        "memory_profile": "hf",
        "reference_tag": "block_hammer_beat_s0",
        "output_dir": Path("/output"),
    },
    "franka": {
        "task_id": 1,
        "task_name": "vla_grasp",
        "instruction": "Use bounded analytic motion",
        "success_criteria": "object follows the gripper",
        "constraints": "1. Keep translation at or below 0.02 m",
        "output_dir": Path("/output"),
    },
    "dual_franka": {
        "task_id": 1,
        "task_name": "vla_grasp",
        "instruction": "Use bounded analytic motion",
        "setup": "",
        "success_criteria": "object follows the chosen gripper",
        "constraints": "1. Move exactly one arm per call",
        "output_dir": Path("/output"),
    },
}


def _format_fields(template: str) -> set[str]:
    return {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    }


def test_registry_discovers_exactly_the_source_checkout_robots() -> None:
    assert enumerate_robots() == EXPECTED_ROBOTS
    for name in EXPECTED_ROBOTS:
        spec = get_robot_spec(name)
        assert isinstance(spec, RobotSpec)
        assert spec.name == name
        assert callable(spec.add_cli_args)
        assert callable(spec.parse_config)
        assert callable(spec.init_runtime)
        assert (spec.flywheel is not None) == (
            name in {"libero", "robocasa", "robodojo", "robotwin"}
        )


@pytest.mark.parametrize("robot_name", EXPECTED_ROBOTS)
def test_robot_prompts_render_from_public_spec(robot_name: str, tmp_path) -> None:
    spec = get_robot_spec(robot_name)

    variables = dict(PROMPT_VARIABLES[robot_name])
    if robot_name == "robocasa":
        memory_dir = tmp_path / "robocasa"
        (memory_dir / "global").mkdir(parents=True)
        (memory_dir / "global/GLOBAL_MEMORY.md").write_text("# Global memory\n")
        variables["memory_dir"] = str(memory_dir)
    system = spec.prompts.system(variables)
    user = spec.prompts.user(PROMPT_VARIABLES[robot_name])

    assert system.strip()
    assert user.strip()
    assert "{{" not in system
    assert "{{" not in user


@pytest.mark.parametrize(
    ("mode", "memory_profile"),
    [("eval", "hf"), ("eval", "local"), ("explore", "local")],
)
def test_libero_prompts_observe_task_before_reading_memory(
    mode: str, memory_profile: str
) -> None:
    variables = {
        **PROMPT_VARIABLES["libero"],
        "mode": mode,
        "memory_profile": memory_profile,
        "explore_attempts_per_session": 5,
    }
    prompts = get_robot_spec("libero").prompts
    system = prompts.system(variables)
    user = prompts.user(variables)
    memory_step = "Read memory before acting"
    assert (
        system.index("READ THE GUIDES")
        < system.index("INSPECT INITIAL STATE")
        < system.index(memory_step)
        < system.index("RUN THE MANDATORY PRE-TASK PERCEPTION PASS")
        < system.index("EXECUTE one primitive")
    )
    assert (
        system.index(memory_step)
        < system.index("The current task's published audit and recipe pair")
        < system.index("Matching task-family notes")
        < system.index("Relevant global lessons")
        < system.index("RUN THE MANDATORY PRE-TASK PERCEPTION PASS")
    )
    assert "READ MEMORY FIRST" not in system
    assert (
        'Read the guides, then call `view_env_state({"step": 0})` and inspect\n'
        "`task_language` and `agentview_high.png`. Read matching memory and available\n"
        "seed-0 references for the current task and scene, then localize, plan, and execute."
    ) in system
    assert "view_env_state" not in user


@pytest.mark.parametrize(
    ("robot_name", "task_key"),
    [
        ("libero", "task"),
        ("robocasa", "task_name"),
        ("robodojo", "task"),
        ("robotwin", "task_name"),
        ("dual_franka", "task_id"),
        ("yam", "task_id"),
    ],
)
def test_exploration_memory_binds_current_cell_and_preserves_publication_scope(
    robot_name: str, task_key: str
) -> None:
    prompts = get_robot_spec(robot_name).prompts
    for task_id in (2, 3):
        recipe_tag = f"{robot_name}_cell_{task_id}"
        values = {
            **PROMPT_VARIABLES[robot_name],
            task_key: task_id if task_key != "task_name" else f"task_{task_id}",
            "mode": "explore",
            "recipe_tag": recipe_tag,
            "memory_inbox": f"/memory/_internal/inbox/{recipe_tag}",
            "memory_dir": "/memory",
            "output_dir": f"/output/{recipe_tag}",
            "session_number": 2,
            "session_max": 3,
            "explore_attempts_per_session": 4,
        }
        system = prompts.system(values)
        if robot_name == "yam":
            # YAM keeps its operator-supervised protocol and prose metadata.
            assert f"Your cell is {recipe_tag}, session 2 of 3." in system
            assert f"/memory/_internal/inbox/{recipe_tag}/wip/" in system
            assert f"/memory/_internal/inbox/{recipe_tag}/technique.md" in system
            assert "scope: task-family, suite: yam, regime: real" in system
            assert (
                f"task_id: {task_id}, task_language: the actual env language" in system
            )
            assert f"evidence: {{cells: [{recipe_tag}]}}" in system
            assert "confidence: single-shot" in system
            assert "Only your current inbox is writable" in system
            assert (
                "runner creates the successful current-attempt recipe/audit pair"
                in system
            )
            assert "Unsolved runs do not publish success recipes" in system
            assert (
                "bounded failure lessons may be merged by the common MemoryManager"
                in system
            )
            assert "An operator abort requires finish" in system
            assert "a new attempt or proof of failure" in system
            assert "{{" not in system
            continue
        frontmatter = re.search(r"```yaml\n---\n(.*?)\n---\n```", system, re.DOTALL)
        assert frontmatter is not None
        metadata = yaml.safe_load(frontmatter.group(1))
        assert metadata["scope"] == "task-family"
        assert metadata["task_id"] == values[task_key]
        assert metadata["evidence"]["cells"] == [recipe_tag]
        assert metadata["confidence"] == "single-shot"
        assert f"/memory/_internal/inbox/{recipe_tag}/wip/notes.md" in system
        assert "{{" not in system
        configurations = [
            json.loads(block)
            for block in re.findall(r"```json\n(.*?)\n```", system, re.DOTALL)
        ]
        assert "memory_explore_options" not in system
        assert '"proposal_gate"' not in system
        assert "robot's authoritative success confirmation" in system
        assert metadata["evidence"]["attempts"] == "<observed count>"
        if robot_name == "libero":
            assert "A human reviews and merges the inbox proposals later" in system
            assert "the finalizer merges eligible inbox proposals" not in system
        else:
            assert "the finalizer merges eligible inbox proposals" in system
        if robot_name in {"libero", "robocasa", "robotwin"}:
            assert metadata["evidence"]["solved_seeds"] == [values["seed"]]
        archive = (
            "attempt_s<session>_a<attempt>_failed.json"
            if robot_name == "dual_franka"
            else "attempt_<N>_failed.json"
        )
        assert f"/output/{recipe_tag}/attempts/{archive}" in system
        corroboration = next(
            item for item in configurations if item.get("effect") == "confirmed"
        )
        assert corroboration["cell"] == recipe_tag
        assert f"/memory/_internal/inbox/{recipe_tag}/corroborations.jsonl" in system
        assert "This is session 2 of 3." in system
        assert "Attempt limit for this session: 4." in system
        assert all("environment" not in item for item in configurations)


@pytest.mark.parametrize("robot_name", EXPECTED_ROBOTS)
def test_dashboard_metadata_has_consistent_fields_and_runtime_components(
    robot_name: str,
) -> None:
    spec = get_robot_spec(robot_name)
    parser = ArgumentParser()
    spec.add_cli_args(parser, use_dashboard=True)
    dashboard = spec.resolve_dashboard(parser.parse_args([]))
    assert dashboard is not None

    task = dashboard["task"]
    fields = tuple(field["name"] for field in task["fields"])
    assert task["command"] == "/rpent-task"
    assert len(fields) == len(set(fields))
    assert _format_fields(task["display"]) <= set(fields)
    assert _format_fields(task["output_slug"]) <= set(fields)

    components = dashboard["runtime_components"]
    component_names = tuple(component["name"] for component in components)
    assert len(component_names) == len(set(component_names))
    assert {component["scope"] for component in components} <= {
        "unique",
        "shared",
    }


def test_run_config_and_robot_spec_are_frozen_contract_values() -> None:
    config = RunConfig(
        recipe_tag="recipe",
        output_dir=Path("output"),
        prompt_vars={"seed": 1},
        task_desc={"task": "demo"},
    )
    with pytest.raises(FrozenInstanceError):
        config.recipe_tag = "changed"  # type: ignore[misc]

    spec = get_robot_spec("libero")
    with pytest.raises(FrozenInstanceError):
        spec.name = "changed"  # type: ignore[misc]


def test_robotwin_runtime_contracts_contain_execution_critical_metadata() -> None:
    env_contract = env_runtime_contract(
        task_name="block_hammer_beat",
        task_config="demo_clean",
        seed=7,
        max_episode_steps=123,
    )
    assert env_contract["runtime"] == "rlinf_robotwin_env"
    assert env_contract["seed"] == 7
    assert env_contract["seed_mode"] == "exact"
    assert env_contract["action_layouts"] == ["qpos14", MODEL_SPEC.action_layout]
    assert env_contract["execution"]["step_limit"] == 123
    assert env_contract["extensions"]["render_camera"]["camera_names"] == list(
        ROBOTWIN_CAMERA_NAMES
    )

    vla_contract = vla_runtime_contract()
    assert vla_contract == {
        "runtime": "lingbotvla",
        "policy_name": MODEL_SPEC.policy_name,
        "camera_order": list(MODEL_SPEC.camera_order),
        "state_layout": MODEL_SPEC.state_layout,
        "action_layout": MODEL_SPEC.action_layout,
        "use_length": MODEL_SPEC.use_length,
    }

    env_contract["action_layouts"].append("mutated")
    vla_contract["camera_order"].append("mutated")
    assert (
        "mutated"
        not in env_runtime_contract(
            task_name="block_hammer_beat",
            task_config="demo_clean",
            seed=7,
        )["action_layouts"]
    )
    assert "mutated" not in vla_runtime_contract()["camera_order"]
