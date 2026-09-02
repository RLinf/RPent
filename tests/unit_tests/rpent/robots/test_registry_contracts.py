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

from dataclasses import FrozenInstanceError
from pathlib import Path
from string import Formatter

import pytest

from robots.robotwin.robot_spec import (
    MODEL_SPEC,
    ROBOTWIN_CAMERA_NAMES,
    env_runtime_contract,
    vla_runtime_contract,
)
from rpent.robots import enumerate_robots, get_robot_spec
from rpent.robots.robot_spec import RobotSpec, RunConfig

EXPECTED_ROBOTS = ("libero", "robocasa", "robotwin")

PROMPT_VARIABLES = {
    "libero": {
        "suite": "libero_object_task",
        "task": 2,
        "seed": 3,
        "recipe_tag": "object_task_t2_s3",
        "mode": "eval",
        "memory_profile": "hf",
        "memory_dir": "/memory",
        "reference_tag": "object_task_t2_s0",
        "memory_inbox": "/memory/_inbox/object_task_t2_s3",
        "session_number": 1,
        "session_max": 1,
        "output_dir": Path("/output"),
    },
    "robocasa": {
        "task_name": "OpenDrawer",
        "split": "target",
        "seed": 3,
        "recipe_tag": "OpenDrawer_target_s3",
        "memory_dir": "/memory",
        "output_dir": Path("/output"),
    },
    "robotwin": {
        "task_name": "block_hammer_beat",
        "seed": 3,
        "task_config": "demo_randomized",
        "instruction": "pick up the hammer",
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


@pytest.mark.parametrize("robot_name", EXPECTED_ROBOTS)
def test_robot_prompts_render_from_public_spec(robot_name: str) -> None:
    spec = get_robot_spec(robot_name)

    system = spec.prompts.render("system", variables=PROMPT_VARIABLES[robot_name])
    user = spec.prompts.render("user", variables=PROMPT_VARIABLES[robot_name])

    assert system.strip()
    assert user.strip()
    assert "{{" not in system
    assert "{{" not in user


@pytest.mark.parametrize("robot_name", EXPECTED_ROBOTS)
def test_dashboard_metadata_has_consistent_fields_and_channels(
    robot_name: str,
) -> None:
    dashboard = get_robot_spec(robot_name).dashboard
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

    channels = dashboard["frame_channels"]
    channel_names = tuple(channel["name"] for channel in channels)
    assert len(channel_names) == len(set(channel_names))
    assert all(channel["label"] for channel in channels)


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
