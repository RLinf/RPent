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

import argparse
from dataclasses import FrozenInstanceError
from pathlib import Path
from string import Formatter

import numpy as np
import pytest

from robots.robotwin.robot_spec import (
    MODEL_SPEC,
    ROBOTWIN_CAMERA_NAMES,
    env_runtime_contract,
    vla_runtime_contract,
)
from rpent.robots import enumerate_robots, get_robot_spec
from rpent.robots.robot_spec import RobotSpec, RunConfig

EXPECTED_ROBOTS = ("behavior", "libero", "robocasa", "robotwin")

PROMPT_VARIABLES = {
    "behavior": {
        "behavior_mode": "eval",
        "task_name": "turning_on_radio",
        "task": 0,
        "task_language": "Turn on the radio.",
        "task_instruction": "Turn on the radio.",
        "public_seed": 1,
        "recipe_tag": "turning_on_radio_s1",
        "output_dir": Path("/output"),
        "max_episode_steps": 43200,
        "wall_clock_seconds": 7200,
        "public_capabilities": ["observe", "pi0_nav_pick", "finish"],
        "memory_dir": "/memory",
        "memory_profile": "local",
        "memory_inbox": "/memory/_inbox/turning_on_radio_s1",
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


def test_behavior_uses_the_shared_pi05_registry_and_wire_contract() -> None:
    from robots.behavior.pi05 import PI05_BEHAVIOR_EMBODIMENT
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient

    openpi_config = PI05_BEHAVIOR_EMBODIMENT["openpi"]
    assert openpi_config["config_name"] == "pi05_behavior"
    assert openpi_config["action_chunk"] == 32
    assert openpi_config["action_env_dim"] == 23
    assert PI05_BEHAVIOR_EMBODIMENT["openpi_data"]["norm_stats_path"] == (
        "assets/behavior-1k/2025-challenge-demos/norm_stats.json"
    )

    class FakeRpcClient:
        def call(self, method, *, args, timeout_s):
            assert method == "vla.predict"
            observation, options = args
            assert observation["main_images"].shape == (1, 720, 720, 3)
            assert observation["wrist_images"].shape == (1, 2, 480, 480, 3)
            assert observation["states"].shape == (1, 256)
            assert observation["task_descriptions"] == ["turn on the radio"]
            assert options == {"mode": "eval"}
            assert timeout_s == 600.0
            return np.zeros((1, 32, 23), dtype=np.float32)

    model = Pi05VLAClient(FakeRpcClient(), embodiment="behavior")
    action = model.predict(
        {
            "main_images": np.zeros((720, 720, 3), dtype=np.uint8),
            "wrist_images": np.zeros((2, 480, 480, 3), dtype=np.uint8),
            "states": np.zeros(256, dtype=np.float32),
            "task_descriptions": "turn on the radio",
        },
        options={"mode": "eval"},
    )
    assert action.shape == (32, 23)
    assert action.dtype == np.float32


@pytest.mark.parametrize(
    ("mode", "task_name", "public_seed", "role_title"),
    [
        ("eval", "turning_on_radio", 1, "ROLE AND EVALUATION"),
        ("explore", "picking_up_trash", 0, "ROLE AND MODE"),
    ],
)
def test_behavior_prompts_strictly_render_real_run_config(
    tmp_path: Path,
    mode: str,
    task_name: str,
    public_seed: int,
    role_title: str,
) -> None:
    spec = get_robot_spec("behavior")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--memory-profile", choices=["hf", "local"], default=None)
    parser.add_argument("--memory-dir")
    spec.add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(
        [
            "--task-name",
            task_name,
            "--public-seed",
            str(public_seed),
            "--behavior-mode",
            mode,
            "--output-dir",
            str(tmp_path / "output"),
            "--memory-dir",
            str(tmp_path / "memory"),
        ]
    )
    variables = spec.parse_config(args).prompt_vars

    system = spec.prompts.render("system", variables=variables)
    user = spec.prompts.render("user", variables=variables)

    assert "{{" not in system
    assert "{{" not in user
    for value in (
        task_name,
        variables["task_instruction"],
        str(public_seed),
        mode,
        variables["recipe_tag"],
        variables["output_dir"],
        str(variables["max_episode_steps"]),
        str(variables["wall_clock_seconds"]),
        str(variables["public_capabilities"]),
        variables["memory_profile"],
        variables["memory_dir"],
        variables["memory_inbox"],
    ):
        assert value in system or value in user

    ordered_sections = [
        role_title,
        "CURRENT INVOCATION",
        "INVOCATION MODEL",
        "RUNTIME",
        "YOUR GOAL",
        "MEMORY",
        "EVIDENCE",
        "PLANNER TOOLS",
        "TERMINATION",
        "OUTPUT DISCIPLINE",
    ]
    positions = [system.index(title) for title in ordered_sections]
    assert positions == sorted(positions)
    assert [user.index(title) for title in ("CELL", "MODE", "BEGIN")] == sorted(
        user.index(title) for title in ("CELL", "MODE", "BEGIN")
    )

    required = {
        "behavior_mode",
        "task_name",
        "task_instruction",
        "public_seed",
        "recipe_tag",
        "output_dir",
        "max_episode_steps",
        "wall_clock_seconds",
        "public_capabilities",
        "memory_profile",
        "memory_dir",
        "memory_inbox",
    }
    for key in required:
        incomplete = {name: value for name, value in variables.items() if name != key}
        with pytest.raises(KeyError, match=key):
            spec.prompts.render("system", variables=incomplete)

    literal = {**variables, "task_instruction": "literal {{do_not_expand}}"}
    assert "literal {{do_not_expand}}" in spec.prompts.render(
        "system", variables=literal
    )

    with pytest.raises(ValueError, match="unsupported BEHAVIOR prompt mode"):
        spec.prompts.render(
            "system", variables={**variables, "behavior_mode": "unknown"}
        )
