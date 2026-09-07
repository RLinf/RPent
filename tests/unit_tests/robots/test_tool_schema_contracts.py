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

from importlib import import_module

import pytest

ROBOT_NAMES = ("libero", "robocasa", "robotwin")


def _schemas(robot_name):
    module = import_module(f"robots.{robot_name}.tools")
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in getattr(module, f"{robot_name.upper()}_TOOLS")
    ]


EXPECTED_TOOL_NAMES = {
    "libero": {
        "finish",
        "reset",
        "view_env_state",
        "move_to",
        "pi0_pick",
        "pi0_doubled",
        "release",
        "set_gripper",
        "rotate_wrist",
        "rotate_pitch",
        "move_pose",
        "view_camera_meta",
        "segment",
        "back_project",
    },
    "robocasa": {
        "move_to",
        "move_delta",
        "rotate_pitch",
        "set_gripper",
        "release",
        "scripted_grasp",
        "rldx_skill",
        "rldx_arm",
        "navigate_to",
        "move_base",
        "reset",
        "view_env_state",
        "back_project_batch",
        "query_world_map",
        "finish",
    },
    "robotwin": {
        "view_env_state",
        "render",
        "sample_world_xyz",
        "query_world_map",
        "lingbot_act",
        "move_to",
        "rotate_wrist",
        "set_gripper",
        "release",
        "finish",
    },
}


@pytest.mark.parametrize("robot_name", ROBOT_NAMES)
def test_robot_tool_names_are_an_explicit_unique_contract(robot_name: str) -> None:
    specs = _schemas(robot_name)
    names = [spec["name"] for spec in specs]

    assert set(names) == EXPECTED_TOOL_NAMES[robot_name]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("robot_name", ROBOT_NAMES)
def test_robot_tool_schemas_have_valid_object_inputs(robot_name: str) -> None:
    for spec in _schemas(robot_name):
        assert set(spec) >= {"name", "description", "input_schema"}
        assert isinstance(spec["description"], str) and spec["description"].strip()

        input_schema = spec["input_schema"]
        assert input_schema["type"] == "object"
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        assert isinstance(properties, dict)
        assert len(required) == len(set(required))
        assert set(required) <= set(properties)


def test_robot_action_schemas_keep_bounded_vector_shapes() -> None:
    schema_sets = [
        {spec["name"]: spec for spec in _schemas("libero")}["move_to"],
        {spec["name"]: spec for spec in _schemas("robotwin")}["move_to"],
    ]
    for spec in schema_sets:
        xyz = spec["input_schema"]["properties"]["xyz"]
        assert xyz["type"] == "array"
        assert xyz["minItems"] == xyz["maxItems"] == 3
