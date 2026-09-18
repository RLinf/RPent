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

import inspect
import json
from importlib import import_module
from pathlib import Path

import pytest

from rpent.tools import Tool
from rpent.tools.common_tools import COMMON_TOOLS

ROBOT_NAMES = ("franka", "dual_franka", "libero", "robocasa", "robotwin")

# These existing optional inputs now explicitly advertise their None value.
NULLABLE_INPUTS = {
    "franka": {
        "back_project": ("step",),
        "back_project_correspondence": (
            "third_person_row",
            "third_person_col",
            "wrist_row",
            "wrist_col",
            "pixels",
            "step",
        ),
    },
    "dual_franka": {"back_project": ("step",), "segment": ("point", "step")},
}


def _tools(group: str) -> tuple[Tool, ...]:
    if group == "common":
        return COMMON_TOOLS
    module = import_module(f"robots.{group}.tools")
    return getattr(module, f"{group.upper()}_TOOLS")


@pytest.mark.parametrize("group", ("common", *ROBOT_NAMES))
def test_input_schemas_match_before_native_migration(group: str) -> None:
    # Extracted from historical TOOLS_SPEC declarations and the API image reader;
    # expected schemas must not be regenerated from the native parameter models.
    root = Path(__file__).parent
    path = (
        root / "fixtures/common_tool_contracts.json"
        if group == "common"
        else root / group / "fixtures/pre_native_tool_contracts.json"
    )
    baseline = json.loads(path.read_text(encoding="utf-8"))
    expected = baseline["schemas"]
    if group == "common":
        # list_dir now resolves the directory from the invocation context.
        expected["list_dir"]["properties"]["path"]["description"] = (
            "Directory path. Defaults to the current task's output directory."
        )
    elif group == "libero":
        # Task-card replay added these parameters after the historical snapshot.
        expected["pi0_pick"]["properties"].update(
            {
                "gripper_open_thresh": {
                    "type": "number",
                    "description": "Minimum finger separation accepted as a held object (default 0.0)",
                },
                "descent_thresh": {
                    "type": "number",
                    "description": "Required descent before lift detection, m (default 0.10)",
                },
            }
        )
    elif group == "robocasa":
        # The preparation PR removed these unimplemented perception tools.
        del expected["view_camera_meta"]
        del expected["back_project"]
    elif group == "robotwin":
        # The native finish declaration adds parameter documentation.
        expected["finish"]["properties"]["status"]["description"] = (
            "Requested task outcome."
        )
        expected["finish"]["properties"]["summary"]["description"] = (
            "Summary of what worked and what failed."
        )

    for name, parameters in NULLABLE_INPUTS.get(group, {}).items():
        for parameter in parameters:
            schema = expected[name]["properties"][parameter]
            schema["type"] = [schema["type"], "null"]

    tools = _tools(group)
    actual = {item.name: item.input_schema for item in tools}
    assert len(actual) == len(tools)
    assert actual.keys() == expected.keys()
    for name, schema in actual.items():
        # Native tool calls now reject unknown top-level arguments.
        expected[name]["additionalProperties"] = False
        assert schema == expected[name], f"{group}.{name} input schema changed"

    if "descriptions" in baseline:
        assert {item.name: item.description for item in tools} == baseline[
            "descriptions"
        ]

    for item in tools:
        for name, parameter in actual[item.name]["properties"].items():
            if "default" in parameter:
                assert (
                    item.args_schema.model_fields[name].default == parameter["default"]
                ), f"{group}.{item.name}.{name} default differs from its schema"


@pytest.mark.parametrize("robot_name", ROBOT_NAMES)
def test_robot_tool_schemas_have_valid_object_inputs(robot_name: str) -> None:
    for item in _tools(robot_name):
        assert isinstance(item.description, str) and item.description.strip()

        input_schema = item.input_schema
        assert input_schema["type"] == "object"
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        assert isinstance(properties, dict)
        assert len(required) == len(set(required))
        assert set(required) <= set(properties)


@pytest.mark.parametrize("robot_name", ROBOT_NAMES)
def test_owned_tool_collections_satisfy_executor_invariants(robot_name):
    tools = (*COMMON_TOOLS, *_tools(robot_name))
    names = [item.name for item in tools]
    assert len(names) == len(set(names))
    for item in tools:
        parameter = inspect.signature(item.handler).parameters["ctx"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
        assert "ctx" not in item.input_schema["properties"]
