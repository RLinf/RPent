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

from rpent.tools.common_tools import COMMON_TOOLS

ROBOT_NAMES = ("libero", "robocasa", "robotwin")


@pytest.mark.parametrize("group", ("common", *ROBOT_NAMES))
def test_input_schemas_match_before_native_migration(group: str) -> None:
    # Extracted from historical TOOLS_SPEC declarations and the API image reader;
    # expected schemas must not be regenerated from the native parameter models.
    root = Path(__file__).parent
    path = (
        root / "fixtures/common_tool_schemas.json"
        if group == "common"
        else root / group / "fixtures/pre_native_tool_schemas.json"
    )
    expected = json.loads(path.read_text(encoding="utf-8"))["schemas"]
    if group == "common":
        # list_dir now resolves the directory from the invocation context.
        expected["list_dir"]["properties"]["path"]["description"] = (
            "Directory path. Defaults to the current task's output directory."
        )
        actual = {item.name: item.input_schema for item in COMMON_TOOLS}
    else:
        actual = {spec["name"]: spec["input_schema"] for spec in _schemas(group)}
        if group == "libero":
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

    assert actual.keys() == expected.keys()
    for name, schema in actual.items():
        # Native tool calls now reject unknown top-level arguments.
        expected[name]["additionalProperties"] = False
        assert schema == expected[name], f"{group}.{name} input schema changed"


def _schemas(robot_name):
    module = import_module(f"robots.{robot_name}.tools")
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in getattr(module, f"{robot_name.upper()}_TOOLS")
    ]


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


@pytest.mark.parametrize("robot_name", ROBOT_NAMES)
def test_owned_tool_collections_satisfy_executor_invariants(robot_name):
    module = import_module(f"robots.{robot_name}.tools")
    tools = (*COMMON_TOOLS, *getattr(module, f"{robot_name.upper()}_TOOLS"))
    names = [item.name for item in tools]
    assert len(names) == len(set(names))
    for item in tools:
        parameter = inspect.signature(item.handler).parameters["ctx"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
        assert "ctx" not in item.input_schema["properties"]
