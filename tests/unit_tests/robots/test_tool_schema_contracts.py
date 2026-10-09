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
from pathlib import Path

import pytest

from robots.dual_franka.toolkit import DualFrankaToolkit
from robots.franka import perception as franka_perception
from robots.franka import tools as franka_tools
from robots.libero import tools as libero_tools
from robots.libero.toolkit import LiberoToolkit
from robots.robocasa import tools as robocasa_tools
from robots.robocasa.primitives import RoboCasaPrimitives
from robots.robodojo import tools as robodojo_tools
from robots.robotwin import tools as robotwin_tools
from robots.robotwin.primitives import RoboTwinPrimitives
from robots.robotwin.toolkit import RoboTwinToolkit
from robots.yam import tools as yam_tools
from robots.yam.toolkit import YamToolkit
from rpent.tools import iter_tools
from rpent.tools.common import CommonTools

TOOL_DECLARATIONS = {
    "common": list(iter_tools(CommonTools)),
    "franka": list(
        iter_tools(franka_tools.FrankaPrimitives, franka_tools, franka_perception)
    ),
    "dual_franka": DualFrankaToolkit.declared_tools(),
    "yam": list(iter_tools(YamToolkit, yam_tools)),
    "libero": list(
        iter_tools(libero_tools, libero_tools.LiberoPrimitives, LiberoToolkit)
    ),
    "robocasa": list(iter_tools(robocasa_tools, RoboCasaPrimitives)),
    "robodojo": list(iter_tools(robodojo_tools)),
    "robotwin": list(iter_tools(robotwin_tools, RoboTwinPrimitives, RoboTwinToolkit)),
}


@pytest.mark.parametrize("group", sorted(TOOL_DECLARATIONS))
def test_tool_declarations_match_pre_refactor_contracts(group: str) -> None:
    """Compare against fixed main TOOLS_SPEC data, including registration notes."""
    baseline = json.loads(
        (Path(__file__).parent / "fixtures" / "tool_schemas.json").read_text()
    )
    path = "rpent/tools/common.py" if group == "common" else f"robots/{group}/tools.py"
    expected = {item["name"]: item for item in baseline["tools"][path]}
    declarations = TOOL_DECLARATIONS[group]
    actual = {
        item.name: {
            "name": item.name,
            "description": item.description,
            "input_schema": item.input_schema,
        }
        for item in declarations
    }
    assert len(actual) == len(declarations)
    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name] == expected[name], f"{group}/{name}"


@pytest.mark.parametrize("robot_name", sorted(TOOL_DECLARATIONS))
def test_robot_tool_schemas_have_valid_object_inputs(robot_name: str) -> None:
    for spec in TOOL_DECLARATIONS[robot_name]:
        assert isinstance(spec.description, str) and spec.description.strip()

        input_schema = spec.input_schema
        assert input_schema["type"] == "object"
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        assert isinstance(properties, dict)
        assert len(required) == len(set(required))
        assert set(required) <= set(properties)


def test_robot_action_schemas_keep_bounded_vector_shapes() -> None:
    schema_sets = [
        {
            spec.name: spec
            for spec in list(
                iter_tools(libero_tools, libero_tools.LiberoPrimitives, LiberoToolkit)
            )
        }["move_to"],
        {
            spec.name: spec
            for spec in list(
                iter_tools(robotwin_tools, RoboTwinPrimitives, RoboTwinToolkit)
            )
        }["move_to"],
    ]
    for spec in schema_sets:
        xyz = spec.input_schema["properties"]["xyz"]
        assert xyz["type"] == "array"
        assert xyz["minItems"] == xyz["maxItems"] == 3
