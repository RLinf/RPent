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

import pytest

from robots.libero import tools as libero_tools
from robots.libero.toolkit import LiberoToolkit
from robots.robocasa import tools as robocasa_tools
from robots.robocasa.primitives import RoboCasaPrimitives
from robots.robotwin import tools as robotwin_tools
from robots.robotwin.primitives import RoboTwinPrimitives
from robots.robotwin.toolkit import RoboTwinToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, iter_tools

ROBOT_SCHEMAS = {
    "libero": [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in iter_tools(
            libero_tools.LiberoPrimitives, libero_tools, LiberoToolkit
        )
    ],
    "robocasa": [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in iter_tools(RoboCasaPrimitives, robocasa_tools)
    ],
    "robotwin": [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in iter_tools(RoboTwinPrimitives, robotwin_tools, RoboTwinToolkit)
    ],
}

EXPECTED_TOOL_NAMES = {
    "libero": {
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


@pytest.mark.parametrize("robot_name", sorted(ROBOT_SCHEMAS))
def test_robot_tool_names_are_an_explicit_unique_contract(robot_name: str) -> None:
    specs = ROBOT_SCHEMAS[robot_name]
    names = [spec["name"] for spec in specs]

    assert set(names) == EXPECTED_TOOL_NAMES[robot_name]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("robot_name", sorted(ROBOT_SCHEMAS))
def test_robot_tool_schemas_have_valid_object_inputs(robot_name: str) -> None:
    for spec in ROBOT_SCHEMAS[robot_name]:
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
        {"input_schema": libero_tools.LiberoPrimitives.move_to.input_schema},
        {"input_schema": RoboTwinPrimitives.move_to.input_schema},
    ]
    for spec in schema_sets:
        xyz = spec["input_schema"]["properties"]["xyz"]
        assert xyz["type"] == "array"
        assert xyz["minItems"] == xyz["maxItems"] == 3


@pytest.mark.parametrize(
    ("definition", "arguments", "parameter", "length"),
    [
        (libero_tools.LiberoPrimitives.segment, {"prompt": "cup"}, "point", 2),
        (RoboTwinPrimitives.move_to, {"arm": "left", "xyz": [0, 0, 0]}, "quat", 4),
    ],
)
def test_optional_vectors_accept_null_and_reject_wrong_lengths_before_execution(
    tmp_path, monkeypatch, definition, arguments, parameter, length
) -> None:
    calls = []
    captures = []

    def handler(**kwargs):
        calls.append(kwargs)
        return ToolResult(data=kwargs)

    def capture(**kwargs):
        captures.append(kwargs)
        return ToolResult(data=kwargs["result"])

    toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state=EnvState(tmp_path / "state"),
    )
    monkeypatch.setattr(toolkit, "get_env_state", capture)
    toolkit.add_tool(definition.with_handler(handler))

    for value in (None, [0] * length):
        result = toolkit.execute_tool(definition.name, {**arguments, parameter: value})
        assert not result.is_error
        assert result.data[parameter] == value
    assert len(calls) == 2
    assert len(captures) == (0 if definition.readonly else 2)

    result = toolkit.execute_tool(
        definition.name, {**arguments, parameter: [0] * (length - 1)}
    )
    assert result.is_error
    assert result.data["errors"]
    assert len(calls) == 2
    assert len(captures) == (0 if definition.readonly else 2)


@pytest.mark.parametrize(
    "group", ["common", "franka", "dual_franka", "libero", "robocasa", "robotwin"]
)
def test_generated_schemas_exactly_match_pre_refactor_contracts(group: str) -> None:
    """The fixture is extracted from 8f9d498 TOOLS_SPEC, never from generated models."""
    import json
    from pathlib import Path

    from robots.dual_franka.toolkit import DualFrankaToolkit
    from robots.franka import perception as franka_perception
    from robots.franka import tools as franka_tools
    from rpent.tools.common import CommonTools

    owners = {
        "common": (CommonTools,),
        "franka": (franka_tools.FrankaPrimitives, franka_tools, franka_perception),
        "libero": (libero_tools.LiberoPrimitives, libero_tools, LiberoToolkit),
        "robocasa": (RoboCasaPrimitives, robocasa_tools),
        "robotwin": (RoboTwinPrimitives, robotwin_tools, RoboTwinToolkit),
    }
    declarations = (
        DualFrankaToolkit.declared_tools()
        if group == "dual_franka"
        else list(iter_tools(*owners[group]))
    )
    actual = {item.name: item.input_schema for item in declarations}
    assert len(actual) == len(declarations)
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "tool_schemas.json").read_text()
    )
    path = "rpent/tools/common.py" if group == "common" else f"robots/{group}/tools.py"
    expected = {item["name"]: item["input_schema"] for item in fixture["tools"][path]}
    assert actual == expected


@pytest.mark.parametrize(
    ("module", "artifacts"),
    [
        (
            libero_tools,
            ("agentview_policy.png", "agentview_high.png", "wrist_high.png"),
        ),
        (robocasa_tools, ("agentview_high.png", "navview.png", "wrist_high.png")),
        (robotwin_tools, ("head_rgb.png", "left_wrist_rgb.png", "right_wrist_rgb.png")),
    ],
    ids=["libero", "robocasa", "robotwin"],
)
def test_observation_images_preserve_camera_order_outside_text(
    tmp_path, module, artifacts
):
    import json

    import numpy as np

    state = EnvState(tmp_path)
    with state.record_step(state={"position": [0, 1, 2]}):
        for index, name in enumerate(artifacts):
            state.save(name, np.full((2, 2, 3), index * 80, dtype=np.uint8))
    observed = module.view_env_state(state=state)
    assert not observed.is_error
    assert observed.images == [state.load_bytes(name) for name in artifacts]
    assert json.loads(observed.to_text()) == observed.data
    assert not any(isinstance(value, bytes) for value in observed.data.values())

    # Missing artifacts must not shift surviving images into the text payload.
    state.artifact_path(artifacts[1]).unlink()
    observed = module.view_env_state(state=state)
    assert observed.images == [
        state.load_bytes(name) for name in (artifacts[0], artifacts[2])
    ]


def test_robotwin_artifact_errors_expose_text_and_structured_details(tmp_path):
    state = EnvState(tmp_path)
    result = robotwin_tools.sample_world_xyz(state, view="head", pixels=[[0, 0]])
    assert result.is_error
    assert result.error == "The requested RoboTwin state artifact does not exist."
    assert result.data == {"success": False, "code": "state_not_found", "step": -1}
