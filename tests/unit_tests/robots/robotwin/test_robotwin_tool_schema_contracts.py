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

"""Robot-specific schema contracts for RoboTwin tools."""

import inspect

import pytest

from robots.robotwin import tools


def test_perception_schemas_use_the_same_view_coordinate_space() -> None:
    by_name = {t.name: t for t in tools.ROBOTWIN_TOOLS}

    for tool_name, coordinate_name in (
        ("sample_world_xyz", "pixels"),
        ("query_world_map", "bbox"),
    ):
        schema = by_name[tool_name].input_schema
        assert schema["required"] == ["view", coordinate_name]
        assert schema["properties"]["view"]["type"] == "string"


def test_native_handlers_require_injected_context_and_keep_fixed_chunk_length():
    for native_tool in tools.ROBOTWIN_TOOLS:
        ctx = inspect.signature(native_tool.handler).parameters["ctx"]
        assert ctx.kind is inspect.Parameter.KEYWORD_ONLY
        assert ctx.default is inspect.Parameter.empty
        assert "ctx" not in native_tool.input_schema["properties"]
    use_length = tools.lingbot_act.input_schema["properties"]["use_length"]
    assert use_length["const"] == use_length["default"] == 50


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("lingbot_act", {"chunks": 0}),
        ("lingbot_act", {"use_length": 10}),
        ("move_to", {"arm": "both", "xyz": [1, 2, 3]}),
        ("move_to", {"arm": "left", "xyz": [1, 2]}),
        ("move_to", {"arm": "left", "xyz": [1, 2, 3], "quat": [1, 0, 0]}),
        ("move_to", {"arm": "left", "xyz": [1, 2, 3], "substeps": -1}),
        ("set_gripper", {"arm": "left", "val": 1.1}),
        ("release", {"arm": "right", "steps": 0}),
        ("sample_world_xyz", {"view": "head", "pixels": []}),
        ("sample_world_xyz", {"view": "head", "pixels": [[0]]}),
        ("sample_world_xyz", {"view": "head", "pixels": [[0, 0]], "neighborhood": 33}),
        ("query_world_map", {"view": "head", "bbox": [0, 0, 1]}),
        ("query_world_map", {"view": "head", "bbox": [0, 0, 1, 1], "max_points": 4097}),
    ],
)
def test_invalid_arguments_are_rejected_before_execution_or_capture(
    robotwin, name, arguments
):
    result = robotwin.toolkit.execute_tool(name, arguments)
    assert result.is_error and result.error.startswith("Invalid arguments")
    assert not robotwin.env.steps and not robotwin.env.chunks and not robotwin.env.plans
    assert robotwin.toolkit.state.latest_step == 0
    assert len(robotwin.env.renders) == 3
