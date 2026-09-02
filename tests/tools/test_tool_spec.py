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

"""Tests for the normalized :class:`ToolSpec` representation."""

from __future__ import annotations

import pytest

from rpent.tools.func_tool import toolspec
from rpent.tools.tool_spec import ToolSpec, readonly
from rpent.utils.logging import get_output_dir


def _spec_dict(**overrides) -> dict:
    spec = {
        "name": "sample_tool",
        "description": "Sample tool description",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "number"}},
            "required": ["x"],
        },
    }
    spec.update(overrides)
    return spec


def test_from_spec_extracts_fields():
    ts = ToolSpec.from_spec(_spec_dict(), lambda x: {"x": x})

    assert ts.name == "sample_tool"
    assert ts.description == "Sample tool description"
    assert ts.input_schema["type"] == "object"
    assert ts.input_schema["properties"]["x"] == {"type": "number"}
    assert ts.input_schema["required"] == ["x"]


def test_from_spec_infers_readonly_from_marker():
    @readonly
    def read_only(x):
        return x

    def stateful(x):
        return x

    assert ToolSpec.from_spec(_spec_dict(), read_only).readonly is True
    assert ToolSpec.from_spec(_spec_dict(), stateful).readonly is False


def test_from_spec_explicit_readonly_overrides_marker():
    @readonly
    def marked(x):
        return x

    ts = ToolSpec.from_spec(_spec_dict(), marked, readonly=False)
    assert ts.readonly is False


def test_toolspec_co_location_matches_from_spec():
    @toolspec(
        description="Sample tool description",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "number"}},
            "required": ["x"],
        },
    )
    def sample_tool(x):
        return {"x": x}

    assert isinstance(sample_tool, ToolSpec)
    assert sample_tool.name == "sample_tool"
    assert sample_tool.description == "Sample tool description"
    assert sample_tool.input_schema == _spec_dict()["input_schema"]
    # The co-location decorator and normalizing the equivalent hand-written
    # dict must produce the same ToolSpec (handlers are the same function).
    plain = ToolSpec.from_spec(_spec_dict(), sample_tool.handler)
    assert plain == sample_tool


def test_resolved_substitutes_without_mutating():
    ts = ToolSpec(
        name="list_dir",
        description="See {{output_dir}}",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Default: {{output_dir}}"}
            },
        },
        handler=lambda: None,
    )

    resolved = ts.resolved()
    output_dir = str(get_output_dir())

    assert resolved is not ts
    assert resolved.name == ts.name
    assert resolved.description == f"See {output_dir}"
    assert resolved.input_schema["properties"]["path"]["description"] == (
        f"Default: {output_dir}"
    )
    # The original must be untouched.
    assert ts.description == "See {{output_dir}}"
    assert ts.input_schema["properties"]["path"]["description"] == (
        "Default: {{output_dir}}"
    )


def test_resolved_accepts_extra_variables():
    ts = ToolSpec(
        name="t",
        description="dir {{output_dir}} tag {{tag}}",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: None,
    )
    resolved = ts.resolved({"tag": "abc"})
    assert resolved.description == f"dir {get_output_dir()} tag abc"


def test_tool_spec_is_frozen():
    ts = ToolSpec.from_spec(_spec_dict(), lambda x: x)
    with pytest.raises(AttributeError):
        ts.name = "renamed"
