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

"""CPU-only examples and checks for signature-derived native tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Literal

import pytest
from pydantic import Field

from rpent.tools import (
    Tool,
    ToolContext,
    ToolResult,
    parallel,
    readonly,
    tool,
)

if TYPE_CHECKING:
    from robots.libero.toolkit import LiberoRuntime


@tool
def move(
    xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    substeps: Annotated[int, Field(ge=0)] = 25,
    *,
    ctx: ToolContext,
) -> ToolResult:
    """Move to a world-frame position in metres.

    Args:
        xyz: World metres.
        substeps: Number of steps.
    """
    return ToolResult(data={"xyz": xyz, "substeps": substeps})


def test_signature_and_docstring_define_schema_and_validation() -> None:
    assert isinstance(move, Tool)
    assert move.name == "move"
    assert move.description == "Move to a world-frame position in metres."
    assert not move.parallel
    assert not move.readonly
    assert move.input_schema == {
        "type": "object",
        "properties": {
            "xyz": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
                "description": "World metres.",
            },
            "substeps": {
                "type": "integer",
                "minimum": 0,
                "description": "Number of steps.",
            },
        },
        "required": ["xyz"],
    }
    assert move.args_schema.model_validate({"xyz": [0, 0, 0]}).substeps == 25
    assert "ctx" not in move.input_schema["properties"]


def test_export_preserves_nullable_constraints_and_explicit_defaults() -> None:
    @tool
    @readonly
    def exported(
        vector: Annotated[list[float] | None, Field(min_length=3, max_length=3)] = None,
        choice: Literal["auto"] | None = None,
        step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
        *,
        ctx: ToolContext,
    ) -> ToolResult:
        """Export constrained nullable parameters."""
        return ToolResult()

    schema = exported.input_schema
    assert "title" not in schema
    assert "description" not in schema
    properties = schema["properties"]
    assert properties["vector"] == {
        "type": ["array", "null"],
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
    }
    assert properties["step"] == {"type": "integer", "default": -1}

    assert properties["choice"] == {
        "anyOf": [{"const": "auto", "type": "string"}, {"type": "null"}]
    }
    assert exported.args_schema.model_validate({}).step == -1


@pytest.mark.parametrize(
    ("markers", "is_readonly", "is_parallel"),
    [
        ([], False, False),
        ([readonly], True, False),
        ([parallel], False, True),
        ([readonly, parallel], True, True),
        ([parallel, readonly], True, True),
    ],
)
def test_policy_markers_are_independent_and_preserve_handler(
    markers, is_readonly, is_parallel
) -> None:
    def handler(
        limit: Annotated[int, Field(ge=1)] = 10, *, ctx: ToolContext
    ) -> ToolResult:
        """Return the requested limit.

        Args:
            limit: Maximum number of results.
        """
        return ToolResult(data={"limit": limit})

    original = tool(handler)
    for marker in markers:
        handler = marker(handler)
    definition = tool(handler)
    assert definition.handler is original.handler
    assert definition.readonly is is_readonly
    assert definition.parallel is is_parallel
    assert definition.description == original.description
    assert definition.input_schema == original.input_schema
    assert definition.args_schema.model_validate({}).limit == 10
    assert definition.handler(3, ctx=None).data == {"limit": 3}


def test_injected_and_return_annotations_need_not_resolve() -> None:
    @tool
    @readonly
    def handler(value: int, *, ctx: ToolContext[LiberoRuntime]) -> LiberoRuntime:
        """Context and return types are not model inputs.

        Args:
            value: A model-supplied integer.
            ctx: Internal resources, hidden from the model.
        """
        return ToolResult(data={"value": value})

    assert handler.handler(3, ctx=None).data == {"value": 3}
    assert set(handler.input_schema["properties"]) == {"value"}


@pytest.mark.parametrize("error", ["", "Failed."])
def test_explicit_error_serializes_without_mutating_data(error):
    result = ToolResult(data={"step": 2}, error=error)
    assert result.is_error
    assert json.loads(result.to_text()) == {"step": 2, "error": error}
    assert result.data == {"step": 2}


def test_business_error_key_does_not_mark_tool_failure():
    result = ToolResult(data={"error": {"count": 0}})
    assert not result.is_error
    assert json.loads(result.to_text()) == result.data
