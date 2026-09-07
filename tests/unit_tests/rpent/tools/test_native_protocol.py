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
import threading
from dataclasses import FrozenInstanceError, dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import (
    Tool,
    ToolCancelled,
    ToolContext,
    ToolResult,
    parallel,
    readonly,
    tool,
)

if TYPE_CHECKING:
    from robots.libero.toolkit import LiberoRuntime


@dataclass
class FakeRobot:
    received: list[tuple[list[float], int]] = field(default_factory=list)


@tool
def move(
    xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    substeps: Annotated[int, Field(ge=0)] = 25,
    *,
    ctx: ToolContext[FakeRobot],
) -> ToolResult:
    """Move to a world-frame position in metres.

    Args:
        xyz: World metres.
        substeps: Number of steps.
    """
    ctx.check_cancelled()
    ctx.robot.received.append((xyz, substeps))
    return ToolResult(data={"xyz": xyz, "substeps": substeps})


def test_handler_keeps_typed_parameters_and_current_resources(tmp_path: Path) -> None:
    state = EnvState(tmp_path)
    memory = MemoryManager(tmp_path / "memory")
    robot = FakeRobot()
    ctx = ToolContext(
        state=state,
        memory=memory,
        robot=robot,
        output_dir=tmp_path,
        record_frame=lambda frame: None,
        _cancel_event=threading.Event(),
    )
    args = move.args_schema.model_validate(
        {"xyz": ["1", 2, 3], "substeps": "2", "ctx": "untrusted", "extra": True}
    )
    result = move.handler(xyz=args.xyz, substeps=args.substeps, ctx=ctx)
    assert robot.received[0][0] is args.xyz
    assert result.data == {"xyz": [1.0, 2.0, 3.0], "substeps": 2}
    assert ctx.state is state
    assert ctx.memory is memory
    assert ctx.robot is robot
    assert ctx.output_dir == tmp_path
    assert result.images == []
    assert not result.is_error

    with pytest.raises(FrozenInstanceError):
        ctx.robot = FakeRobot()
    ctx._cancel_event.set()
    with pytest.raises(ToolCancelled):
        move.handler(xyz=args.xyz, substeps=args.substeps, ctx=ctx)
    assert len(robot.received) == 1
    next_ctx = replace(ctx, _cancel_event=threading.Event())
    move.handler(xyz=args.xyz, substeps=args.substeps, ctx=next_ctx)
    assert robot.received == [(args.xyz, 2), (args.xyz, 2)]


def test_signature_and_docstring_define_schema_and_validation() -> None:
    assert isinstance(move, Tool)
    assert move.name == "move"
    assert move.description == "Move to a world-frame position in metres."
    assert not move.parallel
    assert not move.readonly
    assert move.args_schema.model_config == {}
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
    schema = move.input_schema
    schema["properties"].clear()
    assert "xyz" in move.input_schema["properties"]
    with pytest.raises(FrozenInstanceError):
        move.name = "changed"


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"xyz": [1, 2]},
        {"xyz": [1, 2, 3, 4]},
        {"xyz": [1, 2, "bad"]},
        {"xyz": [1, 2, 3], "substeps": -1},
    ],
)
def test_invalid_arguments_fail_model_validation(arguments: dict) -> None:
    with pytest.raises(ValidationError):
        move.args_schema.model_validate(arguments)


class Position(BaseModel):
    """A nested business value, not a top-level tool parameter declaration."""

    xyz: list[float] = Field(min_length=3, max_length=3)
    substeps: int = 25


def test_export_preserves_nullable_constraints_and_explicit_defaults() -> None:
    @tool
    @readonly
    def exported(
        vector: Annotated[list[float] | None, Field(min_length=3, max_length=3)] = None,
        choice: Literal["auto"] | None = None,
        nested: Position | None = None,
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
    assert "title" not in schema["$defs"]["Position"]

    assert properties["choice"] == {
        "anyOf": [{"const": "auto", "type": "string"}, {"type": "null"}]
    }
    assert properties["nested"] == {
        "anyOf": [{"$ref": "#/$defs/Position"}, {"type": "null"}]
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


def test_constructed_tool_policies_do_not_change_with_handler_markers() -> None:
    def handler(*, ctx: ToolContext) -> ToolResult:
        """Keep the constructed tool's policies fixed."""
        return ToolResult()

    original = tool(handler)
    changed = tool(readonly(parallel(handler)))
    assert not original.readonly
    assert not original.parallel
    assert changed.readonly
    assert changed.parallel


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


def test_local_annotations_and_multiline_parameter_docstrings() -> None:
    maximum = 100
    Query = str

    @tool
    @readonly
    def search(
        query: Query,
        limit: Annotated[int, Field(ge=1, le=maximum)] = 10,
        *,
        ctx: ToolContext,
    ) -> ToolResult:
        """Search the customer database.

        Use the current customer records.

        Args:
            query: Search terms to look for.
                Multiple terms are accepted.
            limit: Maximum number of results to return.
        """
        return ToolResult(data={"query": query, "limit": limit})

    assert search.description == (
        "Search the customer database.\n\nUse the current customer records."
    )
    schema = search.input_schema["properties"]
    assert schema["query"]["description"] == (
        "Search terms to look for.\nMultiple terms are accepted."
    )
    assert schema["limit"]["minimum"] == 1
    assert schema["limit"]["maximum"] == maximum
    assert search.args_schema.model_validate({"query": "alice"}).limit == 10
    for limit in (0, 101):
        with pytest.raises(ValidationError):
            search.args_schema.model_validate({"query": "alice", "limit": limit})


def test_results_have_independent_defaults_and_no_finish_field() -> None:
    first = ToolResult(data={"status": "success", "_finish": True})
    second = ToolResult()
    first.images.append(b"png")
    assert second.data == {}
    assert second.images == []
    assert [f.name for f in fields(ToolResult)] == ["data", "images", "error"]
    first.error = "Budget exhausted."
    assert first.is_error
    assert not second.is_error
    assert json.loads(first.to_text()) == {
        "status": "success",
        "_finish": True,
        "error": "Budget exhausted.",
    }


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
