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

"""Declarations work with existing stateful Python objects and plain functions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated

import pytest
from pydantic import BaseModel, Field, ValidationError

from rpent.tools import ToolResult, iter_tools, tool

if TYPE_CHECKING:
    from robots.libero.env_client import LiberoEnvClient


def test_signature_defines_schema_defaults_constraints_and_descriptions() -> None:
    @tool
    def move(
        xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
        steps: Annotated[int, Field(ge=1, json_schema_extra={"default": 25})] = 25,
    ) -> ToolResult:
        """Move to a world-frame position.

        Args:
            xyz: Position in metres.
            steps: Number of steps.
        """
        return ToolResult(data={"xyz": xyz, "steps": steps})

    assert move.name == "move"
    assert move.description == "Move to a world-frame position."
    assert move.input_schema == {
        "type": "object",
        "properties": {
            "xyz": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
                "description": "Position in metres.",
            },
            "steps": {
                "type": "integer",
                "minimum": 1,
                "default": 25,
                "description": "Number of steps.",
            },
        },
        "required": ["xyz"],
    }
    arguments = move.args_schema.model_validate({"xyz": [0, 1, 2]})
    assert move(**arguments.model_dump()) == ToolResult(
        data={"xyz": [0.0, 1.0, 2.0], "steps": 25}
    )
    with pytest.raises(ValidationError):
        move.args_schema.model_validate({"xyz": [0, 1]})


def test_instance_binding_inheritance_and_internal_method_calls() -> None:
    class Counter:
        def __init__(self, count: int):
            self.count = count

        @tool
        def increment(self, amount: int = 1) -> ToolResult:
            """Advance this counter."""
            self.count += amount
            return ToolResult(data={"count": self.count})

        @tool(readonly=True)
        def read(self) -> ToolResult:
            """Read this counter."""
            return ToolResult(data={"count": self.count})

        def increment_twice(self) -> ToolResult:
            self.increment()
            return self.increment()

    class Child(Counter):
        pass

    first, second = Counter(0), Child(10)
    first_tool = first.increment
    second_tool = second.increment
    assert first_tool(2) == ToolResult(data={"count": 2})
    assert second_tool(3) == ToolResult(data={"count": 13})
    assert first_tool() == ToolResult(data={"count": 3})
    assert first.increment_twice() == ToolResult(data={"count": 5})
    assert Counter.increment(first, 2) == ToolResult(data={"count": 7})
    assert second.read() == ToolResult(data={"count": 13})
    assert first.read.readonly and second.read.readonly
    assert set(Counter.increment.input_schema["properties"]) == {"amount"}


def test_existing_bound_method_can_be_declared_at_registration() -> None:
    class Counter:
        def __init__(self):
            self.count = 0

        def increment(self, amount: int = 1) -> ToolResult:
            self.count += amount
            return ToolResult(data={"count": self.count})

    instance = Counter()
    declared = tool(instance.increment)
    assert declared(amount=3) == ToolResult(data={"count": 3})
    assert instance.increment() == ToolResult(data={"count": 4})
    assert "self" not in declared.input_schema["properties"]


def test_discovery_binds_inherited_tools_without_evaluating_resources() -> None:
    class Parent:
        @tool
        def inherited(self, value: int) -> ToolResult:
            return ToolResult(data={"value": self.offset + value})

        @tool
        def hidden(self) -> ToolResult:
            return ToolResult()

    class Child(Parent):
        offset = 10

        @property
        def resource(self):
            raise AssertionError("Discovery must not initialize robot resources")

        def hidden(self) -> ToolResult:
            return ToolResult()

        @tool(name="read", readonly=True)
        def _read(self) -> ToolResult:
            return ToolResult(data={"value": self.offset})

    instance = Child()
    tools = {tool.name: tool for tool in iter_tools(instance)}
    assert set(tools) == {"inherited", "read"}
    assert tools["inherited"](2).data == {"value": 12}
    assert tools["read"]().data == {"value": 10}
    assert tools["read"].readonly
    module = SimpleNamespace(read=tools["read"], unrelated=object())
    assert [tool.name for tool in iter_tools(module)] == ["read"]


def test_validation_preserves_nested_models_and_ignores_return_annotations() -> None:
    class Target(BaseModel):
        label: str

    @tool
    def inspect_target(target: Target) -> LiberoEnvClient:
        return ToolResult(data={"label": target.label})

    arguments = inspect_target.args_schema.model_validate({"target": {"label": "bowl"}})
    assert isinstance(arguments.target, Target)
    assert inspect_target(arguments.target).data == {"label": "bowl"}


def test_unsupported_signatures_fail_when_declared() -> None:
    def missing(value):
        return value

    def positional(value: int, /):
        return value

    def variadic(**values: int):
        return values

    with pytest.raises(TypeError, match="type annotation"):
        tool(missing)
    for function in (positional, variadic):
        with pytest.raises(TypeError, match="keyword argument"):
            tool(function)


def test_internal_resources_are_excluded_and_model_required_inputs_stay_required():
    @tool(exclude=("state",), required=("prompt",))
    def act(prompt: str = "", *, state: object) -> ToolResult:
        return ToolResult(data={"prompt": prompt, "state": state})

    assert act.input_schema["required"] == ["prompt"]
    assert set(act.input_schema["properties"]) == {"prompt"}
    with pytest.raises(ValidationError):
        act.args_schema.model_validate({})
    with pytest.raises(ValidationError):
        act.args_schema.model_validate({"prompt": "pick", "state": "injected"})
    assert act(state="internal") == ToolResult(data={"prompt": "", "state": "internal"})
