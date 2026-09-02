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

"""Tests for the co-location and FastMCP conversion decorators."""

from __future__ import annotations

from typing import Annotated, Literal

import pytest
from pydantic import Field

from rpent.tools.func_tool import fastmcp_tool, toolspec
from rpent.tools.tool_spec import ToolSpec, readonly

pytest.importorskip("mcp")


# --------------------------------------------------------------------------
# Path A: ``@toolspec`` co-location of a hand-written spec.
# --------------------------------------------------------------------------


def test_toolspec_builds_tool_spec():
    @toolspec(
        description="A hand-written description",
        input_schema={"type": "object", "properties": {"x": {"type": "number"}}},
        readonly=True,
    )
    def my_tool(x: int = 1) -> dict:
        return {"x": x}

    assert isinstance(my_tool, ToolSpec)
    assert my_tool.name == "my_tool"
    assert my_tool.description == "A hand-written description"
    assert my_tool.input_schema == {
        "type": "object",
        "properties": {"x": {"type": "number"}},
    }
    assert my_tool.readonly is True
    # The raw function stays reachable through ``handler``.
    assert my_tool.handler(x=3) == {"x": 3}


def test_toolspec_name_defaults_to_function_name():
    @toolspec(description="d", input_schema={"type": "object"})
    def implicit_name() -> dict:
        return {}

    assert implicit_name.name == "implicit_name"


def test_toolspec_readonly_inferred_from_marker():
    @toolspec(description="d", input_schema={"type": "object"})
    @readonly
    def observed() -> dict:
        return {}

    assert observed.readonly is True


def test_toolspec_readonly_explicit_override():
    @toolspec(description="d", input_schema={"type": "object"}, readonly=False)
    @readonly
    def forced_stateful() -> dict:
        return {}

    assert forced_stateful.readonly is False


# --------------------------------------------------------------------------
# Path B: ``@fastmcp_tool`` schema derivation from a function.
# --------------------------------------------------------------------------


def test_fastmcp_tool_derives_schema_from_signature_and_docstring():
    @fastmcp_tool()
    def add_numbers(a: int, b: int = 2) -> int:
        """Add two integers together."""
        return a + b

    assert isinstance(add_numbers, ToolSpec)
    assert add_numbers.name == "add_numbers"
    assert add_numbers.description == "Add two integers together."

    props = add_numbers.input_schema["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "integer"
    assert props["b"]["default"] == 2
    # Defaulted args are not required.
    assert add_numbers.input_schema["required"] == ["a"]


def test_fastmcp_tool_literal_becomes_enum():
    @fastmcp_tool()
    def choose(mode: Literal["fast", "slow"]) -> str:
        """Choose a mode."""
        return mode

    prop = choose.input_schema["properties"]["mode"]
    assert set(prop.get("enum", [])) == {"fast", "slow"}


def test_fastmcp_tool_annotated_inline_description():
    @fastmcp_tool()
    def greet(name: Annotated[str, Field(description="The person to greet")]) -> str:
        """Greet someone."""
        return f"hi {name}"

    assert (
        greet.input_schema["properties"]["name"].get("description")
        == "The person to greet"
    )


def test_fastmcp_tool_exclude_drops_bound_args():
    @fastmcp_tool(exclude=("state",))
    def lookup(name: str, *, state=None) -> dict:
        """Look something up."""
        return {"name": name, "state": state}

    props = lookup.input_schema["properties"]
    assert "state" not in props
    assert props["name"]["type"] == "string"
    assert lookup.input_schema["required"] == ["name"]


# --------------------------------------------------------------------------
# ``@toolspec`` and ``@fastmcp_tool`` produce equivalent specs.
# --------------------------------------------------------------------------


def test_toolspec_and_fastmcp_tool_equivalent():
    """Same self-contained function yields an equivalent spec via both paths.

    ``@toolspec`` carries a hand-written schema (kept for precise constraints
    like enum/minItems), ``@fastmcp_tool`` derives it from the signature; for
    a self-contained function the two must agree on name, description, and the
    effective input schema.
    """

    def get_weather(city: str, units: Literal["c", "f"] = "c") -> dict:
        """Get current weather for a city."""
        return {"city": city, "units": units}

    by_hand = toolspec(
        description="Get current weather for a city.",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["c", "f"], "default": "c"},
            },
            "required": ["city"],
        },
    )(get_weather)
    by_fastmcp = fastmcp_tool()(get_weather)

    assert by_hand.name == by_fastmcp.name == "get_weather"
    assert by_hand.description == by_fastmcp.description
    assert by_fastmcp.readonly is False

    def _normalize(schema: dict) -> dict:
        """Drop pydantic-only ``title`` fields so the two schemas compare."""
        props = {
            name: {k: v for k, v in prop.items() if k != "title"}
            for name, prop in schema["properties"].items()
        }
        return {
            "type": schema["type"],
            "properties": props,
            "required": schema.get("required"),
        }

    assert _normalize(by_hand.input_schema) == _normalize(by_fastmcp.input_schema)
