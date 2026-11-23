# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""SGLang's native Qwen XML tool format with bounded syntactic whitespace."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from pydantic_ai.tools import ToolDefinition

MAX_WHITESPACE = 16


def _fractional_bounds_validated_by_tool(schema: Any) -> Any:
    """Keep valid zero-valued numbers expressible in XGrammar 0.2's XML format.

    Its fractional-range grammar rejects integer zero inside [-0.5, 0.5].
    The published schema and pre-execution validator still enforce the bounds.
    """
    if isinstance(schema, list):
        return [_fractional_bounds_validated_by_tool(value) for value in schema]
    if not isinstance(schema, dict):
        return schema
    result = {
        key: _fractional_bounds_validated_by_tool(value)
        for key, value in schema.items()
    }
    bounds = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
    types = result.get("type", [])
    if isinstance(types, str):
        types = [types]
    if "number" in types and any(
        isinstance(result.get(key), float) and not result[key].is_integer()
        for key in bounds
    ):
        for key in bounds:
            result.pop(key, None)
    return result


def bounded_tool_format(
    definitions: Sequence[ToolDefinition],
    *,
    parallel: bool = False,
    require_tool: bool = True,
    fractional_bounds_at_execution: bool = False,
) -> dict[str, Any]:
    """Bound whitespace in native tool XML while retaining its argument schema."""
    tags = [
        {
            "type": "tag",
            "begin": f"<tool_call>\n<function={tool.name}>\n",
            "content": {
                "type": "json_schema",
                "json_schema": (
                    _fractional_bounds_validated_by_tool(tool.parameters_json_schema)
                    if fractional_bounds_at_execution
                    else deepcopy(tool.parameters_json_schema)
                ),
                "style": "qwen_xml",
                "max_whitespace_cnt": MAX_WHITESPACE,
            },
            "end": "\n</function>\n</tool_call>",
        }
        for tool in definitions
    ]
    return {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<tool_call>\n<function="],
            "tags": tags,
            "at_least_one": require_tool,
            "stop_after_first": not parallel,
            "excludes": ["<think>", "</think>"],
        },
    }
