# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""SGLang's native Qwen XML tool format with bounded syntactic whitespace."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from pydantic_ai.tools import ToolDefinition

MAX_WHITESPACE = 16


def bounded_tool_format(
    definitions: Sequence[ToolDefinition], *, parallel: bool = False
) -> dict[str, Any]:
    """Keep required native calls and exact schemas without empty-object loops.

    SGLang's default XML parameter grammar can generate whitespace forever for
    finish(). Supply the same native structural format through response_format,
    bounding syntactic whitespace rather than adding a dummy tool argument.
    The server's reasoning parser still owns the thinking prefix.
    """
    tags = [
        {
            "type": "tag",
            "begin": f"<tool_call>\n<function={tool.name}>\n",
            "content": {
                "type": "json_schema",
                "json_schema": deepcopy(tool.parameters_json_schema),
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
            "at_least_one": True,
            "stop_after_first": not parallel,
            "excludes": ["<think>", "</think>"],
        },
    }
