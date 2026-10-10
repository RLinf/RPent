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

"""Planner tools built from native environment action specifications."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field, create_model

from rpent.tools import Tool, ToolResult


def direct_action_tool(
    specs: dict[str, dict[str, Any]], handler: Callable[..., ToolResult]
) -> Tool:
    """Build a native tool and validator from the connected environment's layouts.

    The ``default`` layout has no action_type argument. Typed environments
    advertise their supported modes as keys, with the first mode the default.
    """
    sizes = [len(spec["low"]) for spec in specs.values()]
    fields = {
        "values": (list[float], Field(min_length=min(sizes), max_length=max(sizes)))
    }
    if list(specs) != ["default"]:
        fields["action_type"] = (Literal[tuple(specs)], next(iter(specs)))
    args_schema = create_model("DirectActionArgs", **fields)
    return Tool(
        name="execute_action",
        description=(
            "Execute one native environment action alongside VLA and scripted primitives. "
            "Use the connected environment's layouts and per-coordinate bounds below "
            "(null means unbounded). " + json.dumps(specs, allow_nan=False)
        ),
        args_schema=args_schema,
        handler=handler,
        _input_schema=args_schema.model_json_schema(),
    )
