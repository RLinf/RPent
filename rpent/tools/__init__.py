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

"""Agent tool declarations, handlers, and result serialization."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rpent.tools.base import Tool, ToolResult, iter_tools, tool, tool_error
    from rpent.tools.toolkit import Toolkit

__all__ = [
    "Tool",
    "Toolkit",
    "ToolResult",
    "tool",
    "tool_error",
    "iter_tools",
]


def __getattr__(name: str) -> Any:
    """Load tool declarations only when requested."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = "rpent.tools.toolkit" if name == "Toolkit" else "rpent.tools.base"
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
