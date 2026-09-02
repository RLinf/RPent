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

"""The normalized tool representation: :class:`ToolSpec`.

Hand-written spec dicts, fastmcp ``@mcp.tool`` objects, and the co-location
decorators in :mod:`rpent.tools.func_tool` all converge on ``ToolSpec``;
:class:`rpent.tools.toolkit.Toolkit` stores and returns tools in this form.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from rpent.utils.templates import default_variables, substitute


def readonly(func):
    """Mark a tool handler as not advancing environment state.

    Tool handlers capture a fresh observation (:meth:`Toolkit.get_env_state`)
    by default. Apply this marker to observational and file/IO tools that do
    not move the robot or otherwise change the environment.
    """
    func._readonly = True
    return func


def _is_readonly(handler: Callable[..., Any]) -> bool:
    """Whether ``handler`` was marked with :func:`readonly`."""
    target = handler
    while isinstance(target, partial):
        target = target.func
    target = getattr(target, "__func__", target)
    return bool(getattr(target, "_readonly", False))


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Normalized representation of one tool: name, schema, and handler.

    Hand-written spec dicts, fastmcp ``@mcp.tool`` objects, and the
    co-location decorators in :mod:`rpent.tools.func_tool` all converge on
    ``ToolSpec``; :class:`Toolkit` stores and returns tools in this form.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    readonly: bool = False

    @classmethod
    def from_spec(
        cls,
        spec: dict[str, Any],
        handler: Callable[..., Any],
        *,
        readonly: bool | None = None,
    ) -> "ToolSpec":
        """Normalize a hand-written spec dict into a ``ToolSpec``.

        ``readonly`` defaults to the :func:`readonly` marker on ``handler``.
        """
        if readonly is None:
            readonly = _is_readonly(handler)
        return cls(
            name=spec["name"],
            description=spec["description"],
            input_schema=spec["input_schema"],
            handler=handler,
            readonly=readonly,
        )

    @classmethod
    def from_fastmcp(
        cls,
        fast_tool: Any,
        *,
        readonly: bool = False,
    ) -> "ToolSpec":
        """Convert a fastmcp ``Tool`` (from ``@mcp.tool``) into a ``ToolSpec``."""
        return cls(
            name=fast_tool.name,
            description=fast_tool.description,
            input_schema=fast_tool.parameters,
            handler=fast_tool.fn,
            readonly=readonly or bool(getattr(fast_tool.fn, "_readonly", False)),
        )

    def resolved(self, variables: Mapping[str, Any] | None = None) -> "ToolSpec":
        """Return a copy with ``{{name}}`` placeholders substituted.

        Substitution happens here (on each ``get_tools_spec()`` call) rather
        than at construction time because ``output_dir`` is only known at run
        time. The built-in variables (e.g. ``output_dir``) are always applied;
        ``variables`` adds or overrides them. The original object is never
        mutated.
        """
        merged = dict(default_variables())
        if variables:
            merged.update(variables)
        return ToolSpec(
            name=substitute(self.name, merged),
            description=substitute(self.description, merged),
            input_schema=substitute(self.input_schema, merged),
            handler=self.handler,
            readonly=self.readonly,
        )

    def to_spec_dict(self) -> dict[str, Any]:
        """Return the Anthropic-shaped spec dict for dict-consuming callers."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
