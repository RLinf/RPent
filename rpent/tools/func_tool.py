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

"""Decorators that turn functions into :class:`ToolSpec`.

Two mechanisms:

- :func:`toolspec`: keep a hand-written dict spec (``name`` / ``description``
  / ``input_schema``) written next to its function.
- :func:`fastmcp_tool`: register the function with a literal
  ``mcp.server.fastmcp.FastMCP`` instance and convert the schema FastMCP
  derives from the function signature back into a :class:`ToolSpec`.

Both return a :class:`~rpent.tools.tool_spec.ToolSpec` ready for
``Toolkit.add_tool_spec``.
"""

from __future__ import annotations

from typing import Any, Callable

from rpent.tools.tool_spec import ToolSpec


def toolspec(
    *,
    name: str | None = None,
    description: str,
    input_schema: dict[str, Any],
    readonly: bool | None = None,
) -> Callable[[Callable[..., Any]], ToolSpec]:
    """Co-locate a hand-written spec with its function as a ``ToolSpec``.

    Args:
        name: Tool name as the LLM sees it; defaults to the function name.
        description: The hand-written description the LLM sees.
        input_schema: Hand-written Anthropic-shaped JSON schema dict.
        readonly: Mark the tool as observation-only (runs concurrently).
            Defaults to the :func:`~rpent.tools.toolkit.readonly` marker on
            the function.
    """

    def _decorate(fn: Callable[..., Any]) -> ToolSpec:
        spec = {
            "name": name or fn.__name__,
            "description": description,
            "input_schema": input_schema,
        }
        return ToolSpec.from_spec(spec, fn, readonly=readonly)

    return _decorate


def fastmcp_tool(
    *,
    name: str | None = None,
    readonly: bool = False,
    exclude: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], ToolSpec]:
    """Register ``fn`` with FastMCP and convert the derived tool to ``ToolSpec``.

    Uses a literal ``mcp.server.fastmcp.FastMCP`` instance purely as a schema
    derivation engine (no server is started or served); ``description`` and
    ``input_schema`` are derived from the function docstring and signature.
    ``exclude`` drops runtime-bound arguments (e.g. ``state=``) from the
    derived input schema.
    """

    def _decorate(fn: Callable[..., Any]) -> ToolSpec:
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("rpent")
        tool_name = name or fn.__name__
        mcp.tool(name=tool_name)(fn)
        fast_tool = mcp._tool_manager.get_tool(tool_name)
        spec = ToolSpec.from_fastmcp(fast_tool, readonly=readonly)
        if exclude:
            spec = _without_excluded(spec, exclude)
        return spec

    return _decorate


def _without_excluded(spec: ToolSpec, exclude: tuple[str, ...]) -> ToolSpec:
    """Return ``spec`` with ``exclude`` params removed from the input schema.

    FastMCP 1.x derives the schema from the whole signature and has no
    per-parameter exclusion hook, so runtime-bound args are dropped here.
    """
    props = dict(spec.input_schema.get("properties", {}))
    for arg in exclude:
        props.pop(arg, None)
    required = spec.input_schema.get("required") or []
    if isinstance(required, list):
        required = [name for name in required if name not in exclude]
    input_schema = dict(spec.input_schema)
    input_schema["properties"] = props
    if required:
        input_schema["required"] = required
    else:
        input_schema.pop("required", None)
    return ToolSpec(
        name=spec.name,
        description=spec.description,
        input_schema=input_schema,
        handler=spec.handler,
        readonly=spec.readonly,
    )
