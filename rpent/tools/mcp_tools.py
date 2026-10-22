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

"""Expose selected site MCP tools as first-class Toolkit tools.

Tools that need coordination with the robot's execution loop do not belong in
the planner layer: the model must not be able to call a charging-dock or
safety interlock out of band. A robot adopts them through
:meth:`~rpent.tools.toolkit.Toolkit.add_mcp_tools`, which registers the named
remote tools on its Toolkit. Every planner then sees them under RPent's own
``mcp__rpent__`` namespace, so all backends behave the same.

Adoption is explicit on purpose. Site configuration can declare a server and
reserve its tools by omitting ``planner_tools``, but only robot code that
knows how to coordinate with the device registers them here.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, create_model

from rpent.tools.base import Tool, ToolResult
from rpent.tools.toolkit import Toolkit
from rpent.utils.mcp_servers import McpServerSpec, request_headers

#: Marks an optional argument the model did not send, so it is not forwarded
#: to the remote server as an explicit ``null``.
_OMITTED = object()

_JSON_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def register_mcp_tools(
    toolkit: Toolkit,
    servers: Sequence[McpServerSpec],
    *,
    tools: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Register selected remote tools from ``servers`` on ``toolkit``.

    Args:
        toolkit: The robot Toolkit receiving the tools.
        servers: Candidate specs, usually ``RunConfig.mcp_servers`` (site-level
            plus robot-declared).
        tools: Server name to the remote tool names this robot adopts. Servers
            that are not listed are ignored.

    Returns:
        The registered tool names, in the order given.

    Raises:
        ValueError: On an unknown server or tool name, or a name that already
            belongs to the toolkit or another adopted server.
        RuntimeError: If the MCP client dependency is missing.
    """
    by_name = {server.name: server for server in servers}
    unknown_servers = sorted(set(tools) - set(by_name))
    if unknown_servers:
        declared = sorted(by_name) or "none"
        raise ValueError(
            f"cannot adopt MCP tools from undeclared server(s) {unknown_servers}; "
            f"declared servers: {declared}"
        )

    registered: list[str] = []
    taken = {tool.name for tool in toolkit.list_tools()}
    for server_name, names in tools.items():
        spec = by_name[server_name]
        # Fail at toolkit construction, not on the first tool call.
        request_headers(spec)
        remote = {tool.name: tool for tool in _list_remote_tools(spec)}
        missing = [name for name in names if name not in remote]
        if missing:
            raise ValueError(
                f"mcp_servers.{server_name} did not expose tool(s) {missing}; "
                f"available: {sorted(remote) or 'none'}"
            )
        for name in names:
            if name in taken:
                raise ValueError(
                    f"cannot adopt MCP tool {name!r} from {server_name!r}: the "
                    "tool name is already registered by the toolkit or another "
                    "server"
                )
            tool = remote[name]
            toolkit.add_tool(_adopted_tool(spec, tool))
            taken.add(name)
            registered.append(name)
    return tuple(registered)


def _adopted_tool(spec: McpServerSpec, tool: Any) -> Tool:
    """Wrap one remote tool descriptor as a native Toolkit declaration."""
    schema = tool.inputSchema or {"type": "object", "properties": {}}
    return Tool(
        name=tool.name,
        description=tool.description or "",
        args_schema=_args_model(tool.name, schema),
        handler=_make_handler(spec, tool.name),
        _input_schema=schema,
        readonly=_is_read_only(tool),
    )


def _args_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Build the validator Toolkit applies to the remote tool's arguments."""
    required = set(schema.get("required") or ())
    fields = {
        prop: (
            _argument_annotation(prop_schema),
            ... if prop in required else _OMITTED,
        )
        for prop, prop_schema in (schema.get("properties") or {}).items()
    }
    return create_model(
        f"{name}Args",
        __config__=ConfigDict(extra="forbid", allow_inf_nan=False),
        **fields,
    )


def _argument_annotation(schema: Any) -> Any:
    """Map a JSON-Schema property onto the annotation Toolkit validates.

    Nested shapes stay permissive: the remote server owns their validation,
    and the model already sees the original schema.
    """
    if not isinstance(schema, dict):
        return Any
    if schema.get("enum"):
        return Literal[tuple(schema["enum"])]
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((item for item in kind if item != "null"), None)
    if kind == "array":
        return list[_argument_annotation(schema.get("items"))]
    if kind == "object":
        return dict[str, Any]
    return _JSON_TYPES.get(kind, Any)


def _is_read_only(tool: Any) -> bool:
    """Whether the server declared this tool as read-only.

    MCP's ``readOnlyHint`` maps onto RPent's :func:`readonly` marker, so read
    tools do not capture a fresh observation after every call while
    side-effecting facility tools do.
    """
    annotations = getattr(tool, "annotations", None)
    return bool(annotations is not None and annotations.readOnlyHint)


def _make_handler(spec: McpServerSpec, name: str) -> Callable[..., ToolResult]:
    """Build the synchronous Toolkit handler for one remote tool."""

    def handler(**arguments: Any) -> ToolResult:
        sent = {key: value for key, value in arguments.items() if value is not _OMITTED}
        return ToolResult(data=_result_payload(_call_remote_tool(spec, name, sent)))

    return handler


def _list_remote_tools(spec: McpServerSpec) -> list[Any]:
    """Fetch one server's tool descriptors."""

    async def list_tools(session: Any) -> list[Any]:
        return list((await session.list_tools()).tools)

    return _run_remote(spec, list_tools)


def _call_remote_tool(spec: McpServerSpec, name: str, arguments: dict[str, Any]) -> Any:
    """Call one remote tool and return the MCP result object."""

    async def call(session: Any) -> Any:
        return await session.call_tool(name, arguments)

    return _run_remote(spec, call)


def _run_remote(spec: McpServerSpec, action: Callable[[Any], Awaitable[Any]]) -> Any:
    """Run one MCP session action to completion.

    Tool handlers execute on planner worker threads, where ``asyncio.run`` is
    safe. If a handler is ever invoked from a thread that already owns a loop,
    the action runs on a short-lived helper thread instead of failing.
    """
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as exc:
        raise RuntimeError(
            "adopting site MCP tools requires the MCP client; install it with "
            '`pip install "pydantic-ai-slim[mcp]"`.'
        ) from exc

    headers = request_headers(spec) or None

    async def main() -> Any:
        async with streamablehttp_client(spec.url, headers=headers) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await action(session)

    def run() -> Any:
        return asyncio.run(main())

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run()

    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = run()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
            outcome["error"] = exc

    thread = threading.Thread(target=target, name=f"mcp-{spec.name}", daemon=True)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _result_payload(result: Any) -> dict[str, Any]:
    """Render an MCP tool result as RPent's serializable result dict."""
    texts: list[str] = []
    blocks: list[dict[str, Any]] = []
    for block in result.content or []:
        if getattr(block, "type", None) == "text":
            texts.append(block.text)
        else:
            blocks.append(block.model_dump(exclude_none=True))

    payload: dict[str, Any] = {}
    if texts:
        payload["text"] = "\n".join(texts)
    if blocks:
        payload["content"] = blocks
    if result.structuredContent is not None:
        payload["structured_content"] = result.structuredContent
    if result.isError:
        payload["error"] = payload.get("text") or "MCP tool reported an error"
    return payload
