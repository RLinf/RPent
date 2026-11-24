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

"""Bridge external streamable-HTTP MCP servers onto :class:`Toolkit`.

Every external MCP tool is registered on the robot's :class:`Toolkit` under
``<server>__<tool>``, so all online planners (``api``, ``claude_code``,
``codex``, and ``gemini_cli``) share one transport, timeout, cancellation,
filtering, and capture path.

By default (``sync_with_robot=False``), remote tools register with
``readonly=True``: they run without locking out other read-only tools and return
their remote text, structured data, and PNG images directly to the model.
Setting ``sync_with_robot=True`` registers them with ``readonly=False`` and
``preserve_result=True`` so they serialize against robot actions, trigger
``get_env_state`` observation capture, and keep their remote payload under
``tool_result``.
"""

from __future__ import annotations

import asyncio
import base64
import builtins
import contextlib
import json
import os
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import timedelta
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, create_model, model_validator

from rpent.tools.base import MAX_TOOL_TEXT_BYTES, Tool, ToolResult
from rpent.tools.toolkit import (
    ToolCancelled,
    Toolkit,
    _ToolOperation,
    current_tool_operation,
)
from rpent.utils.logging import get_logger
from rpent.utils.mcp_servers import ALL_TOOLS, McpServerSpec

logger = get_logger("mcp_tools")

#: Marks an optional argument the model did not send, so it is not forwarded
#: to the remote server as an explicit ``null``.
_OMITTED = object()

#: Default response deadline for one MCP request. The environment variable
#: ``MCP_CALL_TIMEOUT_S`` overrides it.
DEFAULT_CALL_TIMEOUT_S = 60.0

_JSON_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def mcp_tool_name(server_name: str, tool_name: str) -> str:
    """Build the injective ``<server>__<tool>`` Toolkit name for a remote tool."""
    return f"{server_name}__{tool_name}"


def register_mcp_servers(
    toolkit: Toolkit,
    servers: Sequence[McpServerSpec],
) -> tuple[str, ...]:
    """Register every unattached server in ``servers`` onto ``toolkit``.

    Servers already registered via :func:`register_mcp_tools` are skipped so a
    robot integration can customize one server while still accepting shared
    servers from ``--mcp-config``.

    Args:
        toolkit: The robot Toolkit receiving the tools.
        servers: Validated server specs, usually ``RunConfig.mcp_servers``.

    Returns:
        The registered ``<server>__<tool>`` names, in declaration order.
    """
    registered: list[str] = []
    taken = {name.casefold() for name in toolkit._tools}
    for spec in servers:
        if spec.name in toolkit._registered_mcp_servers:
            continue
        registered.extend(
            _register_server(
                toolkit,
                spec,
                selected=None if spec.tools == ALL_TOOLS else spec.tools,
                sync_with_robot=spec.sync_with_robot,
                taken=taken,
            )
        )
    return tuple(registered)


def register_mcp_tools(
    toolkit: Toolkit,
    servers: Sequence[McpServerSpec],
    *,
    tools: Mapping[str, Sequence[str]],
    sync_with_robot: bool | None = None,
) -> tuple[str, ...]:
    """Register selected remote tools from ``servers`` on ``toolkit``.

    Args:
        toolkit: The robot Toolkit receiving the tools.
        servers: Candidate specs, usually ``RunConfig.mcp_servers``.
        tools: Server name to the remote tool names to register. Servers not
            listed are ignored.
        sync_with_robot: Override ``spec.sync_with_robot`` for these tools when
            not ``None``.

    Returns:
        The registered ``<server>__<tool>`` tool names, in the order given.

    Raises:
        ValueError: On an unknown server or tool name, or a name that collides
            case-insensitively with an existing tool.
    """
    by_name = {server.name: server for server in servers}
    unknown_servers = sorted(set(tools) - set(by_name))
    if unknown_servers:
        declared = sorted(by_name) or "none"
        raise ValueError(
            f"cannot register MCP tools from undeclared server(s) {unknown_servers}; "
            f"declared servers: {declared}"
        )

    registered: list[str] = []
    taken = {name.casefold() for name in toolkit._tools}
    for server_name, names in tools.items():
        spec = by_name[server_name]
        sync = spec.sync_with_robot if sync_with_robot is None else sync_with_robot
        registered.extend(
            _register_server(
                toolkit,
                spec,
                selected=tuple(dict.fromkeys(names)),
                sync_with_robot=sync,
                taken=taken,
            )
        )
    return tuple(registered)


def _register_server(
    toolkit: Toolkit,
    spec: McpServerSpec,
    *,
    selected: Sequence[str] | None,
    sync_with_robot: bool,
    taken: set[str],
) -> list[str]:
    """Discover and register tools from one server onto ``toolkit``."""
    discovered = list_remote_tools(spec)
    remote = {tool.name: tool for tool in discovered}
    if selected is None:
        names = [tool.name for tool in discovered]
    else:
        missing = [name for name in selected if name not in remote]
        if missing:
            raise ValueError(
                f"mcp_servers.{spec.name} did not expose tool(s) {missing}; "
                f"available: {sorted(remote) or 'none'}"
            )
        names = list(selected)

    registered: list[str] = []
    for remote_name in names:
        published_name = mcp_tool_name(spec.name, remote_name)
        if published_name.casefold() in taken:
            raise ValueError(
                f"cannot register MCP tool {published_name!r} from {spec.name!r}: "
                "the tool name is already registered on this toolkit"
            )
        toolkit.add_tool(
            _registered_tool(
                spec,
                remote[remote_name],
                published_name=published_name,
                sync_with_robot=sync_with_robot,
            )
        )
        taken.add(published_name.casefold())
        registered.append(published_name)

    toolkit._registered_mcp_servers.add(spec.name)
    return registered


def _registered_tool(
    spec: McpServerSpec,
    tool: Any,
    *,
    published_name: str,
    sync_with_robot: bool,
) -> Tool:
    """Wrap one remote tool descriptor as a native Toolkit declaration."""
    schema = tool.inputSchema or {"type": "object", "properties": {}}
    return Tool(
        name=published_name,
        description=tool.description or "",
        args_schema=_args_model(published_name, schema),
        handler=_make_handler(spec, tool.name),
        _input_schema=schema,
        # The server's advisory readOnlyHint is ignored; callers control robot
        # serialization explicitly through sync_with_robot.
        readonly=not sync_with_robot,
        preserve_result=sync_with_robot,
    )


def _args_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Build the validator Toolkit applies to the remote tool's arguments.

    The remote ``inputSchema`` stays authoritative: this model only rejects
    what the schema rejects. Additional properties are accepted unless the
    schema sets ``additionalProperties: false``, nullable and multi-type
    properties keep every type the schema allows, and nested shapes stay
    permissive.

    Properties pydantic reserves cannot become model fields; see
    :func:`_can_be_a_field`. They are accepted as extras and forwarded
    unchanged, leaving their type and required checks to the server.
    """
    required = set(schema.get("required") or ())
    properties = schema.get("properties") or {}
    fields = {
        prop: (
            _argument_annotation(prop_schema),
            ... if prop in required else _OMITTED,
        )
        for prop, prop_schema in properties.items()
        if _can_be_a_field(prop)
    }
    declared = frozenset(properties)

    def _reject_undeclared(value: Any) -> Any:
        if schema.get("additionalProperties") is False and isinstance(value, dict):
            undeclared = sorted(set(value) - declared)
            if undeclared:
                raise ValueError(
                    f"undeclared argument(s) {undeclared}: the remote schema "
                    "sets additionalProperties: false"
                )
        return value

    return create_model(
        f"{name}Args",
        __config__=ConfigDict(extra="allow", allow_inf_nan=False),
        __validators__={
            "reject_undeclared": model_validator(mode="before")(_reject_undeclared)
        },
        **fields,
    )


def _can_be_a_field(name: str) -> bool:
    """Whether ``create_model`` can declare ``name`` as a field.

    Pydantic reserves ``BaseModel`` members (``model_dump``, ``model_validate``,
    ``schema``, ``copy``) and leading underscores. Those arguments stay valid
    for the remote server and are forwarded as extras instead.
    """
    return not name.startswith("_") and not hasattr(BaseModel, name)


def _argument_annotation(schema: Any) -> Any:
    """Map a JSON-Schema property onto the annotation Toolkit validates."""
    if not isinstance(schema, dict):
        return Any
    enum_values = schema.get("enum")
    if (
        isinstance(enum_values, list)
        and enum_values
        and all(isinstance(item, (str, int, bool, type(None))) for item in enum_values)
    ):
        return Literal[tuple(enum_values)]
    kind = schema.get("type")
    if isinstance(kind, list):
        kinds = [item for item in kind if item != "null"]
        if not kinds:
            return Any
        annotations = tuple(_kind_annotation(item, schema) for item in kinds)
        annotation = annotations[0] if len(annotations) == 1 else Union[annotations]
        return Optional[annotation] if len(kinds) != len(kind) else annotation
    return _kind_annotation(kind, schema)


def _kind_annotation(kind: Any, schema: dict[str, Any]) -> Any:
    """Map one JSON-Schema ``type`` value onto a Python annotation."""
    if kind == "array":
        return list[_argument_annotation(schema.get("items"))]
    if kind == "object":
        return dict[str, Any]
    return _JSON_TYPES.get(kind, Any)


def _make_handler(spec: McpServerSpec, name: str) -> Callable[..., ToolResult]:
    """Build the synchronous Toolkit handler for one remote tool."""

    def handler(**arguments: Any) -> ToolResult:
        sent = {key: value for key, value in arguments.items() if value is not _OMITTED}
        result = _call_remote_tool(spec, name, sent)
        return ToolResult(data=result_payload(result), images=result_images(result))

    return handler


async def _list_tools(session: Any) -> list[Any]:
    """Fetch every page of ``tools/list`` within one request deadline."""
    deadline = call_timeout_s()

    async def collect() -> list[Any]:
        tools: list[Any] = []
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            page = await session.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.nextCursor
            if not cursor or cursor in seen:
                return tools
            seen.add(cursor)

    try:
        return await asyncio.wait_for(collect(), timeout=deadline)
    except asyncio.TimeoutError:
        raise TimeoutError(f"tools/list did not finish within {deadline:g}s") from None


def list_remote_tools(spec: McpServerSpec) -> list[Any]:
    """Fetch every tool descriptor a server exposes, following pagination."""
    return _run_remote(spec, _list_tools)


def _call_remote_tool(spec: McpServerSpec, name: str, arguments: dict[str, Any]) -> Any:
    """Call one remote tool and return the MCP result object."""

    async def call(session: Any) -> Any:
        return await session.call_tool(name, arguments)

    return _run_remote(spec, call)


def _mcp_client() -> tuple[Any, Any]:
    """Import the MCP client lazily so importing ``rpent`` stays light."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    return ClientSession, streamablehttp_client


def call_timeout_s() -> float:
    """Return the response deadline for one MCP request."""
    raw = os.environ.get("MCP_CALL_TIMEOUT_S")
    if raw is None:
        return DEFAULT_CALL_TIMEOUT_S
    try:
        timeout_s = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"MCP_CALL_TIMEOUT_S must be a number of seconds, got {raw!r}"
        ) from exc
    if timeout_s <= 0:
        raise ValueError(f"MCP_CALL_TIMEOUT_S must be positive, got {timeout_s}")
    return timeout_s


def _exception_group_types() -> tuple[type[BaseException], ...]:
    """Return the exception-group types task groups raise on this Python."""
    builtin_group = getattr(builtins, "BaseExceptionGroup", None)
    if builtin_group is not None:
        return (builtin_group,)
    try:  # pragma: no cover - Python 3.10
        from exceptiongroup import BaseExceptionGroup
    except ImportError:  # pragma: no cover - keep the raw group text
        return ()
    return (BaseExceptionGroup,)  # pragma: no cover - Python 3.10


_EXCEPTION_GROUP_TYPES = _exception_group_types()


def _leaf_error(exc: BaseException) -> BaseException:
    """Unwrap single-cause transport ``ExceptionGroup`` wrappers."""
    while isinstance(exc, _EXCEPTION_GROUP_TYPES) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


async def run_remote(
    spec: McpServerSpec,
    action: Callable[[Any], Awaitable[Any]],
    *,
    operation: _ToolOperation | None = None,
) -> Any:
    """Run one action on a fresh MCP session with timeout and cancellation.

    Each call opens its own session and bounds ``initialize``, ``tools/list``,
    and ``tools/call`` by ``MCP_CALL_TIMEOUT_S`` (60 seconds by default). The
    session closes with ``terminate_on_close=False`` so timed-out or cancelled
    calls return immediately instead of waiting on a remote ``DELETE``
    handshake. When ``operation`` is provided, ``Toolkit.cancel_active_and_wait``
    cancels the active asyncio task immediately.
    """
    if operation is not None and operation.cancel_event.is_set():
        raise ToolCancelled("tool operation interrupted")

    ClientSession, streamablehttp_client = _mcp_client()
    headers = dict(spec.headers) or None
    timeout = timedelta(seconds=call_timeout_s())
    # Keep the HTTP read timeout slightly above ClientSession's deadline so
    # ClientSession raises a single timeout error instead of racing with
    # httpx.ReadTimeout inside anyio's TaskGroup, while allowing
    # MCP_CALL_TIMEOUT_S values above mcp's 300 s default.
    sse_read_timeout = timeout + timedelta(seconds=5)

    loop = asyncio.get_running_loop()
    task = asyncio.current_task(loop)
    unregister: Callable[[], None] | None = None
    if operation is not None and task is not None:

        def _request_cancel() -> None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(task.cancel)

        unregister = operation.on_cancel(_request_cancel)

    try:
        async with streamablehttp_client(
            spec.url,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            terminate_on_close=False,
        ) as (read, write, _):
            async with ClientSession(
                read, write, read_timeout_seconds=timeout
            ) as session:
                await session.initialize()
                return await action(session)
    except asyncio.CancelledError:
        if operation is not None and operation.cancel_event.is_set():
            raise ToolCancelled("tool operation interrupted") from None
        raise
    except _EXCEPTION_GROUP_TYPES as exc:
        if operation is not None and operation.cancel_event.is_set():
            raise ToolCancelled("tool operation interrupted") from None
        raise _leaf_error(exc) from None
    finally:
        if unregister is not None:
            unregister()


def _run_remote(spec: McpServerSpec, action: Callable[[Any], Awaitable[Any]]) -> Any:
    """Run :func:`run_remote` to completion from synchronous toolkit code.

    Captures the active :class:`_ToolOperation` before running (including when
    marshalled to a helper thread from an already-running event loop) so
    cancellation interrupts in-flight requests cleanly.
    """
    operation = current_tool_operation()

    def run() -> Any:
        return asyncio.run(run_remote(spec, action, operation=operation))

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


def result_images(result: Any) -> list[bytes]:
    """Decode ``image/png`` blocks into RPent's image payload."""
    images: list[bytes] = []
    for block in result.content or []:
        if (
            getattr(block, "type", None) != "image"
            or getattr(block, "mimeType", None) != "image/png"
        ):
            continue
        try:
            encoded = "".join(block.data.split())
            images.append(base64.b64decode(encoded, validate=True))
        except ValueError as exc:
            logger.warning("MCP tool returned image/png with invalid base64: %s", exc)
    return images


def cap_result_text(text: str, limit: int = MAX_TOOL_TEXT_BYTES) -> str:
    """Bound an external text payload to ``limit`` UTF-8 bytes."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    suffix = "\n[truncated]"
    budget = limit - len(suffix.encode("utf-8"))
    if budget < 0:
        return encoded[:limit].decode("utf-8", errors="ignore")
    return encoded[:budget].decode("utf-8", errors="ignore") + suffix


def _bounded_json(value: Any) -> Any:
    """Keep a JSON value as data when it fits; otherwise keep capped JSON text."""
    serialized = json.dumps(value)
    if len(serialized.encode("utf-8")) <= MAX_TOOL_TEXT_BYTES:
        return value
    return cap_result_text(serialized)


def _block_metadata(block: Any) -> dict[str, Any]:
    """Describe a non-forwarded result block without its raw payload."""
    metadata = block.model_dump(mode="json", exclude_none=True)
    for mapping in (metadata, metadata.get("resource")):
        if isinstance(mapping, dict):
            for key in ("data", "blob", "text"):
                mapping.pop(key, None)
    metadata["notice"] = "only text and image/png content is forwarded to the model"
    return metadata


def result_payload(result: Any) -> dict[str, Any]:
    """Render an MCP tool result as RPent's bounded, serializable result dict."""
    texts: list[str] = []
    blocks: list[dict[str, Any]] = []
    for block in result.content or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            texts.append(block.text)
            continue
        if kind == "image" and getattr(block, "mimeType", None) == "image/png":
            continue
        blocks.append(_block_metadata(block))

    payload: dict[str, Any] = {}
    if texts:
        payload["text"] = cap_result_text("\n".join(texts))
    if blocks:
        payload["content"] = _bounded_json(blocks)
    if result.structuredContent is not None:
        payload["structured_content"] = _bounded_json(result.structuredContent)
    if result.isError:
        payload["error"] = cap_result_text(
            payload.get("text") or "MCP tool reported an error"
        )
    return payload
