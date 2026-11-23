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

"""Expose selected external MCP tools as first-class Toolkit tools.

Tools that must be ordered against robot actions do not belong with the
planner: the model must not be able to call a charging dock or safety
interlock out of band. A robot registers them through
:meth:`~rpent.tools.toolkit.Toolkit.add_mcp_tools`; they then behave like local
tools, including exclusive execution and observation capture.

Registration is explicit on purpose. The shared config can declare a server
and reserve its tools by omitting ``planner_tools``, but only robot code that
knows how to coordinate with the device registers them here.
"""

from __future__ import annotations

import asyncio
import base64
import builtins
import json
import os
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import timedelta
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, create_model, model_validator

from rpent.tools.base import MAX_TOOL_TEXT_BYTES, Tool, ToolResult
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger
from rpent.utils.mcp_servers import ALL_TOOLS, McpServerSpec, request_headers

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


def register_mcp_tools(
    toolkit: Toolkit,
    servers: Sequence[McpServerSpec],
    *,
    tools: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Register selected remote tools from ``servers`` on ``toolkit``.

    Args:
        toolkit: The robot Toolkit receiving the tools.
        servers: Candidate specs, usually ``RunConfig.mcp_servers`` (shared
            plus robot-declared).
        tools: Server name to the remote tool names this robot registers.
            Servers that are not listed are ignored.

    Returns:
        The registered tool names, in the order given.

    Raises:
        ValueError: On an unknown server or tool name, a name that already
            belongs to the toolkit or another registered server (compared
            case-insensitively, so an external ``Finish`` cannot shadow the
            toolkit's ``finish``), or a tool the planner also attaches.
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
    taken = {tool.name.casefold() for tool in toolkit.list_tools()}
    for server_name, names in tools.items():
        spec = by_name[server_name]
        # A coordinated tool must not also be attached to the planner: the
        # planner would bypass the execution ordering and observation capture
        # that are the reason to register it.
        if spec.planner_tools == ALL_TOOLS:
            raise ValueError(
                f"cannot register MCP tool(s) {sorted(names)} from {server_name!r}: "
                "planner_tools is '*', so every tool is also attached to the "
                "planner; list only the independent tools under planner_tools "
                "and register the coordinated ones"
            )
        overlap = sorted(set(names) & set(spec.planner_tools or ()))
        if overlap:
            raise ValueError(
                f"cannot register MCP tool(s) {overlap} from {server_name!r}: they "
                "are also attached to the planner by planner_tools"
            )
        # Fail at toolkit construction, not on the first tool call.
        request_headers(spec)
        remote = {tool.name: tool for tool in list_remote_tools(spec)}
        missing = [name for name in names if name not in remote]
        if missing:
            raise ValueError(
                f"mcp_servers.{server_name} did not expose tool(s) {missing}; "
                f"available: {sorted(remote) or 'none'}"
            )
        for name in names:
            if name.casefold() in taken:
                raise ValueError(
                    f"cannot register MCP tool {name!r} from {server_name!r}: the "
                    "tool name is already registered by the toolkit or another "
                    "server"
                )
            tool = remote[name]
            toolkit.add_tool(_registered_tool(spec, tool))
            taken.add(name.casefold())
            registered.append(name)
    return tuple(registered)


def _registered_tool(spec: McpServerSpec, tool: Any) -> Tool:
    """Wrap one remote tool descriptor as a native Toolkit declaration."""
    schema = tool.inputSchema or {"type": "object", "properties": {}}
    return Tool(
        name=tool.name,
        description=tool.description or "",
        args_schema=_args_model(tool.name, schema),
        handler=_make_handler(spec, tool.name),
        _input_schema=schema,
        # Coordinated tools run exclusively and capture an observation after
        # the call. The server's ``readOnlyHint`` is advisory; a third-party
        # server that marks a side-effecting device tool read-only must not be
        # able to bypass the ordering this registration exists to provide.
        readonly=False,
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
    Undeclared arguments are therefore rejected by a validator rather than
    ``extra="forbid"``, which would reject those properties too.
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

    Pydantic refuses names that repeat a ``BaseModel`` member (``model_dump``,
    ``model_validate``) and drops names starting with an underscore or naming
    the config slot; other base attributes, such as ``schema`` and ``copy``,
    only warn. All of these are still valid arguments for the remote server,
    so they are accepted as extras instead of declared fields.
    """
    return not name.startswith("_") and not hasattr(BaseModel, name)


def _argument_annotation(schema: Any) -> Any:
    """Map a JSON-Schema property onto the annotation Toolkit validates.

    ``null`` becomes ``Optional`` and a ``type`` list becomes a ``Union`` so
    arguments the remote schema allows are not rejected locally. Nested shapes
    stay permissive: the remote server owns their validation, and the model
    already sees the original schema.
    """
    if not isinstance(schema, dict):
        return Any
    if schema.get("enum"):
        return Literal[tuple(schema["enum"])]
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
    """Fetch every page of ``tools/list`` within one request deadline.

    ``tools/list`` returns one page per request; the SDK does not follow
    ``nextCursor`` for the caller. Toolkit registration and the planner's
    startup check both need the complete list, so a tool on a later page must
    not be reported as missing. Each page is bounded by
    ``MCP_CALL_TIMEOUT_S``, and that deadline also bounds the whole exchange:
    a server that keeps returning new cursors cannot hold start-up. A repeated
    cursor is treated as the end of the list.
    """
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


async def list_remote_tools_async(spec: McpServerSpec) -> list[Any]:
    """Async form of :func:`list_remote_tools` for the planner."""
    return await run_remote(spec, _list_tools)


async def call_remote_tool_async(
    spec: McpServerSpec, name: str, arguments: dict[str, Any]
) -> Any:
    """Call one remote tool from an event loop and return the MCP result."""

    async def call(session: Any) -> Any:
        return await session.call_tool(name, arguments)

    return await run_remote(spec, call)


def _call_remote_tool(spec: McpServerSpec, name: str, arguments: dict[str, Any]) -> Any:
    """Call one remote tool and return the MCP result object."""

    async def call(session: Any) -> Any:
        return await session.call_tool(name, arguments)

    return _run_remote(spec, call)


def _mcp_client() -> tuple[Any, Any]:
    """Import the MCP client lazily, so importing ``rpent`` stays light.

    ``mcp`` is a core dependency; this is an import-cost choice, not an
    optional install. ``streamablehttp_client`` is the SDK's deprecated
    spelling; its replacement, ``streamable_http_client``, takes headers only
    through a pre-built ``httpx.AsyncClient`` and drops ``sse_read_timeout``,
    so the idle-event-stream guard would need a hand-rolled client here.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    return ClientSession, streamablehttp_client


def call_timeout_s() -> float:
    """Return the response deadline for one MCP request.

    The HTTP transport's connect timeout and the SSE idle read timeout do not
    bound a request: a server that keeps sending keep-alives, or that ends the
    event stream without an event id, leaves the pending request open forever.
    The session deadline is what makes cancellation and cleanup terminate.
    """
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
    """Return the exception-group types task groups raise on this Python.

    ``BaseExceptionGroup`` is a builtin from Python 3.11. On Python 3.10 the
    transports raise the ``exceptiongroup`` backport bundled with anyio, so
    the builtin name alone would not match — or even exist.
    """
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
    """Return the single exception wrapped by transport task groups.

    Closing the SDK's transport after a failed request wraps the real error in
    nested ``ExceptionGroup``s. Unwrapping a single-cause group keeps the
    message the model and the planner error report show (a timeout, a refused
    connection) instead of ``unhandled errors in a TaskGroup``.
    """
    while isinstance(exc, _EXCEPTION_GROUP_TYPES) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


async def run_remote(
    spec: McpServerSpec, action: Callable[[Any], Awaitable[Any]]
) -> Any:
    """Run one action on a fresh MCP session.

    Each call opens its own session and bounds every request — ``initialize``,
    ``tools/list``, and ``tools/call`` — by ``MCP_CALL_TIMEOUT_S`` (60 seconds
    by default). Toolkit cancellation is cooperative and the handler has no
    checkpoint while the remote call is in flight, so this deadline is also
    the bound on how long ``Toolkit.cancel_active_and_wait`` waits for an
    registered tool.

    The session is closed without terminating it on the server. The transport's
    ``DELETE`` handshake waits for the server to finish the in-flight request,
    so a server that streams a tool's response for its whole runtime would hold
    a timed-out call for that runtime — the deadline would bound the error
    message but not the call. RPent never reuses a session, and servers expire
    idle ones on their own.

    The transports close a failed session inside an anyio task group, which
    reports the real error as a nested ``ExceptionGroup``; it is unwrapped here
    so callers see the timeout or refused connection instead of the group.
    """
    ClientSession, streamablehttp_client = _mcp_client()
    headers = request_headers(spec) or None
    timeout = timedelta(seconds=call_timeout_s())
    try:
        async with streamablehttp_client(
            spec.url, headers=headers, terminate_on_close=False
        ) as (
            read,
            write,
            _,
        ):
            async with ClientSession(
                read, write, read_timeout_seconds=timeout
            ) as session:
                await session.initialize()
                return await action(session)
    except _EXCEPTION_GROUP_TYPES as exc:
        raise _leaf_error(exc) from None


def _run_remote(spec: McpServerSpec, action: Callable[[Any], Awaitable[Any]]) -> Any:
    """Run :func:`run_remote` to completion from synchronous toolkit code.

    Tool handlers execute on planner worker threads, where ``asyncio.run`` is
    safe. If a handler is ever invoked from a thread that already owns a loop,
    the action runs on a short-lived helper thread instead of failing.
    """

    def run() -> Any:
        return asyncio.run(run_remote(spec, action))

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
    """Decode PNG image blocks into RPent's image payload.

    RPent's planners advertise ``ToolResult.images`` as PNG, so only
    ``image/png`` blocks are forwarded as native images. Line breaks are
    removed before decoding; a payload that is still not valid base64 is
    dropped instead of failing the whole call, so the text of the same result
    still reaches the model.
    """
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
    """Bound one external text part the way local tools bound their payloads.

    Local results reach the model through ``ToolResult.to_text``, which encodes
    ASCII-only, so its limit is both characters and bytes. External text is not
    encoded, so the cap applies to the UTF-8 bytes; otherwise non-ASCII text
    could grow several times larger than the local bound. Without the cap one
    verbose server can exceed the model context and leave the transcript and
    Dashboard payloads unbounded.

    Args:
        text: The text to bound.
        limit: UTF-8 bytes this part may use. A caller forwarding several parts
            of one result passes what the earlier parts left of the limit, so
            the result is bounded as a whole; the transcript path caps each
            part on its own.
    """
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
    """Describe a result block RPent does not forward, without its payload."""
    metadata = block.model_dump(mode="json", exclude_none=True)
    # ``data`` (image, audio), ``blob`` (binary resource), and ``text`` (text
    # resource) are the payloads that were not forwarded.
    for mapping in (metadata, metadata.get("resource")):
        if isinstance(mapping, dict):
            for key in ("data", "blob", "text"):
                mapping.pop(key, None)
    metadata["notice"] = "only text and image/png content is forwarded to the model"
    return metadata


def result_payload(result: Any) -> dict[str, Any]:
    """Render an MCP tool result as RPent's serializable result dict.

    Text and PNG images are the content RPent forwards; PNG blocks travel
    through ``ToolResult.images``. Every other block keeps only its metadata:
    its body (base64 data, or a resource's text) would otherwise reach the
    model as truncated text, and pydantic URL objects would break trajectory
    serialization.

    The payload is what a Toolkit tool returns, and the Toolkit records it in
    the trajectory and the Dashboard, so text and structured content are
    bounded here: a remote server must not decide how much RPent persists.
    """
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
        # The planner raises this text as a retry and the transcript records
        # it, so it needs the same bound as forwarded text.
        payload["error"] = cap_result_text(
            payload.get("text") or "MCP tool reported an error"
        )
    return payload
