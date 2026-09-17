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

"""In-process streamable-HTTP MCP server that wraps a :class:`Toolkit`.

The Codex CLI accepts streamable-HTTP MCP servers via
``mcp_servers.<name>.url``. This module builds an MCP
:class:`~mcp.server.lowlevel.Server` and serves it through a
:class:`~mcp.server.streamable_http_manager.StreamableHTTPSessionManager`
on a background uvicorn thread — no separate subprocess, no second
``Toolkit`` instance.

Usage::

    server = HttpMcpServer(toolkit)
    server.start()  # binds 127.0.0.1 on a free port
    codex_url = server.url  # e.g. "http://127.0.0.1:54321/mcp/"
    ...
    server.stop()
"""

from __future__ import annotations

import asyncio
import base64
import socket
import threading
from typing import Any

import httpx
import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from rpent.planner.base import execute_tool
from rpent.tools import Tool, ToolResult
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("mcp_http")

SERVER_NAME = "rpent"


def _strip_mcp_prefix(name: str) -> str:
    """``mcp__rpent__mcp_list_dir`` -> ``mcp_list_dir`` ; passthrough."""
    prefix = f"mcp__{SERVER_NAME}__"
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def list_mcp_tools(toolkit: Toolkit) -> tuple[Tool, ...]:
    """Codex and Claude use built-in image readers; read_image is API-only."""
    return tuple(tool for tool in toolkit.list_tools() if tool.name != "read_image")


def mcp_result(result: ToolResult) -> dict[str, Any]:
    """Return MCP text/PNG blocks and the native error flag."""
    return {
        "content": [
            {"type": "text", "text": result.to_text()},
            *[
                {
                    "type": "image",
                    "data": base64.b64encode(data).decode("ascii"),
                    "mimeType": "image/png",
                }
                for data in result.images
            ],
        ],
        "isError": result.is_error,
    }


def build_mcp_server(toolkit: Toolkit) -> Server:
    """Build the MCP service shared by Claude and Codex with native validation."""
    mcp_app: Server = Server(SERVER_NAME, version="0.1.0")
    tool_execution_lock = asyncio.Lock()
    exported_tools = list_mcp_tools(toolkit)
    exported_names = {tool.name for tool in exported_tools}

    @mcp_app.list_tools()
    async def _list_tools() -> list[types.Tool]:
        tools: list[types.Tool] = []
        for tool in exported_tools:
            tools.append(
                types.Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.input_schema,
                )
            )
        return tools

    @mcp_app.call_tool(validate_input=False)
    async def _call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        lookup = _strip_mcp_prefix(name)
        if lookup not in exported_names:
            result = ToolResult(error=f"Unknown tool: {lookup}")
        else:
            async with tool_execution_lock:
                result = await execute_tool(toolkit, lookup, arguments or {})
        return types.CallToolResult(**mcp_result(result))

    return mcp_app


def _build_asgi_app(toolkit: Toolkit) -> Any:
    """Build a raw ASGI3 app wrapping the native MCP service."""
    session_manager = StreamableHTTPSessionManager(
        app=build_mcp_server(toolkit),
        stateless=True,
        json_response=True,
    )

    # minimal ASGI wrapper
    async def asgi_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    async with session_manager.run():
                        await send({"type": "lifespan.startup.complete"})
                        shutdown_event = await receive()
                        # Use an explicit check rather than ``assert`` —
                        # ``python -O`` strips asserts and would turn a
                        # protocol violation into a silent mis-handle.
                        if shutdown_event["type"] != "lifespan.shutdown":
                            break
                        await send({"type": "lifespan.shutdown.complete"})
                    break
                elif event["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    break
            return
        if scope["type"] == "http":
            await session_manager.handle_request(scope, receive, send)

    return asgi_app


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_for_ready(url: str, *, timeout_s: float) -> None:
    """POST an MCP ``initialize`` request, retrying connection failures."""
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hc", "version": "0"},
        },
    }
    transport = httpx.HTTPTransport(retries=10)
    with httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(timeout_s, connect=2),
        trust_env=False,
    ) as c:
        resp = c.post(url, json=body, headers={"Accept": "application/json"})
        body_preview = resp.text[:200]
        if not (resp.is_success and "result" in resp.json()):
            raise RuntimeError(
                f"HttpMcpServer not ready: code: {resp.status_code}; content: {body_preview}"
            )


class HttpMcpServer:
    """Run an in-process streamable-HTTP MCP server over a Toolkit.

    The server runs on a background daemon thread with its own asyncio loop;
    callers must invoke :meth:`start` before reading :attr:`url` and
    :meth:`stop` to release the port before the process exits.
    """

    def __init__(
        self,
        toolkit: Toolkit,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        path: str = "/mcp",
    ) -> None:
        self._toolkit = toolkit
        self._host = host
        self._port = port or _pick_free_port(host)
        self._path = path if path.startswith("/") else f"/{path}"
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}{self._path}/"

    def start(self, *, ready_timeout_s: float = 30.0) -> str:
        """Launch uvicorn in a background thread and block until it's serving."""
        if self._thread is not None:
            return self.url

        app = _build_asgi_app(self._toolkit)

        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
            lifespan="on",
        )
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(
            target=self._server.run, name="mcp-http-server", daemon=True
        )
        self._thread.start()

        _wait_for_ready(self.url, timeout_s=ready_timeout_s)
        logger.info("HttpMcpServer ready at %s", self.url)
        return self.url

    def stop(self, *, timeout_s: float = 5.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                logger.warning(
                    "HttpMcpServer thread did not exit within %.1fs; "
                    "leaving references intact",
                    timeout_s,
                )
        self._server = None
        self._thread = None
