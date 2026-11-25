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

"""Offline contract tests for Toolkit-backed external MCP tool integration."""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic_ai.messages import BinaryContent

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.planner.api_loop import _HarnessToolkit
from rpent.planner.claude_code import _build_rpent_server
from rpent.planner.gemini_cli import _write_settings as write_gemini_settings
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.robots.base import get_toolkit
from rpent.robots.robot_spec import RunConfig
from rpent.session import EnvState
from rpent.tools.base import MAX_TOOL_TEXT_BYTES, ToolResult, tool
from rpent.tools.mcp_tools import (
    call_timeout_s,
    list_remote_tools,
    mcp_tool_name,
)
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import init_output_dir
from rpent.utils.mcp_servers import McpServerSpec

# Minimal valid 1x1 PNG.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _running_mcp_server(
    tools: list[types.Tool],
    handler: Callable[[str, dict[str, Any], dict[str, str]], Any],
    *,
    page_size: int | None = None,
    list_delay_s: float = 0.0,
) -> Iterator[tuple[str, list[dict[str, str]]]]:
    """Start a real local streamable-HTTP MCP server on a background thread."""
    mcp_app = Server("external-test", version="0.1.0")
    received_headers: list[dict[str, str]] = []

    if page_size is not None:

        async def _list_paginated(
            req: types.ListToolsRequest | None,
        ) -> types.ServerResult:
            if list_delay_s > 0:
                await asyncio.sleep(list_delay_s)
            cursor = req.params.cursor if req and req.params else None
            start = int(cursor) if cursor is not None else 0
            chunk = tools[start : start + page_size]
            next_cursor = (
                str(start + page_size) if start + page_size < len(tools) else None
            )
            return types.ServerResult(
                types.ListToolsResult(tools=chunk, nextCursor=next_cursor)
            )

        mcp_app.request_handlers[types.ListToolsRequest] = _list_paginated
    else:

        @mcp_app.list_tools()
        async def _list() -> list[types.Tool]:
            if list_delay_s > 0:
                await asyncio.sleep(list_delay_s)
            return list(tools)

    @mcp_app.call_tool(validate_input=False)
    async def _call(name: str, arguments: dict[str, Any]) -> Any:
        return await handler(name, arguments or {}, dict(received_headers[-1]))

    session_manager = StreamableHTTPSessionManager(
        app=mcp_app,
        stateless=True,
        json_response=True,
    )

    async def asgi_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    async with session_manager.run():
                        await send({"type": "lifespan.startup.complete"})
                        shutdown = await receive()
                        if shutdown["type"] != "lifespan.shutdown":
                            break
                        await send({"type": "lifespan.shutdown.complete"})
                    break
                if event["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    break
            return
        if scope["type"] == "http":
            headers = {
                k.decode("latin1").lower(): v.decode("latin1")
                for k, v in scope.get("headers", ())
            }
            received_headers.append(headers)
            await session_manager.handle_request(scope, receive, send)

    port = _free_port()
    config = uvicorn.Config(
        asgi_app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "local MCP test server failed to start"
    try:
        yield f"http://127.0.0.1:{port}/mcp/", received_headers
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


class _RecordingToolkit(Toolkit):
    """Minimal Toolkit tracking get_env_state calls for tests."""

    def __init__(self, tmp_path: Path) -> None:
        init_output_dir(tmp_path / "out")
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        state = EnvState(tmp_path / "state")
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            state=state,
            memory=MemoryManager(root=memory_dir),
        )
        self.env_state_calls: list[dict[str, Any]] = []
        self.add_tool(self.robot_step)

    @tool
    def robot_step(self, label: str) -> ToolResult:
        """Execute a dummy robot action."""
        return ToolResult(data={"moved": label})

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        self.env_state_calls.append(
            {"command": command, "result": result, "elapsed_s": elapsed_s}
        )
        return ToolResult(data={"step": len(self.env_state_calls), "arm": "idle"})


def test_default_zero_code_registration_returns_remote_payload_directly(
    tmp_path: Path,
) -> None:
    seen_args: list[dict[str, Any]] = []
    tools = [
        types.Tool(
            name="read_power",
            description="Read power meter.",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit": {"type": "string", "enum": ["W", "kW"]},
                    "include_png": {"type": "boolean"},
                },
                "required": ["unit"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="list_outlets",
            description="List power outlets.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

    async def handle(
        name: str, arguments: dict[str, Any], headers: dict[str, str]
    ) -> types.CallToolResult:
        seen_args.append(arguments)
        assert headers.get("authorization") == "Bearer lab-token"
        content: list[Any] = [
            types.TextContent(type="text", text=f"{name}: 42 {arguments['unit']}")
        ]
        if arguments.get("include_png"):
            content.append(
                types.ImageContent(
                    type="image",
                    data=base64.b64encode(_PNG_BYTES).decode("ascii"),
                    mimeType="image/png",
                )
            )
        return types.CallToolResult(
            content=content,
            structuredContent={"watts": 42},
        )

    with _running_mcp_server(tools, handle) as (url, _):
        toolkit = _RecordingToolkit(tmp_path)
        spec = McpServerSpec(
            name="power",
            url=url,
            headers={"Authorization": "Bearer lab-token"},
        )
        registered = toolkit.add_mcp_servers((spec,))
        assert registered == ("power__read_power", "power__list_outlets")

        # Default sync_with_robot=False skips get_env_state and returns the
        # remote text, structured content, and PNG directly.
        res = toolkit.execute_tool(
            "power__read_power", {"unit": "W", "include_png": True}
        )
        assert not res.is_error
        assert res.data == {
            "text": "read_power: 42 W",
            "structured_content": {"watts": 42},
        }
        assert res.images == [_PNG_BYTES]
        assert toolkit.env_state_calls == []
        assert seen_args == [{"unit": "W", "include_png": True}]

        # Omitted optional parameter is not forwarded as null; undeclared
        # arguments are rejected when additionalProperties is false.
        res2 = toolkit.execute_tool("power__read_power", {"unit": "kW"})
        assert not res2.is_error
        assert seen_args[-1] == {"unit": "kW"}
        bad = toolkit.execute_tool(
            "power__read_power", {"unit": "W", "unknown": "extra"}
        )
        assert bad.is_error
        assert "bad arguments" in bad.data["error"]


def test_sync_with_robot_captures_observation_and_preserves_remote_result(
    tmp_path: Path,
) -> None:
    tools = [
        types.Tool(
            name="set_dock",
            description="Toggle charging dock.",
            inputSchema={
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
            },
        ),
        types.Tool(
            name="fail_dock",
            description="Fail dock command.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

    async def handle(
        name: str, arguments: dict[str, Any], headers: dict[str, str]
    ) -> types.CallToolResult:
        del headers
        if name == "fail_dock":
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="interlock open")],
                isError=True,
            )
        return types.CallToolResult(
            content=[
                types.TextContent(type="text", text=f"dock={arguments['enabled']}"),
                types.ImageContent(
                    type="image",
                    data=base64.b64encode(_PNG_BYTES).decode("ascii"),
                    mimeType="image/png",
                ),
            ]
        )

    with _running_mcp_server(tools, handle) as (url, _):
        toolkit = _RecordingToolkit(tmp_path)
        spec = McpServerSpec(
            name="dock",
            url=url,
            tools=("set_dock", "fail_dock"),
            sync_with_robot=True,
        )
        registered = toolkit.add_mcp_servers((spec,))
        assert registered == ("dock__set_dock", "dock__fail_dock")

        ok = toolkit.execute_tool("dock__set_dock", {"enabled": True})
        assert not ok.is_error
        assert ok.data == {
            "step": 1,
            "arm": "idle",
            "tool_result": {"text": "dock=True"},
        }
        assert ok.images == [_PNG_BYTES]
        assert len(toolkit.env_state_calls) == 1
        assert toolkit.env_state_calls[0]["command"] == {
            "action": "dock__set_dock",
            "enabled": True,
        }

        err = toolkit.execute_tool("dock__fail_dock", {})
        assert err.is_error
        assert err.data["step"] == 2
        assert err.data["error"] == "interlock open"


def test_explicit_add_mcp_tools_skips_auto_registration_duplicate(
    tmp_path: Path,
) -> None:
    tools = [
        types.Tool(name="a", description="A", inputSchema={"type": "object"}),
        types.Tool(name="b", description="B", inputSchema={"type": "object"}),
    ]

    async def handle(
        name: str, arguments: dict[str, Any], headers: dict[str, str]
    ) -> types.CallToolResult:
        del arguments, headers
        return types.CallToolResult(content=[types.TextContent(type="text", text=name)])

    with _running_mcp_server(tools, handle) as (url, _):
        toolkit = _RecordingToolkit(tmp_path)
        spec = McpServerSpec(name="lab", url=url)
        first = toolkit.add_mcp_tools(
            (spec,), tools={"lab": ["a"]}, sync_with_robot=True
        )
        assert first == ("lab__a",)
        # Automatic attachment skips servers already registered by the robot.
        assert toolkit.add_mcp_servers((spec,)) == ()
        names = {t.name for t in toolkit.list_tools()}
        assert "lab__a" in names
        assert "lab__b" not in names


def test_explicit_add_mcp_tools_enforces_yaml_tools_allowlist(tmp_path: Path) -> None:
    tools = [
        types.Tool(name="a", description="A", inputSchema={"type": "object"}),
        types.Tool(name="b", description="B", inputSchema={"type": "object"}),
    ]

    async def handle(
        name: str, arguments: dict[str, Any], headers: dict[str, str]
    ) -> types.CallToolResult:
        del arguments, headers
        return types.CallToolResult(content=[types.TextContent(type="text", text=name)])

    with _running_mcp_server(tools, handle) as (url, _):
        toolkit = _RecordingToolkit(tmp_path)
        restricted = McpServerSpec(name="lab", url=url, tools=("a",))

        with pytest.raises(
            ValueError,
            match="not allowed by mcp_servers.lab.tools",
        ):
            toolkit.add_mcp_tools((restricted,), tools={"lab": ["b"]})

        assert toolkit.add_mcp_tools((restricted,), tools={"lab": ["a"]}) == ("lab__a",)
        assert toolkit.add_mcp_servers((restricted,)) == ()


def test_missing_tool_or_unknown_server_raises_clear_error(tmp_path: Path) -> None:
    tools = [
        types.Tool(name="ping", description="Ping", inputSchema={"type": "object"})
    ]

    async def handle(
        name: str, arguments: dict[str, Any], headers: dict[str, str]
    ) -> types.CallToolResult:
        del name, arguments, headers
        return types.CallToolResult(content=[types.TextContent(type="text", text="ok")])

    with _running_mcp_server(tools, handle) as (url, _):
        toolkit = _RecordingToolkit(tmp_path)
        bad_spec = McpServerSpec(name="lab", url=url, tools=("missing_tool",))
        with pytest.raises(
            ValueError, match="did not expose tool\\(s\\).*missing_tool"
        ):
            toolkit.add_mcp_servers((bad_spec,))
        with pytest.raises(ValueError, match="undeclared server"):
            toolkit.add_mcp_tools((), tools={"lab": ["ping"]})


def test_get_toolkit_auto_registers_run_config_mcp_servers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tools = [
        types.Tool(name="read_temp", description="Temp", inputSchema={"type": "object"})
    ]

    async def handle(
        name: str, arguments: dict[str, Any], headers: dict[str, str]
    ) -> types.CallToolResult:
        del name, arguments, headers
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="23.5 C")]
        )

    with _running_mcp_server(tools, handle) as (url, _):
        fake_robot = SimpleNamespace(
            get_toolkit=lambda **kwargs: _RecordingToolkit(tmp_path)
        )
        monkeypatch.setattr("rpent.robots.base._resolve_robot", lambda name: fake_robot)
        config = RunConfig(
            recipe_tag="run",
            output_dir=tmp_path,
            prompt_vars={},
            task_desc={},
            mcp_servers=(McpServerSpec(name="sensor", url=url),),
        )
        tk = get_toolkit("fake", config=config)
        assert "sensor__read_temp" in {t.name for t in tk.list_tools()}
        assert tk.execute_tool("sensor__read_temp", {}).data == {"text": "23.5 C"}


def test_external_mcp_tools_work_across_all_online_planner_adapters(
    tmp_path: Path,
) -> None:
    tools = [
        types.Tool(
            name="read_meter",
            description="Read meter.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="finish",
            description="External finish tool that must not shadow RPent finish.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

    async def handle(
        name: str, arguments: dict[str, Any], headers: dict[str, str]
    ) -> types.CallToolResult:
        del arguments, headers
        return types.CallToolResult(
            content=[
                types.TextContent(type="text", text=f"remote:{name}"),
                types.ImageContent(
                    type="image",
                    data=base64.b64encode(_PNG_BYTES).decode("ascii"),
                    mimeType="image/png",
                ),
            ]
        )

    with _running_mcp_server(tools, handle) as (url, _):
        toolkit = _RecordingToolkit(tmp_path)
        toolkit.add_mcp_servers((McpServerSpec(name="power", url=url),))
        tool_name = mcp_tool_name("power", "read_meter")
        external_finish = mcp_tool_name("power", "finish")

        # 1. API planner (_HarnessToolkit) respects images and --no-images.
        harness_with_img = _HarnessToolkit(
            toolkit,
            dashboard_events=NullDashboardEventSink(),
            no_images=False,
        )
        ctx = SimpleNamespace(tool_call_id="call-1")
        out = asyncio.run(harness_with_img.call(tool_name, ctx))
        assert json.loads(out.return_value[0]) == {"text": "remote:read_meter"}
        assert isinstance(out.return_value[1], BinaryContent)

        harness_no_img = _HarnessToolkit(
            toolkit,
            dashboard_events=NullDashboardEventSink(),
            no_images=True,
        )
        out_no_img = asyncio.run(harness_no_img.call(tool_name, ctx))
        assert "--no-images" in out_no_img.return_value[1]

        # External finish does not collide with RPent's finish output tool.
        finish_out = asyncio.run(harness_with_img.call(external_finish, ctx))
        assert json.loads(finish_out.return_value[0]) == {"text": "remote:finish"}
        assert harness_with_img.finish_result is None

        # 2. Claude Code SDK bridge exposes the prefixed tool on server "rpent".
        captured_sdk_tools: dict[str, Any] = {}

        class _FakeSdk:
            @staticmethod
            def tool(name: str, description: str, schema: dict[str, Any]):
                del description, schema

                def decorate(fn):
                    captured_sdk_tools[name] = fn
                    return fn

                return decorate

            @staticmethod
            def create_sdk_mcp_server(name: str, version: str, tools: list[Any]):
                return {"name": name, "version": version, "tools": tools}

        _build_rpent_server(_FakeSdk(), toolkit=toolkit)
        assert tool_name in captured_sdk_tools
        claude_res = asyncio.run(captured_sdk_tools[tool_name]({}))
        assert not claude_res.get("isError")
        assert "remote:read_meter" in claude_res["content"][0]["text"]

        # 3. Codex & Gemini CLI share HttpMcpServer; Gemini settings allowlist it.
        http_server = HttpMcpServer(toolkit)
        http_url = http_server.start()
        try:
            exposed = {
                t.name
                for t in list_remote_tools(McpServerSpec(name="bridge", url=http_url))
            }
            assert tool_name in exposed
            settings_path = write_gemini_settings(
                tmp_path / "gemini",
                mcp_url=http_url,
                max_turns=5,
                tool_names=(t.name for t in toolkit.list_tools()),
            )
            settings = json.loads(settings_path.read_text())
            assert f"mcp_rpent_{tool_name}" in settings["tools"]["core"]
        finally:
            http_server.stop()


def test_cancel_active_and_wait_interrupts_inflight_mcp_call_promptly(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    tools = [
        types.Tool(
            name="slow_op",
            description="Blocks for a long time.",
            inputSchema={"type": "object", "properties": {}},
        )
    ]

    async def handle(
        name: str, arguments: dict[str, Any], headers: dict[str, str]
    ) -> types.CallToolResult:
        del name, arguments, headers
        started.set()
        await asyncio.sleep(10.0)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="done")]
        )

    with _running_mcp_server(tools, handle) as (url, _):
        toolkit = _RecordingToolkit(tmp_path)
        toolkit.add_mcp_servers((McpServerSpec(name="slow", url=url),))

        result_box: dict[str, ToolResult] = {}

        def worker() -> None:
            result_box["result"] = toolkit.execute_tool("slow__slow_op", {})

        thread = threading.Thread(target=worker)
        thread.start()
        assert started.wait(timeout=5.0)

        t0 = time.monotonic()
        toolkit.cancel_active_and_wait()
        elapsed = time.monotonic() - t0
        thread.join(timeout=5.0)

        assert elapsed < 3.0
        assert result_box["result"].data["code"] == "tool_cancelled"
        assert result_box["result"].data["interrupted"] is True


def test_timeout_pagination_reserved_fields_and_payload_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tools = [
        types.Tool(
            name="page_one",
            description="Tool 1",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_dump": {"type": "string"},
                    "_private": {"type": "integer"},
                    "nullable": {"type": ["string", "null"]},
                },
            },
        ),
        types.Tool(
            name="page_two",
            description="Tool 2",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="hang",
            description="Hangs",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

    async def handle(
        name: str, arguments: dict[str, Any], headers: dict[str, str]
    ) -> types.CallToolResult:
        del headers
        if name == "hang":
            await asyncio.sleep(5.0)
            return types.CallToolResult(content=[])
        if name == "page_one":
            return types.CallToolResult(
                content=[
                    types.TextContent(type="text", text=json.dumps(arguments)),
                    types.ImageContent(
                        type="image",
                        data=base64.b64encode(b"jpegbytes").decode("ascii"),
                        mimeType="image/jpeg",
                    ),
                ]
            )
        return types.CallToolResult(
            content=[
                types.TextContent(type="text", text="x" * (MAX_TOOL_TEXT_BYTES + 100))
            ]
        )

    with _running_mcp_server(tools, handle, page_size=1) as (url, _):
        toolkit = _RecordingToolkit(tmp_path)
        spec = McpServerSpec(name="paged", url=url)
        registered = toolkit.add_mcp_servers((spec,))
        assert registered == ("paged__page_one", "paged__page_two", "paged__hang")

        # Reserved pydantic field names and nullable types forward intact;
        # non-PNG image blocks keep metadata without raw base64 payload.
        res1 = toolkit.execute_tool(
            "paged__page_one",
            {"model_dump": "ok", "_private": 7, "nullable": None},
        )
        assert json.loads(res1.data["text"]) == {
            "model_dump": "ok",
            "_private": 7,
            "nullable": None,
        }
        assert res1.images == []
        assert res1.data["content"][0]["mimeType"] == "image/jpeg"
        assert "data" not in res1.data["content"][0]

        # Oversized text is capped to MAX_TOOL_TEXT_BYTES.
        res2 = toolkit.execute_tool("paged__page_two", {})
        assert len(res2.data["text"].encode("utf-8")) <= MAX_TOOL_TEXT_BYTES
        assert res2.data["text"].endswith("[truncated]")

        # MCP_CALL_TIMEOUT_S bounds hanging remote calls.
        monkeypatch.setenv("MCP_CALL_TIMEOUT_S", "0.3")
        assert call_timeout_s() == 0.3
        res3 = toolkit.execute_tool("paged__hang", {})
        assert res3.is_error

        # Per-server YAML timeout_s overrides MCP_CALL_TIMEOUT_S.
        monkeypatch.setenv("MCP_CALL_TIMEOUT_S", "60")
        yaml_timed = McpServerSpec(name="yaml_timed", url=url, timeout_s=0.3)
        assert call_timeout_s(yaml_timed) == 0.3
        toolkit.add_mcp_servers((yaml_timed,))
        res4 = toolkit.execute_tool("yaml_timed__hang", {})
        assert res4.is_error


@pytest.mark.parametrize("raw_timeout", ["inf", "-inf", "nan", "0", "-1", "abc"])
def test_call_timeout_s_rejects_invalid_env_values(
    monkeypatch: pytest.MonkeyPatch, raw_timeout: str
) -> None:
    monkeypatch.setenv("MCP_CALL_TIMEOUT_S", raw_timeout)
    with pytest.raises(ValueError, match="MCP_CALL_TIMEOUT_S"):
        call_timeout_s()


def test_list_remote_tools_honors_per_server_timeout_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = [
        types.Tool(name=f"tool_{idx}", description="T", inputSchema={"type": "object"})
        for idx in range(3)
    ]

    async def handle(
        name: str, arguments: dict[str, Any], headers: dict[str, str]
    ) -> types.CallToolResult:
        del name, arguments, headers
        return types.CallToolResult(content=[])

    with _running_mcp_server(tools, handle, page_size=1, list_delay_s=0.15) as (url, _):
        # Lower per-server timeout_s bounds multi-page discovery even when env is higher.
        monkeypatch.setenv("MCP_CALL_TIMEOUT_S", "60")
        tight_spec = McpServerSpec(name="tight", url=url, timeout_s=0.25)
        with pytest.raises(TimeoutError, match="tools/list did not finish"):
            list_remote_tools(tight_spec)

        # Higher per-server timeout_s overrides a tighter global MCP_CALL_TIMEOUT_S.
        monkeypatch.setenv("MCP_CALL_TIMEOUT_S", "0.2")
        relaxed_spec = McpServerSpec(name="relaxed", url=url, timeout_s=3.0)
        discovered = list_remote_tools(relaxed_spec)
        assert [t.name for t in discovered] == ["tool_0", "tool_1", "tool_2"]


def test_get_toolkit_closes_toolkit_when_mcp_registration_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tools = [
        types.Tool(name="ping", description="Ping", inputSchema={"type": "object"})
    ]

    async def handle(
        name: str, arguments: dict[str, Any], headers: dict[str, str]
    ) -> types.CallToolResult:
        del name, arguments, headers
        return types.CallToolResult(content=[types.TextContent(type="text", text="ok")])

    class _ClosableToolkit(_RecordingToolkit):
        def __init__(self, root: Path, *, fail_close: bool = False) -> None:
            super().__init__(root)
            self.closed = False
            self._fail_close = fail_close

        def close(self) -> None:
            self.closed = True
            super().close()
            if self._fail_close:
                raise RuntimeError("close failed")

    with _running_mcp_server(tools, handle) as (url, _):
        config = RunConfig(
            recipe_tag="run",
            output_dir=tmp_path,
            prompt_vars={},
            task_desc={},
            mcp_servers=(McpServerSpec(name="lab", url=url, tools=("missing",)),),
        )

        # Normal failure path closes the constructed toolkit before re-raising.
        tk_ok = _ClosableToolkit(tmp_path / "ok")
        monkeypatch.setattr(
            "rpent.robots.base._resolve_robot",
            lambda name: SimpleNamespace(get_toolkit=lambda **kwargs: tk_ok),
        )
        with pytest.raises(ValueError, match="did not expose tool\\(s\\).*missing"):
            get_toolkit("fake", config=config)
        assert tk_ok.closed is True

        # Secondary cleanup failure must not mask the original MCP error.
        tk_bad = _ClosableToolkit(tmp_path / "bad", fail_close=True)
        monkeypatch.setattr(
            "rpent.robots.base._resolve_robot",
            lambda name: SimpleNamespace(get_toolkit=lambda **kwargs: tk_bad),
        )
        with pytest.raises(ValueError, match="did not expose tool\\(s\\).*missing"):
            get_toolkit("fake", config=config)
        assert tk_bad.closed is True


def test_non_primitive_json_schema_enum_does_not_crash_registration(
    tmp_path: Path,
) -> None:
    tools = [
        types.Tool(
            name="complex_enum",
            description="Accepts complex enum values.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pose": {
                        "type": "object",
                        "enum": [{"x": 0}, {"x": 1}],
                    },
                    "scale": {
                        "type": "number",
                        "enum": [0.5, 1.5],
                    },
                    "tags": {
                        "type": "array",
                        "enum": [["a"], ["b"]],
                    },
                },
            },
        )
    ]

    async def handle(
        name: str, arguments: dict[str, Any], headers: dict[str, str]
    ) -> types.CallToolResult:
        del name, headers
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(arguments))]
        )

    with _running_mcp_server(tools, handle) as (url, _):
        toolkit = _RecordingToolkit(tmp_path)
        registered = toolkit.add_mcp_servers((McpServerSpec(name="geo", url=url),))
        assert registered == ("geo__complex_enum",)
        res = toolkit.execute_tool(
            "geo__complex_enum",
            {"pose": {"x": 1}, "scale": 1.5, "tags": ["a"]},
        )
        assert not res.is_error
        assert json.loads(res.data["text"]) == {
            "pose": {"x": 1},
            "scale": 1.5,
            "tags": ["a"],
        }
