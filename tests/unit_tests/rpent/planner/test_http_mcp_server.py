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

from __future__ import annotations

import asyncio
import itertools
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server

from rpent.dashboard.events import DashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.planner.utils import http_mcp_server
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, tool
from rpent.tools.human_in_the_loop import HumanInTheLoopInput
from rpent.utils.logging import init_output_dir

CONCURRENT_CALLS = [
    ("list_dir", {"path": "resources/libero/memory"}),
    ("read_text_file", {"path": "robots/libero/guides/strict_hybrid_guide.md"}),
    ("read_text_file", {"path": "robots/libero/guides/pro_hybrid_guide.md"}),
    ("read_text_file", {"path": "robots/libero/guides/env_calibration.md"}),
    ("view_env_state", {"step": 0}),
]
_REQUEST_IDS = itertools.count(1)


class RecordingSink(DashboardEventSink):
    def __init__(self) -> None:
        self.events: list[Any] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event: Any) -> None:
        self.events.append(event)


class FakeToolkit(Toolkit):
    """Minimal toolkit whose tools sleep to widen the overlap window."""

    def __init__(self, state_dir: Path) -> None:
        super().__init__(
            dashboard_events=RecordingSink(),
            state=EnvState(state_dir),
            memory=MemoryManager(state_dir / "memory"),
        )
        self.overlap_errors: list[tuple[str, dict[str, Any]]] = []
        self._register_fake_tools()

    def _register_fake_tools(self) -> None:
        @tool(readonly=True)
        def read_text_file(path: str, max_chars: int = 40000) -> ToolResult:
            time.sleep(0.05)
            p = Path(path)
            return ToolResult(
                data={"path": str(p), "size": 0, "content": "fake content"}
            )

        @tool(readonly=True)
        def list_dir(path: str = "") -> ToolResult:
            time.sleep(0.05)
            return ToolResult(data={"path": path, "count": 0, "files": []})

        @tool
        def view_env_state(step: int = -1) -> ToolResult:
            time.sleep(0.3)
            return ToolResult(data={"step": step, "mode": "evaluation"})

        self.add_tool(read_text_file, replace=True)
        self.add_tool(list_dir, replace=True)
        self.add_tool(view_env_state)

    def execute_tool(
        self,
        name: str,
        input_dict: dict[str, Any],
        *,
        cancel_event: threading.Event | None = None,
    ) -> ToolResult:
        result = super().execute_tool(name, input_dict, cancel_event=cancel_event)
        if result.data.get("error") == "another tool operation is still active":
            self.overlap_errors.append((name, dict(input_dict)))
        return result

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        return ToolResult(data={"observed": True})

    def solved(self) -> bool:
        return False


def test_stop_timeout_retains_references_for_retry(tmp_path):
    from types import SimpleNamespace

    server = HttpMcpServer(FakeToolkit(tmp_path))

    class Thread:
        alive = True

        def join(self, timeout):
            pass

        def is_alive(self):
            return self.alive

    thread = Thread()
    uvicorn_server = SimpleNamespace(should_exit=False)
    server._thread = thread
    server._server = uvicorn_server
    server.stop(timeout_s=0)
    assert server._thread is thread and server._server is uvicorn_server
    assert uvicorn_server.should_exit
    thread.alive = False
    server.stop(timeout_s=0)
    assert server._thread is None and server._server is None


def test_http_mcp_server_serializes_concurrent_tool_calls(tmp_path: Path) -> None:
    init_output_dir(tmp_path / "log")
    toolkit = FakeToolkit(tmp_path)
    server = HttpMcpServer(toolkit)
    try:
        url = server.start()
        rejected = asyncio.run(_fire_concurrent(url))
    finally:
        server.stop()

    assert rejected == 0
    assert toolkit.overlap_errors == []


async def _fire_concurrent(url: str) -> int:
    rejected = 0
    # trust_env=False keeps the SDK client on loopback even when the host
    # advertises an HTTP proxy (e.g. macOS system settings), matching the
    # production readiness probe in http_mcp_server._wait_for_ready.
    async with httpx.AsyncClient(trust_env=False) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (
            read,
            write,
            _get_session_id,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                results = await asyncio.gather(
                    *(session.call_tool(name, args) for name, args in CONCURRENT_CALLS)
                )
                for result in results:
                    if "another tool operation is still active" in json.dumps(
                        result.content, default=str
                    ):
                        rejected += 1
    return rejected


def test_http_mcp_preserves_native_validation_and_recovers_for_next_call(tmp_path):
    init_output_dir(tmp_path / "log")
    toolkit = FakeToolkit(tmp_path)
    invalid = {"path": 123, "unexpected": True}
    expected_error = toolkit.execute_tool("read_text_file", invalid)
    assert expected_error.is_error
    server = HttpMcpServer(toolkit)

    async def run(url):
        async with httpx.AsyncClient(trust_env=False) as http_client:
            async with streamable_http_client(url, http_client=http_client) as (
                read,
                write,
                _get_session_id,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    declarations = await session.list_tools()
                    published = next(
                        item
                        for item in declarations.tools
                        if item.name == "read_text_file"
                    )
                    native = next(
                        item
                        for item in toolkit.list_tools()
                        if item.name == "read_text_file"
                    )
                    assert published.inputSchema == native.input_schema
                    rejected = await session.call_tool("read_text_file", invalid)
                    assert rejected.isError
                    assert json.loads(rejected.content[0].text) == json.loads(
                        expected_error.to_text()
                    )
                    valid = await session.call_tool("read_text_file", {"path": "valid"})
                    assert not valid.isError
                    assert json.loads(valid.content[0].text)["path"] == "valid"

    try:
        asyncio.run(run(server.start()))
    finally:
        server.stop()
    assert toolkit.overlap_errors == []


class OperatorToolkit(FakeToolkit):
    """Wait for a real input broker before counting a fake reset."""

    def __init__(self, state_dir: Path) -> None:
        super().__init__(state_dir)
        self.operator = HumanInTheLoopInput(interactive=True)
        self.resets = 0
        self.probe_calls = 0
        self.cleanup_started = threading.Event()
        self.cleanup_release = threading.Event()
        self.cleanup_release.set()

        @tool(readonly=True)
        def request_scene_reset() -> ToolResult:
            """Wait for confirmation before a fake reset."""
            try:
                response = self.operator.request(
                    "Confirm a fake reset.", self.raise_if_cancelled, kind="reset"
                )
                self.raise_if_cancelled()
                if response is not None and response.split()[0] == "done":
                    self.resets += 1
                return ToolResult(data={"resets": self.resets})
            finally:
                self.cleanup_started.set()
                assert self.cleanup_release.wait(5), (
                    "test worker cleanup was not released"
                )

        self.add_tool(request_scene_reset)

        @tool(readonly=True)
        def probe() -> ToolResult:
            """Count an observation request."""
            self.probe_calls += 1
            return ToolResult(data={"observed": True})

        self.add_tool(probe)


@pytest.fixture
def operator_server(tmp_path: Path):
    init_output_dir(tmp_path / "log")
    toolkit = OperatorToolkit(tmp_path)
    server = HttpMcpServer(toolkit)
    try:
        yield toolkit, server.start()
    finally:
        toolkit.cleanup_release.set()
        toolkit.operator.cancel_pending()
        toolkit.cancel_active_and_wait()
        server.stop()


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=3)


async def _call_over_http(
    client: httpx.AsyncClient,
    url: str,
    name: str,
    *,
    timeout: float = 5,
) -> dict[str, Any]:
    await _initialize_http_session(client, url)
    response = await client.post(
        url,
        json={
            "jsonrpc": "2.0",
            "id": next(_REQUEST_IDS),
            "method": "tools/call",
            "params": {"name": name, "arguments": {}},
        },
        headers={"Accept": "application/json, text/event-stream"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["result"]


async def _initialize_http_session(client: httpx.AsyncClient, url: str) -> None:
    if "Mcp-Session-Id" in client.headers:
        return
    client.headers["Accept"] = "application/json, text/event-stream"
    response = await client.post(
        url,
        json={
            "jsonrpc": "2.0",
            "id": next(_REQUEST_IDS),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "cancellation-test", "version": "1"},
            },
        },
    )
    response.raise_for_status()
    client.headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
    client.headers["Mcp-Protocol-Version"] = response.json()["result"][
        "protocolVersion"
    ]
    response = await client.post(
        url,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    response.raise_for_status()


def test_http_disconnect_retires_operator_request_without_late_reset(operator_server):
    toolkit, url = operator_server

    async def run() -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            request = asyncio.create_task(
                _call_over_http(client, url, "request_scene_reset", timeout=0.25)
            )
            await _wait_until(lambda: toolkit.operator.pending_kind == "reset")
            request_id = toolkit.operator._pending[0]
            with pytest.raises(httpx.ReadTimeout):
                await request
            await _wait_until(lambda: toolkit.operator.pending_kind is None)
            assert toolkit.operator.route_line(f"/operator {request_id} done")
            assert toolkit.operator.route_line("/done")
            assert toolkit.resets == 0

            next_request = asyncio.create_task(
                _call_over_http(client, url, "request_scene_reset")
            )
            await _wait_until(lambda: toolkit.operator.pending_kind == "reset")
            assert toolkit.operator._pending[0] != request_id
            toolkit.operator.route_line(f"/operator {request_id} done")
            assert toolkit.operator.pending_kind == "reset"
            toolkit.operator.route_line("/done current request confirmed")
            result = await next_request
            assert not result.get("isError", False)
            assert toolkit.resets == 1
            assert toolkit.operator.pending_kind is None

    asyncio.run(run())


def test_http_disconnect_keeps_serialization_until_worker_settles(operator_server):
    toolkit, url = operator_server
    toolkit.cleanup_release.clear()

    async def run() -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            request = asyncio.create_task(
                _call_over_http(client, url, "request_scene_reset", timeout=0.25)
            )
            await _wait_until(lambda: toolkit.operator.pending_kind == "reset")
            with pytest.raises(httpx.ReadTimeout):
                await request
            await _wait_until(toolkit.cleanup_started.is_set)
            probe = asyncio.create_task(_call_over_http(client, url, "probe"))
            await asyncio.sleep(0.1)
            assert not probe.done()
            assert toolkit.probe_calls == 0
            toolkit.cleanup_release.set()
            result = await probe
            assert not result.get("isError", False)
            assert toolkit.probe_calls == 1
            assert toolkit.overlap_errors == []
            assert toolkit.resets == 0

    asyncio.run(run())


def test_queued_http_disconnect_does_not_cancel_active_operator_request(
    operator_server,
):
    toolkit, url = operator_server

    async def run() -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            request = asyncio.create_task(
                _call_over_http(client, url, "request_scene_reset")
            )
            await _wait_until(lambda: toolkit.operator.pending_kind == "reset")
            with pytest.raises(httpx.ReadTimeout):
                await _call_over_http(client, url, "probe", timeout=0.1)
            await asyncio.sleep(0.1)
            assert toolkit.operator.pending_kind == "reset"
            assert toolkit.probe_calls == 0
            toolkit.operator.route_line("/done")
            result = await request
            assert not result.get("isError", False)
            await asyncio.sleep(0.1)
            assert toolkit.probe_calls == 0
            assert toolkit.resets == 1
            result = await _call_over_http(client, url, "probe")
            assert not result.get("isError", False)
            assert toolkit.probe_calls == 1

    asyncio.run(run())


@pytest.mark.parametrize("cancel_twice", [False, True])
def test_cancelled_mcp_coroutine_waits_for_sync_worker_cleanup(
    tmp_path, monkeypatch, cancel_twice
):
    init_output_dir(tmp_path / "log")
    toolkit = OperatorToolkit(tmp_path)
    toolkit.cleanup_release.clear()
    handlers = []

    async def receive() -> dict[str, Any]:
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    class CapturingServer(Server):
        @property
        def request_context(self):
            return SimpleNamespace(request=SimpleNamespace(receive=receive))

        def call_tool(self, **kwargs):
            register = super().call_tool(**kwargs)

            def capture(handler):
                handlers.append(handler)
                return register(handler)

            return capture

    monkeypatch.setattr(http_mcp_server, "Server", CapturingServer)
    http_mcp_server._build_asgi_app(toolkit)
    call_tool = handlers[0]

    async def run() -> None:
        request = asyncio.create_task(call_tool("request_scene_reset", {}))
        try:
            await _wait_until(lambda: toolkit.operator.pending_kind == "reset")
            request.cancel()
            await _wait_until(toolkit.cleanup_started.is_set)
            if cancel_twice:
                request.cancel()
                await asyncio.sleep(0.05)
            assert toolkit.operator.pending_kind is None
            assert not request.done()
            probe = asyncio.create_task(call_tool("probe", {}))
            await asyncio.sleep(0.1)
            assert not probe.done()
            assert toolkit.probe_calls == 0
            toolkit.cleanup_release.set()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert not (await probe).isError
            assert toolkit.probe_calls == 1
            assert toolkit.resets == 0
            assert toolkit.overlap_errors == []
        finally:
            toolkit.cleanup_release.set()
            toolkit.operator.cancel_pending()
            toolkit.cancel_active_and_wait()
            await asyncio.gather(request, return_exceptions=True)

    asyncio.run(run())


def test_mcp_cancel_notification_retires_confirmation_without_http_disconnect(
    operator_server,
):
    toolkit, url = operator_server

    async def run() -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            await _initialize_http_session(client, url)
            request_id = next(_REQUEST_IDS)
            request = asyncio.create_task(
                client.post(
                    url,
                    json={
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "tools/call",
                        "params": {"name": "request_scene_reset", "arguments": {}},
                    },
                )
            )
            await _wait_until(lambda: toolkit.operator.pending_kind == "reset")
            operator_id = toolkit.operator._pending[0]
            response = await client.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": request_id, "reason": "tool deadline"},
                },
            )
            response.raise_for_status()
            response = await request
            response.raise_for_status()
            assert response.json()["error"]["message"] == "Request cancelled"
            await _wait_until(lambda: toolkit.operator.pending_kind is None)
            toolkit.operator.route_line(f"/operator {operator_id} done")
            toolkit.operator.route_line("/done")
            assert toolkit.resets == 0
            result = await _call_over_http(client, url, "probe")
            assert not result.get("isError", False)
            assert toolkit.probe_calls == 1
            assert toolkit.overlap_errors == []

    asyncio.run(run())


def test_server_tool_deadline_retires_confirmation_with_client_still_connected(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CODEX_MCP_TOOL_TIMEOUT_S", "0.1")
    init_output_dir(tmp_path / "log")
    toolkit = OperatorToolkit(tmp_path)
    server = HttpMcpServer(toolkit)

    async def run(url: str) -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            request = asyncio.create_task(
                _call_over_http(client, url, "request_scene_reset")
            )
            await _wait_until(lambda: toolkit.operator.pending_kind == "reset")
            operator_id = toolkit.operator._pending[0]
            result = await request
            assert result["isError"]
            assert json.loads(result["content"][0]["text"])["code"] == "tool_cancelled"
            assert toolkit.operator.pending_kind is None
            toolkit.operator.route_line(f"/operator {operator_id} done")
            toolkit.operator.route_line("/done")
            assert toolkit.resets == 0
            result = await _call_over_http(client, url, "probe")
            assert not result.get("isError", False)
            assert toolkit.probe_calls == 1

    try:
        asyncio.run(run(server.start()))
    finally:
        toolkit.operator.cancel_pending()
        toolkit.cancel_active_and_wait()
        server.stop()
