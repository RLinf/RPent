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
import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from rpent.dashboard.events import DashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.session import EnvState
from rpent.tools import ToolResult, tool
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import init_output_dir

CONCURRENT_CALLS = [
    ("list_dir", {"path": "resources/libero/memory"}),
    ("read_text_file", {"path": "robots/libero/guides/strict_hybrid_guide.md"}),
    ("read_text_file", {"path": "robots/libero/guides/pro_hybrid_guide.md"}),
    ("read_text_file", {"path": "robots/libero/guides/env_calibration.md"}),
    ("view_env_state", {"step": 0}),
]


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

        def view_env_state(step: int = -1) -> ToolResult:
            time.sleep(0.3)
            return ToolResult(data={"step": step, "mode": "evaluation"})

        self.add_tool(read_text_file, replace=True)
        self.add_tool(list_dir, replace=True)
        self.add_tool(tool(view_env_state))

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        result = super().execute_tool(name, input_dict)
        if result.to_dict().get("error") == "another tool operation is still active":
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


def test_native_method_results_and_finish_cross_the_mcp_boundary(tmp_path) -> None:
    from rpent.planner.api_loop import _ApiRunObserver
    from rpent.planner.claude_code import _Recorder as ClaudeRecorder
    from rpent.planner.codex import _Recorder as CodexRecorder

    init_output_dir(tmp_path / "log")
    toolkit = FakeToolkit(tmp_path)
    calls = []

    class Primitives:
        @tool(readonly=True)
        def inspect_scene(self, count: int) -> ToolResult:
            """Read the saved camera view."""
            calls.append(count)
            return ToolResult(data={"count": count}, images=[b"camera pixels"])

        @tool(readonly=True)
        def finish(self, status: str) -> ToolResult:
            """Finish only after confirmation."""
            if status != "success":
                return ToolResult(error="operator confirmation required")
            return ToolResult(data={"status": status, "operator_notes": "confirmed"})

    primitives = Primitives()
    toolkit.add_tool(primitives.inspect_scene)
    toolkit.add_tool(primitives.finish, replace=True)
    recorders = [
        _ApiRunObserver(
            toolkit=toolkit, dashboard_events=RecordingSink(), messages=[], max_turns=2
        ),
        ClaudeRecorder(toolkit=toolkit, dashboard_events=RecordingSink(), max_turns=2),
        CodexRecorder(toolkit=toolkit, dashboard_events=RecordingSink(), max_turns=2),
    ]

    async def exercise(url):
        async with httpx.AsyncClient(trust_env=False) as http_client:
            async with streamable_http_client(url, http_client=http_client) as (
                read,
                write,
                _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("inspect_scene", {"count": 2})
                    assert not result.isError
                    assert json.loads(result.content[0].text) == {"count": 2}
                    assert base64.b64decode(result.content[1].data) == b"camera pixels"
                    for count in ("3", True, 3.0):
                        rejected = await session.call_tool(
                            "inspect_scene", {"count": count}
                        )
                        assert rejected.isError and calls == [2]
                        assert json.loads(rejected.content[0].text)["error"] == (
                            "bad arguments for inspect_scene"
                        )
                    rejected = await session.call_tool("inspect_scene", {"self": {}})
                    assert rejected.isError and calls == [2]
                    assert json.loads(rejected.content[0].text)["error"] == (
                        "bad arguments for inspect_scene"
                    )
                    refused = await session.call_tool("finish", {"status": "failure"})
                    assert refused.isError
                    assert all(recorder.finish_result is None for recorder in recorders)
                    accepted = await session.call_tool("finish", {"status": "success"})
                    assert not accepted.isError
                    assert all(
                        recorder.finish_result
                        == {"status": "success", "operator_notes": "confirmed"}
                        for recorder in recorders
                    )

    server = HttpMcpServer(toolkit)
    try:
        asyncio.run(exercise(server.start()))
    finally:
        server.stop()


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
