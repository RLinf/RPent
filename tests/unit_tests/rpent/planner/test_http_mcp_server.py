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
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from mcp import ClientSession, McpError, types
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.planner.utils.http_mcp_server import HttpMcpServer, build_mcp_server
from rpent.session import EnvState, Payload
from rpent.session.env_capture import CapturingEnvClient
from rpent.tools import Toolkit, ToolResult, tool
from rpent.utils.logging import init_output_dir

CONCURRENT_CALLS = [
    ("list_dir", {"path": "resources/libero/memory"}),
    ("read_text_file", {"path": "robots/libero/guides/strict_hybrid_guide.md"}),
    ("read_text_file", {"path": "robots/libero/guides/pro_hybrid_guide.md"}),
    ("read_text_file", {"path": "robots/libero/guides/env_calibration.md"}),
    ("view_env_state", {"step": 0}),
]


class FakeToolkit(Toolkit):
    """Minimal toolkit whose tools sleep to widen the overlap window."""

    def __init__(self, state_dir: Path) -> None:
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            state=EnvState(state_dir),
            memory=MemoryManager(state_dir / "memory"),
        )
        self.overlap_errors: list[tuple[str, dict[str, Any]]] = []
        self._register_fake_tools()

    def _register_fake_tools(self) -> None:
        @tool(readonly=True)
        def read_text_file(path: str, max_chars: int = 40000) -> dict:
            time.sleep(0.05)
            p = Path(path)
            return ToolResult(
                data={"path": str(p), "size": 0, "content": "fake content"}
            )

        @tool(readonly=True)
        def list_dir(path: str = "") -> dict:
            time.sleep(0.05)
            return ToolResult(data={"path": path, "count": 0, "files": []})

        @tool
        def view_env_state(step: int = -1) -> dict:
            time.sleep(0.3)
            return ToolResult(data={"step": step, "mode": "evaluation"})

        self.add_tool(read_text_file, replace=True)
        self.add_tool(list_dir, replace=True)
        self.add_tool(view_env_state)

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        result = super().execute_tool(name, input_dict)
        if result.data.get("error") == "another tool operation is still active":
            self.overlap_errors.append((name, dict(input_dict)))
        return result

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        return ToolResult(data={"observed": True})

    def solved(self) -> bool:
        return False


def test_http_mcp_server_schedules_mixed_concurrent_tool_calls(tmp_path: Path) -> None:
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


@pytest.mark.parametrize("capture_enabled", [False, True])
def test_http_mcp_readonly_calls_execute_concurrently(tmp_path, capture_enabled):
    init_output_dir(tmp_path / "log")
    toolkit = FakeToolkit(tmp_path)
    if capture_enabled:
        toolkit.state.start_task(instruction="Sense.", config=Payload("test", {}))
    rendezvous = threading.Barrier(2)

    @tool(readonly=True)
    def sense(label: str) -> ToolResult:
        rendezvous.wait(timeout=5)
        return ToolResult(data={"label": label})

    toolkit.add_tool(sense)
    server = HttpMcpServer(toolkit)

    async def scenario(url):
        async with httpx.AsyncClient(trust_env=False) as http_client:
            async with streamable_http_client(url, http_client=http_client) as (
                read,
                write,
                _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    results = await asyncio.gather(
                        *(
                            session.call_tool(
                                "sense",
                                {"label": label},
                                meta={"codex_bridge_mcp_call_id": label},
                            )
                            for label in ("left", "right")
                        )
                    )
                    assert not any(result.isError for result in results)
                    assert [
                        json.loads(result.content[0].text)["label"]
                        for result in results
                    ] == ["left", "right"]

    try:
        asyncio.run(scenario(server.start()))
    finally:
        toolkit.close()
        server.stop()

    calls = toolkit.state.tool_call_records()
    if capture_enabled:
        assert len(calls) == 2
        for call in calls:
            label = call.arguments.data["label"]
            assert call.status == "completed"
            assert call.source_ids == {"codex_bridge_mcp_call_id": label}
            assert call.raw_result.data["data"] == {"label": label}
            assert json.loads(call.feedback.data["content"][0]["text"]) == {
                "label": label
            }
    else:
        assert calls == []


@pytest.fixture
def captured_toolkit(tmp_path):
    init_output_dir(tmp_path / "log")
    toolkit = FakeToolkit(tmp_path)
    toolkit.state.start_task(instruction="Move.", config=Payload("test", {}))
    try:
        yield toolkit
    finally:
        toolkit.close()


def test_mcp_capture_preserves_arguments_feedback_and_environment(captured_toolkit):
    toolkit = captured_toolkit
    state = toolkit.state

    def step(action, *, capture):
        assert capture is True
        return {
            "result": {"moved": True},
            "before_observation": {
                "format": "test.observation",
                "data": {"image": b"before", "position": 0},
            },
            "steps": [
                {
                    "action": {"format": "test.action", "data": action},
                    "after_observation": {
                        "format": "test.observation",
                        "data": {"image": b"after", "position": 1},
                    },
                    "result": {"format": "test.result", "data": {"reward": 1}},
                }
            ],
        }

    env = CapturingEnvClient(SimpleNamespace(step=step), state)

    @tool
    def move(instruction: str, distance: int = 1) -> ToolResult:
        with env.prediction(instruction):
            result = env.step([distance])
        return ToolResult(data=result, images=[b"png"])

    toolkit.add_tool(move)

    async def scenario():
        async with create_connected_server_and_client_session(
            build_mcp_server(toolkit)
        ) as session:
            return await session.call_tool(
                "mcp__rpent__move",
                {"instruction": "Move."},
                meta={"codex_bridge_mcp_call_id": "bridge-1"},
            )

    result = asyncio.run(scenario())
    assert not result.isError
    (call,) = state.tool_call_records()
    assert call.name == "move"
    assert call.planner_id is None and call.origin == "unknown"
    assert call.source_ids == {"codex_bridge_mcp_call_id": "bridge-1"}
    assert call.arguments == Payload(
        "mcp.tool_args", {"instruction": "Move."}, completeness="complete"
    )
    assert call.status == "completed" and call.ended_at >= call.started_at
    assert call.raw_result.data == {"data": {"moved": True}, "images": [b"png"]}
    assert call.feedback.data == result.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )
    assert call.feedback.base64_paths == {"/content/1/data": ""}
    assert json.loads(result.content[0].text) == {"observed": True}
    assert base64.b64decode(result.content[1].data) == b"png"
    assert set(call.missing) == {"planner_id"}

    (action,) = state.action_records()
    (prediction,) = state.prediction_records()
    assert action.tool_call_id == prediction.tool_call_id == call.tool_call_id
    assert prediction.planner_id is None
    assert prediction.instruction == "Move."
    assert prediction.action_ids == [action.action_id]
    assert action.source == "vla" and action.action.data == [1]
    assert action.before_observation_id == call.before_observation_id
    assert action.after_observation_id == call.after_observation_id


@pytest.mark.parametrize(
    "failure", ["unknown_tool", "bad_arguments", "paused", "handler"]
)
def test_mcp_capture_finishes_failed_calls(captured_toolkit, failure):
    toolkit = captured_toolkit

    @tool(readonly=True)
    def fail() -> ToolResult:
        raise RuntimeError("handler failed")

    toolkit.add_tool(fail)
    name, arguments = {
        "unknown_tool": ("unknown", {}),
        "bad_arguments": ("view_env_state", {"step": "invalid"}),
        "paused": ("view_env_state", {"step": 0}),
        "handler": ("fail", {}),
    }[failure]
    if failure == "paused":
        toolkit.cancel_active_and_wait()

    async def scenario():
        async with create_connected_server_and_client_session(
            build_mcp_server(toolkit)
        ) as session:
            return await session.call_tool(name, arguments)

    result = asyncio.run(scenario())
    assert result.isError
    (call,) = toolkit.state.tool_call_records()
    assert call.arguments.data == arguments
    assert call.source_ids == {}
    assert call.feedback.data == result.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )
    assert "feedback" not in call.missing
    if failure == "handler":
        assert call.status == "failed" and call.ended_at >= call.started_at
        assert call.raw_result.data["data"]["error"] == "handler failed"
    else:
        assert call.status == "not_executed"
        assert call.started_at is None and call.ended_at is None
        assert call.raw_result is None
        assert call.missing["raw_result"].reason == "not_applicable"
    assert toolkit.state.action_records() == []


def test_mcp_capture_settles_cancelled_request_after_worker_cleanup(captured_toolkit):
    toolkit = captured_toolkit
    entered = threading.Event()
    stopped = threading.Event()
    request_ids = []

    @tool(readonly=True)
    def wait_for_cancel() -> ToolResult:
        request_ids.append(app.request_context.request_id)
        entered.set()
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                toolkit.raise_if_cancelled()
                time.sleep(0.01)
            raise RuntimeError("cancellation did not reach the worker")
        finally:
            stopped.set()

    toolkit.add_tool(wait_for_cancel)
    app = build_mcp_server(toolkit)

    async def scenario():
        async with create_connected_server_and_client_session(app) as session:
            request = asyncio.create_task(session.call_tool("wait_for_cancel", {}))
            assert await asyncio.to_thread(entered.wait, 3)
            await session.send_notification(
                types.ClientNotification(
                    types.CancelledNotification(
                        params=types.CancelledNotificationParams(
                            requestId=request_ids[0], reason="Stop the task."
                        )
                    )
                )
            )
            with pytest.raises(McpError, match="Request cancelled"):
                await request

    asyncio.run(scenario())
    assert stopped.is_set()
    (call,) = toolkit.state.tool_call_records()
    assert call.status == "cancelled" and call.ended_at >= call.started_at
    assert call.raw_result.data["data"]["code"] == "tool_cancelled"
    assert call.feedback is None
    assert call.missing["feedback"].reason == "not_recorded"


def test_mcp_calls_work_without_environment_state(tmp_path):
    init_output_dir(tmp_path / "log")
    toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )

    @tool(readonly=True)
    def sense() -> ToolResult:
        return ToolResult(data={"observed": True})

    toolkit.add_tool(sense)

    async def scenario():
        async with create_connected_server_and_client_session(
            build_mcp_server(toolkit)
        ) as session:
            return await session.call_tool("sense", {})

    try:
        result = asyncio.run(scenario())
    finally:
        toolkit.close()
    assert not result.isError
    assert json.loads(result.content[0].text) == {"observed": True}
