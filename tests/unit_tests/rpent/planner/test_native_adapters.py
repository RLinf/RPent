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
from pathlib import Path

import claude_agent_sdk
import numpy as np
import pytest
from mcp import types
from pydantic_ai import BinaryContent, ToolReturn
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from rpent.dashboard.events import NullDashboardEventSink, TranscriptEvent
from rpent.memory import MemoryManager
from rpent.planner.api_loop import (
    ApiAgentLoop,
    _ApiRunObserver,
    _build_tools,
    _make_tool_function,
)
from rpent.planner.claude_code import (
    ClaudeCodePlanner,
    _build_rpent_server,
    _ClaudeSessionDriver,
)
from rpent.planner.claude_code import _Recorder as ClaudeRecorder
from rpent.planner.codex import _Recorder as CodexRecorder
from rpent.planner.utils.http_mcp_server import mcp_result
from rpent.session import EnvState
from rpent.tools import ToolContext, Toolkit, ToolResult, readonly, tool
from rpent.tools.base import MAX_TOOL_TEXT_BYTES

from ._native_helpers import PNG, RecordingSink, call_sdk_tool, read


@tool
@readonly
def finish(status: str, summary: str, *, ctx: ToolContext) -> ToolResult:
    """Accept the requested outcome for this test toolkit."""
    return ToolResult(data={"_finish": True, "status": status, "summary": summary})


def test_read_image_is_available_to_api_and_not_callable_through_mcp(
    make_toolkit,
):
    toolkit = make_toolkit()
    with toolkit.state.record_step(state={}):
        toolkit.state.save("frame.png", np.zeros((3, 4, 3), dtype=np.uint8))
    path = toolkit.state.artifact_path("frame.png")

    async def scenario():
        api_reader = next(t for t in _build_tools(toolkit) if t.name == "read_image")
        result = await api_reader.function_schema.function(name="frame.png")
        assert isinstance(result, ToolReturn)
        assert json.loads(result.return_value) == {
            "artifact": "frame.png",
            "step": 0,
        }
        assert result.content == [
            BinaryContent(data=path.read_bytes(), media_type="image/png")
        ]
        assert toolkit.calls == [("read_image", {"name": "frame.png"})]

        server = _build_rpent_server(toolkit=toolkit)
        specs = await server["instance"].request_handlers[types.ListToolsRequest](
            types.ListToolsRequest(method="tools/list")
        )
        assert "read_image" not in {tool.name for tool in specs.root.tools}
        for name in ("read_image", "mcp__rpent__read_image"):
            rejected = await call_sdk_tool(server, name, {"name": "frame.png"})
            assert rejected.isError
            assert json.loads(rejected.content[0].text) == {
                "error": "Unknown tool: read_image"
            }
        assert toolkit.calls == [("read_image", {"name": "frame.png"})]

    try:
        asyncio.run(scenario())
    finally:
        toolkit.close()


def test_result_budget_leaves_data_images_and_paths_intact():
    native = ToolResult(
        data={
            "large": "测" * 60000,
            "path": "/tmp/output.json",
            "_finish": True,
            "status": "failure",
            "summary": "verified",
        },
        error="business failure",
        images=[PNG, PNG],
    )
    text = native.to_text()
    assert len(text.encode("utf-8")) <= MAX_TOOL_TEXT_BYTES
    assert text.endswith("[truncated]")
    assert native.data["status"] == "failure"
    assert native.data["summary"] == "verified"
    assert len(native.data["large"]) == 60000
    converted = mcp_result(native)
    assert converted["isError"]
    assert [
        base64.b64decode(block["data"]) for block in converted["content"][1:]
    ] == native.images
    native.data.pop("large")
    payload = json.loads(native.to_text())
    assert payload == {
        "path": "/tmp/output.json",
        "error": "business failure",
        "_finish": True,
        "status": "failure",
        "summary": "verified",
    }


@pytest.mark.parametrize("backend", ["api", "claude"])
def test_file_and_finish_responses_keep_original_fields(backend, tmp_path):
    toolkit = Toolkit(
        state=EnvState(tmp_path),
        memory=MemoryManager(tmp_path / "memory"),
        robot=None,
        output_dir=tmp_path,
        tools=(finish,),
    )

    async def scenario():
        server = _build_rpent_server(toolkit=toolkit)

        async def call(name, **arguments):
            if backend == "api":
                return json.loads(await _make_tool_function(toolkit, name)(**arguments))
            result = await call_sdk_tool(server, name, arguments)
            assert not result.isError
            return json.loads(result.content[0].text)

        path = tmp_path / "note.txt"
        assert await call("write_text_file", path=str(path), content="hello") == {
            "path": str(path),
            "bytes_written": 5,
        }
        assert await call("read_text_file", path=str(path)) == {
            "path": str(path),
            "size": 5,
            "content": "hello",
        }
        directory = await call("list_dir", path=str(path.parent))
        assert set(directory) == {"path", "count", "files"}
        assert "note.txt" in directory["files"]
        assert await call("finish", status="success", summary="done") == {
            "_finish": True,
            "status": "success",
            "summary": "done",
        }
        assert toolkit.finish_result == {"status": "success", "summary": "done"}

    try:
        asyncio.run(scenario())
    finally:
        toolkit.close()


def test_cancellation_keeps_original_error_fields_without_mutating_native_result():
    native = ToolResult(
        data={"code": "tool_cancelled", "interrupted": True},
        error="Tool execution cancelled.",
    )
    before = dict(native.data)
    assert json.loads(native.to_text()) == {
        "error": "Tool execution cancelled.",
        "code": "tool_cancelled",
        "interrupted": True,
    }
    assert native.data == before
    assert native.data["code"] == "tool_cancelled"


@pytest.mark.parametrize(
    "recorder_type", [ClaudeRecorder, CodexRecorder, _ApiRunObserver]
)
@pytest.mark.parametrize(
    "summary", ["environment verification failed", "任务未完成" * 12000]
)
def test_recorders_read_verified_finish_without_a_matching_provider_event(
    make_toolkit,
    recorder_type,
    summary,
):
    toolkit = make_toolkit(accepted={"status": "failure", "summary": summary})
    recorder = recorder_type(
        toolkit=toolkit,
        max_turns=3,
        dashboard_events=NullDashboardEventSink(),
        **({"messages": []} if recorder_type is _ApiRunObserver else {}),
    )
    assert recorder.finish_result is None
    try:
        result = toolkit.execute_tool(
            "finish", {"status": "success", "summary": "model claims success"}
        )
        assert not result.is_error
        text = result.to_text()
        assert len(text.encode("utf-8")) <= MAX_TOOL_TEXT_BYTES
        if len(summary) > 10000:
            assert text.endswith("[truncated]")
        assert recorder.finish_result == {"status": "failure", "summary": summary}
        assert recorder.finish_result == toolkit.finish_result
    finally:
        toolkit.close()


@pytest.mark.parametrize("backend", ["api", "claude"])
def test_shared_calls_reach_scheduler_through_both_adapters(tmp_path: Path, backend):
    toolkit = Toolkit(
        state=EnvState(tmp_path),
        memory=MemoryManager(tmp_path / "memory"),
        robot=threading.Barrier(2),
        output_dir=tmp_path,
        tools=(finish, read),
    )

    async def scenario():
        if backend == "api":
            results = await asyncio.gather(
                *[_make_tool_function(toolkit, "read")(number=str(n)) for n in [1, 2]]
            )
            assert [json.loads(result)["number"] for result in results] == [
                1,
                2,
            ]
        else:
            server = _build_rpent_server(toolkit=toolkit)
            results = await asyncio.gather(
                *[call_sdk_tool(server, "read", {"number": str(n)}) for n in [1, 2]]
            )
            assert all(not result.isError for result in results)
            assert [
                json.loads(result.content[0].text)["number"] for result in results
            ] == [1, 2]

    try:
        asyncio.run(scenario())
    finally:
        toolkit.close()


def test_api_sdk_dispatch_is_parallel_and_accepts_verified_finish(tmp_path):
    toolkit = Toolkit(
        state=EnvState(tmp_path),
        memory=MemoryManager(tmp_path / "memory"),
        robot=threading.Barrier(2),
        output_dir=tmp_path,
        tools=(finish, read),
    )

    def model(messages, info):
        returned = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returned:
            return ModelResponse(
                parts=[
                    ToolCallPart("read", {"number": "1"}, "r1"),
                    ToolCallPart("read", {"number": 2}, "r2"),
                ]
            )
        assert all("error" not in json.loads(part.content) for part in returned)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "finish", {"status": "failure", "summary": "complete"}, "f"
                )
            ]
        )

    result = ApiAgentLoop(
        FunctionModel(model), dashboard_events=NullDashboardEventSink()
    ).solve(system_prompt="", user_message="read", toolkit=toolkit, max_turns=3)
    assert result.error is None
    assert result.finish_result == {"status": "failure", "summary": "complete"}
    assert result.stats["tool_calls"] == 3


def test_api_failure_preserves_images_and_emits_error_without_replaying(
    make_toolkit,
):
    toolkit = make_toolkit({"error": "cannot execute"}, images=[PNG])
    sink = RecordingSink()

    def model(messages, info):
        returned = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if returned:
            assert json.loads(returned[0].content) == {"error": "cannot execute"}
            return ModelResponse(parts=[TextPart("failed")])
        return ModelResponse(parts=[ToolCallPart("inspect_scene", {}, "x")])

    result = ApiAgentLoop(FunctionModel(model), dashboard_events=sink).solve(
        system_prompt="", user_message="inspect", toolkit=toolkit, max_turns=3
    )
    assert result.error is None
    assert len(toolkit.calls) == 1
    tool_events = [
        e
        for e in sink.events
        if isinstance(e, TranscriptEvent) and e.payload.get("type") == "tool_result"
    ]
    assert tool_events[0].payload["result"]["is_error"] is True


def test_claude_error_cleanup_retains_actual_finish(
    make_toolkit, tmp_path, monkeypatch
):
    toolkit = make_toolkit(
        accepted={"status": "failure", "summary": "verified failure"}
    )

    async def query(*, prompt, options):
        await call_sdk_tool(
            options.mcp_servers["rpent"],
            "finish",
            {"status": "success", "summary": "requested"},
        )
        raise RuntimeError("transport ended before tool event")
        yield

    monkeypatch.setattr(claude_agent_sdk, "query", query)
    result = ClaudeCodePlanner(
        output_dir=str(tmp_path),
        output_path=tmp_path / "claude.out",
        dashboard_events=NullDashboardEventSink(),
    ).solve(system_prompt="", user_message="finish", toolkit=toolkit, max_turns=3)
    assert result.finish_result == {"status": "failure", "summary": "verified failure"}
    assert "transport ended" in result.error


def test_claude_interrupt_waits_for_result_message_before_resuming(
    make_toolkit,
):
    async def scenario():
        toolkit = make_toolkit()
        recorder = ClaudeRecorder(
            toolkit=toolkit, max_turns=3, dashboard_events=NullDashboardEventSink()
        )
        driver = _ClaudeSessionDriver(
            sdk=None, options=None, recorder=recorder, emit=recorder.observe
        )
        messages = asyncio.Queue()
        acknowledgement = asyncio.Event()

        class Client:
            async def query(self, text):
                pass

            async def interrupt(self):
                acknowledgement.set()

            async def receive_messages(self):
                while (message := await messages.get()) is not None:
                    yield message

        class Adapter:
            async def on_message(self, driver, message):
                raise AssertionError("old SDK completion must not flush new input")

        driver._client = Client()
        await driver.query("old turn")
        consumer = asyncio.create_task(driver._consume(Adapter()))
        interrupted = asyncio.create_task(driver.interrupt())
        await acknowledgement.wait()
        assert not interrupted.done()
        await messages.put({"type": "ResultMessage", "is_error": True})
        assert await asyncio.wait_for(interrupted, timeout=2) == 1
        assert recorder.error is None
        await messages.put(None)
        await consumer

    asyncio.run(scenario())


def test_api_finish_accepted_during_interrupt_seals_dashboard():
    from types import SimpleNamespace

    from rpent.planner.api_loop import _ApiDashboardSession

    async def scenario():
        ended = asyncio.Event()
        control = SimpleNamespace(end=ended.set)
        session = _ApiDashboardSession(
            agent=None,
            control=control,
            observer=SimpleNamespace(
                finish_result={"status": "failure", "summary": "verified"}
            ),
            max_turns=3,
            no_images=False,
        )
        session._active_prompt = True
        session._run_task = asyncio.create_task(asyncio.Event().wait())
        assert await session.interrupt() == 1
        assert ended.is_set()
        assert session._run_task is None

    asyncio.run(scenario())
