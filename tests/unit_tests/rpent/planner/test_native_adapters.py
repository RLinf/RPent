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
import json
import threading

import pytest
from mcp import types
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.api_loop import (
    ApiAgentLoop,
    _ApiRunObserver,
    _build_tools,
)
from rpent.planner.claude_code import (
    _build_rpent_server,
    _ClaudeSessionDriver,
)
from rpent.planner.claude_code import _Recorder as ClaudeRecorder
from rpent.planner.codex import _Recorder as CodexRecorder
from rpent.session import EnvState
from rpent.tools import ToolContext, Toolkit, ToolResult, tool
from rpent.tools.base import MAX_TOOL_TEXT_BYTES

from ._native_helpers import call_sdk_tool, read


@tool
def finish(status: str, summary: str, *, ctx: ToolContext) -> ToolResult:
    """Accept the requested outcome for this test toolkit."""
    return ToolResult(data={"_finish": True, "status": status, "summary": summary})


def test_read_image_is_available_to_api_and_not_callable_through_mcp(make_toolkit):
    toolkit = make_toolkit()
    assert "read_image" in {definition.name for definition in _build_tools(toolkit)}

    async def scenario():
        server = _build_rpent_server(toolkit=toolkit)
        specs = await server["instance"].request_handlers[types.ListToolsRequest](
            types.ListToolsRequest(method="tools/list")
        )
        assert "read_image" not in {definition.name for definition in specs.root.tools}
        rejected = await call_sdk_tool(server, "read_image", {"name": "frame.png"})
        assert rejected.isError
        assert json.loads(rejected.content[0].text) == {
            "error": "Unknown tool: read_image"
        }
        assert toolkit.calls == []

    asyncio.run(scenario())


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


def test_claude_sdk_dispatch_is_parallel(tmp_path):
    toolkit = Toolkit(
        state=EnvState(tmp_path),
        memory=MemoryManager(tmp_path / "memory"),
        robot=threading.Barrier(2),
        output_dir=tmp_path,
        tools=(finish, read),
    )

    async def scenario():
        server = _build_rpent_server(toolkit=toolkit)
        results = await asyncio.gather(
            call_sdk_tool(server, "read", {"number": "1"}),
            call_sdk_tool(server, "read", {"number": "2"}),
        )
        assert all(not result.isError for result in results)
        assert [json.loads(result.content[0].text)["number"] for result in results] == [
            1,
            2,
        ]

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
