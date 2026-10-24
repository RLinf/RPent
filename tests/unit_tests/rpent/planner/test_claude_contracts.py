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
import queue
from pathlib import Path
from typing import Any

import claude_agent_sdk
import pytest

from rpent.dashboard.events import TranscriptEvent, UsageEvent
from rpent.planner.claude_code import (
    ClaudeCodePlanner,
    _build_rpent_server,
    _ClaudeSessionDriver,
    _Recorder,
    _tool_result_to_mcp,
)
from rpent.tools import ToolResult, tool
from rpent.tools.common import CommonTools
from rpent.utils.logging import init_output_dir


@pytest.fixture(autouse=True)
def logging_directory(tmp_path):
    init_output_dir(tmp_path / "logs")


def patch_client_stream(monkeypatch, stream_factory):
    """Exercise the persistent-client surface without starting a real CLI."""

    class Client:
        def __init__(self, *, options):
            self.options = options
            self.stream = None

        async def __aenter__(self):
            self.owner = asyncio.current_task()
            return self

        async def __aexit__(self, *args):
            assert asyncio.current_task() is self.owner
            if self.stream is not None:
                await self.stream.aclose()

        async def query(self, prompt):
            self.stream = stream_factory(prompt=prompt, options=self.options)

        async def receive_messages(self):
            async for message in self.stream:
                yield message

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)

    def forbidden_query(**kwargs):
        pytest.fail("Planner tasks must use ClaudeSDKClient in every mode")

    monkeypatch.setattr(claude_agent_sdk, "query", forbidden_query)


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event: Any) -> None:
        self.events.append(event)


class FakeToolkit:
    def __init__(self, result: dict[str, Any] | None = None, *, images=None) -> None:
        self.result = result or {"value": "ok"}
        self.images = images or []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.cancel_calls = 0
        self._tools = (
            self.inspect_scene,
            CommonTools.finish.with_handler(lambda **kwargs: ToolResult(data=kwargs)),
        )

    @tool(readonly=True)
    def inspect_scene(self, detail: str = "") -> ToolResult:
        """Inspect the current scene."""
        return ToolResult(data=dict(self.result), images=list(self.images))

    def list_tools(self):
        return self._tools

    def execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls.append((name, args))
        return ToolResult(data=dict(self.result), images=list(self.images))

    def cancel_active_and_wait(self) -> None:
        self.cancel_calls += 1

    def resume_calls(self) -> None:
        pass


class FakeSdkTools:
    def __init__(self) -> None:
        self.options: dict[str, Any] | None = None
        self.created_server: dict[str, Any] | None = None

    def ClaudeAgentOptions(self, **kwargs: Any) -> dict[str, Any]:
        self.options = kwargs
        return kwargs

    @staticmethod
    def tool(name: str, description: str, schema: dict[str, Any]):
        def decorate(function: Any) -> Any:
            function.sdk_name = name
            function.sdk_description = description
            function.sdk_schema = schema
            return function

        return decorate

    def create_sdk_mcp_server(
        self,
        *,
        name: str,
        version: str,
        tools: list[Any],
    ) -> dict[str, Any]:
        self.created_server = {"name": name, "version": version, "tools": tools}
        return self.created_server


def patch_sdk_surface(monkeypatch: pytest.MonkeyPatch, fake: FakeSdkTools) -> None:
    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", fake.ClaudeAgentOptions)
    monkeypatch.setattr(claude_agent_sdk, "tool", fake.tool)
    monkeypatch.setattr(
        claude_agent_sdk,
        "create_sdk_mcp_server",
        fake.create_sdk_mcp_server,
    )


def make_planner(tmp_path: Path, sink: RecordingSink, *, timeout_s: float = 1):
    return ClaudeCodePlanner(
        output_dir=str(tmp_path),
        repo_root=tmp_path,
        model="fake-claude",
        allowed_tools="Read,Grep,mcp__external__keep",
        timeout_s=timeout_s,
        max_budget_usd=2.5,
        extra_dirs=[str(tmp_path / "memory")],
        output_path=tmp_path / "claude.out",
        dashboard_events=sink,
    )


def test_options_translate_builtin_and_rpent_tools_without_mutating_specs(
    tmp_path: Path,
) -> None:
    sink = RecordingSink()
    planner = make_planner(tmp_path, sink)
    toolkit = FakeToolkit()
    original_specs = toolkit.list_tools()
    fake_sdk = FakeSdkTools()

    options = planner._build_options(fake_sdk, toolkit=toolkit, max_turns=4)

    assert options is fake_sdk.options
    assert options["cwd"] == str(tmp_path)
    assert options["model"] == "fake-claude"
    assert options["max_turns"] == 4
    assert options["max_budget_usd"] == 2.5
    assert options["tools"] == ["Read", "Grep"]
    assert options["allowed_tools"] == [
        "Read",
        "Grep",
        "mcp__external__keep",
        "mcp__rpent__inspect_scene",
        "mcp__rpent__finish",
    ]
    assert options["add_dirs"] == [str(tmp_path), str(tmp_path / "memory")]
    assert options["setting_sources"] == []
    assert toolkit.list_tools() == original_specs


def test_options_construct_with_the_installed_claude_sdk(tmp_path: Path) -> None:
    planner = make_planner(tmp_path, RecordingSink())

    options = planner._build_options(
        claude_agent_sdk,
        toolkit=FakeToolkit(),
        max_turns=4,
    )

    assert isinstance(options, claude_agent_sdk.ClaudeAgentOptions)
    assert options.cwd == str(tmp_path)
    assert options.model == "fake-claude"
    assert options.max_turns == 4
    assert options.allowed_tools[-2:] == [
        "mcp__rpent__inspect_scene",
        "mcp__rpent__finish",
    ]


def test_in_process_mcp_bridge_maps_schema_dispatch_and_errors() -> None:
    toolkit = FakeToolkit({"error": "rejected"}, images=[b"contract-image"])
    fake_sdk = FakeSdkTools()

    server = _build_rpent_server(fake_sdk, toolkit=toolkit)

    assert server["name"] == "rpent"
    assert server["version"] == "0.1.0"
    tools = {tool.sdk_name: tool for tool in server["tools"]}
    assert tools["inspect_scene"].sdk_description == "Inspect the current scene."
    assert tools["inspect_scene"].sdk_schema == toolkit.list_tools()[0].input_schema

    response = asyncio.run(tools["inspect_scene"]({"detail": "high"}))

    assert toolkit.calls == [("inspect_scene", {"detail": "high"})]
    assert response["is_error"] is True
    assert [block["type"] for block in response["content"]] == ["text", "image"]
    assert response["content"][1]["mimeType"] == "image/png"


def test_tool_result_conversion_preserves_data_and_images() -> None:
    plain = _tool_result_to_mcp(ToolResult(data={"value": "plain"}))
    assert json.loads(plain["content"][0]["text"]) == {"value": "plain"}

    result = ToolResult(
        data={"value": "visible"},
        images=[b"pixels"],
    )
    converted = _tool_result_to_mcp(result)

    assert converted["content"][0] == {
        "type": "text",
        "text": '{\n  "value": "visible"\n}',
    }
    assert converted["content"][1]["type"] == "image"
    assert "is_error" not in converted


def test_successful_fake_sdk_stream_accounts_for_finish_and_hides_image_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sdk = FakeSdkTools()
    patch_sdk_surface(monkeypatch, fake_sdk)
    embedded = "embedded-image-secret"

    async def query(*, prompt: str, options: Any):
        assert prompt == "system rules\n\nuser task"
        assert options is fake_sdk.options
        yield {
            "type": "AssistantMessage",
            "message_id": "assistant-1",
            "parent_tool_use_id": None,
            "usage": {"input_tokens": 5, "output_tokens": 2},
            "content": [
                {"type": "TextBlock", "text": "working"},
                {
                    "type": "ToolUseBlock",
                    "id": "finish-1",
                    "name": "mcp__rpent__finish",
                    "input": {"status": "success", "summary": "done"},
                },
            ],
        }
        handler = next(
            tool
            for tool in options["mcp_servers"]["rpent"]["tools"]
            if tool.sdk_name == "finish"
        )
        await handler({"status": "success", "summary": "done"})
        yield {
            "type": "UserMessage",
            "parent_tool_use_id": "finish-1",
            "content": [
                {"type": "image", "source": {"data": embedded}},
            ],
        }

    patch_client_stream(monkeypatch, query)
    sink = RecordingSink()
    result = make_planner(tmp_path, sink).solve(
        system_prompt="system rules",
        user_message="user task",
        toolkit=FakeToolkit({"_finish": True, "status": "success", "summary": "done"}),
        max_turns=3,
    )

    assert result.finish_result == {
        "_finish": True,
        "status": "success",
        "summary": "done",
    }
    assert result.error is None
    assert result.stats["backend"] == "claude_agent_sdk"
    assert result.stats["turns_used"] == 1
    assert result.stats["tool_calls"] == 1
    assert result.stats["total_input_tokens"] == 5
    assert result.stats["total_output_tokens"] == 2
    transcript = json.dumps(result.messages)
    assert "working" in transcript
    assert "images" in transcript
    assert embedded not in transcript
    assert any(isinstance(event, TranscriptEvent) for event in sink.events)
    assert any(isinstance(event, UsageEvent) for event in sink.events)


def test_rejected_finish_result_is_not_promoted(tmp_path: Path) -> None:
    recorder = _Recorder(max_turns=2, dashboard_events=RecordingSink())
    recorder.observe(
        {
            "type": "AssistantMessage",
            "message_id": "assistant-1",
            "parent_tool_use_id": None,
            "content": [
                {
                    "type": "ToolUseBlock",
                    "id": "finish-1",
                    "name": "mcp__rpent__finish",
                    "input": {"status": "success", "summary": "too early"},
                }
            ],
        }
    )

    rendered = recorder.observe(
        {
            "type": "UserMessage",
            "parent_tool_use_id": None,
            "content": [
                {
                    "type": "ToolResultBlock",
                    "tool_use_id": "finish-1",
                    "content": "finish refused",
                    "is_error": True,
                }
            ],
        }
    )

    assert recorder.finish_result is None
    assert recorder.tool_calls == 1
    assert "is_error" in rendered
    assert str(tmp_path) not in rendered


def test_fake_sdk_failure_and_timeout_are_returned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sdk = FakeSdkTools()
    patch_sdk_surface(monkeypatch, fake_sdk)

    async def failed_query(*, prompt: str, options: Any):
        del prompt, options
        raise RuntimeError("sdk unavailable")
        yield  # pragma: no cover

    patch_client_stream(monkeypatch, failed_query)
    failure = make_planner(tmp_path / "failure", RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=1,
    )

    slow_stream_closed = False

    async def slow_query(*, prompt: str, options: Any):
        nonlocal slow_stream_closed
        del prompt, options
        try:
            await asyncio.sleep(10)
            yield  # pragma: no cover
        finally:
            slow_stream_closed = True

    patch_client_stream(monkeypatch, slow_query)
    timeout = make_planner(
        tmp_path / "timeout",
        RecordingSink(),
        timeout_s=0.01,
    ).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=1,
    )

    assert failure.error == "RuntimeError: sdk unavailable"
    assert timeout.error == "Claude Agent SDK timed out after 0.01s"
    assert slow_stream_closed is True


def test_noninteractive_timeout_cancels_active_toolkit_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sdk = FakeSdkTools()
    patch_sdk_surface(monkeypatch, fake_sdk)
    stream_closed = False

    async def slow_query(*, prompt: str, options: Any):
        nonlocal stream_closed
        del prompt, options
        try:
            await asyncio.sleep(10)
            yield  # pragma: no cover
        finally:
            stream_closed = True

    patch_client_stream(monkeypatch, slow_query)
    toolkit = FakeToolkit()
    result = make_planner(
        tmp_path,
        RecordingSink(),
        timeout_s=0.01,
    ).solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=1,
    )

    assert result.error == "Claude Agent SDK timed out after 0.01s"
    assert stream_closed is True
    assert toolkit.cancel_calls == 1


def test_stateful_sdk_driver_drains_tools_before_closing_tasks_and_client():
    events = []

    class Client:
        async def __aenter__(self):
            self.owner = asyncio.current_task()
            events.append("client-enter")
            return self

        async def __aexit__(self, *args):
            assert asyncio.current_task() is self.owner
            events.append("client-exit")

        async def query(self, text):
            events.append(f"query:{text}")

        async def receive_messages(self):
            events.append("consumer-started")
            try:
                await asyncio.Event().wait()
                yield  # pragma: no cover
            finally:
                events.append("consumer-closed")

    class Sdk:
        @staticmethod
        def ClaudeSDKClient(*, options):
            return Client()

    class Control:
        async def start(self):
            events.append("initial-succeeded")

        async def run(self, driver):
            events.append("command-pump-complete")

        def end(self):
            events.append("control-end")

    class Toolkit(FakeToolkit):
        def cancel_active_and_wait(self):
            events.append("tools-drained")

    driver = _ClaudeSessionDriver(
        sdk=Sdk(),
        options={},
        recorder=_Recorder(max_turns=1, dashboard_events=RecordingSink()),
        emit=lambda message: None,
        toolkit=Toolkit(),
        control=Control(),
    )
    asyncio.run(driver.run("initial prompt"))

    assert events[:3] == ["client-enter", "query:initial prompt", "initial-succeeded"]
    assert "consumer-started" in events
    assert "command-pump-complete" in events
    assert events[-3:] == ["tools-drained", "consumer-closed", "client-exit"]


def test_queue_and_dashboard_are_mutually_exclusive_before_sdk_use(
    tmp_path: Path,
) -> None:
    planner = make_planner(tmp_path, RecordingSink())

    with pytest.raises(ValueError, match="cannot be used together"):
        planner.solve(
            system_prompt="",
            user_message="task",
            toolkit=FakeToolkit(),
            max_turns=1,
            input_queue=queue.Queue(),
            dashboard_interaction=object(),
        )


def test_claude_bridge_readonly_calls_execute_concurrently(tmp_path):
    import threading

    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.tools import Toolkit
    from rpent.utils.logging import init_output_dir

    init_output_dir(tmp_path / "log")
    toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    rendezvous = threading.Barrier(2)

    @tool(readonly=True)
    def sense(label: str) -> ToolResult:
        rendezvous.wait(timeout=5)
        return ToolResult(data={"label": label})

    toolkit.add_tool(sense)
    server = _build_rpent_server(FakeSdkTools(), toolkit=toolkit)
    handler = next(item for item in server["tools"] if item.sdk_name == "sense")

    async def scenario():
        results = await asyncio.gather(
            handler({"label": "left"}), handler({"label": "right"})
        )
        assert all(not result.get("is_error") for result in results)
        assert [
            json.loads(result["content"][0]["text"])["label"] for result in results
        ] == ["left", "right"]

    try:
        asyncio.run(scenario())
    finally:
        toolkit.close()


def test_claude_interrupt_waits_for_old_results_before_next_query():
    async def scenario():
        messages = asyncio.Queue()
        acknowledged = asyncio.Event()
        interrupted = asyncio.Event()
        queries = []

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def query(self, text):
                queries.append(text)

            async def interrupt(self):
                acknowledged.set()

            async def receive_messages(self):
                while True:
                    yield await messages.get()

        client = Client()

        class Sdk:
            @staticmethod
            def ClaudeSDKClient(**kwargs):
                return client

        class Control:
            async def start(self):
                pass

            async def run(self, driver):
                assert await driver.interrupt() == 1
                interrupted.set()
                await driver.query("follow-up")

            async def complete(self, driver):
                raise AssertionError("Interrupted results must not flush new input")

            def end(self):
                pass

        driver = _ClaudeSessionDriver(
            sdk=Sdk(),
            options={},
            recorder=_Recorder(max_turns=3, dashboard_events=RecordingSink()),
            emit=lambda message: None,
            toolkit=FakeToolkit(),
            control=Control(),
        )
        running = asyncio.create_task(driver.run("initial"))
        try:
            await asyncio.wait_for(acknowledged.wait(), timeout=2)
            assert not interrupted.is_set()
            assert queries == ["initial"]
            await messages.put({"type": "ResultMessage"})
            await asyncio.wait_for(running, timeout=2)
            assert interrupted.is_set()
            assert queries == ["initial", "follow-up"]
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "verdict",
    [
        {"_finish": True, "status": "success", "summary": "verified"},
        {"_finish": True, "status": "failure", "summary": "operator aborted"},
        {"status": "pending", "awaiting_operator": True},
        {"error": "finish refused", "_finish": True, "status": "success"},
    ],
)
def test_finish_uses_actual_tool_verdict(tmp_path, monkeypatch, verdict):
    fake_sdk = FakeSdkTools()
    patch_sdk_surface(monkeypatch, fake_sdk)
    toolkit = FakeToolkit(verdict)
    continued = []

    async def stream(*, prompt, options):
        yield {
            "type": "AssistantMessage",
            "content": [
                {
                    "type": "ToolUseBlock",
                    "id": "finish-1",
                    "name": "mcp__rpent__finish",
                    "input": {"status": "success", "summary": "model claim"},
                }
            ],
        }
        handler = next(
            tool
            for tool in options["mcp_servers"]["rpent"]["tools"]
            if tool.sdk_name == "finish"
        )
        response = await handler({"status": "success", "summary": "model claim"})
        yield {
            "type": "UserMessage",
            "content": [
                {
                    "type": "ToolResultBlock",
                    "tool_use_id": "finish-1",
                    **response,
                }
            ],
        }
        continued.append(True)
        yield {"type": "ResultMessage", "subtype": "success"}

    patch_client_stream(monkeypatch, stream)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=3,
    )
    accepted = verdict.get("_finish") is True and "error" not in verdict
    assert result.finish_result == (verdict if accepted else None)
    assert continued == ([] if accepted else [True])
    assert result.error is None
    assert toolkit.cancel_calls == 1


def test_noninteractive_client_stops_at_first_result(tmp_path, monkeypatch):
    fake_sdk = FakeSdkTools()
    patch_sdk_surface(monkeypatch, fake_sdk)

    async def stream(**kwargs):
        yield {"type": "ResultMessage", "subtype": "success"}
        pytest.fail("A non-interactive task must not wait for another user turn")

    patch_client_stream(monkeypatch, stream)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=3,
    )
    assert result.error is None
    assert result.finish_result is None


@pytest.mark.parametrize("stage", ["connect", "query"])
def test_terminal_startup_failure_closes_input_and_drains_tools(
    tmp_path, monkeypatch, stage
):
    fake_sdk = FakeSdkTools()
    patch_sdk_surface(monkeypatch, fake_sdk)
    closed = []
    toolkit = FakeToolkit()

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            if stage == "connect":
                raise RuntimeError("connect failed")
            return self

        async def __aexit__(self, *args):
            assert toolkit.cancel_calls == 1
            closed.append(True)

        async def query(self, prompt):
            raise RuntimeError("query failed")

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=3,
        input_queue=queue.Queue(),
    )
    assert result.error == f"RuntimeError: {stage} failed"
    assert toolkit.cancel_calls == 1
    assert closed == ([True] if stage == "query" else [])
