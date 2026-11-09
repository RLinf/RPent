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
import queue
import threading
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import claude_agent_sdk
import pytest
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from mcp.shared.memory import create_connected_server_and_client_session

from rpent.dashboard.events import TranscriptEvent, UsageEvent
from rpent.memory import MemoryManager
from rpent.planner.claude_code import (
    ClaudeCodePlanner,
    _build_rpent_server,
    _ClaudeSessionDriver,
    _Recorder,
)
from rpent.session import EnvState, Payload
from rpent.tools import Toolkit, ToolResult, tool
from rpent.tools.common import CommonTools
from rpent.utils.logging import init_output_dir


@pytest.fixture(autouse=True)
def _output_dir(tmp_path: Path) -> None:
    init_output_dir(tmp_path / "log")


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

    @property
    def state(self) -> EnvState:
        raise RuntimeError("toolkit has no environment state")

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
        yield {
            "type": "UserMessage",
            "parent_tool_use_id": "finish-1",
            "content": [
                {"type": "image", "source": {"data": embedded}},
            ],
        }

    monkeypatch.setattr(claude_agent_sdk, "query", query)
    sink = RecordingSink()
    result = make_planner(tmp_path, sink).solve(
        system_prompt="system rules",
        user_message="user task",
        toolkit=FakeToolkit(),
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

    monkeypatch.setattr(claude_agent_sdk, "query", failed_query)
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

    monkeypatch.setattr(claude_agent_sdk, "query", slow_query)
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


def test_terminal_timeout_cancels_active_toolkit_work(
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

    monkeypatch.setattr(claude_agent_sdk, "query", slow_query)
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


def test_stateful_sdk_driver_closes_adapter_tasks_and_client() -> None:
    events: list[str] = []

    class Client:
        async def __aenter__(self):
            events.append("client-enter")
            return self

        async def __aexit__(self, *args: Any) -> None:
            events.append("client-exit")

        async def query(self, text: str) -> None:
            events.append(f"query:{text}")

        async def receive_messages(self):
            events.append("consumer-started")
            await asyncio.Event().wait()
            yield  # pragma: no cover

    client = Client()

    class Sdk:
        @staticmethod
        def ClaudeSDKClient(*, options: Any) -> Client:
            assert options == {"fake": "options"}
            return client

    class Adapter:
        async def initial_query_succeeded(self, driver: Any) -> None:
            del driver
            events.append("initial-succeeded")

        async def run(self, driver: Any) -> None:
            del driver
            events.append("command-pump-complete")

        async def on_message(self, driver: Any, message: Any) -> None:
            del driver, message

        async def close(self) -> None:
            events.append("adapter-close")

    driver = _ClaudeSessionDriver(
        sdk=Sdk(),
        options={"fake": "options"},
        recorder=_Recorder(max_turns=1, dashboard_events=RecordingSink()),
        emit=lambda message: events.append(f"message:{message}"),
    )

    asyncio.run(driver.run("initial prompt", Adapter()))

    assert events[:3] == [
        "client-enter",
        "query:initial prompt",
        "initial-succeeded",
    ]
    assert "consumer-started" in events
    assert "command-pump-complete" in events
    assert events[-2:] == ["adapter-close", "client-exit"]


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


@pytest.fixture
def runtime_toolkit(tmp_path):
    toolkit = Toolkit(
        dashboard_events=RecordingSink(),
        state=EnvState(tmp_path / "state"),
        memory=MemoryManager(tmp_path / "memory"),
    )
    try:
        yield toolkit
    finally:
        toolkit.close()


@pytest.fixture
def capture_cli(monkeypatch):
    monkeypatch.setattr(claude_agent_sdk, "__version__", "0.2.154")
    monkeypatch.setattr(SubprocessCLITransport, "_find_cli", lambda self: "claude")
    version = Mock(return_value="2.1.274 (Claude Code)")
    monkeypatch.setattr("rpent.session.claude_capture.subprocess.check_output", version)
    return version


@pytest.mark.parametrize("capture_enabled", [False, True])
def test_claude_bridge_readonly_calls_execute_concurrently(
    tmp_path, monkeypatch, runtime_toolkit, capture_cli, capture_enabled
):
    state = runtime_toolkit.state
    if capture_enabled:
        state.start_task(instruction="Sense.", config=Payload("test", {}))
    first_started = threading.Event()
    release_first = threading.Event()
    images = iter((b"first", b"second"))

    @tool(readonly=True)
    def sense(label: str) -> ToolResult:
        image = next(images)
        if image == b"first":
            first_started.set()
            assert release_first.wait(5)
        return ToolResult(data={"label": label}, images=[image])

    runtime_toolkit.add_tool(sense)
    args = {"label": "front"}
    replies = {}

    async def query(*, prompt, options, transport=None):
        assert (transport is not None) == capture_enabled
        assert ("OTEL_LOG_RAW_API_BODIES" in options.env) == capture_enabled
        app = options.mcp_servers["rpent"]["instance"]
        async with create_connected_server_and_client_session(app) as session:
            first = asyncio.create_task(
                session.call_tool("sense", args, meta={"claudecode/toolUseId": "first"})
            )
            try:
                assert await asyncio.to_thread(first_started.wait, 5)
                replies["second"] = await session.call_tool(
                    "sense", args, meta={"claudecode/toolUseId": "second"}
                )
            finally:
                release_first.set()
            replies["first"] = await first
        yield {"type": "ResultMessage"}

    monkeypatch.setattr(claude_agent_sdk, "query", query)
    result = make_planner(tmp_path, RecordingSink(), timeout_s=10).solve(
        system_prompt="", user_message="Sense.", toolkit=runtime_toolkit, max_turns=1
    )
    assert result.error is None
    assert list(replies) == ["second", "first"]
    for call_id, reply in replies.items():
        assert not reply.isError
        assert json.loads(reply.content[0].text) == args
        assert base64.b64decode(reply.content[1].data) == call_id.encode()

    calls = state.tool_call_records()
    if capture_enabled:
        assert result.stats["claude_trace"]["planners"] == 0
        assert (
            "No Claude request recordings" in result.stats["claude_trace"]["errors"][0]
        )
        assert len(calls) == 2
        assert len({call.source_ids["claude.run_id"] for call in calls}) == 1
        assert {
            call.source_ids["claude.tool_use_id"] for call in calls
        } == replies.keys()
        for call in calls:
            call_id = call.source_ids["claude.tool_use_id"]
            assert call.name == "sense" and call.arguments.data == args
            assert call.status == "completed" and call.planner_id is None
            assert call.missing["planner_id"].reason == "not_recorded"
            assert call.raw_result.data == {"data": args, "images": [call_id.encode()]}
            assert call.feedback.data == {
                "content": [
                    block.model_dump(by_alias=True, exclude_none=True)
                    for block in replies[call_id].content
                ]
            }
            assert call.feedback.base64_paths == {"/content/1/data": ""}
    else:
        assert calls == []
        assert "claude_trace" not in result.stats
        capture_cli.assert_not_called()


@pytest.mark.parametrize(
    ("sdk_version", "cli_version", "required"),
    [
        ("0.2.153", "2.1.274", "claude-agent-sdk >= 0.2.154, found 0.2.153"),
        ("0.2.154", "2.1.273", "Claude Code >= 2.1.274, found 2.1.273"),
    ],
)
def test_capture_checks_versions_before_query(
    tmp_path,
    monkeypatch,
    runtime_toolkit,
    capture_cli,
    sdk_version,
    cli_version,
    required,
):
    monkeypatch.setattr(claude_agent_sdk, "__version__", sdk_version)
    capture_cli.return_value = f"{cli_version} (Claude Code)"
    monkeypatch.setattr(
        claude_agent_sdk, "query", lambda **kw: pytest.fail("unexpected query")
    )
    state = runtime_toolkit.state
    state.start_task(instruction="Inspect.", config=Payload("test", {}))

    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="", user_message="Inspect.", toolkit=runtime_toolkit, max_turns=1
    )

    assert required in result.error
    assert not (state.trajectory_dir / "claude").exists()


@pytest.mark.parametrize("cancelled", [False, True])
def test_capture_imports_last_write_after_finish_or_cancel(
    tmp_path, monkeypatch, runtime_toolkit, capture_cli, cancelled
):
    state = runtime_toolkit.state
    state.start_task(instruction="Finish.", config=Payload("test", {}))

    async def query(*, prompt, options, transport):
        location = options.env["OTEL_LOG_RAW_API_BODIES"]
        run_dir = Path(location.removeprefix("file:"))
        assert location.startswith("file:") and run_dir.is_absolute()
        assert run_dir.parent == state.trajectory_dir / "claude"
        closed = asyncio.Event()

        class Writer:
            async def wait(self):
                await asyncio.wait_for(closed.wait(), timeout=2)
                (run_dir / "last.request.json").write_text('{"messages": []}')

        transport._process = Writer()
        try:
            if cancelled:
                raise asyncio.CancelledError
            yield {
                "type": "AssistantMessage",
                "content": [
                    {
                        "type": "ToolUseBlock",
                        "id": "end",
                        "name": "finish",
                        "input": {"status": "success"},
                    }
                ],
            }
            yield {
                "type": "UserMessage",
                "parent_tool_use_id": "end",
                "content": "done",
            }
            pytest.fail("finish requested another SDK message")
        finally:
            closed.set()

    monkeypatch.setattr(claude_agent_sdk, "query", query)
    planner = make_planner(tmp_path, RecordingSink())
    args = {
        "system_prompt": "",
        "user_message": "Finish.",
        "toolkit": runtime_toolkit,
        "max_turns": 1,
    }
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            planner.solve(**args)
    else:
        result = planner.solve(**args)
        assert result.error is None and result.finish_result["status"] == "success"
        assert result.stats["claude_trace"] == {
            "sdk_version": "0.2.154",
            "cli_version": "2.1.274",
            "planners": 1,
            "errors": [],
        }
    (record,) = state.planner_records()
    assert record.request.data == {"messages": []}
    assert record.response is None
    assert state.get_task().status == "running"


def test_capture_import_error_preserves_sdk_error(
    tmp_path, monkeypatch, runtime_toolkit, capture_cli
):
    runtime_toolkit.state.start_task(instruction="Inspect.", config=Payload("test", {}))

    async def query(**kwargs):
        raise RuntimeError("model unavailable")
        yield  # pragma: no cover

    def fail_import(state, run_dir):
        raise OSError("unreadable trace")

    monkeypatch.setattr(claude_agent_sdk, "query", query)
    monkeypatch.setattr("rpent.session.claude_capture.import_claude_trace", fail_import)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="", user_message="Inspect.", toolkit=runtime_toolkit, max_turns=1
    )
    assert result.error == "RuntimeError: model unavailable"
    assert result.stats["claude_trace"]["errors"] == ["OSError: unreadable trace"]


def test_claude_tool_errors_preserve_execution_boundary(runtime_toolkit):
    state = runtime_toolkit.state
    state.start_task(instruction="Sense.", config=Payload("test", {}))

    @tool(readonly=True)
    def sense(label: str) -> ToolResult:
        return ToolResult(data={"error": label})

    runtime_toolkit.add_tool(sense)
    server = _build_rpent_server(
        claude_agent_sdk, toolkit=runtime_toolkit, run_id="run-1"
    )

    async def scenario():
        async with create_connected_server_and_client_session(
            server["instance"]
        ) as session:
            rejected = await session.call_tool("sense", {"label": []})
            assert rejected.isError and state.tool_call_records() == []
            failed = await session.call_tool("sense", {"label": "offline"})
            runtime_toolkit.cancel_active_and_wait()
            paused = await session.call_tool("sense", {"label": "paused"})
            return failed, paused

    replies = asyncio.run(scenario())
    failed, paused = state.tool_call_records()
    assert failed.status == "failed" and failed.raw_result.data["data"] == {
        "error": "offline"
    }
    assert (
        paused.status == "not_executed"
        and paused.started_at is None
        and paused.raw_result is None
    )
    assert paused.missing["raw_result"].reason == "not_applicable"
    for call, reply in zip((failed, paused), replies, strict=True):
        assert reply.isError
        assert call.feedback.data == {
            "is_error": True,
            "content": [{"type": "text", "text": reply.content[0].text}],
        }
        assert call.source_ids == {"claude.run_id": "run-1"}
        assert call.missing["planner_id"].reason == "not_exposed"
    assert state.action_records() == []


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

        class Adapter:
            async def initial_query_succeeded(self, driver):
                pass

            async def run(self, driver):
                assert await driver.interrupt() == 1
                interrupted.set()
                await driver.query("follow-up")

            async def on_message(self, driver, message):
                raise AssertionError("Interrupted results must not flush new input")

            async def close(self):
                pass

        driver = _ClaudeSessionDriver(
            sdk=Sdk(),
            options={},
            recorder=_Recorder(max_turns=3, dashboard_events=RecordingSink()),
            emit=lambda message: None,
        )
        running = asyncio.create_task(driver.run("initial", Adapter()))
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
