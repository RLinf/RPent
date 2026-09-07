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
import threading
import time
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
from rpent.tools.tool_spec import ToolSpec
from rpent.tools.toolkit import Toolkit, ToolResult


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event: Any) -> None:
        self.events.append(event)


class FakeToolkit:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result or {"value": "ok"}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.cancel_calls = 0
        self._specs = [
            ToolSpec(
                name="inspect_scene",
                description="Inspect the current scene.",
                input_schema={"type": "object", "properties": {"detail": {"type": "string"}}},
                handler=lambda **kwargs: {},
                readonly=False,
            ),
            ToolSpec(
                name="finish",
                description="Finish the task.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["status", "summary"],
                },
                handler=lambda **kwargs: {},
                readonly=False,
            ),
        ]

    def get_tools_spec(self) -> list[ToolSpec]:
        return self._specs

    def execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls.append((name, args))
        return ToolResult(name, dict(self.result))

    def cancel_active_and_wait(self) -> None:
        self.cancel_calls += 1


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
    original_specs = toolkit.get_tools_spec()
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
    assert toolkit.get_tools_spec() == original_specs


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
    toolkit = FakeToolkit({"error": "rejected", "_image_bytes": b"contract-image"})
    fake_sdk = FakeSdkTools()

    server = _build_rpent_server(fake_sdk, toolkit=toolkit)

    assert server["name"] == "rpent"
    assert server["version"] == "0.1.0"
    tools = {tool.sdk_name: tool for tool in server["tools"]}
    assert tools["inspect_scene"].sdk_description == "Inspect the current scene."
    assert (
        tools["inspect_scene"].sdk_schema == toolkit.get_tools_spec()[0].input_schema
    )

    response = asyncio.run(tools["inspect_scene"]({"detail": "high"}))

    assert toolkit.calls == [("inspect_scene", {"detail": "high"})]
    assert response["is_error"] is True
    assert [block["type"] for block in response["content"]] == ["text", "image"]
    assert response["content"][1]["mimeType"] == "image/png"


def test_tool_result_conversion_supports_plain_values_and_content_blocks() -> None:
    assert _tool_result_to_mcp("plain") == {
        "content": [{"type": "text", "text": "plain"}]
    }

    result = ToolResult(
        "inspect_scene",
        {"value": "visible", "_image_bytes": b"pixels"},
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


class _ConcurrentToolkit(Toolkit):
    """Real ``Toolkit`` (RWLock in ``execute_tool``) with probe tools.

    The other tests use :class:`FakeToolkit`, whose stub ``execute_tool`` never
    serializes. The SDK bridge tests need the real concurrency semantics, so
    they register read-only (parallel) and stateful (exclusive) probe tools.
    """

    _FRAME_ARTIFACTS = {}

    def __init__(self) -> None:
        super().__init__(
            dashboard_events=RecordingSink(),
            state=_FakeState(),
            memory=_FakeMemory(),
        )
        self._tools = {}  # isolate from the common file tools
        self.captured_commands: list[dict[str, Any]] = []

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        del result, elapsed_s
        self.captured_commands.append(command)
        return {"state": "captured"}


class _FakeState:
    def latest_record(self) -> None:
        return None


class _FakeMemory:
    def get_common_tool_bindings(self) -> dict[str, Any]:
        return {}


def test_rpent_server_dispatches_readonly_in_parallel_and_stateful_serialized() -> None:
    gate = threading.Barrier(2)
    toolkit = _ConcurrentToolkit()
    toolkit.add_tool_spec(
        ToolSpec(
            name="rd",
            description="read-only probe",
            input_schema={"type": "object", "properties": {}},
            handler=lambda x: (gate.wait(timeout=2), {"value": x})[1],
            readonly=True,
        )
    )
    toolkit.add_tool_spec(
        ToolSpec(
            name="st",
            description="stateful probe",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: (time.sleep(0.2), {"ok": True})[1],
            readonly=False,
        )
    )
    server = _build_rpent_server(FakeSdkTools(), toolkit=toolkit)
    tools = {tool.sdk_name: tool for tool in server["tools"]}

    async def run_readonly() -> tuple[float, list[dict[str, Any]]]:
        start = time.perf_counter()
        results = await asyncio.gather(
            tools["rd"]({"x": "a"}),
            tools["rd"]({"x": "b"}),
        )
        return time.perf_counter() - start, results

    async def run_stateful() -> float:
        start = time.perf_counter()
        await asyncio.gather(tools["st"]({}), tools["st"]({}))
        return time.perf_counter() - start

    parallel_elapsed, read_results = asyncio.run(run_readonly())
    # ``Barrier(2)`` proves both read-only handlers ran at once; a serialized
    # dispatch (or the old global-lock rejection) could not satisfy it.
    assert parallel_elapsed < 0.5, (
        f"read-only dispatch should overlap, took {parallel_elapsed:.3f}s"
    )
    values = {
        json.loads(res["content"][0]["text"])["value"] for res in read_results
    }
    assert values == {"a", "b"}
    assert all(
        "another tool operation" not in res["content"][0]["text"]
        for res in read_results
    )

    stateful_elapsed = asyncio.run(run_stateful())
    assert stateful_elapsed >= 0.35, (
        f"stateful dispatch must serialize, took {stateful_elapsed:.3f}s"
    )
    assert [c["action"] for c in toolkit.captured_commands] == ["st", "st"]
