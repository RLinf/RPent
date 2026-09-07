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
import os
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

import rpent.planner.codex as codex_module
from rpent.dashboard.events import TranscriptEvent, UsageEvent
from rpent.planner.codex import (
    PROVIDER_ENV_KEY,
    PROVIDER_ID,
    CodexPlanner,
    _codex_mcp_config_overrides,
    _interrupt,
)
from rpent.planner.utils.http_mcp_server import (
    HttpMcpServer,
    _toolkit_to_mcp_content,
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
    def __init__(self) -> None:
        self.cancel_calls = 0

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "finish",
                "description": "Finish the task.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                },
            }
        ]

    def execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(name, {"name": name, "args": args})

    def cancel_active_and_wait(self) -> None:
        self.cancel_calls += 1


class FakeMcpServer:
    instances: list[FakeMcpServer] = []

    def __init__(self, toolkit: FakeToolkit) -> None:
        self.toolkit = toolkit
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def start(self) -> str:
        self.started = True
        return "http://fake.invalid/mcp/"

    def stop(self) -> None:
        self.stopped = True


class FakeTurn:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.interrupt_calls = 0
        self.steered: list[str] = []

    def stream(self):
        yield from self.events

    def interrupt(self) -> None:
        self.interrupt_calls += 1

    def steer(self, text: str) -> None:
        self.steered.append(text)


class FakeThread:
    def __init__(self, turn: FakeTurn) -> None:
        self.fake_turn = turn
        self.turn_prompts: list[tuple[str, dict[str, Any]]] = []

    def turn(self, prompt: str, **options: Any) -> FakeTurn:
        self.turn_prompts.append((prompt, options))
        return self.fake_turn


class FakeCodex:
    instances: list[FakeCodex] = []
    events: list[dict[str, Any]] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.closed = False
        self.thread = FakeThread(FakeTurn(list(self.events)))
        self.instances.append(self)

    def __enter__(self) -> FakeCodex:
        return self

    def __exit__(self, *args: Any) -> None:
        self.closed = True

    def thread_start(self, **options: Any) -> FakeThread:
        self.thread_options = options
        return self.thread

    def close(self) -> None:
        self.closed = True


def make_planner(tmp_path: Path, sink: RecordingSink, *, timeout_s: float = 1):
    return CodexPlanner(
        output_dir=str(tmp_path),
        repo_root=tmp_path,
        timeout_s=timeout_s,
        extra_dirs=[str(tmp_path / "memory")],
        output_path=tmp_path / "codex.out",
        model="fake-codex",
        dashboard_events=sink,
    )


def install_fake_backend(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    def codex_config(**kwargs: Any) -> dict[str, Any]:
        configs.append(kwargs)
        return kwargs

    FakeMcpServer.instances.clear()
    FakeCodex.instances.clear()
    monkeypatch.setattr(FakeCodex, "events", [])
    monkeypatch.setattr(codex_module, "HttpMcpServer", FakeMcpServer)
    monkeypatch.setattr(codex_module.openai_codex, "CodexConfig", codex_config)
    monkeypatch.setattr(codex_module.openai_codex, "Codex", FakeCodex)
    return configs


def test_mcp_content_conversion_preserves_text_images_and_error_status() -> None:
    plain, plain_error = _toolkit_to_mcp_content("plain")
    assert plain_error is False
    assert plain[0].type == "text"
    assert plain[0].text == "plain"

    result = ToolResult(
        "finish",
        {"error": "finish refused", "_image_bytes": b"image bytes"},
    )
    content, is_error = _toolkit_to_mcp_content(result)

    assert [block.type for block in content] == ["text", "image"]
    assert json.loads(content[0].text) == {"error": "finish refused"}
    assert content[1].mimeType == "image/png"
    assert is_error is True


def test_http_mcp_readiness_ignores_environment_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    server = HttpMcpServer(FakeToolkit())

    try:
        url = server.start(ready_timeout_s=3.0)
    finally:
        server.stop()

    assert url.startswith("http://127.0.0.1:")


def test_config_overrides_normalize_provider_url() -> None:
    assert _codex_mcp_config_overrides(
        mcp_url="http://fake.invalid/mcp/",
        base_url=None,
    ) == ['mcp_servers.rpent.url="http://fake.invalid/mcp/"']

    overrides = _codex_mcp_config_overrides(
        mcp_url="http://fake.invalid/mcp/",
        base_url="https://provider.invalid/root/",
    )

    assert overrides == [
        'mcp_servers.rpent.url="http://fake.invalid/mcp/"',
        f'model_provider="{PROVIDER_ID}"',
        f'model_providers.{PROVIDER_ID}.name="{PROVIDER_ID}"',
        f'model_providers.{PROVIDER_ID}.base_url="https://provider.invalid/root/v1"',
        f'model_providers.{PROVIDER_ID}.wire_api="responses"',
        f'model_providers.{PROVIDER_ID}.env_key="{PROVIDER_ENV_KEY}"',
    ]


def test_build_config_translates_environment_without_contacting_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_BASE_URL", "https://provider.invalid")
    monkeypatch.setenv("CODEX_API_KEY", "contract-key")
    monkeypatch.setenv("CODEX_BIN", "/fake/codex")
    captured: list[dict[str, Any]] = []

    def codex_config(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(codex_module.openai_codex, "CodexConfig", codex_config)
    planner = make_planner(tmp_path, RecordingSink())

    config = planner._build_config("http://fake.invalid/mcp/")

    assert config is captured[0]
    assert config["cwd"] == str(tmp_path)
    assert config["experimental_api"] is True
    assert config["codex_bin"] == "/fake/codex"
    assert config["env"][PROVIDER_ENV_KEY] == "contract-key"
    assert f'model_provider="{PROVIDER_ID}"' in config["config_overrides"]


def test_build_config_constructs_with_the_installed_codex_sdk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_BASE_URL", "https://provider.invalid")
    monkeypatch.setenv("CODEX_API_KEY", "contract-key")
    monkeypatch.delenv("CODEX_BIN", raising=False)
    planner = make_planner(tmp_path, RecordingSink())

    config = planner._build_config("http://fake.invalid/mcp/")

    assert isinstance(config, codex_module.openai_codex.CodexConfig)
    assert config.cwd == str(tmp_path)
    assert config.experimental_api is True
    assert config.env[PROVIDER_ENV_KEY] == "contract-key"
    assert f'model_provider="{PROVIDER_ID}"' in config.config_overrides


def test_build_config_scopes_loopback_no_proxy_to_codex_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("NO_PROXY", "metadata.internal")
    monkeypatch.setenv("no_proxy", ".corp.invalid")
    captured: list[dict[str, Any]] = []

    def codex_config(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(codex_module.openai_codex, "CodexConfig", codex_config)
    planner = make_planner(tmp_path, RecordingSink())

    config = planner._build_config("http://127.0.0.1:8765/mcp/")

    child_env = config["env"]
    assert child_env["HTTP_PROXY"] == "http://proxy.invalid:8080"
    assert child_env["HTTPS_PROXY"] == "http://proxy.invalid:8080"
    assert child_env["NO_PROXY"].split(",") == [
        "metadata.internal",
        ".corp.invalid",
        "127.0.0.1",
        "localhost",
    ]
    assert child_env["no_proxy"] == child_env["NO_PROXY"]
    assert os.environ["NO_PROXY"] == "metadata.internal"
    assert os.environ["no_proxy"] == ".corp.invalid"


def test_successful_fake_codex_lifecycle_uses_fake_mcp_and_accounts_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = install_fake_backend(monkeypatch)
    embedded = "data:image/png;base64,embedded-image-secret"
    FakeCodex.events = [
        {
            "method": "item/completed",
            "payload": {"item": {"type": "userMessage", "content": embedded}},
        },
        {
            "method": "item/completed",
            "payload": {"item": {"type": "agentMessage", "text": "working"}},
        },
        {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": "mcp__rpent__finish",
                    "status": "completed",
                    "arguments": {"status": "success", "summary": "done"},
                    "result": "accepted",
                }
            },
        },
        {
            "method": "thread/tokenUsage/updated",
            "payload": {
                "token_usage": {
                    "total": {
                        "input_tokens": 11,
                        "cached_input_tokens": 4,
                        "output_tokens": 6,
                        "reasoning_output_tokens": 2,
                    }
                }
            },
        },
        {
            "method": "turn/completed",
            "payload": {"turn": {"status": "completed", "duration_ms": 1250}},
        },
    ]
    sink = RecordingSink()
    result = make_planner(tmp_path, sink).solve(
        system_prompt="system rules",
        user_message="user task",
        toolkit=FakeToolkit(),
        max_turns=3,
    )

    assert configs
    assert FakeMcpServer.instances[0].started is True
    assert FakeMcpServer.instances[0].stopped is True
    fake_codex = FakeCodex.instances[0]
    assert fake_codex.closed is True
    assert fake_codex.thread.turn_prompts[0][0] == "system rules\n\nuser task"
    assert result.finish_result == {
        "_finish": True,
        "status": "success",
        "summary": "done",
    }
    assert result.error is None
    assert result.stats["backend"] == "codex_sdk"
    assert result.stats["turns_used"] == 1
    assert result.stats["tool_calls"] == 1
    assert result.stats["total_input_tokens"] == 11
    assert result.stats["total_output_tokens"] == 6
    transcript = json.dumps(result.messages)
    assert "<image omitted>" in transcript
    assert "embedded-image-secret" not in transcript
    assert (tmp_path / "codex.out.last").read_text() == "working"
    assert any(isinstance(event, TranscriptEvent) for event in sink.events)
    assert any(isinstance(event, UsageEvent) for event in sink.events)


def test_rejected_finish_item_is_not_promoted() -> None:
    from rpent.planner.codex import _Recorder

    recorder = _Recorder(max_turns=2, dashboard_events=RecordingSink())

    rendered = recorder.observe(
        {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": "mcp__rpent__finish",
                    "status": "failed",
                    "arguments": {"status": "success", "summary": "too early"},
                    "error": "finish refused",
                }
            },
        }
    )

    assert recorder.finish_result is None
    assert recorder.tool_calls == 1
    assert "finish" in rendered


def test_fake_codex_backend_failure_stops_mcp_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)

    class FailingCodex:
        def __init__(self, config: Any) -> None:
            del config

        def __enter__(self):
            raise RuntimeError("codex unavailable")

        def __exit__(self, *args: Any) -> None:
            return None

    monkeypatch.setattr(codex_module.openai_codex, "Codex", FailingCodex)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=1,
    )

    assert result.error == "RuntimeError: codex unavailable"
    assert FakeMcpServer.instances[0].stopped is True


def test_timeout_interrupts_without_starting_a_worker_or_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)

    class TimeoutThread:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.join_calls: list[float | None] = []

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            self.join_calls.append(timeout)

        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr(
        codex_module,
        "threading",
        SimpleNamespace(Thread=TimeoutThread),
    )
    result = make_planner(
        tmp_path,
        RecordingSink(),
        timeout_s=0.01,
    ).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=1,
    )

    assert result.error == "Codex SDK timed out after 0.01s"
    assert FakeMcpServer.instances[0].stopped is True


def test_terminal_timeout_cancels_active_toolkit_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)

    class TimeoutThread:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr(
        codex_module,
        "threading",
        SimpleNamespace(Thread=TimeoutThread),
    )
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

    assert result.error == "Codex SDK timed out after 0.01s"
    assert FakeMcpServer.instances[0].stopped is True
    assert toolkit.cancel_calls == 1


def test_queue_and_dashboard_are_rejected_before_mcp_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnexpectedMcpServer:
        def __init__(self, toolkit: Any) -> None:
            raise AssertionError(f"MCP should not be constructed: {toolkit}")

    monkeypatch.setattr(codex_module, "HttpMcpServer", UnexpectedMcpServer)
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


def test_interrupt_attempts_both_turn_and_codex_cleanup() -> None:
    events: list[str] = []

    class Turn:
        def interrupt(self) -> None:
            events.append("turn-interrupt")
            raise RuntimeError("already complete")

    class Codex:
        def close(self) -> None:
            events.append("codex-close")

    _interrupt({"turn": Turn(), "codex": Codex()})

    assert events == ["turn-interrupt", "codex-close"]


class _ConcurrentToolkit(Toolkit):
    """Real ``Toolkit`` (RWLock in ``execute_tool``) with probe tools.

    The codex contract tests otherwise use :class:`FakeToolkit`, whose stub
    ``execute_tool`` never serializes. These tests need the real concurrency
    semantics, so they register read-only (parallel) and stateful (exclusive)
    probe tools on the base toolkit.
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


def test_http_mcp_dispatch_runs_readonly_in_parallel_and_stateful_serialized(
    tmp_path: Path,
) -> None:
    from rpent.utils.logging import init_output_dir

    init_output_dir(str(tmp_path))
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
    server = HttpMcpServer(toolkit)

    async def call_pair(
        name: str, arguments: list[dict[str, Any]]
    ) -> tuple[float, list[Any]]:
        async with streamable_http_client(server.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                start = time.perf_counter()
                results = await asyncio.gather(
                    *(session.call_tool(name, args) for args in arguments)
                )
                elapsed = time.perf_counter() - start
                return elapsed, results

    try:
        server.start()
        parallel_elapsed, read_results = asyncio.run(
            call_pair("rd", [{"x": "a"}, {"x": "b"}])
        )
        # ``Barrier(2)`` proves both read-only handlers ran at once; a
        # serialized dispatch (or the old global-lock rejection) could not
        # satisfy it without hitting the 2s barrier timeout.
        assert parallel_elapsed < 0.5, (
            f"read-only dispatch should overlap, took {parallel_elapsed:.3f}s"
        )
        for result in read_results:
            body = json.dumps(result.model_dump())
            assert "another tool operation" not in body
            text = next(
                block.text for block in result.content if block.type == "text"
            )
            assert json.loads(text)["value"] in ("a", "b")

        stateful_elapsed, _ = asyncio.run(call_pair("st", [{}, {}]))
        assert stateful_elapsed >= 0.35, (
            f"stateful dispatch must serialize, took {stateful_elapsed:.3f}s"
        )
    finally:
        server.stop()

    assert [c["action"] for c in toolkit.captured_commands] == ["st", "st"]
