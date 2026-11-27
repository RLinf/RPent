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
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from openai_codex import JsonRpcError, TransportClosedError

import rpent.planner.codex as codex_module
import rpent.planner.utils.agent_session as interactive_module
from rpent.dashboard.events import TranscriptEvent, UsageEvent
from rpent.dashboard.state import DashboardState
from rpent.planner.base import is_recoverable_planner_result
from rpent.planner.codex import (
    PROVIDER_ENV_KEY,
    PROVIDER_ID,
    CodexPlanner,
    _codex_mcp_config_overrides,
)
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.session import EnvState, MissingInfo, Payload, codex_capture
from rpent.tools import Toolkit, ToolResult
from rpent.tools.common import CommonTools


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event: Any) -> None:
        self.events.append(event)


class FakeToolkit(Toolkit):
    def __init__(self, state: EnvState | None = None) -> None:
        super().__init__(dashboard_events=RecordingSink(), memory=Mock(), state=state)
        self.cancel_calls = 0

    def _register_common_tools(self):
        self.add_tool(
            CommonTools.finish.with_handler(
                lambda **kwargs: ToolResult(data={"_finish": True, **kwargs})
            )
        )

    def cancel_active_and_wait(self) -> None:
        self.cancel_calls += 1
        super().cancel_active_and_wait()


class FakeMcpServer:
    instances: list[FakeMcpServer] = []

    def __init__(self, toolkit: FakeToolkit, *, on_tool_result=None) -> None:
        self.toolkit = toolkit
        self.on_tool_result = on_tool_result
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

    async def stream(self):
        for event in self.events:
            yield event

    async def interrupt(self) -> None:
        self.interrupt_calls += 1

    def steer(self, text: str) -> None:
        self.steered.append(text)


class FakeThread:
    def __init__(self, turn: FakeTurn) -> None:
        self.fake_turn = turn
        self.turn_prompts: list[tuple[str, dict[str, Any]]] = []

    async def turn(self, prompt: str, **options: Any) -> FakeTurn:
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

    async def thread_start(self, **options: Any) -> FakeThread:
        self.thread_options = options
        return self.thread

    async def close(self) -> None:
        self.closed = True


def make_planner(
    tmp_path: Path,
    sink: RecordingSink,
    *,
    timeout_s: float = 1,
    sandbox: str = "full-access",
):
    return CodexPlanner(
        output_dir=str(tmp_path),
        repo_root=tmp_path,
        timeout_s=timeout_s,
        extra_dirs=[str(tmp_path / "memory")],
        output_path=tmp_path / "codex.out",
        model="fake-codex",
        dashboard_events=sink,
        sandbox=sandbox,
    )


def install_fake_backend(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    def codex_config(**kwargs: Any) -> dict[str, Any]:
        configs.append(kwargs)
        return kwargs

    FakeMcpServer.instances.clear()
    FakeCodex.instances.clear()
    monkeypatch.setattr(FakeCodex, "events", [])
    monkeypatch.setattr(interactive_module, "HttpMcpServer", FakeMcpServer)
    monkeypatch.setattr(codex_module.openai_codex, "CodexConfig", codex_config)
    monkeypatch.setattr(codex_module.openai_codex, "AsyncCodex", FakeCodex)
    return configs


@pytest.mark.parametrize("dashboard", [False, True], ids=["cli", "dashboard"])
@pytest.mark.parametrize("capture", ["no_state", "no_task", "active"])
def test_trace_root_is_scoped_to_each_captured_solve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dashboard: bool,
    capture: str,
) -> None:
    install_fake_backend(monkeypatch)
    FakeCodex.events = [
        {"method": "turn/completed", "payload": {"turn": {"status": "completed"}}}
    ]
    monkeypatch.delenv("CODEX_ROLLOUT_TRACE_ROOT", raising=False)
    state = None if capture == "no_state" else EnvState(tmp_path / "session")
    if capture == "active":
        state.start_task(instruction="Move.", config=Payload("test.config", {}))
    toolkit = FakeToolkit(state)
    planner = make_planner(tmp_path, RecordingSink())
    session_factory = Mock(
        return_value=SimpleNamespace(
            run=AsyncMock(return_value=None),
            close=AsyncMock(return_value=None),
            stop_reason="completed",
            diagnostics=[],
            cleanup_complete=None,
        )
    )
    if dashboard:
        monkeypatch.setattr(codex_module, "_CodexSession", session_factory)
    import_trace = codex_capture.import_codex_trace

    def import_after_cleanup(state: EnvState, root: Path) -> dict[str, Any]:
        assert state.get_task().status == "running"
        assert FakeMcpServer.instances[-1].stopped
        assert toolkit.cancel_calls > 0
        if dashboard:
            assert session_factory.return_value.close.await_count == len(
                FakeMcpServer.instances
            )
        else:
            assert FakeCodex.instances[-1].closed
        return import_trace(state, root)

    importer = Mock(side_effect=import_after_cleanup)
    monkeypatch.setattr(codex_capture, "import_codex_trace", importer)

    for _ in range(2):
        result = planner.solve(
            system_prompt="",
            user_message="Move.",
            toolkit=toolkit,
            max_turns=1,
            dashboard_interaction=Mock() if dashboard else None,
        )
        assert result.error is None
        if capture == "active":
            assert result.stats["codex_trace"]["sdk_version"] == (
                codex_module.openai_codex.__version__
            )
            assert result.stats["codex_trace"]["unlinked_tool_calls"] == 0
            assert "No Codex trace.jsonl" in result.stats["codex_trace"]["errors"][0]
        else:
            assert "codex_trace" not in result.stats

    configs = (
        [call.kwargs["config"] for call in session_factory.call_args_list]
        if dashboard
        else [client.config for client in FakeCodex.instances]
    )
    roots = [config["env"].get("CODEX_ROLLOUT_TRACE_ROOT") for config in configs]
    assert len(roots) == 2
    if capture == "active":
        expected_parent = (state.trajectory_dir / "codex").resolve()
        assert all(Path(root).parent == expected_parent for root in roots)
        assert roots[0] != roots[1]
    else:
        assert roots == [None, None]
    assert importer.call_count == (2 if capture == "active" else 0)
    assert "CODEX_ROLLOUT_TRACE_ROOT" not in os.environ


@pytest.mark.parametrize("dashboard", [False, True], ids=["cli", "dashboard"])
@pytest.mark.parametrize("outcome", ["error", "timeout", "cancelled"])
def test_exit_closes_session_before_trace_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, dashboard: bool
) -> None:
    install_fake_backend(monkeypatch)
    state = EnvState(tmp_path)
    state.start_task(instruction="Move.", config=Payload("test.config", {}))
    toolkit = FakeToolkit(state)
    failure = {
        "error": RuntimeError("backend failed"),
        "timeout": asyncio.TimeoutError(),
        "cancelled": asyncio.CancelledError(),
    }[outcome]
    session = SimpleNamespace(
        run=AsyncMock(side_effect=failure),
        close=AsyncMock(),
        diagnostics=[],
        cleanup_complete=None,
        stop_reason=None,
    )
    monkeypatch.setattr(codex_module, "_CodexSession", Mock(return_value=session))
    import_trace = codex_capture.import_codex_trace
    cleanup = []

    def import_after_cleanup(state: EnvState, root: Path) -> dict[str, Any]:
        cleanup.append(
            (
                session.close.await_count,
                toolkit.cancel_calls,
                FakeMcpServer.instances[0].stopped,
                state.get_task().status,
            )
        )
        return import_trace(state, root)

    importer = Mock(side_effect=import_after_cleanup)
    monkeypatch.setattr(codex_capture, "import_codex_trace", importer)
    planner = make_planner(tmp_path, RecordingSink())
    kwargs = {
        "system_prompt": "",
        "user_message": "Move.",
        "toolkit": toolkit,
        "max_turns": 1,
        "dashboard_interaction": Mock() if dashboard else None,
    }
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            planner.solve(**kwargs)
    else:
        result = planner.solve(**kwargs)
        assert result.error == (
            "RuntimeError: backend failed" if outcome == "error" else "TimeoutError: "
        )
        assert result.stop_reason == "planner_error"
        assert not is_recoverable_planner_result(result)
        assert result.stats["codex_trace"]["bundles"] == 0
    importer.assert_called_once()
    assert cleanup == [(1, 1, True, "running")]


def test_trace_import_failure_preserves_execution_result_and_unlinked_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_backend(monkeypatch)
    FakeCodex.events = [
        {"method": "turn/completed", "payload": {"turn": {"status": "completed"}}}
    ]
    state = EnvState(tmp_path)
    state.start_task(instruction="Move.", config=Payload("test.config", {}))
    tool_id = state.register_tool_call(
        planner_id=None,
        origin="unknown",
        name="move",
        arguments=Payload("mcp.tool_args", {}),
        status="completed",
        raw_result=Payload("test.result", b"image"),
        missing={"planner_id": MissingInfo("pending", "Waiting for trace.")},
    )
    monkeypatch.setattr(
        codex_capture,
        "import_codex_trace",
        Mock(side_effect=ValueError("unsupported trace schema")),
    )

    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="Move.",
        toolkit=FakeToolkit(state),
        max_turns=1,
    )

    assert result.error is None
    assert result.stats["codex_trace"] == {
        "sdk_version": codex_module.openai_codex.__version__,
        "errors": ["ValueError: unsupported trace schema"],
        "unlinked_tool_calls": 1,
    }
    call = state.get_tool_call(tool_id)
    assert call.planner_id is None and call.status == "completed"
    assert call.raw_result.data == b"image"
    assert call.missing["planner_id"].reason == "not_recorded"
    assert state.get_task().status == "running"


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


def test_config_overrides_normalize_provider_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_MODEL_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("CODEX_AUTO_COMPACT_TOKEN_LIMIT", raising=False)
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


def test_restricted_sandboxes_approve_the_planner_owned_mcp_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_MODEL_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("CODEX_AUTO_COMPACT_TOKEN_LIMIT", raising=False)

    assert _codex_mcp_config_overrides(
        mcp_url="http://fake.invalid/mcp/",
        base_url=None,
        sandbox="read-only",
    ) == [
        'mcp_servers.rpent.url="http://fake.invalid/mcp/"',
        'mcp_servers.rpent.default_tools_approval_mode="approve"',
    ]
    assert _codex_mcp_config_overrides(
        mcp_url="http://fake.invalid/mcp/",
        base_url=None,
        sandbox="full-access",
    ) == ['mcp_servers.rpent.url="http://fake.invalid/mcp/"']


def test_config_overrides_include_optional_context_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_MODEL_CONTEXT_WINDOW", "262144")
    monkeypatch.setenv("CODEX_AUTO_COMPACT_TOKEN_LIMIT", "230000")

    overrides = _codex_mcp_config_overrides(mcp_url=None, base_url=None)

    assert overrides == [
        "model_context_window=262144",
        "model_auto_compact_token_limit=230000",
    ]


@pytest.mark.parametrize("trace_enabled", [False, True])
def test_build_config_translates_environment_without_contacting_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trace_enabled: bool,
) -> None:
    monkeypatch.setenv("CODEX_BASE_URL", "https://provider.invalid")
    monkeypatch.setenv("CODEX_API_KEY", "contract-key")
    monkeypatch.setenv("CODEX_BIN", "/fake/codex")
    monkeypatch.setenv("CODEX_ROLLOUT_TRACE_ROOT", "parent-trace")
    monkeypatch.chdir(tmp_path)
    captured: list[dict[str, Any]] = []

    def codex_config(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(codex_module.openai_codex, "CodexConfig", codex_config)
    planner = make_planner(tmp_path, RecordingSink())
    trace_root = Path("trace") if trace_enabled else None

    config = planner._build_config("http://fake.invalid/mcp/", trace_root=trace_root)

    assert config is captured[0]
    assert config["cwd"] == str(tmp_path)
    assert config["experimental_api"] is True
    assert config["codex_bin"] == "/fake/codex"
    assert config["env"][PROVIDER_ENV_KEY] == "contract-key"
    assert config["env"]["CODEX_ROLLOUT_TRACE_ROOT"] == (
        str((tmp_path / "trace").resolve()) if trace_enabled else "parent-trace"
    )
    assert os.environ["CODEX_ROLLOUT_TRACE_ROOT"] == "parent-trace"
    assert f'model_provider="{PROVIDER_ID}"' in config["config_overrides"]


def test_build_config_constructs_with_the_installed_codex_sdk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_BASE_URL", "https://provider.invalid")
    monkeypatch.setenv("CODEX_API_KEY", "contract-key")
    monkeypatch.delenv("CODEX_BIN", raising=False)
    planner = make_planner(tmp_path, RecordingSink())
    trace_root = tmp_path / "trace"

    config = planner._build_config("http://fake.invalid/mcp/", trace_root=trace_root)

    assert isinstance(config, codex_module.openai_codex.CodexConfig)
    assert config.cwd == str(tmp_path)
    assert config.experimental_api is True
    assert config.env[PROVIDER_ENV_KEY] == "contract-key"
    assert config.env["CODEX_ROLLOUT_TRACE_ROOT"] == str(trace_root.resolve())
    assert f'model_provider="{PROVIDER_ID}"' in config.config_overrides
    assert "project_doc_max_bytes=0" in config.config_overrides
    assert not any(
        entry.startswith("skills.config=") for entry in config.config_overrides
    )


def test_build_config_disables_symlinked_project_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_BIN", raising=False)
    skill = tmp_path / "shared" / "review-代码" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: review-pr\ndescription: Review a PR.\n---\n")
    linked_skill = tmp_path / ".agents" / "skills" / "review-pr"
    linked_skill.parent.mkdir(parents=True)
    linked_skill.symlink_to(skill.parent, target_is_directory=True)
    planner = make_planner(tmp_path, RecordingSink())

    config = planner._build_config("http://fake.invalid/mcp/")

    assert config.cwd == str(tmp_path)
    assert "project_doc_max_bytes=0" in config.config_overrides
    assert (
        "skills.config=["
        f"{{ path = {json.dumps(str(skill.resolve()), ensure_ascii=False)}, enabled = false }}]"
    ) in config.config_overrides


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


def test_planner_forwards_configured_service_tier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)
    monkeypatch.setenv("CODEX_SERVICE_TIER", "fast")
    FakeCodex.events = [
        {
            "method": "turn/completed",
            "payload": {"turn": {"status": "completed", "duration_ms": 10}},
        },
    ]

    make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="system rules",
        user_message="user task",
        toolkit=FakeToolkit(),
        max_turns=1,
    )

    fake_codex = FakeCodex.instances[0]
    assert fake_codex.thread_options["service_tier"] == "fast"
    assert fake_codex.thread.turn_prompts[0][1]["service_tier"] == "fast"


def test_efe_sandbox_choices_match_the_codex_sdk() -> None:
    from robots.efe.constants import SANDBOX_CHOICES

    assert set(SANDBOX_CHOICES) == {
        sandbox.value for sandbox in codex_module.openai_codex.Sandbox
    }


def test_planner_applies_the_configured_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)
    FakeCodex.events = [
        {
            "method": "turn/completed",
            "payload": {"turn": {"status": "completed", "duration_ms": 10}},
        },
    ]

    make_planner(tmp_path, RecordingSink(), sandbox="read-only").solve(
        system_prompt="system rules",
        user_message="user task",
        toolkit=FakeToolkit(),
        max_turns=1,
    )

    assert (
        FakeCodex.instances[0].thread_options["sandbox"]
        is codex_module.openai_codex.Sandbox.read_only
    )


def test_planner_rejects_an_unknown_sandbox(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sandbox"):
        make_planner(tmp_path, RecordingSink(), sandbox="none")


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
    assert result.stats["turns_used"] == 0  # Recorded SDK events did not execute tools.
    assert result.stats["tool_calls"] == 1
    assert result.stats["total_input_tokens"] == 11
    assert result.stats["total_output_tokens"] == 6
    transcript = json.dumps(result.messages)
    assert "<image omitted>" in transcript
    assert "embedded-image-secret" not in transcript
    assert (tmp_path / "codex.out.last").read_text() == "working"
    assert any(isinstance(event, TranscriptEvent) for event in sink.events)
    assert any(isinstance(event, UsageEvent) for event in sink.events)


def _model_response_events(index: int, *, text: bool = False, finish: bool = False):
    yield {
        "method": "item/completed",
        "payload": {"item": {"type": "reasoning", "summary": []}},
    }
    for message in ("working", "still working") if text else ("",):
        yield {
            "method": "item/completed",
            "payload": {"item": {"type": "agentMessage", "text": message}},
        }
    for tool in ("read_text_file", "finish" if finish else "list_dir"):
        yield {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": tool,
                    "status": "completed",
                    "arguments": {"status": "stuck", "summary": "done"}
                    if tool == "finish"
                    else {},
                    "result": "accepted",
                }
            },
        }
    usage = {
        "method": "thread/tokenUsage/updated",
        "payload": {
            "token_usage": {
                "total": {"input_tokens": 10 * index, "output_tokens": 2 * index}
            }
        },
    }
    yield usage
    yield usage  # A repeated SDK notification is not another model response.


@pytest.mark.parametrize(
    "status,expected_reason",
    [("interrupted", "planner_error"), ("completed", "completed")],
)
def test_codex_terminal_without_local_interrupt(
    tmp_path, monkeypatch, status, expected_reason
):
    install_fake_backend(monkeypatch)
    terminal = {"status": status}
    if status == "interrupted":
        terminal["error"] = {"message": "interrupted model"}
    FakeCodex.events = [{"method": "turn/completed", "payload": {"turn": terminal}}]
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=2,
    )
    assert result.stop_reason == expected_reason
    assert result.error == ("interrupted model" if status == "interrupted" else None)
    assert "interruption_diagnostics" not in result.stats


@pytest.mark.parametrize("dashboard", [False, True], ids=["cli", "dashboard"])
@pytest.mark.parametrize("finish_on_last", [False, True])
@pytest.mark.parametrize(
    "rpc_error,status,detail,expected_error",
    [
        (None, "completed", None, None),
        (None, "failed", "provider failed", "provider failed"),
        (None, "interrupted", "interrupted model", None),
        (JsonRpcError(-32600, "no active turn to interrupt"), "completed", None, None),
        (
            JsonRpcError(-32600, "no active turn to interrupt"),
            "failed",
            "provider failed",
            "provider failed",
        ),
        (
            JsonRpcError(-32600, "thread not found"),
            "completed",
            None,
            "Codex interrupt failed: JSON-RPC error -32600: thread not found",
        ),
        (
            TransportClosedError("transport closed"),
            "completed",
            None,
            "Codex interrupt failed: transport closed",
        ),
        (
            RuntimeError("no active turn to interrupt"),
            "completed",
            None,
            "Codex interrupt failed: no active turn to interrupt",
        ),
    ],
)
def test_tool_budget_drains_and_preserves_verdict(
    tmp_path,
    monkeypatch,
    active_codex_dashboard,
    dashboard,
    finish_on_last,
    rpc_error,
    status,
    detail,
    expected_error,
):
    from rpent.planner.base import execute_tool
    from rpent.tools import tool

    toolkit = FakeToolkit()
    executed = []

    @tool(readonly=True)
    def inspect_scene() -> ToolResult:
        executed.append("inspect")
        return ToolResult(data={"observed": True})

    toolkit.add_tool(inspect_scene)

    async def stream(self):
        # Model responses alone must not exhaust the tool budget.
        for index in range(1, 4):
            for event in _model_response_events(index, text=True):
                if event["method"] == "thread/tokenUsage/updated" or (
                    event["method"] == "item/completed"
                    and event["payload"]["item"]["type"] == "agentMessage"
                ):
                    yield event
        server = FakeMcpServer.instances[0]
        for name in ("inspect_scene", "finish" if finish_on_last else "inspect_scene"):
            args = {"status": "success", "summary": "done"} if name == "finish" else {}
            result = await execute_tool(toolkit, name, args)
            server.on_tool_result(name, args, result)
        while not self.interrupt_calls:
            await asyncio.sleep(0)
        yield _terminal_event(status, detail)

    async def interrupt(self):
        self.interrupt_calls += 1
        if rpc_error is not None:
            raise rpc_error

    monkeypatch.setattr(FakeTurn, "stream", stream)
    monkeypatch.setattr(FakeTurn, "interrupt", interrupt)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=2,
        dashboard_interaction=active_codex_dashboard if dashboard else None,
    )
    assert result.stats["turns_used"] == 2
    assert executed == ["inspect"] * (1 if finish_on_last else 2)
    assert (result.finish_result is not None) == finish_on_last
    assert result.error == expected_error
    if (
        rpc_error is not None
        and expected_error
        and expected_error.startswith("Codex interrupt failed")
    ):
        assert result.stop_info["cleanup_complete"] is False
    assert result.stats.get("interruption_diagnostics", []) == (
        ["interrupted model"] if status == "interrupted" else []
    )
    assert result.stats["total_input_tokens"] == 30
    assert (tmp_path / "codex.out.last").read_text() == "still working"
    assert _terminal_event(status, detail) in [
        json.loads(line)
        for line in (tmp_path / "codex.out.stream.jsonl").read_text().splitlines()
    ]
    assert result.stop_reason == (
        "planner_error"
        if result.error
        else "finish"
        if finish_on_last
        else "tool_limit"
    )
    assert FakeCodex.instances[0].thread.fake_turn.interrupt_calls == 1
    assert FakeCodex.instances[0].closed
    assert FakeMcpServer.instances[0].stopped


def _terminal_event(status, detail=None):
    turn = {"status": status}
    if detail:
        turn["error"] = {"message": detail}
    return {"method": "turn/completed", "payload": {"turn": turn}}


@pytest.fixture
def active_codex_dashboard(tmp_path, monkeypatch):
    install_fake_backend(monkeypatch)
    dashboard = DashboardState(
        output_dir=tmp_path,
        dashboard_spec={
            "task": {"display": "test", "output_slug": "test"},
            "runtime_components": (),
            "primitives": (),
        },
    )
    dashboard.shared_services_ready()
    dashboard.request_task({})
    assert dashboard.wait_for_task(timeout=0) is not None
    return dashboard


@pytest.mark.parametrize(
    "terminal_first", [True, False], ids=["terminal-first", "rpc-first"]
)
@pytest.mark.parametrize(
    "rpc_error,status,detail,expected_error",
    [
        (JsonRpcError(-32600, "no active turn to interrupt"), "completed", None, None),
        (
            JsonRpcError(-32600, "no active turn to interrupt"),
            "interrupted",
            "remote stop",
            "remote stop",
        ),
        (None, "interrupted", "local stop", None),
        (None, "failed", "provider failure", "provider failure"),
        (
            TransportClosedError("lost connection"),
            "completed",
            None,
            "TransportClosedError: lost connection",
        ),
        (
            RuntimeError("no active turn to interrupt"),
            "completed",
            None,
            "RuntimeError: no active turn to interrupt",
        ),
    ],
)
def test_dashboard_replacement_drains_the_codex_turn(
    tmp_path,
    monkeypatch,
    active_codex_dashboard,
    terminal_first,
    rpc_error,
    status,
    detail,
    expected_error,
):
    dashboard = active_codex_dashboard
    rpc_started, rpc_returned, terminal_arrived = (asyncio.Event() for _ in range(3))
    observe = codex_module._Recorder.observe
    request_interrupt = codex_module._CodexSession._request_interrupt

    def record(recorder, event):
        rendered = observe(recorder, event)
        if event["method"] == "turn/completed":
            terminal_arrived.set()
        return rendered

    async def request(session, turn):
        try:
            await request_interrupt(session, turn)
        finally:
            rpc_returned.set()

    async def stream(self):
        dashboard.request_task({})
        await rpc_started.wait()
        if not terminal_first:
            await rpc_returned.wait()
        yield _terminal_event(status, detail)

    async def interrupt(self):
        self.interrupt_calls += 1
        rpc_started.set()
        if terminal_first:
            await terminal_arrived.wait()
        if rpc_error:
            raise rpc_error

    monkeypatch.setattr(codex_module._Recorder, "observe", record)
    monkeypatch.setattr(codex_module._CodexSession, "_request_interrupt", request)
    monkeypatch.setattr(FakeTurn, "stream", stream)
    monkeypatch.setattr(FakeTurn, "interrupt", interrupt)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=2,
        dashboard_interaction=dashboard,
    )
    assert result.error == expected_error
    assert result.stop_reason == ("planner_error" if expected_error else status)
    assert result.stats.get("interruption_diagnostics", []) == (
        ["local stop"] if detail == "local stop" else []
    )
    assert terminal_arrived.is_set() and rpc_returned.is_set()
    assert FakeCodex.instances[0].thread.fake_turn.interrupt_calls == 1
    assert FakeCodex.instances[0].closed and FakeMcpServer.instances[0].stopped


def test_no_active_rpc_without_a_terminal_fails_task_replacement(
    tmp_path, monkeypatch, active_codex_dashboard
):

    async def stream(self):
        active_codex_dashboard.request_task({})
        await asyncio.Event().wait()
        yield

    interrupt = AsyncMock(
        side_effect=JsonRpcError(-32600, "no active turn to interrupt")
    )
    monkeypatch.setattr(FakeTurn, "interrupt", interrupt)
    monkeypatch.setattr(FakeTurn, "stream", stream)
    monkeypatch.setattr(codex_module, "_INTERRUPT_TIMEOUT_S", 0.03)
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=2,
        dashboard_interaction=active_codex_dashboard,
    )
    assert result.stop_reason == "planner_error"
    assert "Codex did not finish the interrupted turn" in result.error
    assert result.stop_info["cleanup_complete"] is False
    assert not is_recoverable_planner_result(result)
    interrupt.assert_awaited_once()
    assert len(FakeCodex.instances[0].thread.turn_prompts) == 1
    assert FakeCodex.instances[0].closed and FakeMcpServer.instances[0].stopped


async def _start_fake_turn(stream, interrupt, emit_event):
    session = codex_module._CodexSession(
        config=None,
        thread_options={},
        turn_options={},
        recorder=codex_module._Recorder(dashboard_events=RecordingSink()),
        emit_event=emit_event,
        control=None,
    )
    session._thread = FakeThread(SimpleNamespace(stream=stream, interrupt=interrupt))
    await session._submit_text("task")
    return session


def test_cancelling_interrupt_joins_rpc_and_drains_terminal():
    async def scenario():
        rpc_started, terminal_arrived, rpc_finished = (
            asyncio.Event() for _ in range(3)
        )

        async def interrupt():
            rpc_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                rpc_finished.set()

        async def stream():
            await rpc_started.wait()
            yield _terminal_event("completed")

        session = await _start_fake_turn(
            stream, interrupt, lambda event: terminal_arrived.set()
        )
        requesting = asyncio.create_task(session.interrupt())
        await terminal_arrived.wait()
        requesting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await requesting
        assert rpc_finished.is_set()
        await session.close()
        assert session.error is None
        assert session.stop_reason == "completed"

    asyncio.run(asyncio.wait_for(scenario(), timeout=1))


def test_operator_can_retry_interrupt_after_rpc_failure():
    async def scenario():
        accepted = asyncio.Event()
        failure = TransportClosedError("disconnected")

        async def stream():
            await accepted.wait()
            yield _terminal_event("interrupted", "local stop")

        async def interrupt():
            if rpc.await_count == 1:
                raise failure
            accepted.set()

        rpc = AsyncMock(side_effect=interrupt)
        session = await _start_fake_turn(stream, rpc, lambda event: None)
        with pytest.raises(TransportClosedError, match="disconnected"):
            await session.interrupt()
        await session.interrupt()
        await session.close()
        assert rpc.await_count == 2
        assert session.error is None
        assert session.stop_reason == "interrupted"
        assert session._recorder.stats()["interruption_diagnostics"] == ["local stop"]

    asyncio.run(asyncio.wait_for(scenario(), timeout=1))


def test_response_count_ignores_items_and_duplicate_usage():
    from rpent.planner.codex import _Recorder

    recorder = _Recorder(dashboard_events=RecordingSink())
    usage_events = []
    for index, text in [(1, False), (2, True)]:
        events = list(_model_response_events(index, text=text))
        # Reasoning, empty/text messages and multiple tools belong to one response.
        for event in events[:-2]:
            recorder.observe(event)
        assert recorder.turns == index - 1
        for event in events[-2:]:
            recorder.observe(event)
        assert recorder.turns == index
        usage_events.append(events[-1])
    assert recorder.tool_calls == 4
    assert recorder.final_response == "still working"

    for event in reversed(usage_events):
        recorder.observe(event)
    recorder.observe(
        {
            "method": "thread/tokenUsage/updated",
            "payload": {"token_usage": {"total": {}}},
        }
    )
    assert recorder.turns == 2
    assert recorder.stats()["total_input_tokens"] == 20


def test_stats_distinguish_unobserved_usage_from_zero_tokens():
    from rpent.planner.codex import _Recorder

    recorder = _Recorder(dashboard_events=RecordingSink())

    assert recorder.stats()["usage_observed"] is False
    assert recorder.stats()["total_input_tokens"] == 0

    for event in _model_response_events(1):
        recorder.observe(event)

    assert recorder.stats()["usage_observed"] is True
    assert recorder.stats()["total_input_tokens"] == 10


def test_rejected_finish_item_is_not_promoted() -> None:
    from rpent.planner.codex import _Recorder

    recorder = _Recorder(dashboard_events=RecordingSink())

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


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "payload, is_error, accepted",
    [
        ({"_finish": True, "status": "failure"}, False, True),
        ({"_finish": True, "status": "failure"}, True, False),
        ({"status": "pending"}, False, False),
        ({"_finish": False}, False, False),
        ({"error": "verdict pending"}, False, False),
    ],
)
def test_finish_uses_accepted_tool_result_over_requested_success(
    wrapped, payload, is_error, accepted
):
    recorder = codex_module._Recorder(dashboard_events=RecordingSink())
    result = (
        {
            "isError": is_error,
            "content": [{"type": "text", "text": json.dumps(payload)}],
        }
        if wrapped
        else payload
    )
    recorder.observe(
        {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": "mcp__rpent__finish",
                    "status": "failed" if is_error and not wrapped else "completed",
                    "arguments": {"status": "success"},
                    "result": result,
                }
            },
        }
    )
    assert recorder.finish_result == (payload if accepted else None)
    assert recorder.tool_calls == 1


def test_fake_codex_backend_failure_stops_mcp_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)

    class FailingCodex:
        def __init__(self, config: Any) -> None:
            del config

        async def thread_start(self, **kwargs):
            raise RuntimeError("codex unavailable")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(codex_module.openai_codex, "AsyncCodex", FailingCodex)
    state = EnvState(tmp_path)
    state.start_task(instruction="Move.", config=Payload("test.config", {}))
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(state),
        max_turns=1,
    )

    assert result.error == "RuntimeError: codex unavailable"
    assert result.stats["codex_trace"]["bundles"] == 0
    assert FakeMcpServer.instances[0].stopped is True
    assert state.get_task().status == "running"


@pytest.mark.parametrize("sdk_close_fails", [False, True])
@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("initializing", [False, True])
def test_timeout_interrupts_and_drains_turn_before_closing(
    tmp_path, monkeypatch, interactive, initializing, sdk_close_fails
):
    install_fake_backend(monkeypatch)
    stopped = asyncio.Event()
    events = []

    async def stream(self):
        await stopped.wait()
        events.append("turn_completed")
        yield {
            "method": "turn/completed",
            "payload": {"turn": {"status": "interrupted"}},
        }

    async def interrupt(self):
        events.append("interrupt")
        self.interrupt_calls += 1
        stopped.set()

    async def close(self):
        events.append("close")
        self.closed = True
        if sdk_close_fails:
            raise RuntimeError("SDK close failed")

    async def thread_start(self, **options):
        await asyncio.Event().wait()

    if initializing:
        monkeypatch.setattr(FakeCodex, "thread_start", thread_start)
    monkeypatch.setattr(FakeTurn, "stream", stream)
    monkeypatch.setattr(FakeTurn, "interrupt", interrupt)
    monkeypatch.setattr(FakeCodex, "close", close)
    state = EnvState(tmp_path / "session")
    state.start_task(instruction="Move.", config=Payload("test.config", {}))
    toolkit = FakeToolkit(state)
    import_trace = codex_capture.import_codex_trace

    def import_after_cleanup(state, root):
        assert FakeCodex.instances[0].closed
        assert FakeMcpServer.instances[0].stopped
        assert toolkit.cancel_calls == 1
        events.append("import")
        return import_trace(state, root)

    monkeypatch.setattr(codex_capture, "import_codex_trace", import_after_cleanup)
    result = make_planner(tmp_path, RecordingSink(), timeout_s=0.05).solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=1,
        input_queue=queue.Queue() if interactive else None,
    )
    assert result.error == "Codex SDK timed out after 0.05s" + (
        "; Session cleanup failed: SDK close failed" if sdk_close_fails else ""
    )
    assert result.stop_reason == ("planner_error" if sdk_close_fails else "timeout")
    assert (result.stop_info or {}).get("cleanup_complete") is (
        False if sdk_close_fails else None
    )
    assert is_recoverable_planner_result(result) is (not sdk_close_fails)
    assert events == (
        ["close", "import"]
        if initializing
        else ["interrupt", "turn_completed", "close", "import"]
    )
    assert FakeMcpServer.instances[0].stopped is True
    assert toolkit.cancel_calls == 1


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("provider TIMEOUT"), RuntimeError("fetch failed / TRANSPORT")],
)
def test_provider_failures_are_not_planner_deadlines(tmp_path, monkeypatch, failure):
    install_fake_backend(monkeypatch)
    monkeypatch.setattr(FakeCodex, "thread_start", AsyncMock(side_effect=failure))
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="", user_message="task", toolkit=FakeToolkit(), max_turns=1
    )
    assert result.stop_reason == "planner_error"
    assert not is_recoverable_planner_result(result)
    assert result.stop_info.get("cleanup_complete") is None
    assert result.error == f"{type(failure).__name__}: {failure}"
    assert FakeCodex.instances[0].closed
    assert FakeMcpServer.instances[0].stopped


@pytest.mark.parametrize("stage", ["toolkit", "session", "mcp", "cancelled_task"])
def test_deadline_cleanup_failure_preserves_both_causes(tmp_path, monkeypatch, stage):
    install_fake_backend(monkeypatch)

    async def run(prompt):
        try:
            await asyncio.Event().wait()
        finally:
            if stage == "cancelled_task":
                raise RuntimeError("cleanup failed: drain failed")

    session = SimpleNamespace(
        run=run,
        close=AsyncMock(),
        diagnostics=[],
        cleanup_complete=None,
        stop_reason=None,
    )
    monkeypatch.setattr(codex_module, "_CodexSession", Mock(return_value=session))
    toolkit = FakeToolkit()
    failure = RuntimeError("drain failed")
    if stage == "toolkit":
        monkeypatch.setattr(
            toolkit, "cancel_active_and_wait", Mock(side_effect=failure)
        )
    elif stage == "session":
        session.close.side_effect = failure
    elif stage == "mcp":
        monkeypatch.setattr(FakeMcpServer, "stop", Mock(side_effect=failure))
    result = make_planner(tmp_path, RecordingSink(), timeout_s=0.01).solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=1
    )
    assert result.stop_reason == "planner_error"
    assert not is_recoverable_planner_result(result)
    assert "Codex SDK timed out after 0.01s" in result.error
    assert "cleanup failed: drain failed" in result.error
    assert result.stop_info["cleanup_complete"] is False
    session.close.assert_awaited_once()


def test_provider_error_before_deadline_is_not_lost(tmp_path, monkeypatch):
    install_fake_backend(monkeypatch)
    stopped = asyncio.Event()

    async def stream(self):
        yield {
            "method": "error",
            "payload": {"message": "TRANSPORT", "will_retry": False},
        }
        await stopped.wait()
        yield {
            "method": "turn/completed",
            "payload": {
                "turn": {
                    "status": "interrupted",
                    "error": {"message": "interrupted model"},
                }
            },
        }

    async def interrupt(self):
        stopped.set()

    monkeypatch.setattr(FakeTurn, "stream", stream)
    monkeypatch.setattr(FakeTurn, "interrupt", interrupt)
    result = make_planner(tmp_path, RecordingSink(), timeout_s=0.05).solve(
        system_prompt="", user_message="task", toolkit=FakeToolkit(), max_turns=2
    )
    assert result.stop_reason == "planner_error"
    assert not is_recoverable_planner_result(result)
    assert result.stats["interruption_diagnostics"] == ["interrupted model"]
    assert "TRANSPORT" in result.error
    assert "Codex SDK timed out after 0.05s" in result.error


@pytest.mark.parametrize(
    "outcome,last_error",
    [
        ("stream", "RuntimeError: TRANSPORT"),
        ("terminal", "TRANSPORT"),
        ("missing", "Codex did not finish the interrupted turn"),
    ],
)
def test_deadline_preserves_each_provider_stream_and_cleanup_error_once(
    tmp_path, monkeypatch, outcome, last_error
):
    install_fake_backend(monkeypatch)
    interrupt_failed = asyncio.Event()
    payload = {"message": "RuntimeError: TRANSPORT; exhausted"}

    async def stream(self):
        for _ in range(2):
            yield {"method": "error", "payload": payload}
        await interrupt_failed.wait()
        if outcome == "missing":
            await asyncio.Event().wait()
        if outcome == "stream":
            raise RuntimeError("TRANSPORT")
        yield _terminal_event("failed", last_error)

    async def interrupt(self):
        interrupt_failed.set()
        raise TransportClosedError("connection closed")

    monkeypatch.setattr(codex_module, "_INTERRUPT_TIMEOUT_S", 0.05)
    monkeypatch.setattr(FakeTurn, "interrupt", interrupt)
    monkeypatch.setattr(FakeTurn, "stream", stream)
    result = make_planner(tmp_path, RecordingSink(), timeout_s=0.05).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=2,
    )
    provider_error = f"Codex SDK timed out after 0.05s; {json.dumps(payload)}"
    cleanup_error = "Codex interrupt failed: connection closed"
    assert result.error in {
        f"{provider_error}; {cleanup_error}; {last_error}",
        f"{provider_error}; {last_error}; {cleanup_error}",
    }
    assert result.stop_reason == "planner_error"
    assert result.stop_info["cleanup_complete"] is False
    assert not is_recoverable_planner_result(result)
    assert FakeCodex.instances[0].closed and FakeMcpServer.instances[0].stopped


def test_queue_and_dashboard_are_rejected_before_mcp_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnexpectedMcpServer:
        def __init__(self, toolkit: Any) -> None:
            raise AssertionError(f"MCP should not be constructed: {toolkit}")

    monkeypatch.setattr(interactive_module, "HttpMcpServer", UnexpectedMcpServer)
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


def test_session_close_attempts_codex_cleanup_after_interrupt_error():
    events = []

    async def scenario():
        class Turn:
            async def interrupt(self):
                events.append("turn-interrupt")
                raise RuntimeError("already complete")

        class Codex:
            async def close(self):
                events.append("codex-close")

        session = codex_module._CodexSession(
            config=None,
            thread_options={},
            turn_options={},
            recorder=None,
            emit_event=None,
            control=None,
        )
        session._codex = Codex()
        session._turn = Turn()
        session._turn_done = asyncio.Event()
        session._turn_done.set()
        await session.close()

    asyncio.run(scenario())
    assert events == ["turn-interrupt", "codex-close"]


# ---------------------------------------------------------------------------
# Connectivity probe: SDK turn-consumption contract
# ---------------------------------------------------------------------------


def test_turn_handle_is_not_iterable_and_exposes_run_and_stream() -> None:
    """Consume async SDK handles through run or stream, not direct iteration."""
    from openai_codex import AsyncTurnHandle

    assert callable(AsyncTurnHandle.run)
    assert callable(AsyncTurnHandle.stream)
    assert not hasattr(AsyncTurnHandle, "__iter__")
    assert not hasattr(AsyncTurnHandle, "__aiter__")


def test_turn_result_carries_the_fields_the_probe_reads() -> None:
    from openai_codex import TurnResult

    annotations = TurnResult.__annotations__
    for field in ("status", "error", "final_response"):
        assert field in annotations, f"TurnResult lost {field!r}"


class FakeProbeTurn:
    """A turn handle shaped like the real one: run/stream, never iterable."""

    def __init__(
        self,
        result: Any = None,
        *,
        blocks: asyncio.Event | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.result = result
        self.blocks = blocks
        self.raises = raises
        self.interrupt_calls = 0

    async def run(self) -> Any:
        if self.raises is not None:
            raise self.raises
        if self.blocks is not None:
            await self.blocks.wait()
        return self.result

    def stream(self):
        raise AssertionError("the probe must use run(), not stream()")

    async def interrupt(self) -> None:
        self.interrupt_calls += 1


class FakeProbeCodex:
    """Minimal Codex session double for the probe path."""

    instances: list[FakeProbeCodex] = []
    turn_factory: Any = None

    def __init__(self, config: Any) -> None:
        self.config = config
        self.closed = False
        self.thread_options: dict[str, Any] = {}
        self.turn_prompts: list[tuple[str, dict[str, Any]]] = []
        self.fake_turn = type(self).turn_factory()
        FakeProbeCodex.instances.append(self)

    def __enter__(self) -> FakeProbeCodex:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.closed = True

    async def close(self) -> None:
        self.closed = True

    async def thread_start(self, **options: Any) -> FakeProbeCodex:
        self.thread_options = options
        return self

    async def turn(self, prompt: str, **options: Any) -> FakeProbeTurn:
        self.turn_prompts.append((prompt, options))
        return self.fake_turn


@pytest.fixture
def probe_codex(monkeypatch: pytest.MonkeyPatch):
    """Install FakeProbeCodex and hand back its instance list."""
    FakeProbeCodex.instances = []

    def _install(turn_factory: Any) -> list[FakeProbeCodex]:
        FakeProbeCodex.turn_factory = turn_factory
        monkeypatch.setattr(codex_module.openai_codex, "AsyncCodex", FakeProbeCodex)
        return FakeProbeCodex.instances

    return _install


def test_probe_consumes_the_turn_with_run_and_returns_final_response(
    probe_codex: Any,
) -> None:
    result = SimpleNamespace(status="completed", error=None, final_response="ok")
    instances = probe_codex(lambda: FakeProbeTurn(result))

    reply = codex_module.run_probe_turn(
        object(), prompt="ping", model="gpt-5.5", timeout_s=30
    )

    assert reply == "ok"
    codex = instances[0]
    assert codex.turn_prompts[0][0] == "ping"
    assert codex.thread_options["model"] == "gpt-5.5"
    assert codex.closed is True


def test_probe_omits_model_when_none_so_the_sdk_default_applies(
    probe_codex: Any,
) -> None:
    result = SimpleNamespace(status="completed", error=None, final_response="ok")
    instances = probe_codex(lambda: FakeProbeTurn(result))

    codex_module.run_probe_turn(object(), prompt="ping", model=None, timeout_s=30)

    assert "model" not in instances[0].thread_options


def test_probe_runs_read_only_and_denies_approvals(probe_codex: Any) -> None:
    result = SimpleNamespace(status="completed", error=None, final_response="ok")
    instances = probe_codex(lambda: FakeProbeTurn(result))

    codex_module.run_probe_turn(object(), prompt="ping", model=None, timeout_s=30)

    options = instances[0].thread_options
    assert options["approval_mode"] is codex_module.openai_codex.ApprovalMode.deny_all
    assert options["sandbox"] is codex_module.openai_codex.Sandbox.read_only


def test_probe_forwards_configured_service_tier(
    probe_codex: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_SERVICE_TIER", "fast")
    result = SimpleNamespace(status="completed", error=None, final_response="ok")
    instances = probe_codex(lambda: FakeProbeTurn(result))

    codex_module.run_probe_turn(object(), prompt="ping", model=None, timeout_s=30)

    assert instances[0].thread_options["service_tier"] == "fast"
    assert instances[0].turn_prompts[0][1]["service_tier"] == "fast"


def test_probe_config_attaches_no_mcp_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEX_BASE_URL", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_MODEL_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("CODEX_AUTO_COMPACT_TOKEN_LIMIT", raising=False)

    overrides = _codex_mcp_config_overrides(mcp_url=None, base_url=None)

    assert overrides == []
    assert not any("mcp_servers" in entry for entry in overrides)


def test_probe_config_keeps_the_provider_override_without_mcp() -> None:
    overrides = _codex_mcp_config_overrides(mcp_url=None, base_url="https://gw.example")

    assert not any("mcp_servers" in entry for entry in overrides)
    assert any("model_providers" in entry for entry in overrides)


def test_probe_timeout_interrupts_the_turn_and_closes_the_session(
    probe_codex: Any,
) -> None:
    release = asyncio.Event()
    turns: list[FakeProbeTurn] = []

    def factory() -> FakeProbeTurn:
        turn = FakeProbeTurn(None, blocks=release)
        turns.append(turn)
        return turn

    instances = probe_codex(factory)
    try:
        with pytest.raises(TimeoutError) as excinfo:
            codex_module.run_probe_turn(
                object(), prompt="ping", model=None, timeout_s=1
            )
        assert "1s" in str(excinfo.value)
        assert turns[0].interrupt_calls == 1
        assert instances[0].closed is True
    finally:
        release.set()


def test_probe_raises_when_the_turn_status_is_failed(probe_codex: Any) -> None:
    result = SimpleNamespace(
        status="failed",
        error=SimpleNamespace(message="model provider rejected the request"),
        final_response=None,
    )
    probe_codex(lambda: FakeProbeTurn(result))

    with pytest.raises(RuntimeError, match="model provider rejected"):
        codex_module.run_probe_turn(object(), prompt="ping", model=None, timeout_s=30)


def test_probe_propagates_sdk_exceptions_for_classification(probe_codex: Any) -> None:
    probe_codex(lambda: FakeProbeTurn(raises=RuntimeError("401 Unauthorized")))

    with pytest.raises(RuntimeError, match="401 Unauthorized"):
        codex_module.run_probe_turn(object(), prompt="ping", model=None, timeout_s=30)


def test_probe_returns_empty_string_when_the_model_said_nothing(
    probe_codex: Any,
) -> None:
    result = SimpleNamespace(status="completed", error=None, final_response=None)
    probe_codex(lambda: FakeProbeTurn(result))

    reply = codex_module.run_probe_turn(
        object(), prompt="ping", model=None, timeout_s=30
    )

    assert reply == ""


def test_probe_forwards_image_and_schema(probe_codex: Any) -> None:
    """Visual grounding must preserve the image and structured-output request."""
    result = SimpleNamespace(
        status="completed", error=None, final_response='{"point_xy":null}'
    )
    instances = probe_codex(lambda: FakeProbeTurn(result))
    schema = {"type": "object", "properties": {"point_xy": {"type": "null"}}}
    reply = codex_module.run_probe_turn(
        object(),
        prompt="select a point",
        model="test-model",
        timeout_s=30,
        image_bytes=b"test-image",
        output_schema=schema,
    )
    payload, options = instances[0].turn_prompts[0]
    assert isinstance(payload[0], codex_module.openai_codex.TextInput)
    assert isinstance(payload[1], codex_module.openai_codex.ImageInput)
    assert options["output_schema"] == schema
    assert reply == '{"point_xy":null}'


def test_retry_error_is_visible_without_poisoning_successful_turn():
    from rpent.planner.codex import _Recorder

    sink = RecordingSink()
    recorder = _Recorder(dashboard_events=sink)
    recorder.observe(
        {
            "method": "error",
            "payload": {
                "will_retry": True,
                "error": {
                    "message": "Reconnecting 4/5",
                    "additional_details": "Broken pipe",
                },
            },
        }
    )
    assert recorder.error is None
    assert "Broken pipe" in str(sink.events[-1])
    recorder.observe(
        {
            "method": "error",
            "payload": {
                "will_retry": False,
                "error": {"message": "Retries exhausted"},
            },
        }
    )
    assert "Retries exhausted" in recorder.error
    assert "Model connection failed" in str(sink.events[-1])
