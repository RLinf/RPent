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
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import acp
import pytest

import rpent.planner.utils.agent_session as session_module
from rpent.dashboard.events import NullDashboardEventSink, TranscriptEvent, UsageEvent
from rpent.memory import MemoryManager
from rpent.planner.gemini_cli import GeminiCliPlanner
from rpent.planner.utils.acp import AcpRecorder
from rpent.tools import Toolkit, ToolResult
from rpent.utils.logging import init_output_dir


@pytest.fixture(autouse=True)
def logging_directory(tmp_path):
    init_output_dir(tmp_path)


def make_planner(tmp_path: Path, *, timeout_s=2) -> GeminiCliPlanner:
    return GeminiCliPlanner(
        output_dir=str(tmp_path),
        output_path=tmp_path / "gemini.out",
        model="gemini-upstream-id",
        timeout_s=timeout_s,
        dashboard_events=NullDashboardEventSink(),
    )


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        (None, "none"),
        ("gemini-upstream-id", "low"),
        ("gemini-upstream-id", "medium"),
        ("gemini-upstream-id", "high"),
    ],
)
def test_launch_uses_exact_model_and_trusted_workspace_environment(
    tmp_path, monkeypatch, acp_backend, model, effort
) -> None:
    monkeypatch.setenv("NO_PROXY", "gateway.example")
    monkeypatch.delenv("no_proxy", raising=False)
    planner = GeminiCliPlanner(
        output_dir=str(tmp_path),
        model=model,
        reasoning_effort=effort,
        dashboard_events=NullDashboardEventSink(),
    )
    assert planner.probe("ping") == "done"
    settings = acp_backend["settings"]
    if effort == "none":
        assert "modelConfigs" not in settings
    else:
        override = settings["modelConfigs"]["overrides"][0]
        assert override["match"] == {"isChatModel": True}
        assert override["modelConfig"]["generateContentConfig"]["thinkingConfig"] == {
            "thinkingLevel": effort.upper(),
            "thinkingBudget": None,
        }

    assert acp_backend["command"] == [
        "/fake/gemini",
        "--acp",
        "--allowed-mcp-server-names",
        "rpent",
        "--extensions",
        "none",
    ] + (["--model", model] if model else [])
    env = acp_backend["env"]
    assert env["GEMINI_CLI_TRUST_WORKSPACE"] == "true"
    assert env["GEMINI_CLI_NO_RELAUNCH"] == "1"
    assert env["NO_PROXY"] == "gateway.example,127.0.0.1,localhost"
    assert env["no_proxy"] == env["NO_PROXY"]
    assert "mcpServers" not in settings
    acp_backend["server_factory"].assert_not_called()
    assert acp_backend["events"][-1] == "close"
    assert not acp_backend["cwd"].exists()


def test_gemini_rejects_unsupported_effort_before_launch(tmp_path):
    planner = GeminiCliPlanner(
        output_dir=tmp_path,
        reasoning_effort="xhigh",
        dashboard_events=NullDashboardEventSink(),
    )
    with pytest.raises(ValueError, match="supports --reasoning-effort"):
        planner.probe("ping")


@pytest.mark.parametrize("tool_error", [False, True])
def test_recorder_uses_actual_finish_and_accumulates_usage(tool_error):
    events = []
    recorder = AcpRecorder(SimpleNamespace(enabled=True, emit=events.append))
    recorder.observe(
        {
            "type": "rpent_tool",
            "name": "finish",
            "args": {"status": "success", "summary": "done"},
            "result": {"_finish": True, "status": "success", "summary": "done"},
            "is_error": tool_error,
        }
    )
    recorder.observe(
        {
            "type": "acp_update",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "complete"},
            },
        }
    )
    for _ in range(2):
        recorder.observe(
            {
                "type": "acp_result",
                "stopReason": "end_turn",
                "usage": {"inputTokens": 12, "outputTokens": 4, "cachedReadTokens": 3},
            }
        )
    assert recorder.finish_result == (
        None
        if tool_error
        else {"_finish": True, "status": "success", "summary": "done"}
    )
    assert recorder.final_response == "complete"
    assert recorder.stats() == {
        "tool_calls": 1,
        "total_input_tokens": 24,
        "total_output_tokens": 8,
        "total_cached_input_tokens": 6,
    }
    assert any(isinstance(event, TranscriptEvent) for event in events)
    assert any(isinstance(event, UsageEvent) for event in events)


def test_secret_redactor_snapshots_configuration_once_per_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "dotenv-secret-value"
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir()
    dotenv = gemini_dir / ".env"
    dotenv.write_text(
        f"export GEMINI_API_KEY='{secret}'\nGOOGLE_API_KEY=dotenv-google-secret\n"
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "environment-google-secret")
    read_text = Path.read_text
    reads = []

    def read(path):
        reads.append(path)
        return read_text(path)

    monkeypatch.setattr(Path, "read_text", read)
    redact = GeminiCliPlanner._load_secret_redactor()
    dotenv.write_text("GEMINI_API_KEY=replacement-secret\n")
    for _ in range(3):
        assert redact(f"request failed for {secret}") == (
            "request failed for <GEMINI_API_KEY>"
        )
        assert redact("environment-google-secret") == "<GOOGLE_API_KEY>"
        assert redact("dotenv-google-secret") == "dotenv-google-secret"
    assert reads == [dotenv]
    assert GeminiCliPlanner._load_secret_redactor()("replacement-secret") == (
        "<GEMINI_API_KEY>"
    )
    assert reads == [dotenv, dotenv]


@pytest.fixture
def acp_backend(monkeypatch):
    seen = {"events": [], "reply": "done", "stderr": ""}
    server_factory = Mock()
    server_factory.return_value.start.return_value = "http://127.0.0.1:43210/mcp/"
    seen["server_factory"] = server_factory
    monkeypatch.setenv("GEMINI_BIN", "/fake/gemini")
    monkeypatch.setattr(session_module, "HttpMcpServer", server_factory)

    @asynccontextmanager
    async def spawn(client, binary, *args, cwd, **kwargs):
        seen["command"] = [binary, *args]
        seen["env"] = kwargs["env"]
        seen["settings"] = json.loads((cwd / ".gemini" / "settings.json").read_text())
        seen["cwd"] = cwd
        stderr = asyncio.StreamReader()
        stderr.feed_data(seen["stderr"].encode())
        stderr.feed_eof()
        stopped = asyncio.Event()

        class Connection:
            async def initialize(self, **kwargs):
                seen["events"].append("initialize")
                if "startup_error" in seen:
                    raise seen["startup_error"]
                return acp.schema.InitializeResponse(
                    protocol_version=acp.PROTOCOL_VERSION,
                    agent_capabilities=acp.schema.AgentCapabilities(
                        mcp_capabilities=acp.schema.McpCapabilities(http=True)
                    ),
                )

            async def authenticate(self, **kwargs):
                pytest.fail("RPent must not change CLI authentication")

            async def new_session(self, **kwargs):
                session_cwd = Path(kwargs["cwd"])
                assert session_cwd.is_absolute()
                assert session_cwd == cwd
                assert (session_cwd / ".gemini" / "settings.json").is_file()
                assert kwargs["mcp_servers"] == (
                    [
                        acp.schema.HttpMcpServer(
                            type="http",
                            name="rpent",
                            url=server_factory.return_value.start.return_value,
                            headers=[],
                        )
                    ]
                    if server_factory.called
                    else []
                )
                seen["events"].append("new_session")
                return acp.schema.NewSessionResponse(session_id="same-session")

            async def prompt(self, session_id, prompt):
                assert session_id == "same-session"
                seen["prompt"] = prompt[0].text
                seen["events"].append("prompt")
                if "prompt_error" in seen:
                    raise seen["prompt_error"]
                if seen.get("block"):
                    await stopped.wait()
                elif server_factory.called and seen.get("finish", True):
                    server_factory.call_args.kwargs["on_tool_result"](
                        "finish",
                        {"status": "success", "summary": "done"},
                        ToolResult(
                            data={
                                "_finish": True,
                                "status": "success",
                                "summary": "done",
                            }
                        ),
                    )
                    if seen.get("block_after_finish"):
                        await stopped.wait()
                await client.session_update(
                    session_id,
                    SimpleNamespace(
                        model_dump=lambda **kwargs: {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": seen["reply"]},
                        }
                    ),
                )
                seen["events"].append("prompt_completed")
                return SimpleNamespace(
                    stop_reason="end_turn",
                    model_dump=lambda **kwargs: {
                        "stopReason": "end_turn",
                        "usage": {"inputTokens": 8, "outputTokens": 2},
                    },
                )

            async def cancel(self, session_id):
                seen["events"].append("cancel")
                stopped.set()

        try:
            yield Connection(), SimpleNamespace(stderr=stderr)
        finally:
            seen["events"].append("close")

    monkeypatch.setattr(acp, "spawn_agent_process", spawn)
    return seen


@pytest.mark.parametrize(
    "outcome",
    [
        "finish_then_block",
        "no_finish",
        "startup_error",
        "prompt_error",
        "timeout",
    ],
)
def test_session_result_artifacts_and_cleanup(
    tmp_path, monkeypatch, acp_backend, outcome
):
    secret = "environment-secret-value"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    acp_backend["reply"] = f"request failed for {secret}"
    acp_backend["stderr"] = f"provider rejected {secret}\n"
    acp_backend["finish"] = outcome == "finish_then_block"
    acp_backend["block_after_finish"] = outcome == "finish_then_block"
    acp_backend["block"] = outcome == "timeout"
    if outcome in {"startup_error", "prompt_error"}:
        acp_backend[outcome] = RuntimeError(f"gemini unavailable: {secret}")
    toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    toolkit.cancel_active_and_wait = Mock(
        side_effect=lambda: acp_backend["events"].append("cancel_tools")
    )
    result = make_planner(
        Path("logs/run"), timeout_s=0.05 if outcome == "timeout" else 2
    ).solve(
        system_prompt="system rules",
        user_message="user task",
        toolkit=toolkit,
        max_turns=6,
    )
    server = acp_backend["server_factory"].return_value
    server.start.assert_called_once()
    server.stop.assert_called_once()
    toolkit.cancel_active_and_wait.assert_called_once()
    settings = acp_backend["settings"]
    assert settings["model"]["maxSessionTurns"] == -1
    assert settings["tools"]["core"] == [
        f"mcp_rpent_{tool.name}" for tool in toolkit.list_tools()
    ]
    assert settings["skills"]["enabled"] is False
    assert settings["hooksConfig"]["enabled"] is False
    assert settings["context"]["includeDirectoryTree"] is False
    assert "mcpServers" not in settings
    assert "GEMINI_API_KEY" not in json.dumps(settings)
    assert acp_backend["events"][-1] == "close"
    assert not acp_backend["cwd"].exists()
    assert result.finish_result == (
        {"_finish": True, "status": "success", "summary": "done"}
        if acp_backend["finish"]
        else None
    )
    assert result.stats["backend"] == "gemini_cli"
    assert secret not in str(result.messages)
    for path in ("output_path", "raw_stream_path", "stderr_path"):
        text = Path(result.stats[path]).read_text()
        assert secret not in text
        if outcome != "timeout":
            assert "<GEMINI_API_KEY>" in text
    if outcome == "timeout":
        assert result.error == "Gemini CLI timed out after 0.05s"
    elif outcome in {"startup_error", "prompt_error"}:
        assert result.error == "RuntimeError: gemini unavailable: <GEMINI_API_KEY>"
    else:
        assert result.error is None
        assert result.stats["total_input_tokens"] == 8
        assert acp_backend["prompt"] == "system rules\n\nuser task"
    if outcome in {"timeout", "finish_then_block"}:
        assert acp_backend["events"][-4:] == [
            "cancel_tools",
            "cancel",
            "prompt_completed",
            "close",
        ]


@pytest.mark.parametrize("entrypoint", ["probe", "check"])
def test_prompt_auth_required_preserves_type_and_redacts_key(
    monkeypatch, acp_backend, entrypoint
):
    from rpent.planner.check import (
        STATUS_MISSING_API_KEY,
        LlmCheckRequest,
        check_llm,
    )

    secret = "gemini-probe-secret"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    error = acp.RequestError.auth_required()
    error.args = (f"Authentication required: {secret}",)
    acp_backend["prompt_error"] = error

    if entrypoint == "probe":
        with pytest.raises(acp.RequestError) as caught:
            GeminiCliPlanner(
                output_dir=".",
                dashboard_events=NullDashboardEventSink(),
                timeout_s=1,
            ).probe("ping")
        assert caught.value.code == error.code
        detail = str(caught.value)
    else:
        result = check_llm(LlmCheckRequest(planner="gemini_cli", timeout_s=1))
        assert result.status == STATUS_MISSING_API_KEY
        detail = result.detail
    assert "<GEMINI_API_KEY>" in detail
    assert secret not in detail
    assert acp_backend["events"][-1] == "close"


@pytest.mark.parametrize(
    ("outcome", "error", "message"),
    [
        ("timeout", asyncio.TimeoutError, None),
        ("auth_error", RuntimeError, "401 Unauthorized"),
    ],
)
def test_failed_probe_is_tool_free_and_closes_transport(
    acp_backend, outcome, error, message
):
    planner = GeminiCliPlanner(
        output_dir=".",
        dashboard_events=NullDashboardEventSink(),
        timeout_s=0.05 if outcome == "timeout" else 1,
    )
    acp_backend["block"] = outcome == "timeout"
    if outcome == "auth_error":
        acp_backend["prompt_error"] = RuntimeError("401 Unauthorized")
    with pytest.raises(error, match=message):
        planner.probe("ping")
    assert "mcpServers" not in acp_backend["settings"]
    acp_backend["server_factory"].assert_not_called()
    assert acp_backend["events"][-1] == "close"
    if outcome == "timeout":
        assert acp_backend["events"][-3:] == ["cancel", "prompt_completed", "close"]


def test_simultaneous_interactive_channels_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        make_planner(tmp_path).solve(
            system_prompt="",
            user_message="task",
            toolkit=Mock(),
            max_turns=2,
            input_queue=object(),
            dashboard_interaction=object(),
        )
