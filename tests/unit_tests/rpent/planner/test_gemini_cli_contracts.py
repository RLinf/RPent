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

import rpent.planner.gemini_cli as gemini_module
import rpent.planner.utils.agent_session as session_module
from rpent.dashboard.events import NullDashboardEventSink, TranscriptEvent, UsageEvent
from rpent.planner.gemini_cli import GeminiCliPlanner, _Recorder
from rpent.tools import ToolResult


class FakeMcpServer:
    instances: list[FakeMcpServer] = []

    def __init__(self, toolkit, *, on_tool_result=None) -> None:
        self.on_tool_result = on_tool_result
        self.toolkit = toolkit
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def start(self) -> str:
        self.started = True
        return "http://127.0.0.1:43210/mcp/"

    def stop(self) -> None:
        self.stopped = True


def make_planner(tmp_path: Path, *, timeout_s=2) -> GeminiCliPlanner:
    return GeminiCliPlanner(
        output_dir=str(tmp_path),
        output_path=tmp_path / "gemini.out",
        model="gemini-upstream-id",
        timeout_s=timeout_s,
        dashboard_events=NullDashboardEventSink(),
    )


def test_workspace_settings_expose_only_rpent_mcp_and_set_turn_limit(
    tmp_path: Path,
) -> None:
    path = gemini_module._write_settings(
        tmp_path,
        mcp_url="http://127.0.0.1:43210/mcp/",
        max_turns=7,
        tool_names=("inspect_scene", "finish"),
    )

    settings = json.loads(path.read_text())
    assert settings["model"]["maxSessionTurns"] == 7
    assert settings["tools"]["core"] == [
        "mcp_rpent_inspect_scene",
        "mcp_rpent_finish",
    ]
    assert settings["skills"]["enabled"] is False
    assert settings["hooksConfig"]["enabled"] is False
    assert settings["context"]["includeDirectoryTree"] is False
    assert settings["mcpServers"] == {
        "rpent": {
            "httpUrl": "http://127.0.0.1:43210/mcp/",
            "trust": True,
        }
    }
    assert "GEMINI_API_KEY" not in path.read_text()


def test_command_uses_acp_and_exact_optional_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_BIN", "/fake/gemini")

    inherited = gemini_module._build_command(model=None)
    overridden = gemini_module._build_command(model="gemini-upstream-id")

    assert inherited == [
        "/fake/gemini",
        "--acp",
        "--allowed-mcp-server-names",
        "rpent",
        "--extensions",
        "none",
    ]
    assert overridden[-2:] == ["--model", "gemini-upstream-id"]


def test_environment_trusts_generated_workspace_and_bypasses_loopback_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_PROXY", "gateway.example")
    monkeypatch.delenv("no_proxy", raising=False)

    env = gemini_module._gemini_environment()

    assert env["GEMINI_CLI_TRUST_WORKSPACE"] == "true"
    assert env["GEMINI_CLI_NO_RELAUNCH"] == "1"
    assert env["NO_PROXY"] == "gateway.example,127.0.0.1,localhost"
    assert env["no_proxy"] == env["NO_PROXY"]


@pytest.mark.parametrize("tool_error", [False, True])
def test_recorder_uses_actual_finish_and_accumulates_usage(tool_error):
    events = []
    recorder = _Recorder(SimpleNamespace(enabled=True, emit=events.append))
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


def test_stderr_redacts_key_loaded_from_gemini_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "dotenv-secret-value"
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir()
    (gemini_dir / ".env").write_text(f"GEMINI_API_KEY={secret}\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert gemini_module._redact_secrets(f"request failed for {secret}") == (
        "request failed for <GEMINI_API_KEY>"
    )


@pytest.fixture
def acp_backend(monkeypatch):
    seen = {"events": [], "reply": "done", "stderr": ""}
    FakeMcpServer.instances.clear()
    monkeypatch.setenv("GEMINI_BIN", "/fake/gemini")
    monkeypatch.setattr(session_module, "HttpMcpServer", FakeMcpServer)

    @asynccontextmanager
    async def spawn(client, binary, *args, cwd, **kwargs):
        seen["command"] = [binary, *args]
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

            async def authenticate(self, **kwargs):
                pytest.fail("RPent must not change CLI authentication")

            async def new_session(self, **kwargs):
                seen["events"].append("new_session")
                return SimpleNamespace(session_id="same-session")

            async def prompt(self, session_id, prompt):
                assert session_id == "same-session"
                seen["prompt"] = prompt[0].text
                seen["events"].append("prompt")
                if "prompt_error" in seen:
                    raise seen["prompt_error"]
                if seen.get("block"):
                    await stopped.wait()
                elif FakeMcpServer.instances and seen.get("finish", True):
                    server = FakeMcpServer.instances[-1]
                    server.on_tool_result(
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
    "outcome", ["finish", "no_finish", "startup_error", "prompt_error", "timeout"]
)
def test_session_result_artifacts_and_cleanup(
    tmp_path, monkeypatch, acp_backend, outcome
):
    secret = "environment-secret-value"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    acp_backend["reply"] = f"request failed for {secret}"
    acp_backend["stderr"] = f"provider rejected {secret}\n"
    acp_backend["finish"] = outcome == "finish"
    acp_backend["block"] = outcome == "timeout"
    if outcome in {"startup_error", "prompt_error"}:
        acp_backend[outcome] = RuntimeError(f"gemini unavailable: {secret}")
    toolkit = Mock()
    toolkit.list_tools.return_value = (SimpleNamespace(name="finish"),)
    toolkit.cancel_active_and_wait.side_effect = lambda: acp_backend["events"].append(
        "cancel_tools"
    )
    result = make_planner(
        tmp_path, timeout_s=0.05 if outcome == "timeout" else 2
    ).solve(
        system_prompt="system rules",
        user_message="user task",
        toolkit=toolkit,
        max_turns=6,
    )
    server = FakeMcpServer.instances[0]
    assert server.started and server.stopped
    toolkit.cancel_active_and_wait.assert_called_once()
    assert acp_backend["settings"]["model"]["maxSessionTurns"] == 6
    assert acp_backend["settings"]["mcpServers"]["rpent"]["httpUrl"].endswith("/mcp/")
    assert acp_backend["command"][1] == "--acp"
    assert acp_backend["events"][-1] == "close"
    assert not acp_backend["cwd"].exists()
    assert result.finish_result == (
        {"_finish": True, "status": "success", "summary": "done"}
        if outcome == "finish"
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
        assert acp_backend["events"][-4:] == [
            "cancel_tools",
            "cancel",
            "prompt_completed",
            "close",
        ]
    elif outcome in {"startup_error", "prompt_error"}:
        assert result.error == "RuntimeError: gemini unavailable: <GEMINI_API_KEY>"
    else:
        assert result.error is None
        assert result.stats["total_input_tokens"] == 8
        assert acp_backend["prompt"] == "system rules\n\nuser task"


@pytest.mark.parametrize("outcome", ["success", "timeout", "auth_error"])
def test_probe_is_tool_free_and_closes_transport(acp_backend, outcome):
    acp_backend["block"] = outcome == "timeout"
    if outcome == "auth_error":
        acp_backend["prompt_error"] = RuntimeError("401 Unauthorized")
    if outcome == "success":
        assert (
            gemini_module.run_probe_turn(prompt="ping", model=None, timeout_s=1)
            == "done"
        )
    else:
        error, message = (
            (TimeoutError, "did not finish within")
            if outcome == "timeout"
            else (RuntimeError, "401 Unauthorized")
        )
        with pytest.raises(error, match=message):
            gemini_module.run_probe_turn(prompt="ping", model=None, timeout_s=0.05)
    assert "--model" not in acp_backend["command"]
    assert "mcpServers" not in acp_backend["settings"]
    assert not FakeMcpServer.instances
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
