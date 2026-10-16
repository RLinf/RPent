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
from typing import Any

import acp
import pytest

import rpent.planner.gemini_cli as gemini_module
import rpent.planner.utils.agent_session as session_module
from rpent.dashboard.events import TranscriptEvent, UsageEvent
from rpent.planner.gemini_cli import GeminiCliPlanner, _Recorder
from rpent.tools import ToolResult


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

    def cancel_active_and_wait(self) -> None:
        self.cancel_calls += 1

    def resume_calls(self) -> None:
        pass

    def list_tools(self) -> tuple[Any, ...]:
        return (
            type("Tool", (), {"name": "inspect_scene"})(),
            type("Tool", (), {"name": "finish"})(),
        )


class FakeMcpServer:
    instances: list[FakeMcpServer] = []

    def __init__(self, toolkit: FakeToolkit, *, on_tool_result=None) -> None:
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


def make_planner(tmp_path: Path, sink: RecordingSink) -> GeminiCliPlanner:
    return GeminiCliPlanner(
        output_dir=str(tmp_path),
        output_path=tmp_path / "gemini.out",
        model="gemini-upstream-id",
        timeout_s=2,
        dashboard_events=sink,
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


def test_recorder_captures_successful_finish_and_usage():
    sink = RecordingSink()
    recorder = _Recorder(sink)
    recorder.observe(
        {
            "type": "rpent_tool",
            "name": "finish",
            "args": {"status": "success", "summary": "done"},
            "result": {"_finish": True, "status": "success", "summary": "done"},
            "is_error": False,
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
    recorder.observe(
        {
            "type": "acp_result",
            "stopReason": "end_turn",
            "usage": {"inputTokens": 12, "outputTokens": 4, "cachedReadTokens": 3},
        }
    )
    assert recorder.finish_result == {
        "_finish": True,
        "status": "success",
        "summary": "done",
    }
    assert recorder.final_response == "complete"
    assert recorder.stats() == {
        "tool_calls": 1,
        "total_input_tokens": 12,
        "total_output_tokens": 4,
        "total_cached_input_tokens": 3,
    }
    assert any(isinstance(event, TranscriptEvent) for event in sink.events)
    assert any(isinstance(event, UsageEvent) for event in sink.events)


def test_failed_finish_tool_is_not_accepted():
    recorder = _Recorder(RecordingSink())
    recorder.observe(
        {
            "type": "rpent_tool",
            "name": "finish",
            "args": {"status": "success"},
            "is_error": True,
            "result": {"_finish": True, "status": "success"},
        }
    )
    assert recorder.finish_result is None


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


def test_session_artifacts_redact_key_echoed_by_cli(tmp_path, monkeypatch, acp_backend):
    secret = "environment-secret-value"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    acp_backend["reply"] = f"request failed for {secret}"
    acp_backend["stderr"] = f"provider rejected {secret}\n"
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=FakeToolkit(),
        max_turns=2,
    )
    for path in ("output_path", "raw_stream_path", "stderr_path"):
        text = Path(result.stats[path]).read_text()
        assert secret not in text
        assert "<GEMINI_API_KEY>" in text
    assert secret not in str(result.messages)


@pytest.mark.parametrize("with_finish", [False, True])
def test_planner_lifecycle_writes_config_and_cleans_resources(
    tmp_path, acp_backend, with_finish
):
    toolkit = FakeToolkit()
    acp_backend["finish"] = with_finish
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="system rules",
        user_message="user task",
        toolkit=toolkit,
        max_turns=6,
    )
    server = FakeMcpServer.instances[0]
    assert server.started and server.stopped
    assert toolkit.cancel_calls == 1
    assert acp_backend["settings"]["model"]["maxSessionTurns"] == 6
    assert acp_backend["settings"]["mcpServers"]["rpent"]["httpUrl"].endswith("/mcp/")
    assert acp_backend["command"][1] == "--acp"
    assert acp_backend["prompt"] == "system rules\n\nuser task"
    assert acp_backend["events"] == [
        "initialize",
        "new_session",
        "prompt",
        "prompt_completed",
        "close",
    ]
    assert not acp_backend["cwd"].exists()
    assert result.finish_result == (
        {"_finish": True, "status": "success", "summary": "done"}
        if with_finish
        else None
    )
    assert result.error is None
    assert result.stats["backend"] == "gemini_cli"
    assert result.stats["total_input_tokens"] == 8
    assert (tmp_path / "gemini.out.stream.jsonl").is_file()


@pytest.mark.parametrize("stage", ["startup_error", "prompt_error"])
def test_backend_failure_still_stops_mcp_server(tmp_path, acp_backend, stage):
    toolkit = FakeToolkit()
    acp_backend[stage] = RuntimeError("gemini unavailable")
    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=2,
    )
    assert result.error == "RuntimeError: gemini unavailable"
    assert FakeMcpServer.instances[0].stopped
    assert toolkit.cancel_calls == 1
    assert acp_backend["events"][-1] == "close"


def test_probe_uses_acp_without_mcp_and_preserves_configuration(acp_backend):
    assert (
        gemini_module.run_probe_turn(prompt="ping", model=None, timeout_s=1) == "done"
    )
    assert acp_backend["prompt"] == "ping"
    assert "--model" not in acp_backend["command"]
    assert "mcpServers" not in acp_backend["settings"]
    assert not FakeMcpServer.instances
    assert acp_backend["events"][-1] == "close"


def test_probe_timeout_cancels_prompt_and_closes_transport(acp_backend):
    acp_backend["block"] = True
    with pytest.raises(TimeoutError, match="did not finish within"):
        gemini_module.run_probe_turn(prompt="ping", model=None, timeout_s=0.05)
    assert acp_backend["events"][-3:] == ["cancel", "prompt_completed", "close"]


def test_planner_timeout_cancels_tools_before_acp_cancel(tmp_path, acp_backend):
    acp_backend["block"] = True
    toolkit = FakeToolkit()
    toolkit.cancel_active_and_wait = lambda: acp_backend["events"].append(
        "cancel_tools"
    )
    planner = GeminiCliPlanner(
        output_dir=str(tmp_path),
        dashboard_events=RecordingSink(),
        timeout_s=0.05,
    )
    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=2,
    )
    assert result.error == "Gemini CLI timed out after 0.05s"
    assert acp_backend["events"][-4:] == [
        "cancel_tools",
        "cancel",
        "prompt_completed",
        "close",
    ]
    assert FakeMcpServer.instances[0].stopped


def test_probe_propagates_authentication_errors(acp_backend):
    acp_backend["prompt_error"] = RuntimeError("401 Unauthorized")
    with pytest.raises(RuntimeError, match="401 Unauthorized"):
        gemini_module.run_probe_turn(prompt="ping", model=None, timeout_s=1)
    assert acp_backend["events"][-1] == "close"


def test_simultaneous_interactive_channels_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        make_planner(tmp_path, RecordingSink()).solve(
            system_prompt="",
            user_message="task",
            toolkit=FakeToolkit(),  # type: ignore[arg-type]
            max_turns=2,
            input_queue=object(),
            dashboard_interaction=object(),
        )
