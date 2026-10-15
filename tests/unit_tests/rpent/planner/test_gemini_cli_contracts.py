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

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

import rpent.planner.gemini_cli as gemini_module
from rpent.dashboard.events import TranscriptEvent, UsageEvent
from rpent.planner.gemini_cli import GeminiCliPlanner, _Recorder


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

    def list_tools(self) -> tuple[Any, ...]:
        return (
            type("Tool", (), {"name": "inspect_scene"})(),
            type("Tool", (), {"name": "finish"})(),
        )


class FakeMcpServer:
    instances: list[FakeMcpServer] = []

    def __init__(self, toolkit: FakeToolkit) -> None:
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
    assert settings["mcp"] == {"allowed": ["rpent"]}
    assert settings["mcpServers"] == {
        "rpent": {
            "httpUrl": "http://127.0.0.1:43210/mcp/",
            "trust": True,
        }
    }
    assert "GEMINI_API_KEY" not in path.read_text()


def test_command_uses_exact_optional_model_and_stable_headless_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_BIN", "/fake/gemini")

    inherited = gemini_module._build_command(prompt="task", model=None)
    overridden = gemini_module._build_command(prompt="task", model="gemini-upstream-id")

    assert inherited == [
        "/fake/gemini",
        "--prompt",
        "task",
        "--output-format",
        "stream-json",
        "--approval-mode",
        "yolo",
        "--skip-trust",
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


def test_recorder_captures_successful_finish_and_usage() -> None:
    sink = RecordingSink()
    recorder = _Recorder(sink)

    recorder.observe(
        {
            "type": "tool_use",
            "tool_name": "mcp_rpent_finish",
            "tool_id": "call-1",
            "parameters": {"status": "success", "summary": "done"},
        }
    )
    recorder.observe(
        {
            "type": "tool_result",
            "tool_id": "call-1",
            "status": "success",
            "output": json.dumps(
                [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"_finish": True, "status": "success", "summary": "done"}
                        ),
                    }
                ]
            ),
        }
    )
    recorder.observe({"type": "message", "role": "assistant", "content": "complete"})
    recorder.observe(
        {
            "type": "result",
            "status": "success",
            "stats": {
                "input_tokens": 12,
                "output_tokens": 4,
                "cached": 3,
                "tool_calls": 1,
            },
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


def test_failed_finish_tool_is_not_accepted() -> None:
    recorder = _Recorder(RecordingSink())
    recorder.observe(
        {
            "type": "tool_use",
            "tool_name": "mcp_rpent_finish",
            "tool_id": "call-1",
            "parameters": {"status": "success", "summary": "not accepted"},
        }
    )
    recorder.observe(
        {
            "type": "tool_result",
            "tool_id": "call-1",
            "status": "error",
            "output": '{"_finish": true, "status": "success"}',
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

    assert gemini_module._safe_stderr(f"request failed for {secret}") == (
        "request failed for <GEMINI_API_KEY>"
    )


def test_base_url_is_loaded_from_gemini_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir()
    (gemini_dir / ".env").write_text("GEMINI_BASE_URL=https://gateway.example\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)

    assert gemini_module.configured_base_url() == "https://gateway.example"


def test_session_artifacts_redact_key_echoed_by_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "environment-secret-value"
    event = json.dumps(
        {
            "type": "error",
            "severity": "warning",
            "message": f"request failed for {secret}",
        }
    )

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = StringIO(event + "\n")
            self.stderr = StringIO(f"provider rejected {secret}\n")

        def wait(self) -> int:
            return 0

    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setattr(
        gemini_module.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    output_path = tmp_path / "gemini.out"
    raw_stream_path = tmp_path / "gemini.out.stream.jsonl"
    stderr_path = tmp_path / "gemini.out.stderr.log"
    state: dict[str, Any] = {}

    gemini_module._run_session(
        command=["gemini"],
        cwd=tmp_path,
        recorder=_Recorder(RecordingSink()),
        output_path=output_path,
        raw_stream_path=raw_stream_path,
        stderr_path=stderr_path,
        state=state,
    )

    for text in (
        output_path.read_text(),
        raw_stream_path.read_text(),
        stderr_path.read_text(),
        str(state),
    ):
        assert secret not in text
        assert "<GEMINI_API_KEY>" in text


def test_planner_lifecycle_writes_config_and_cleans_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = RecordingSink()
    toolkit = FakeToolkit()
    seen: dict[str, Any] = {}
    FakeMcpServer.instances.clear()
    monkeypatch.setenv("GEMINI_BIN", "/fake/gemini")
    monkeypatch.setattr(gemini_module, "HttpMcpServer", FakeMcpServer)

    def fake_run_session(**kwargs: Any) -> None:
        seen["command"] = kwargs["command"]
        seen["settings"] = json.loads(
            (kwargs["cwd"] / ".gemini" / "settings.json").read_text()
        )
        recorder = kwargs["recorder"]
        events = [
            {
                "type": "tool_use",
                "tool_name": "mcp_rpent_finish",
                "tool_id": "finish-1",
                "parameters": {"status": "success", "summary": "done"},
            },
            {
                "type": "tool_result",
                "tool_id": "finish-1",
                "status": "success",
                "output": '{"_finish": true, "status": "success", "summary": "done"}',
            },
            {"type": "message", "role": "assistant", "content": "done"},
            {
                "type": "result",
                "status": "success",
                "stats": {"input_tokens": 8, "output_tokens": 2, "tool_calls": 1},
            },
        ]
        rendered = "".join(recorder.observe(event) for event in events)
        kwargs["output_path"].write_text(rendered)
        kwargs["raw_stream_path"].write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )
        kwargs["stderr_path"].write_text("")
        kwargs["state"]["text"] = rendered

    monkeypatch.setattr(gemini_module, "_run_session", fake_run_session)

    result = make_planner(tmp_path, sink).solve(
        system_prompt="system rules",
        user_message="user task",
        toolkit=toolkit,  # type: ignore[arg-type]
        max_turns=6,
    )

    server = FakeMcpServer.instances[0]
    assert server.started is True
    assert server.stopped is True
    assert toolkit.cancel_calls == 1
    assert seen["settings"]["model"]["maxSessionTurns"] == 6
    assert seen["settings"]["mcpServers"]["rpent"]["httpUrl"].endswith("/mcp/")
    assert seen["command"][1:3] == [
        "--prompt",
        "system rules\n\nuser task",
    ]
    assert result.finish_result == {
        "_finish": True,
        "status": "success",
        "summary": "done",
    }
    assert result.error is None
    assert result.stats["backend"] == "gemini_cli"
    assert (tmp_path / "gemini.out.stream.jsonl").is_file()


def test_backend_failure_still_stops_mcp_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolkit = FakeToolkit()
    FakeMcpServer.instances.clear()
    monkeypatch.setenv("GEMINI_BIN", "/fake/gemini")
    monkeypatch.setattr(gemini_module, "HttpMcpServer", FakeMcpServer)

    def fail(**kwargs: Any) -> None:
        kwargs["state"]["error"] = RuntimeError("gemini unavailable")

    monkeypatch.setattr(gemini_module, "_run_session", fail)

    result = make_planner(tmp_path, RecordingSink()).solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,  # type: ignore[arg-type]
        max_turns=2,
    )

    assert result.error == "RuntimeError: gemini unavailable"
    assert FakeMcpServer.instances[0].stopped is True
    assert toolkit.cancel_calls == 1


@pytest.mark.parametrize("channel", ["input_queue", "dashboard_interaction"])
def test_interactive_channels_are_rejected_before_starting_mcp(
    tmp_path: Path,
    channel: str,
) -> None:
    kwargs: dict[str, Any] = {channel: object()}

    with pytest.raises(ValueError, match="non-interactive"):
        make_planner(tmp_path, RecordingSink()).solve(
            system_prompt="",
            user_message="task",
            toolkit=FakeToolkit(),  # type: ignore[arg-type]
            max_turns=2,
            **kwargs,
        )
