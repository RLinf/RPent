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

import asyncio
import json
from types import SimpleNamespace

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.utils import agent_session
from rpent.planner.utils.agent_runtime import AgentArtifacts


@pytest.fixture
def lifecycle(monkeypatch):
    events = []

    def start():
        events.append("start")
        return "http://127.0.0.1:43210/mcp/"

    server = SimpleNamespace(start=start, stop=lambda: events.append("stop"))
    toolkit = SimpleNamespace(
        cancel_active_and_wait=lambda: events.append("cancel"),
        resume_calls=lambda: None,
    )
    monkeypatch.setattr(agent_session, "HttpMcpServer", lambda toolkit: server)

    def run(artifacts, work, *, timeout_s=1, redact=str):
        class Session:
            error = None

            async def run(self, prompt):
                events.append("run")
                await work()

            async def close(self):
                events.append("close")

        def create_session(url, control, emit):
            assert url == "http://127.0.0.1:43210/mcp/"
            assert control is None
            return Session()

        return asyncio.run(
            agent_session.run_agent(
                prompt="task",
                initial_user_text="task",
                toolkit=toolkit,
                artifacts=artifacts,
                recorder=SimpleNamespace(finish_result=None),
                events=NullDashboardEventSink(),
                create_session=create_session,
                timeout_s=timeout_s,
                label="Test agent",
                input_queue=None,
                interaction=None,
                redact=redact,
            )
        )

    return events, server, toolkit, run


@pytest.mark.parametrize("backend", ["codex_sdk", "gemini_cli"])
def test_session_and_result_preserve_artifacts(tmp_path, lifecycle, backend):
    events, _, _, run = lifecycle
    artifacts = AgentArtifacts(backend, tmp_path / "logs" / "agent.out")

    async def work():
        pass

    error = run(artifacts, work)
    result = artifacts.result(
        text="",
        finish_result={"_finish": True},
        final_response="done",
        stats={"tool_calls": 1},
        error=error,
        stderr_path=artifacts.sidecar(".stderr.log"),
    )
    assert events == ["start", "run", "cancel", "close", "stop"]
    assert result.error is None
    assert result.finish_result == {"_finish": True}
    assert result.messages == [
        {"role": backend, "content": artifacts.output_path.read_text()}
    ]
    assert result.stats["output_chars"] == len(artifacts.output_path.read_text())
    assert result.stats["last_message_chars"] == 4
    assert result.stats["tool_calls"] == 1
    assert result.stats["elapsed_s"] >= 0
    assert result.stats["raw_stream_path"].endswith("agent.out.stream.jsonl")
    assert result.stats["stderr_path"].endswith("agent.out.stderr.log")


def test_start_failure_still_cancels_tools_and_stops_server(tmp_path, lifecycle):
    events, server, _, run = lifecycle

    def fail():
        events.append("start")
        raise RuntimeError("MCP did not start")

    async def work():
        pytest.fail("agent must not start")

    server.start = fail
    error = run(AgentArtifacts("test", tmp_path / "agent.out"), work)
    assert error == "RuntimeError: MCP did not start"
    assert events == ["start", "cancel", "stop"]


@pytest.mark.parametrize("backend_fails", [False, True])
def test_cleanup_failure_preserves_backend_error_and_redacts_secrets(
    tmp_path, lifecycle, backend_fails, caplog
):
    events, _, toolkit, run = lifecycle
    artifacts = AgentArtifacts("test", tmp_path / "agent.out")
    secret = "private-key-value"

    async def work():
        if backend_fails:
            raise RuntimeError(f"backend failed: {secret}")

    def cancel():
        events.append("cancel")
        raise RuntimeError(f"cancel failed: {secret}")

    toolkit.cancel_active_and_wait = cancel
    error = run(artifacts, work, redact=lambda text: text.replace(secret, "<redacted>"))
    expected = (
        "RuntimeError: backend failed: <redacted>"
        if backend_fails
        else "Toolkit cleanup failed: cancel failed: <redacted>"
    )
    assert error == expected
    assert events[-3:] == ["cancel", "close", "stop"]
    assert secret not in caplog.text
    assert secret not in artifacts.output_path.read_text()
    assert json.loads(artifacts.raw_stream_path.read_text()) == {
        "type": "error",
        "message": expected,
    }


def test_timeout_cancels_tools_before_backend_close(tmp_path, lifecycle):
    events, _, _, run = lifecycle
    artifacts = AgentArtifacts("test", tmp_path / "agent.out")

    async def work():
        await asyncio.Event().wait()

    error = run(artifacts, work, timeout_s=0.05)
    assert events == ["start", "run", "cancel", "close", "stop"]
    assert error == "Test agent timed out after 0.05s"
    assert json.loads(artifacts.raw_stream_path.read_text())["message"] == error


def test_temporary_output_uses_requested_directory(tmp_path):
    directory = tmp_path / "artifacts"
    artifacts = AgentArtifacts("gemini_cli", None, directory=directory)
    assert artifacts.output_path.parent == directory
    assert artifacts.output_path.is_file()
    assert artifacts.output_path.name.startswith("gemini_cli_task_")
