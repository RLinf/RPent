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
import queue
from contextlib import asynccontextmanager
from types import SimpleNamespace

import acp
import claude_agent_sdk
import pytest
from acp.schema import PromptResponse

from rpent.dashboard.events import NullDashboardEventSink, RunStartedEvent
from rpent.dashboard.state import DashboardState
from rpent.planner import claude_code, codex, gemini_cli
from rpent.planner.utils import agent_session as interactive
from rpent.planner.utils.agent_runtime import AgentArtifacts
from rpent.tools import ToolResult
from rpent.tools.common import CommonTools


@pytest.fixture
def dashboard(tmp_path):
    state = DashboardState(
        output_dir=tmp_path,
        dashboard_spec={
            "task": {
                "command": "/rpent-task",
                "usage": "/rpent-task",
                "fields": (),
                "display": "test",
                "output_slug": "test",
            },
            "runtime_components": (),
            "primitives": (),
        },
    )
    state.shared_services_ready()
    state.submit_input("/rpent-task")
    assert state.wait_for_task(timeout=0) is not None
    state.emit(RunStartedEvent())
    return state


@pytest.mark.parametrize("backend", ["codex", "gemini_cli", "claude_code"])
@pytest.mark.parametrize("channel", ["terminal", "dashboard"])
def test_queue_interrupt_and_continuation_share_one_session(
    tmp_path, monkeypatch, backend, channel, dashboard
):
    events = []
    turns = []
    sessions = []
    input_queue = queue.Queue()

    class Toolkit:
        def list_tools(self):
            return []

        def cancel_active_and_wait(self):
            events.append("cancel_tools")

        def resume_calls(self):
            events.append("resume_tools")

    class Server:
        def __init__(self, toolkit, **kwargs):
            pass

        def start(self):
            events.append("mcp_start")
            return "http://127.0.0.1:12345/mcp/"

        def stop(self):
            events.append("mcp_stop")

    class Turn:
        def __init__(self, text):
            self.text = text
            self.result = asyncio.get_running_loop().create_future()
            turns.append(self)
            events.append(f"submit:{text}")

        async def stream(self):
            status = await self.result
            yield {"method": "turn/completed", "payload": {"turn": {"status": status}}}

        async def interrupt(self):
            events.append("interrupt")
            if not self.result.done():
                self.result.set_result("interrupted")

        async def steer(self, text):
            pytest.fail("queued input must not steer an active turn")

    class Thread:
        async def turn(self, text, **kwargs):
            return Turn(text)

    class Codex:
        def __init__(self, config):
            pass

        async def thread_start(self, **kwargs):
            sessions.append("codex-thread")
            return Thread()

        async def close(self):
            events.append("agent_close")

    class Connection:
        async def initialize(self, **kwargs):
            pass

        async def new_session(self, **kwargs):
            sessions.append("gemini-session")
            return SimpleNamespace(session_id="gemini-session")

        async def prompt(self, session_id, prompt):
            assert session_id == "gemini-session"
            turn = Turn(prompt[0].text)
            status = await turn.result
            return PromptResponse(
                stop_reason="cancelled" if status == "interrupted" else "end_turn"
            )

        async def cancel(self, session_id):
            assert session_id == "gemini-session"
            await turns[-1].interrupt()

    class ClaudeClient:
        def __init__(self, *, options):
            self.pending = asyncio.Queue()

        async def __aenter__(self):
            self.owner = asyncio.current_task()
            sessions.append("claude-session")
            return self

        async def __aexit__(self, *args):
            assert asyncio.current_task() is self.owner
            events.append("agent_close")

        async def query(self, text):
            await self.pending.put(Turn(text))

        async def receive_messages(self):
            while True:
                turn = await self.pending.get()
                status = await turn.result
                yield {
                    "type": "ResultMessage",
                    "is_error": status == "interrupted",
                }

        async def interrupt(self):
            await turns[-1].interrupt()

    @asynccontextmanager
    async def spawn(client, binary, *args, **kwargs):
        assert "--acp" in args
        stderr = asyncio.StreamReader()
        stderr.feed_eof()
        try:
            yield Connection(), SimpleNamespace(stderr=stderr)
        finally:
            events.append("agent_close")

    monkeypatch.setattr(interactive, "HttpMcpServer", Server)
    monkeypatch.setattr(codex.openai_codex, "AsyncCodex", Codex)
    monkeypatch.setattr(acp, "spawn_agent_process", spawn)
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", ClaudeClient)
    monkeypatch.setenv("GEMINI_BIN", "/fake/gemini")
    cls = {
        "codex": codex.CodexPlanner,
        "gemini_cli": gemini_cli.GeminiCliPlanner,
        "claude_code": claude_code.ClaudeCodePlanner,
    }[backend]
    planner = cls(
        output_dir=str(tmp_path),
        output_path=tmp_path / "agent.out",
        dashboard_events=NullDashboardEventSink(),
        timeout_s=5,
    )

    async def wait_until(condition):
        async def poll():
            while not condition():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(poll(), timeout=2)

    async def scenario():
        solve = (
            planner._solve_async(
                "initial",
                initial_user_text="initial",
                toolkit=Toolkit(),
                max_turns=20,
                input_queue=input_queue if channel == "terminal" else None,
                dashboard_interaction=dashboard if channel == "dashboard" else None,
            )
            if backend == "claude_code"
            else planner._solve(
                prompt="initial",
                initial_user_text="initial",
                toolkit=Toolkit(),
                max_turns=20,
                input_queue=input_queue if channel == "terminal" else None,
                interaction=dashboard if channel == "dashboard" else None,
            )
        )
        task = asyncio.create_task(solve)
        try:
            await wait_until(lambda: len(turns) == 1)
            if channel == "terminal":
                input_queue.put("ordinary")
            else:
                dashboard.submit_input("ordinary")
            await wait_until(input_queue.empty)
            await asyncio.sleep(0.02)
            assert [turn.text for turn in turns] == ["initial"]
            assert "interrupt" not in events
            turns[0].result.set_result("completed")
            await wait_until(lambda: len(turns) == 2)
            assert turns[1].text == "ordinary"
            if channel == "terminal":
                input_queue.put("later")
                input_queue.put("/interrupt urgent")
            else:
                dashboard.submit_input("urgent")
                dashboard.submit_input("later")
                assert dashboard.request_interrupt() == "accepted"
            await wait_until(lambda: len(turns) == 3)
            assert turns[2].text == "urgent"
            assert events.index("cancel_tools") < events.index("interrupt")
            assert (
                events.index("interrupt")
                < events.index("resume_tools")
                < events.index("submit:urgent")
            )
            turns[2].result.set_result("completed")
            await wait_until(lambda: len(turns) == 4)
            assert turns[3].text == "later"
        finally:
            if channel == "terminal":
                input_queue.put(None)
            else:
                dashboard.seal_interaction()
            result = await asyncio.wait_for(task, timeout=3)
        assert result.error is None
        assert len(sessions) == 1
        assert events[-1] == ("agent_close" if backend == "claude_code" else "mcp_stop")
        if backend == "claude_code":
            assert "mcp_start" not in events
        assert "agent_close" in events
        assert "[user] ordinary" in (tmp_path / "agent.out").read_text()
        assert "[user] urgent" in (tmp_path / "agent.out").read_text()

    asyncio.run(scenario())


@pytest.mark.parametrize("channel", ["terminal", "dashboard"])
def test_claude_confirmed_finish_does_not_send_pending_input(
    tmp_path, monkeypatch, channel, dashboard
):
    # Use the real SDK MCP registration with a fake client, not a model request.
    input_queue = queue.Queue()
    queries = []
    verdict = {"_finish": True, "status": "failure", "summary": "operator aborted"}

    class Toolkit:
        def list_tools(self):
            return [CommonTools.finish]

        def execute_tool(self, name, args):
            return ToolResult(data=verdict)

        def cancel_active_and_wait(self):
            pass

        def resume_calls(self):
            pass

    class Client:
        def __init__(self, *, options):
            self.options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def query(self, text):
            queries.append(text)

        async def receive_messages(self):
            if channel == "terminal":
                input_queue.put("must remain unsent")
            else:
                dashboard.submit_input("must remain unsent")
            await asyncio.sleep(0.02)
            # Invoke the SDK server's registered handler through its MCP API.
            server = self.options.mcp_servers["rpent"]["instance"]
            from mcp.types import CallToolRequest, CallToolRequestParams

            response = await server.request_handlers[CallToolRequest](
                CallToolRequest(
                    params=CallToolRequestParams(
                        name="finish",
                        arguments={"status": "success", "summary": "claim"},
                    )
                )
            )
            assert not response.root.isError
            yield {"type": "ResultMessage", "subtype": "success"}

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    planner = claude_code.ClaudeCodePlanner(
        output_dir=str(tmp_path),
        dashboard_events=NullDashboardEventSink(),
    )
    result = planner.solve(
        system_prompt="",
        user_message="initial",
        toolkit=Toolkit(),
        max_turns=3,
        input_queue=input_queue if channel == "terminal" else None,
        dashboard_interaction=dashboard if channel == "dashboard" else None,
    )
    assert result.error is None
    assert result.finish_result == verdict
    assert queries == ["initial"]
    if channel == "dashboard":
        message = dashboard.snapshot()["interaction"]["messages"][0]
        assert message["status"] == "unsent"


@pytest.mark.parametrize(
    "outcome, expected_error",
    [
        ("mcp_error", "RuntimeError: MCP did not start"),
        ("startup_error", "RuntimeError: backend startup failed"),
        ("timeout", "test agent timed out after 0.05s"),
        ("cleanup_error", "Toolkit cleanup failed: <redacted>"),
        ("both_errors", "RuntimeError: backend startup failed"),
        ("finish", None),
    ],
)
def test_interactive_cleanup_and_finish_preserve_unsent_input(
    tmp_path, monkeypatch, dashboard, outcome, expected_error, caplog
):
    events = []

    class Server:
        def __init__(self, toolkit):
            pass

        def start(self):
            events.append("mcp_start")
            if outcome == "mcp_error":
                raise RuntimeError("MCP did not start")
            return "http://127.0.0.1:12345/mcp/"

        def stop(self):
            events.append("mcp_stop")

    class Session:
        error = None

        def __init__(self, url, control, emit):
            self.control, self.emit = control, emit

        async def run(self, prompt):
            if outcome in {"startup_error", "both_errors"}:
                raise RuntimeError("backend startup failed")
            await self.control.start()
            dashboard.submit_input("never send after finish or timeout")
            if outcome == "finish":
                self.emit(
                    {
                        "type": "rpent_tool",
                        "name": "finish",
                        "args": {},
                        "is_error": False,
                        "result": {"_finish": True, "status": "success"},
                    }
                )
                await self.control.complete(self)
            elif outcome == "timeout":
                await asyncio.Event().wait()

        async def submit(self, message):
            pytest.fail("pending input must not be sent after task termination")

        async def close(self):
            events.append("agent_close")

    monkeypatch.setattr(interactive, "HttpMcpServer", Server)
    artifacts = AgentArtifacts("test", tmp_path / "agent.out")
    recorder = gemini_cli._Recorder(NullDashboardEventSink())

    def cancel():
        events.append("cancel_tools")
        if outcome in {"cleanup_error", "both_errors"}:
            raise RuntimeError("private-key-value")

    error = asyncio.run(
        interactive.run_agent(
            prompt="initial",
            initial_user_text="initial",
            toolkit=SimpleNamespace(
                cancel_active_and_wait=cancel,
                resume_calls=lambda: events.append("resume_tools"),
            ),
            artifacts=artifacts,
            recorder=recorder,
            events=NullDashboardEventSink(),
            create_session=Session,
            timeout_s=0.05,
            label="test agent",
            input_queue=None,
            interaction=dashboard,
            redact=lambda text: text.replace("private-key-value", "<redacted>"),
        )
    )
    expected_events = ["mcp_start", "cancel_tools"]
    if outcome != "mcp_error":
        expected_events.append("agent_close")
    assert events == [*expected_events, "mcp_stop"]
    assert error == expected_error
    assert dashboard.planner_activity == "ended"
    assert "private-key-value" not in caplog.text
    for path in (artifacts.output_path, artifacts.raw_stream_path):
        assert "private-key-value" not in path.read_text()
        if error:
            assert error in path.read_text()
    if outcome not in {"mcp_error", "startup_error", "both_errors"}:
        assert dashboard.snapshot()["interaction"]["messages"][0]["status"] == "unsent"
    if outcome == "finish":
        assert recorder.finish_result["status"] == "success"
