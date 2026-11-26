# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Persistent Python execution and cancellation across a real child process."""

import socket
import threading
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from rpent.cli.policy import planner_policy
from rpent.code_policy import runtime
from rpent.code_policy.runtime import CodeSession
from rpent.code_policy.toolkit import CodePolicyTools
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.prompt.code_policy import code_policy_api_reference
from rpent.robots.prompt_bundle import PromptBundle
from rpent.session import EnvState, Payload
from rpent.session.env_capture import ToolCallCapture
from rpent.tools import Toolkit, ToolResult, tool
from rpent.utils import logging as logging_module


class _BlockingToolkit(Toolkit):
    def __init__(self, root: Path) -> None:
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            state=EnvState(root),
            memory=MemoryManager(root / "memory"),
        )
        self.entered = threading.Event()
        self.release = threading.Event()
        self.stopping = threading.Event()
        self.cancel_count = 0
        self.observations = []
        self.add_tools((self.block, self.move))
        self.add_tool(self.finish, replace=True)

    @tool(readonly=True)
    def block(self) -> ToolResult:
        """Wait for a simulated blocking RPC to return before checking cancellation."""
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError("Test did not release the blocking call")
        self.raise_if_cancelled()
        return ToolResult(data={"ok": True})

    @tool
    def move(self, distance: int) -> ToolResult:
        return ToolResult(data={"distance": distance})

    @tool
    def finish(self) -> ToolResult:
        return ToolResult(data={"_finish": True})

    def get_env_state(self, *, command, result, elapsed_s):
        self.observations.append(command)
        return ToolResult(data=result, images=[b"observation"])

    def cancel_active_and_wait(self) -> None:
        self.cancel_count += 1
        self.stopping.set()
        super().cancel_active_and_wait()


@pytest.fixture
def policy(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_module, "_output_dir", tmp_path)
    robot = _BlockingToolkit(tmp_path / "robot")
    provider = CodePolicyTools(robot, output_dir=tmp_path)
    try:
        yield provider, robot
    finally:
        robot.release.set()
        provider.close()


def _run(provider, code):
    return provider.execute_tool("python", {"code": code}).data


def test_default_code_prompt_and_dispatch_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_module, "_output_dir", tmp_path)
    robot = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        state=EnvState(tmp_path / "state"),
        memory=None,
        action_model_available=False,
    )

    @tool(readonly=True)
    def native_status() -> ToolResult:
        return ToolResult(data={"ready": True})

    @tool(readonly=True, requires_action_model=True)
    def use_model() -> ToolResult:
        pytest.fail("A disabled model tool was executed")

    robot.add_tools((native_status, use_model))
    args = Namespace(
        policy_mode="code", code_timeout_s=10, code_max_calls=4, no_images=False
    )
    bundle = PromptBundle(
        system=lambda _: "Use direct tools.", user=lambda _: "Observe."
    )
    with planner_policy(robot, args, tmp_path, {"policy_mode": "code"}) as (
        policy,
        variables,
    ):
        prompt = bundle.render("system", variables=variables)
        assert "persistent Python session" in prompt
        assert "robo.native_status" in prompt
        assert "Use direct tools." not in prompt and "robo.use_model" not in prompt
        assert (
            _run(policy, 'print(robo.native_status()["ready"])')["stdout"] == "True\n"
        )
        assert _run(policy, "robo.use_model()")["exception"]["type"] == "AttributeError"
        assert robot.execute_tool("use_model", {}).is_error


def test_variables_survive_python_exceptions(policy):
    provider, robot = policy
    failed = _run(provider, "x = 41\nraise ValueError('retry')")
    assert failed["exception"]["type"] == "ValueError"
    assert not failed["namespace_reset"]
    silent = _run(provider, "x + 1")
    assert silent["exception"] is None
    assert silent["stdout"] == ""
    assert "value" not in silent
    assert _run(provider, "print(x + 1)")["stdout"] == "42\n"
    assert robot.cancel_count == 0


def test_worker_isolates_files_environment_and_network(policy, tmp_path, monkeypatch):
    provider, _ = policy
    marker = tmp_path / "host-only"
    marker.write_text("test", encoding="utf-8")
    monkeypatch.setenv("RPENT_TEST_HOST_ONLY", "test")
    with socket.create_server(("127.0.0.1", 0)) as listener:
        result = _run(
            provider,
            "import os, socket\n"
            "from pathlib import Path\n"
            "import numpy as np\n"
            "Path('cell-data').write_text('scratch')\n"
            f"print(Path({str(marker)!r}).exists())\n"
            "print('RPENT_TEST_HOST_ONLY' in os.environ)\n"
            "try:\n"
            f"    socket.create_connection({listener.getsockname()!r}, timeout=1).close()\n"
            "except OSError:\n"
            "    print('network isolated')\n"
            "print(robo.move(distance=np.int64(2))['distance'])",
        )
    assert result["exception"] is None
    assert result["stdout"] == "False\nFalse\nnetwork isolated\n2\n"
    assert (
        _run(provider, "print(Path('cell-data').read_text())")["stdout"] == "scratch\n"
    )


def test_cell_captures_each_primitive_once_and_returns_latest_feedback(policy):
    provider, robot = policy
    state = robot.state
    state.start_task(instruction="Move twice.", config=Payload("test", {}))
    cell_id = state.register_tool_call(
        planner_id=None, name="python", arguments=Payload("test", {})
    )
    with ToolCallCapture(state, cell_id).bind():
        result = provider.execute_tool(
            "python", {"code": "robo.move(distance=1)\nprint(robo.move(distance=2))"}
        )
    assert result.data["exception"] is None
    assert result.data["primitive_call_count"] == 2
    assert result.data["last_tool_result"] == {"distance": 2}
    assert "'distance': 2" in result.data["stdout"]
    assert result.images == [b"observation"]
    assert robot.observations == [
        {"action": "move", "distance": 1},
        {"action": "move", "distance": 2},
    ]
    assert provider.state is robot.state
    assert {item.name for item in provider.list_tools()} == {"python"}
    assert provider.execute_tool("move", {"distance": 3}).is_error
    cell, first, second = state.tool_call_records()
    assert (
        cell.name == "python"
        and cell.raw_result.data["data"]["primitive_call_count"] == 2
    )
    assert [call.feedback.data for call in (first, second)] == [
        {"distance": 1},
        {"distance": 2},
    ]
    for call in (first, second):
        assert call.origin == "program" and call.status == "completed"
        assert call.source_ids["rpent.parent_tool_call_id"] == cell_id


def test_nested_call_cancelled_before_admission_has_no_execution_times(policy):
    provider, robot = policy
    state = robot.state
    state.start_task(instruction="Move.", config=Payload("test", {}))
    cell_id = state.register_tool_call(
        planner_id=None, name="python", arguments=Payload("test", {})
    )
    robot.cancel_active_and_wait()
    with ToolCallCapture(state, cell_id).bind():
        _run(provider, "robo.move(distance=1)")
    _, child = state.tool_call_records()
    assert child.status == "cancelled"
    assert child.started_at is None and child.ended_at is None
    assert child.feedback.data["code"] == "tool_cancelled"
    assert robot.observations == []


@pytest.mark.parametrize("code", ["while True: pass", "import os; os._exit(2)"])
def test_worker_failure_or_timeout_allows_a_fresh_cell(policy, code):
    provider, robot = policy
    provider._timeout_s = 0.5
    result = _run(provider, "x = 41\n" + code)
    assert result["interrupted"] and result["namespace_reset"]
    assert result["timed_out"] is code.startswith("while")
    assert robot.cancel_count == 1
    assert _run(provider, "print('x' in globals())")["stdout"] == "False\n"
    assert _run(provider, "print(6 * 7)")["stdout"] == "42\n"


@pytest.mark.parametrize("external", [False, True])
def test_cancellation_drains_robot_and_external_stop_requires_resume(policy, external):
    provider, robot = policy
    provider._timeout_s = 5 if external else 0.5
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(_run, provider, "x = 41\nrobo.block()")
        stop = None
        try:
            assert robot.entered.wait(3)
            if external:
                stop = pool.submit(provider.cancel_active_and_wait)
            assert robot.stopping.wait(3)
            assert not running.done()
            if stop is not None:
                assert not stop.done()
        finally:
            robot.release.set()
        result = running.result(timeout=3)
        if stop is not None:
            stop.result(timeout=3)
    assert result["namespace_reset"]
    assert result["timed_out"] is not external
    assert robot.cancel_count == 1
    if external:
        assert _run(provider, "42")["code"] == "tool_cancelled"
        provider.resume_calls()
    assert _run(provider, "print('x' in globals())")["stdout"] == "False\n"


def test_repeated_stop_notifies_once_until_resumed(tmp_path):
    notifications = []
    session = CodeSession(
        tools=[],
        dispatch=lambda name, args: {},
        workspace=tmp_path,
        on_interrupt=lambda: notifications.append("stop"),
    )
    try:
        session.execute("x = 41", timeout_s=3)
        session.cancel()
        session.cancel()
        assert notifications == ["stop"]
        session.resume()
        result = session.execute("print('x' in globals())", timeout_s=3)
        assert result["stdout"] == "False\n"
        session.cancel()
        assert notifications == ["stop", "stop"]
    finally:
        session.close()


def test_cancel_at_cell_completion_keeps_session_stopped(tmp_path, monkeypatch):
    session = CodeSession(tools=[], dispatch=lambda *_: {}, workspace=tmp_path)
    reached = threading.Event()
    release = threading.Event()
    calls = 0

    def pause_at_elapsed_time():
        nonlocal calls
        calls += 1
        if calls == 2:
            reached.set()
            assert release.wait(3)
        return time.monotonic()

    monkeypatch.setattr(
        runtime, "time", SimpleNamespace(monotonic=pause_at_elapsed_time)
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(session.execute, "x = 41", timeout_s=5)
            try:
                assert reached.wait(3)
                session.cancel()
            finally:
                release.set()
            running.result(timeout=3)
        with pytest.raises(RuntimeError, match="resume it first"):
            session.execute("print(1)", timeout_s=3)
        session.resume()
        assert (
            session.execute("print('x' in globals())", timeout_s=3)["stdout"]
            == "False\n"
        )
    finally:
        session.close()


def test_timeout_discards_already_queued_cells(policy):
    provider, robot = policy
    provider._timeout_s = 0.5
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(_run, provider, "robo.block()")
        try:
            assert robot.entered.wait(3)
            waiting = pool.submit(_run, provider, "stale = True")
            with provider._scheduler.condition:
                assert provider._scheduler.condition.wait_for(
                    lambda: len(provider._scheduler.pending) == 1, timeout=3
                )
            assert robot.stopping.wait(3)
        finally:
            robot.release.set()
        assert running.result(timeout=3)["timed_out"]
        assert waiting.result(timeout=3)["code"] == "tool_cancelled"
    assert _run(provider, "print('stale' in globals())")["stdout"] == "False\n"


@pytest.mark.parametrize("include_images", [False, True])
def test_python_returns_finish_without_closing_borrowed_robot(policy, include_images):
    provider, robot = policy
    provider._include_images = include_images
    assert _run(provider, "x = 1")["exception"] is None
    code = "try:\n    robo.finish()\nexcept Exception:\n    pass\nrobo.move(distance=1)"
    result = provider.execute_tool("python", {"code": code})
    assert result.data == {"_finish": True}
    assert result.images == ([b"observation"] if include_images else [])
    assert robot.observations == [{"action": "finish"}]
    assert _run(provider, "x = 2")["code"] == "tool_cancelled"
    provider.close()
    assert not robot.execute_tool("move", {"distance": 1}).is_error


@pytest.mark.parametrize("name", ["request_scene_reset", "request_operator_verdict"])
def test_operator_requests_are_only_available_as_top_level_tools(
    tmp_path, monkeypatch, name
):
    monkeypatch.setattr(logging_module, "_output_dir", tmp_path)
    robot = _BlockingToolkit(tmp_path / "robot")
    requests = []

    @tool(name=name, readonly=True)
    def request(message: str) -> ToolResult:
        requests.append(message)
        return ToolResult(data={"status": "confirmed"})

    robot.add_tool(request)
    provider = CodePolicyTools(robot, output_dir=tmp_path)
    try:
        assert {item.name for item in provider.list_tools()} == {"python", name}
        assert f"robo.{name}" not in code_policy_api_reference(provider.robot_tools())
        result = _run(provider, f"robo.{name}(message='inside Python')")
        assert result["exception"]["type"] == "AttributeError"
        top_level = provider.execute_tool(name, {"message": "top level"})
        assert top_level.data == {"status": "confirmed"}
        assert requests == ["top level"]
    finally:
        provider.close()
        robot.close()
