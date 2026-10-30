# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Persistent Python execution and cancellation across a real child process."""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rpent.code_policy.runtime import CodeSession
from rpent.code_policy.toolkit import CodePolicyTools
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.prompt.code_policy import code_policy_api_reference
from rpent.session import EnvState
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


def test_python_delegates_actions_and_observes_each_primitive_once(policy):
    provider, robot = policy
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


def test_invalid_python_arguments_do_not_start_a_cell(policy):
    provider, robot = policy
    for arguments in ({}, {"code": 42}, {"code": "x = 1", "extra": True}):
        result = provider.execute_tool("python", arguments)
        assert result.is_error
        assert result.data["errors"]
    assert _run(provider, "print(42)")["cell"] == 1
    assert robot.observations == []


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
