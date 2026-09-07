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
from types import SimpleNamespace

import pytest

from rpent.dashboard.events import StepRecordEvent
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import ToolCancelled, ToolContext, Toolkit, ToolResult, readonly, tool


@tool
@readonly
def finish(status: str, summary: str, *, ctx: ToolContext) -> ToolResult:
    """Accept the requested outcome for this test toolkit."""
    return ToolResult(data={"_finish": True, "status": status, "summary": summary})


class EventSink:
    enabled = True

    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


@tool
def action(value: int = 1, *, ctx: ToolContext) -> ToolResult:
    """Execute an action with the supplied value."""
    ctx.robot.received.append((value, ctx))
    if ctx.robot.failure:
        raise ctx.robot.failure
    return ctx.robot.result


@tool
@readonly
def inspect_state(*, ctx: ToolContext) -> ToolResult:
    """Return data without capturing another observation."""
    return ToolResult(data={"ready": True})


class RecordingToolkit(Toolkit):
    def _capture_observation(self, *, command, result, elapsed_s):
        with self.state.record_step(
            state={"position": 3},
            command=command,
            result=result.to_dict(),
            elapsed_s=elapsed_s,
        ) as step:
            return {"step": step, "position": 3}, [b"observation"]


@pytest.fixture
def toolkit(tmp_path):
    instance = RecordingToolkit(
        state=EnvState(tmp_path / "states"),
        memory=MemoryManager(tmp_path / "memory"),
        robot=SimpleNamespace(
            received=[],
            failure=None,
            result=ToolResult(data={"moved": True}, images=[b"action"]),
        ),
        output_dir=tmp_path,
        tools=(finish, action, inspect_state),
        dashboard_events=EventSink(),
    )
    yield instance
    instance.close()


def test_capture_replaces_action_data_appends_images_and_publishes_once(toolkit):
    result = toolkit.execute_tool("action", {"value": "2"})
    assert not result.is_error
    assert result.data == {"step": 0, "position": 3}
    assert result.images == [b"action", b"observation"]
    record = toolkit.state.latest_record()
    assert record.command == {"action": "action", "value": 2}
    assert record.result == {"moved": True}
    assert record.elapsed_s >= 0
    (event,) = toolkit._dashboard_events.events
    assert isinstance(event, StepRecordEvent)
    assert event.record is record
    assert event.env_state is toolkit.state
    assert toolkit.execute_tool("inspect_state", {}).data == {"ready": True}
    assert len(toolkit.state.records()) == 1
    assert len(toolkit._dashboard_events.events) == 1


@pytest.mark.parametrize("arguments", [{"value": "invalid"}, {"value": None}])
def test_validation_rejects_external_arguments_before_execution(toolkit, arguments):
    result = toolkit.execute_tool("action", arguments)
    assert result.is_error
    message, details = result.error.split("\n", 1)
    assert message == "Invalid arguments for action."
    (error,) = json.loads(details)["errors"]
    assert error["loc"] == ["value"]
    assert "input" not in error
    assert toolkit._robot.received == []
    assert toolkit.state.latest_record() is None
    assert toolkit._dashboard_events.events == []


def test_unknown_tool_does_not_execute_or_capture(toolkit):
    assert toolkit.execute_tool("missing", {}).error == "Unknown tool: missing"
    assert toolkit.state.latest_record() is None
    assert toolkit._robot.received == []


@pytest.mark.parametrize(
    "error_type", [ToolCancelled, PermissionError, TypeError, ValueError]
)
def test_handler_failure_is_bounded_logged_and_still_captured(
    toolkit, error_type, caplog
):
    message = "failure " * 100
    toolkit._robot.failure = error_type(message)
    result = toolkit.execute_tool("action", {})
    assert result.error == message[:500]
    assert result.data == {"step": 0, "position": 3}
    assert toolkit.state.latest_record().result == {"error": message[:500]}
    assert message in caplog.text
    assert not toolkit.execute_tool("inspect_state", {}).is_error


@pytest.mark.parametrize("error", [None, "", "action failed"])
def test_capture_failure_preserves_action_data_images_and_error(
    toolkit, monkeypatch, error
):
    toolkit._robot.result.error = error

    def fail(**kwargs):
        raise OSError("camera offline")

    monkeypatch.setattr(toolkit, "_capture_observation", fail)
    result = toolkit.execute_tool("action", {})
    prefix = f"{error}\n" if error is not None else ""
    assert result.error == prefix + "State capture failed: camera offline"
    assert result.data == {"moved": True}
    assert result.images == [b"action"]
    assert toolkit.state.latest_record() is None
    assert not toolkit.execute_tool("inspect_state", {}).is_error


@pytest.mark.parametrize("error", ["", "business failure"])
def test_successful_capture_retains_explicit_action_error(toolkit, error):
    toolkit._robot.result.error = error
    result = toolkit.execute_tool("action", {})
    assert result.is_error
    assert result.error == error
    assert result.data == {"step": 0, "position": 3}
    assert toolkit.state.latest_record().result == {"moved": True, "error": error}


def test_capture_failure_after_recording_still_publishes_saved_step(
    toolkit, monkeypatch
):
    capture = toolkit._capture_observation

    def fail(**kwargs):
        capture(**kwargs)
        raise OSError("image unavailable")

    monkeypatch.setattr(toolkit, "_capture_observation", fail)
    result = toolkit.execute_tool("action", {})
    assert result.error == "State capture failed: image unavailable"
    (event,) = toolkit._dashboard_events.events
    assert event.record is toolkit.state.latest_record()


def test_dashboard_failure_does_not_retry_or_change_result(
    toolkit, monkeypatch, caplog
):
    calls = []

    def fail(event):
        calls.append(event)
        raise RuntimeError("dashboard offline")

    monkeypatch.setattr(toolkit._dashboard_events, "emit", fail)
    result = toolkit.execute_tool("action", {})
    assert not result.is_error
    assert len(calls) == len(toolkit.state.records()) == 1
    assert "dashboard offline" in caplog.text


def test_base_exception_propagates_and_releases_admission(toolkit):
    toolkit._robot.failure = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        toolkit.execute_tool("action", {})
    assert not toolkit.execute_tool("inspect_state", {}).is_error


def test_context_is_fresh_and_cannot_be_replaced_by_external_arguments(toolkit):
    for _ in range(2):
        toolkit.execute_tool("action", {"ctx": "untrusted", "value": "3"})
    first, second = [ctx for value, ctx in toolkit._robot.received]
    assert first is not second
    assert first._cancel_event is not second._cancel_event
    for ctx in (first, second):
        assert ctx.robot is toolkit._robot
        assert ctx.state is toolkit.state
        assert ctx.memory is toolkit.memory
        assert ctx.output_dir == toolkit._task_output_dir
        assert ctx.record_frame == toolkit.record_frame
