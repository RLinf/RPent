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

"""Task capture outcomes at runner cleanup boundaries."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from pydantic_ai.models.test import TestModel

from rpent.cli.main import _build_argparser
from rpent.cli.task_capture import end_api_task, start_api_task
from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.api_loop import ApiAgentLoop
from rpent.planner.base import PlannerResult
from rpent.robots.robot_spec import RunConfig
from rpent.session import EnvState, Payload


@pytest.mark.parametrize("record_trajectory", [False, True])
def test_api_trajectory_capture_requires_explicit_cli_opt_in(
    tmp_path, record_trajectory
):
    args = _build_argparser().parse_args(
        ["--record-trajectory"] if record_trajectory else []
    )
    state = EnvState(tmp_path)
    toolkit = SimpleNamespace(
        state=state,
        list_tools=lambda: (),
        resume_calls=lambda: None,
        cancel_active_and_wait=lambda: None,
    )
    config = RunConfig("test", tmp_path, {}, {"suite": "libero_object", "task": 3})
    captured = start_api_task(toolkit, args, config, "Move the cup.")
    assert (captured is state) is record_trajectory
    args.max_tokens = 1
    result = ApiAgentLoop(
        model=TestModel(call_tools=[], custom_output_text="Done."),
        dashboard_events=NullDashboardEventSink(),
    ).solve(
        system_prompt="Complete the task.",
        user_message="Move the cup.",
        toolkit=toolkit,
        max_turns=1,
    )
    assert result.error is None
    assert result.stop_reason == "model_output"
    assert result.stats["turns_used"] == 1
    end_api_task(
        captured,
        result,
        finish_result=result.finish_result,
        environment_success=False,
        exception=None,
    )
    if record_trajectory:
        task = state.get_task()
        assert task.status == "completed" and task.stop_reason == "model_output"
        assert task.instruction == "Move the cup."
        assert task.config.data["arguments"]["record_trajectory"] is True
        assert task.config.data["arguments"]["max_tokens"] == 8192
        assert task.config.data["run"]["output_dir"] == str(tmp_path)
        root = tmp_path / "trajectories" / task.task_id
        planner = json.loads((root / "planners.jsonl").read_text())
        assert planner["status"] == "completed"
        assert planner["request"] is not None and planner["response"] is not None
        assert (
            json.loads((root / "task.json").read_text())["task"]["task_id"]
            == task.task_id
        )
    else:
        assert state.get_task() is None
        assert state.planner_records() == []
        assert not (tmp_path / "trajectories").exists()


@pytest.mark.parametrize("planner", ["claude_code", "codex", "flash"])
def test_recording_flag_does_not_enable_capture_for_other_planners(tmp_path, planner):
    args = _build_argparser().parse_args(["--planner", planner, "--record-trajectory"])
    state = EnvState(tmp_path)
    config = RunConfig("test", tmp_path, {}, {})
    captured = start_api_task(SimpleNamespace(state=state), args, config, "Move.")
    assert captured is None
    assert state.get_task() is None
    end_api_task(
        captured,
        None,
        finish_result=None,
        environment_success=None,
        exception=RuntimeError("planner failed"),
    )
    assert not (tmp_path / "trajectories").exists()


@pytest.mark.parametrize(
    "exception,cancelled_reason,expected_status,expected_reason",
    [
        (None, None, "failed", "timeout"),
        (KeyboardInterrupt(), None, "cancelled", "operator"),
        (RuntimeError("cleanup failed"), None, "failed", "execution_error"),
        (None, "task_replaced", "cancelled", "task_replaced"),
    ],
)
def test_task_end_keeps_partial_records_and_actual_stop_cause(
    tmp_path, exception, cancelled_reason, expected_status, expected_reason
):
    state = EnvState(tmp_path)
    state.start_task(instruction="Move the cup.", config=Payload("test.config", {}))
    planner_id = state.register_planner(
        backend="api",
        status="completed",
        request=Payload("test", {"messages": ["Move the cup."]}),
    )
    tool_id = state.register_tool_call(
        planner_id=planner_id, name="move", arguments=Payload("json", {})
    )
    result = PlannerResult(
        error="planner timed out", stop_reason="timeout", stats={"turns_used": 1}
    )
    end_api_task(
        state,
        result,
        finish_result=None,
        environment_success=False,
        exception=exception,
        cancelled_reason=cancelled_reason,
    )
    task = state.get_task()
    assert (task.status, task.stop_reason) == (expected_status, expected_reason)
    assert task.result.data["environment_success"] is False
    assert task.ended_at >= task.started_at
    assert state.get_planner(planner_id).status == "completed"
    root = tmp_path / "trajectories" / task.task_id
    saved_task = json.loads((root / "task.json").read_text())["task"]
    assert (saved_task["status"], saved_task["stop_reason"]) == (
        expected_status,
        expected_reason,
    )
    assert saved_task["result"]["data"] == task.result.data
    planner = json.loads((root / "planners.jsonl").read_text())
    assert planner["planner_id"] == planner_id
    assert planner["status"] == "completed"
    assert planner["tool_calls"][0]["tool_call_id"] == tool_id
    assert planner["tool_calls"][0]["status"] == "requested"


@pytest.mark.parametrize(
    ("verdict", "status", "reason"),
    [
        ("operator_finished", "completed", "operator_finished"),
        ("operator_aborted", "cancelled", "operator"),
    ],
)
def test_operator_verdict_is_separate_from_planner_error(
    tmp_path, verdict, status, reason
):
    state = EnvState(tmp_path)
    state.start_task(instruction="Task", config=Payload("test", {}))
    result = PlannerResult(error="interrupted model", stop_reason="planner_error")
    end_api_task(
        state,
        result,
        finish_result={verdict: True},
        environment_success=True,
        exception=None,
    )
    task = state.get_task()
    assert task.status == status and task.stop_reason == reason
    assert task.result.data["error"] == "interrupted model"
    root = tmp_path / "trajectories" / task.task_id
    assert json.loads((root / "task.json").read_text())["task"]["status"] == status


def test_save_failure_is_logged_without_replacing_execution_error(
    tmp_path, monkeypatch, caplog
):
    state = EnvState(tmp_path)
    state.start_task(instruction="Task", config=Payload("test", {}))
    monkeypatch.setattr(logging.getLogger("rpent"), "propagate", True)

    def fail_save():
        raise OSError("disk full")

    monkeypatch.setattr(state, "save_planner_trajectory", fail_save)
    error = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed") as caught:
        try:
            raise error
        finally:
            end_api_task(
                state,
                None,
                finish_result=None,
                environment_success=None,
                exception=error,
            )
    assert caught.value is error
    assert state.get_task().status == "failed"
    assert state.get_task().result.data["error"] == "RuntimeError: model failed"
    assert "Failed to save planner trajectory" in caplog.text
    assert "disk full" in caplog.text
