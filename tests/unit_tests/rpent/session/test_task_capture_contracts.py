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

"""Planner task capture snapshots and outcomes after execution cleanup."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Annotated, Literal

import pytest
from pydantic import Field

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.base import PlannerResult
from rpent.session import EnvState, Payload
from rpent.session.planner_capture import end_planner_task, start_planner_task
from rpent.tools import Toolkit, ToolResult, tool


@pytest.mark.parametrize(
    "system_prompt", ["Use tools.\nPreserve world coordinates.", ""]
)
def test_task_start_saves_prompts_and_tools_without_mutating_sources(
    tmp_path, monkeypatch, system_prompt
):
    """Verify that task capture takes an independent snapshot of the prompt and tools."""
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    state = EnvState(tmp_path)
    toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        state=state,
        memory=MemoryManager(tmp_path / "memory"),
    )

    @tool
    def move(
        row: Annotated[int, Field(ge=0, description="Pixel row in {{output_dir}}")],
        camera: Annotated[
            Literal["wrist", "agentview"], Field(json_schema_extra={"default": "wrist"})
        ] = "wrist",
    ) -> ToolResult:
        """Move using the camera in {{output_dir}}."""
        raise AssertionError("Capturing tool definitions must not execute tools")

    toolkit.add_tool(move)
    definitions = toolkit.list_tools()
    config = {"arguments": {"robot_name": "test"}, "run": {"output_dir": tmp_path}}
    start_planner_task(
        state,
        instruction="Move",
        system_prompt=system_prompt,
        config=config,
        tools=definitions,
    )

    assert "tools" not in config
    assert "system_prompt" not in config
    assert "{{output_dir}}" in move.input_schema["properties"]["row"]["description"]
    toolkit.add_tool(
        replace(move, description="Changed after task start"), replace=True
    )
    move._input_schema["properties"]["row"]["minimum"] = 99
    config["arguments"]["robot_name"] = "changed"

    state.end_task(status="completed", stop_reason="model_output")
    root = state.save_planner_trajectory()
    document = json.loads((root / "task.json").read_text())
    saved = document["task"]["config"]["data"]
    assert document["schema_version"] == 5
    assert document["task"]["instruction"] == "Move"
    assert saved["system_prompt"] == system_prompt
    assert saved["arguments"]["robot_name"] == "test"
    assert saved["run"]["output_dir"] == str(tmp_path)
    assert [item["name"] for item in saved["tools"]] == [d.name for d in definitions]
    by_name = {item["name"]: item for item in saved["tools"]}
    assert by_name["move"]["description"] == f"Move using the camera in {tmp_path}."
    assert by_name["move"]["readonly"] is False
    schema = by_name["move"]["input_schema"]
    assert schema["required"] == ["row"]
    assert schema["properties"]["row"]["type"] == "integer"
    assert schema["properties"]["row"]["minimum"] == 0
    assert schema["properties"]["row"]["description"] == f"Pixel row in {tmp_path}"
    assert schema["properties"]["camera"]["enum"] == ["wrist", "agentview"]
    assert schema["properties"]["camera"]["default"] == "wrist"
    assert by_name["list_dir"]["readonly"] is True
    assert "{{output_dir}}" not in json.dumps(saved["tools"])


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
    end_planner_task(
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
    end_planner_task(
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
            end_planner_task(
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
