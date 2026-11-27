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

import pytest
from pydantic_core import to_jsonable_python

from rpent.planner.base import PlannerResult
from rpent.robots import RunConfig
from rpent.session import EnvState, Payload
from rpent.session.planner_capture import end_planner_task, task_config_snapshot
from rpent.utils.mcp_servers import McpServerSpec


@pytest.mark.parametrize(
    "exception,cancelled_reason,expected_status,expected_reason",
    [
        (None, None, "completed", "timeout"),
        (KeyboardInterrupt(), None, "cancelled", "operator"),
        (RuntimeError("cleanup failed"), None, "failed", "timeout"),
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
@pytest.mark.parametrize("confirmed", [True, False])
def test_operator_verdict_is_separate_from_planner_error(
    tmp_path, verdict, status, reason, confirmed
):
    state = EnvState(tmp_path)
    state.start_task(instruction="Task", config=Payload("test", {}))
    result = PlannerResult(
        stop_reason="cancelled" if confirmed else "planner_error",
        error=None if confirmed else "interrupted model",
        stats={"interruption_diagnostics": ["interrupted model"]} if confirmed else {},
    )
    end_planner_task(
        state,
        result,
        finish_result={verdict: True},
        environment_success=True,
        exception=None,
    )
    task = state.get_task()
    if confirmed:
        assert task.result.data["stats"]["interruption_diagnostics"] == [
            "interrupted model"
        ]
    else:
        status, reason = "failed", "planner_error"
    assert task.status == status and task.stop_reason == reason
    assert task.result.data["error"] == result.error
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


def test_task_config_snapshot_redacts_mcp_header_values(tmp_path) -> None:
    spec = McpServerSpec(
        name="facility",
        url="https://facility.example/mcp?token=secret",
        headers={"Authorization": "Bearer secret"},
        tools=("read_power_meter",),
        sync_with_robot=True,
    )
    run_config = RunConfig(
        recipe_tag="run",
        output_dir=tmp_path,
        prompt_vars={},
        task_desc={},
        mcp_servers=(spec,),
    )

    config = task_config_snapshot(
        {"planner": "api", "shared_mcp_servers": (spec,)}, run_config
    )

    dumped = json.dumps(to_jsonable_python(config))
    assert "Bearer secret" not in dumped
    assert "token=secret" not in dumped
    redacted = {"Authorization": "<redacted>"}
    assert config["run"]["mcp_servers"][0]["headers"] == redacted
    assert config["arguments"]["shared_mcp_servers"][0]["headers"] == redacted
    assert config["run"]["mcp_servers"][0]["url"] == (
        "https://facility.example/mcp?<redacted>"
    )
    assert config["run"]["mcp_servers"][0]["tools"] == ("read_power_meter",)
    assert config["run"]["mcp_servers"][0]["sync_with_robot"] is True

    bare = task_config_snapshot(
        {"planner": "api", "shared_mcp_servers": ()},
        RunConfig(
            recipe_tag="run",
            output_dir=tmp_path,
            prompt_vars={},
            task_desc={},
        ),
    )
    assert "shared_mcp_servers" not in bare["arguments"]
    assert "mcp_servers" not in bare["run"]


@pytest.mark.parametrize(
    "boundary,error,cancelled_reason,verdict,status,reason",
    [
        (
            {"reason": "provider_limit", "category": "error"},
            None,
            None,
            None,
            "failed",
            "provider_limit",
        ),
        (
            {"reason": "operator_pause", "category": "cancelled"},
            None,
            None,
            None,
            "cancelled",
            "operator_pause",
        ),
        (
            {"reason": "step_budget", "category": "budget"},
            None,
            None,
            None,
            "completed",
            "step_budget",
        ),
        (
            {"reason": "finish", "category": "finish"},
            None,
            "task_replaced",
            None,
            "cancelled",
            "task_replaced",
        ),
        (
            {"reason": "provider_limit", "category": "error"},
            None,
            "operator",
            None,
            "failed",
            "provider_limit",
        ),
        (
            {"reason": "finish", "category": "finish"},
            "cleanup failed",
            None,
            None,
            "failed",
            "finish",
        ),
        (
            {"reason": "provider_limit", "category": "error"},
            None,
            None,
            "success",
            "failed",
            "provider_limit",
        ),
        (
            {"reason": "finish", "category": "finish"},
            None,
            None,
            "abort",
            "cancelled",
            "operator",
        ),
    ],
)
def test_structured_stop_priority_matches_finalization(
    tmp_path, boundary, error, cancelled_reason, verdict, status, reason
):
    from robots.libero.result import finalize_libero_result
    from rpent.evaluation import RunFinalizationContext

    state = EnvState(tmp_path)
    state.start_task(instruction="Task", config=Payload("test", {}))
    result = PlannerResult(stop_info=boundary, stop_reason="stale_reason", error=error)
    finish = (
        {"operator_finished": True, "operator_verdict": verdict} if verdict else None
    )
    end_planner_task(
        state,
        result,
        finish_result=finish,
        environment_success=False,
        exception=None,
        cancelled_reason=cancelled_reason,
    )
    task = state.get_task()
    assert (task.status, task.stop_reason) == (status, reason)
    record = json.loads(
        finalize_libero_result(
            RunFinalizationContext(
                output_dir=tmp_path,
                robot_name="libero",
                task_desc={},
                environment_success=False,
                agent_error=error,
                elapsed_s=1,
                planner="api",
                model=None,
                reasoning_effort="high",
                max_turns=1,
                planner_timeout_s=1,
                finish_result=finish,
                stats={},
                stop_info=boundary,
                stop_reason="stale_reason",
                cancelled_reason=cancelled_reason,
            )
        ).read_text()
    )
    assert record["termination_reason"] == task.stop_reason
    assert record["termination_category"] == task.result.data["stop_info"]["category"]


def test_runner_cleanup_diagnostic_is_not_duplicated(tmp_path):
    state = EnvState(tmp_path)
    state.start_task(instruction="Task", config=Payload("test", {}))
    diagnostic = "Planner timed out; RuntimeError: cleanup failed"
    end_planner_task(
        state,
        PlannerResult(error=diagnostic, stop_reason="timeout"),
        finish_result=None,
        environment_success=False,
        exception=RuntimeError("cleanup failed"),
    )
    assert state.get_task().result.data["error"] == diagnostic
    assert state.get_task().result.data["stop_info"]["category"] == "error"
