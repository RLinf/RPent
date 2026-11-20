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

"""Native success stays independent of planner and operator outcomes."""

import json
from dataclasses import replace

import pytest

from robots.libero.result import finalize_libero_result
from robots.libero.robot_spec import get_robot_spec
from rpent.evaluation import RunFinalizationContext


@pytest.fixture
def context(tmp_path):
    return RunFinalizationContext(
        output_dir=tmp_path,
        robot_name="libero",
        task_desc={"suite": "libero_10_task", "task": 0, "seed": 2},
        environment_success=False,
        agent_error=None,
        elapsed_s=12.5,
        planner="deepseek_harness",
        model="gpt-6.1-sol",
        reasoning_effort="xhigh",
        max_turns=100,
        planner_timeout_s=1200,
        finish_result={
            "_finish": True,
            "status": "success",
            "summary": "claimed success",
        },
        stats={"stop_reason": "finish"},
    )


def test_libero_registers_result_finalizer():
    assert get_robot_spec().finalize_run is finalize_libero_result


@pytest.mark.parametrize("native", [True, False, None])
def test_finish_claim_cannot_replace_native_success(context, native):
    path = finalize_libero_result(replace(context, environment_success=native))
    result = json.loads(path.read_text())
    assert result["environment_success"] is native
    assert result["success"] is native
    assert result["environment_result_available"] is (native is not None)
    assert result["success_source"] == (
        "native_terminated" if native is not None else None
    )
    assert result["finish_accepted"] is True
    assert result["finish_status"] == "success"
    assert result["operator_verdict"] is None
    assert result["valid"] is (native is not None)
    assert not (path.parent / ".result.json.tmp").exists()


def test_operator_abort_and_planner_error_preserve_native_evidence(context):
    result = json.loads(
        finalize_libero_result(
            replace(
                context,
                environment_success=True,
                agent_error="private provider diagnostic",
                finish_result={
                    "operator_finished": True,
                    "operator_verdict": "abort",
                    "status": "failure",
                },
                stats={"stop_reason": "interrupted"},
            )
        ).read_text()
    )
    assert result["success"] is True
    assert result["finish_accepted"] is False
    assert result["finish_status"] == "failure"
    assert result["operator_verdict"] == "abort"
    assert result["planner_error"] is True
    assert result["valid"] is False
    assert result["termination_reason"] == "interrupted"
    assert result["termination_category"] == "error"
    assert "private provider diagnostic" not in json.dumps(result)


@pytest.mark.parametrize(
    "reason,category",
    [("completed", "completed"), ("model_output", "completed"), ("budget", "budget")],
)
@pytest.mark.parametrize("native", [True, False])
def test_determinate_nonfinish_attempt_counts_with_its_native_outcome(
    context, reason, category, native
):
    record = json.loads(
        finalize_libero_result(
            replace(
                context,
                environment_success=native,
                finish_result=None,
                stats={},
                stop_reason=reason,
            )
        ).read_text()
    )
    assert record["valid"] is True
    assert record["success"] is native
    assert record["finish_accepted"] is False
    assert record["termination_category"] == category


def test_missing_finish_and_unconfirmed_operator_claim_remain_separate(context):
    result = json.loads(
        finalize_libero_result(
            replace(
                context,
                finish_result={"status": "success", "operator_verdict": "success"},
                stats={},
            )
        ).read_text()
    )
    assert result["finish_accepted"] is False
    assert result["operator_verdict"] is None
    assert result["termination_reason"] == "stopped"
    assert result["stop_info"] is None


@pytest.mark.parametrize(
    "reason", ["control_error", "host_closed", "task_replaced", "compaction_error"]
)
def test_result_preserves_control_reason_and_delivery_boundary(context, reason):
    boundary = {
        "execution_certain": False,
        "result_delivery_complete": False,
        "cleanup_complete": True,
        "allow_new_session": False,
        "process_interrupted": True,
        "pending_tool_call_ids": ["private-runtime-id"],
    }
    result = json.loads(
        finalize_libero_result(
            replace(context, stats={"stop_reason": reason, "stop_info": boundary})
        ).read_text()
    )
    assert result["termination_reason"] == reason
    assert result["stop_info"] == {
        "reason": reason,
        "category": "unknown",
        "execution_certain": False,
        "result_delivery_complete": False,
        "cleanup_complete": True,
        "allow_new_session": False,
        "process_interrupted": True,
    }
    assert "private-runtime-id" not in json.dumps(result)


@pytest.mark.parametrize("planner", ["api", "claude_code", "codex", "flash"])
def test_unknown_backend_reason_is_retained_even_with_accepted_finish(context, planner):
    result = json.loads(
        finalize_libero_result(
            replace(
                context,
                planner=planner,
                stats={"stop_reason": "provider_specific_stop"},
            )
        ).read_text()
    )
    assert result["termination_reason"] == "provider_specific_stop"
    assert result["termination_category"] == "unknown"
    assert result["finish_accepted"] is True


def test_explicit_category_and_raw_boundary_reason_take_precedence(context):
    result = json.loads(
        finalize_libero_result(
            replace(
                context,
                stats={
                    "stop_reason": "stale_reason",
                    "stop_info": {
                        "reason": "provider_limit",
                        "category": "error",
                        "execution_certain": True,
                        "allow_new_session": False,
                    },
                },
            )
        ).read_text()
    )
    assert result["termination_reason"] == "provider_limit"
    assert result["termination_category"] == "error"
    assert result["stop_info"]["allow_new_session"] is False
    assert result["planner_error"] is True
    assert result["valid"] is False


def test_context_stop_info_takes_precedence_over_legacy_stats(context):
    result = json.loads(
        finalize_libero_result(
            replace(
                context,
                stop_info={"reason": "operator_pause", "category": "cancelled"},
                stop_reason="timeout",
                stats={
                    "stop_reason": "finish",
                    "stop_info": {"reason": "finish", "category": "finish"},
                },
            )
        ).read_text()
    )
    assert result["termination_reason"] == "operator_pause"
    assert result["termination_category"] == "cancelled"


@pytest.mark.parametrize(
    "reason,category,error",
    [
        ("request_limit", "budget", None),
        ("timeout", "timeout", "Planner timed out"),
        ("cancelled", "cancelled", None),
    ],
)
def test_standalone_reason_precedes_legacy_stats_without_inventing_safety(
    context, reason, category, error
):
    result = json.loads(
        finalize_libero_result(
            replace(context, stop_reason=reason, agent_error=error)
        ).read_text()
    )
    assert result["termination_reason"] == reason
    assert result["termination_category"] == category
    assert result["stop_info"] is None


@pytest.mark.parametrize(
    "reason,category,error,valid,planner_error",
    [
        ("finish", "finish", None, True, False),
        ("request_limit", "budget", None, True, False),
        ("timeout", "timeout", "Planner timed out", True, False),
        ("timeout", "timeout", None, True, False),
        ("provider_timeout", "error", "TIMEOUT", False, True),
        ("provider_transport", "error", "fetch failed", False, True),
        ("provider_limit", "error", None, False, True),
        ("cancelled", "cancelled", None, False, False),
        ("future_reason", "unknown", None, False, False),
        ("finish", "finish", "unexpected failure", False, True),
        ("timeout", "error", "cleanup failed", False, True),
    ],
)
def test_validity_uses_category_and_keeps_success_independent(
    context, reason, category, error, valid, planner_error
):
    boundary = {
        "reason": reason,
        "category": category,
        "execution_certain": True,
        "result_delivery_complete": True,
        "cleanup_complete": True,
        "allow_new_session": False,
    }
    result = json.loads(
        finalize_libero_result(
            replace(context, stop_info=boundary, agent_error=error)
        ).read_text()
    )
    assert result["valid"] is valid
    assert result["planner_error"] is planner_error
    assert result["environment_success"] is False
    assert result["finish_accepted"] is True
    assert result["termination_reason"] == reason
    assert boundary["category"] == category


@pytest.mark.parametrize(
    "field", ["execution_certain", "result_delivery_complete", "cleanup_complete"]
)
@pytest.mark.parametrize("value", [False, None, 1, "true", "missing"])
def test_incomplete_boundary_excludes_otherwise_normal_results(context, field, value):
    boundary = dict.fromkeys(
        ["execution_certain", "result_delivery_complete", "cleanup_complete"], True
    )
    if value == "missing":
        del boundary[field]
    else:
        boundary[field] = value
    result = json.loads(
        finalize_libero_result(replace(context, stop_info=boundary)).read_text()
    )
    assert result["valid"] is False
    assert result["environment_success"] is False


@pytest.mark.parametrize("reason", ["task_replaced", "operator", "shutdown"])
def test_runner_cancellation_supersedes_old_finish(context, reason):
    result = json.loads(
        finalize_libero_result(replace(context, cancelled_reason=reason)).read_text()
    )
    assert result["valid"] is False
    assert result["planner_error"] is False
    assert result["termination_category"] == "cancelled"
    assert result["termination_reason"] == reason


@pytest.mark.parametrize(
    "verdict,valid", [("success", True), ("failure", True), ("abort", False)]
)
def test_operator_verdict_is_not_native_success(context, verdict, valid):
    result = json.loads(
        finalize_libero_result(
            replace(
                context,
                finish_result={"operator_finished": True, "operator_verdict": verdict},
            )
        ).read_text()
    )
    assert result["valid"] is valid
    assert result["operator_verdict"] == verdict
    assert result["finish_accepted"] is False
    assert result["success"] is False


def test_flash_completion_has_known_reason_without_fabricating_finish(
    context, monkeypatch
):
    from types import SimpleNamespace

    from rpent.planner import flash

    monkeypatch.setattr(
        flash,
        "get_robot_spec",
        lambda _: SimpleNamespace(run_flash=lambda *args: {"done": False, "plan": 2}),
    )
    planner = flash.FlashPlanner(recipe_tag="test", robot_name="libero")
    outcome = planner.solve(
        system_prompt="", user_message="", toolkit=None, max_turns=1
    )
    result = json.loads(
        finalize_libero_result(
            replace(
                context,
                planner="flash",
                finish_result=outcome.finish_result,
                stop_reason=outcome.stop_reason,
                stats=outcome.stats,
            )
        ).read_text()
    )
    assert result["valid"] is True
    assert result["termination_category"] == "finish"
    assert result["finish_accepted"] is False
