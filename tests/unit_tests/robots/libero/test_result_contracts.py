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
    assert "private provider diagnostic" not in json.dumps(result)


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
        "execution_certain": False,
        "result_delivery_complete": False,
        "cleanup_complete": True,
        "allow_new_session": False,
        "process_interrupted": True,
    }
    assert "private-runtime-id" not in json.dumps(result)
