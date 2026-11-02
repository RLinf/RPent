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

"""LIBERO results with independent environment, planner, and operator outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rpent.evaluation import RunFinalizationContext, write_json_atomic


def finalize_libero_result(context: RunFinalizationContext) -> Path:
    """Persist native success without inferring it from finish or operator text.

    Args:
        context: Runner evidence captured before toolkit and runtime cleanup.

    Returns:
        Path to the atomically written result.json artifact.
    """
    finish = context.finish_result or {}
    accepted = finish.get("_finish") is True
    status = finish.get("status")
    if status not in ("success", "failure", "stuck"):
        status = "other" if status is not None else None
    verdict = finish.get("operator_verdict")
    if finish.get("operator_finished") is not True or verdict not in (
        "success",
        "failure",
        "abort",
    ):
        verdict = None
    stop_reason = context.stats.get("stop_reason")
    known_reasons = {
        "finish",
        "budget",
        "timeout",
        "interrupted",
        "runtime_error",
        "missing_finish",
        "model_mismatch",
        "tool_transport_error",
        "image_pipeline_error",
        "compaction_error",
        "control_error",
        "host_closed",
        "task_replaced",
    }
    if stop_reason not in known_reasons:
        stop_reason = (
            "planner_error"
            if context.agent_error
            else ("finish" if accepted else "stopped")
        )
    boundary = context.stats.get("stop_info")
    stop_info = None
    if isinstance(boundary, Mapping):
        stop_info = {
            "reason": stop_reason,
            **{
                name: boundary.get(name)
                if isinstance(boundary.get(name), bool)
                else None
                for name in (
                    "execution_certain",
                    "result_delivery_complete",
                    "cleanup_complete",
                    "allow_new_session",
                    "process_interrupted",
                )
            },
        }
    return write_json_atomic(
        context.output_dir / "result.json",
        {
            "schema_version": "1.0",
            "robot": "libero",
            "task": {
                name: context.task_desc.get(name) for name in ("suite", "task", "seed")
            },
            "success": context.environment_success,
            "environment_success": context.environment_success,
            "environment_result_available": context.environment_success is not None,
            "success_source": "native_terminated",
            "valid": context.environment_success is not None
            and context.agent_error is None,
            "finish_accepted": accepted,
            "finish_status": status,
            "operator_verdict": verdict,
            "planner_error": context.agent_error is not None,
            "termination_reason": stop_reason,
            "stop_info": stop_info,
            "elapsed_s": round(context.elapsed_s, 3),
            "planner": {
                "backend": context.planner,
                "model": context.model,
                "reasoning_effort": context.reasoning_effort,
                "max_turns": context.max_turns,
                "timeout_s": context.planner_timeout_s,
            },
            "evidence": {
                "transcripts": sorted(
                    path.name
                    for path in context.output_dir.glob("transcript*.json")
                    if path.is_file()
                ),
                "memory_metadata": "memory.json"
                if (context.output_dir / "memory.json").is_file()
                else None,
                "memory_source": "memory-source.json"
                if (context.output_dir / "memory-source.json").is_file()
                else None,
            },
        },
    )
