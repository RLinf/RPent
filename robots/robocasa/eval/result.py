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

"""Machine-readable result records for RoboCasa evaluation cells."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rpent.evaluation import RunFinalizationContext, write_json_atomic

RESULT_SCHEMA_VERSION = "1.1"
TARGET50_MANIFEST = Path(__file__).with_name("target50_v2.json")


def _target50_identity(
    task_name: str, environment_split: str, seed: int
) -> tuple[str | None, str | None]:
    """Return ``(protocol_id, evaluation_split)`` for a Target50 cell."""
    if environment_split != "target":
        return None, None
    manifest = json.loads(TARGET50_MANIFEST.read_text(encoding="utf-8"))
    for split_name, split in manifest["splits"].items():
        if task_name in split["tasks"] and seed in split["seeds"]:
            return manifest["protocol_id"], split_name
    return None, None


def _termination_reason(agent_error: str | None, success: bool) -> str:
    """Classify completion without persisting provider error details."""
    if success or not agent_error:
        return "completed"
    lowered = agent_error.lower()
    if "timed out" in lowered and (
        "planner" in lowered or "agent sdk" in lowered or "codex sdk" in lowered
    ):
        return "planner_timeout"
    return "infrastructure_error"


def build_cell_result(
    *,
    task_name: str,
    environment_split: str,
    seed: int,
    success: bool,
    environment_result_available: bool,
    agent_error: str | None,
    elapsed_s: float,
    planner: str,
    model: str | None,
    reasoning_effort: str,
    max_turns: int,
    cell_timeout_seconds: int | None,
    memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one sanitized, environment-authoritative cell result."""
    protocol_id, evaluation_split = _target50_identity(
        task_name, environment_split, seed
    )
    reason = _termination_reason(agent_error, success)
    memory = memory or {}
    max_chunks = int(os.environ.get("RLDX_MAX_CHUNKS", "70"))
    settle_patience = int(os.environ.get("RLDX_SETTLE_PATIENCE", "999"))
    action_steps = int(os.environ.get("RLDX_ACTION_STEPS_PER_CHUNK", "8"))
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "evaluation_split": evaluation_split,
        "task_name": task_name,
        "environment_split": environment_split,
        "seed": seed,
        "valid": environment_result_available and reason != "infrastructure_error",
        "success": bool(success),
        "success_source": "state.success",
        "termination_reason": reason,
        "elapsed_s": round(elapsed_s, 1),
        "planner": {
            "backend": planner,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_turns": max_turns,
        },
        "runtime": {
            "cell_timeout_seconds": cell_timeout_seconds,
            "rldx_max_chunks": max_chunks,
            "rldx_settle_patience": settle_patience,
            "rldx_action_steps_per_chunk": action_steps,
        },
        "memory": memory,
    }


def finalize_cell_result(context: RunFinalizationContext) -> Path:
    """Adapt shared run state to the RoboCasa Target50 result schema."""
    task = context.task_desc
    memory = {}
    try:
        metadata = json.loads((context.output_dir / "memory.json").read_text())
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("selected_files"), list)
            or not all(isinstance(name, str) for name in metadata["selected_files"])
        ):
            raise ValueError("invalid memory metadata")
        memory = metadata
        # Only this run's read log can establish a completed read.
        memory.pop("read_files", None)
        reads = [
            json.loads(line)
            for line in (context.output_dir / "memory_reads.jsonl")
            .read_text()
            .splitlines()
        ]
        if any(
            not isinstance(event, dict)
            or not isinstance(event.get("path"), str)
            or not isinstance(event.get("complete"), bool)
            or event["path"] not in memory["selected_files"]
            for event in reads
        ):
            raise ValueError("invalid memory read audit")
        memory["read_files"] = sorted(
            {event["path"] for event in reads if event["complete"]}
        )
        memory["audit_status"] = "ok"
    except FileNotFoundError:
        memory["audit_status"] = "missing"
    except (OSError, ValueError):
        memory["audit_status"] = "invalid"
    record = build_cell_result(
        task_name=str(task["task_name"]),
        environment_split=str(task["split"]),
        seed=int(task["seed"]),
        success=bool(context.environment_success),
        environment_result_available=context.environment_success is not None,
        agent_error=context.agent_error,
        elapsed_s=context.elapsed_s,
        planner=context.planner,
        model=context.model,
        reasoning_effort=context.reasoning_effort,
        max_turns=context.max_turns,
        cell_timeout_seconds=context.planner_timeout_s,
        memory=memory,
    )
    return write_json_atomic(context.output_dir / "result.json", record)
