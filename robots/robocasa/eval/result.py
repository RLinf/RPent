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

RESULT_SCHEMA_VERSION = "1.0"
TARGET50_MANIFEST = Path(__file__).with_name("target50.json")


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
) -> dict[str, Any]:
    """Build one sanitized, environment-authoritative cell result."""
    protocol_id, evaluation_split = _target50_identity(
        task_name, environment_split, seed
    )
    reason = _termination_reason(agent_error, success)
    max_chunks = int(os.environ.get("RLDX_MAX_CHUNKS", "40"))
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
    }


def write_cell_result(output_dir: Path | str, **kwargs: Any) -> Path:
    """Atomically write ``result.json`` and return its path."""
    destination = Path(output_dir) / "result.json"
    temporary = destination.with_name(".result.json.tmp")
    record = build_cell_result(**kwargs)
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)
            file.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
