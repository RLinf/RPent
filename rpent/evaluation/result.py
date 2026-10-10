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

"""Shared contracts for robot-specific run-result artifacts."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpent.utils.logging import get_logger


@dataclass(frozen=True)
class RunFinalizationContext:
    """Structured runner state supplied to a robot result finalizer."""

    output_dir: Path
    robot_name: str
    task_desc: Mapping[str, Any]
    environment_success: bool | None
    agent_error: str | None
    elapsed_s: float
    planner: str
    model: str | None
    reasoning_effort: str
    max_turns: int
    planner_timeout_s: int | None
    finish_result: Mapping[str, Any] | None
    stats: Mapping[str, Any]


RunFinalizer = Callable[[RunFinalizationContext], Path | str | None]


def record_operator_outcome(
    *,
    memory_root: Path,
    recipe_tag: str,
    run_name: str,
    session_number: int,
    state_output_dir: str | Path,
    verdict: dict[str, Any],
    messages: list[dict[str, Any]],
    planner_error: str | None,
) -> Path:
    """Archive an operator outcome without overwriting a previous attempt.

    Args:
        memory_root: Root of the run's memory store.
        recipe_tag: Cell whose inbox receives the evidence.
        run_name: Run identifier used to distinguish evidence files.
        session_number: Attempt number within the run.
        state_output_dir: Directory containing the attempt's observations.
        verdict: Operator-confirmed result and notes.
        messages: Serializable planner transcript.
        planner_error: Error recorded before operator finalization, if any.

    Returns:
        Path to the archived evidence JSON.
    """
    outcome = verdict["operator_verdict"]
    path = memory_root / "_internal" / "inbox" / recipe_tag / "wip"
    path /= f"{run_name}-session-{session_number:03d}-{outcome}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(
            {
                "verdict": verdict,
                "session": session_number,
                "evidence_dir": str(state_output_dir),
                "messages": messages,
                "planner_error": planner_error,
                "note": "Operator-confirmed outcome; explanations still require evidence review.",
            },
            file,
            indent=2,
            default=str,
        )
    get_logger("evaluation").info("Attempt evidence saved: %s", path)
    return path


def write_json_atomic(
    destination: Path | str,
    record: Mapping[str, Any],
) -> Path:
    """Write one JSON object atomically and return its destination path."""
    path = Path(destination)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
