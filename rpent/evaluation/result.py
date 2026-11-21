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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

#: Who establishes the outcome a run is judged by: the environment it drives,
#: or the operator supervising a real robot.
TruthSource = Literal["environment", "operator"]

TRUTH_SOURCES: tuple[str, ...] = ("environment", "operator")


@dataclass(frozen=True)
class RunFinalizationContext:
    """Structured runner state supplied to a robot result finalizer.

    ``environment_success`` is the raw task verdict from the run's truth source;
    ``None`` means the run produced no verdict (for example, a real robot run
    nobody judged). The verdict fields below derive from it so a reader of a
    result artifact never has to reconstruct the mapping.
    """

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
    environment_metrics: Mapping[str, Any] = field(default_factory=dict)
    #: Who established ``environment_success``; None when there is no verdict.
    truth_source: TruthSource | None = None

    def __post_init__(self) -> None:
        if self.truth_source not in (None, *TRUTH_SOURCES):
            raise ValueError(
                f"truth_source must be one of {TRUTH_SOURCES} or None, "
                f"got {self.truth_source!r}"
            )

    @property
    def accepted(self) -> bool | None:
        """Whether the run's task outcome is a success; None when unjudged."""
        return self.environment_success

    @property
    def retryable(self) -> bool:
        """Whether a re-run can still produce a verdict."""
        return self.accepted is None

    @property
    def failure_category(self) -> str | None:
        """Why the run is not an accepted success; None when it is."""
        if self.accepted is True:
            return None
        if self.agent_error:
            return "agent_error"
        return "task_failed" if self.accepted is False else "no_verdict"

    @property
    def stop_reason(self) -> str:
        """Why the planner session ended."""
        finish = self.finish_result or {}
        if finish.get("operator_aborted"):
            return "operator_aborted"
        if finish.get("operator_finished"):
            return "operator_finished"
        if self.accepted is True:
            return "solved"
        if self.agent_error:
            return "planner_error"
        if finish:
            return "finished"
        return "unfinished"


def verdict_fields(context: RunFinalizationContext) -> dict[str, Any]:
    """Return the run verdict recorded by transcripts and result artifacts."""
    return {
        "accepted": context.accepted,
        "retryable": context.retryable,
        "failure_category": context.failure_category,
        "stop_reason": context.stop_reason,
        "truth_source": context.truth_source,
    }


RunFinalizer = Callable[[RunFinalizationContext], Path | str | None]


def write_json_atomic(
    destination: Path | str,
    record: Mapping[str, Any],
    *,
    default: Callable[[Any], Any] | None = None,
) -> Path:
    """Write one JSON object atomically and return its destination path.

    ``default`` serializes values the planner produced that JSON cannot encode
    on its own; callers that own a closed schema leave it unset.
    """
    path = Path(destination)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(record, file, indent=2, default=default)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
