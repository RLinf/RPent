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

"""Shared bookkeeping for robot-specific exploration lifecycles."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal, TypeVar


class ResetAfterSuccess(Enum):
    """Whether an adapter preserves or refuses its existing post-success reset."""

    ALLOW = auto()
    REFUSE = auto()


class ResetRefusal(Enum):
    """Mechanical reasons a requested exploration reset cannot begin."""

    TASK_SOLVED = auto()
    ATTEMPT_BUDGET_SPENT = auto()


class ResetPhase(Enum):
    """Bookkeeping state for the most recent reset invocation."""

    READY = auto()
    IN_PROGRESS = auto()
    FAILED = auto()


@dataclass(frozen=True, slots=True)
class FinishRefusal:
    """Attempt counts needed to render an adapter-specific finish refusal."""

    remaining: int
    budget: int


RecordT = TypeVar("RecordT")
ResetResultT = TypeVar("ResetResultT")


def _is_successful_reset(record: Any) -> bool:
    command = getattr(record, "command", None)
    if not isinstance(command, dict) or command.get("action") != "reset":
        return False
    result = getattr(record, "result", None)
    return not (isinstance(result, dict) and result.get("error"))


def records_after_latest_successful_reset(
    records: Iterable[RecordT], *, after_step: int = -1
) -> list[RecordT]:
    """Return records strictly after the latest non-error reset command."""
    materialized = list(records)
    boundary = max(
        (
            getattr(record, "step_idx")
            for record in materialized
            if _is_successful_reset(record)
        ),
        default=after_step,
    )
    boundary = max(boundary, after_step)
    return [record for record in materialized if getattr(record, "step_idx") > boundary]


def refusal_result(action: Literal["finish", "reset"], reason: str) -> dict[str, str]:
    """Build the refusal envelope shared by exploration tools."""
    return {"error": f"{action} refused", "reason": reason}


class ExplorationLifecycle:
    """Own common attempt, reset, finish, and trace-boundary bookkeeping.

    The caller still owns the physical reset operation and all semantic
    validation of its result. This object only decides whether a reset may
    start, accounts for the attempt before invoking the supplied operation,
    and records whether that invocation completed or raised.
    """

    def __init__(
        self,
        *,
        mode: str,
        attempts_per_session: int,
        adapter_name: str,
        reset_after_success: ResetAfterSuccess,
    ) -> None:
        if mode not in {"evaluation", "exploration"}:
            raise ValueError(f"unsupported {adapter_name} toolkit mode: {mode!r}")
        self._mode = mode
        self._attempts_per_session = max(0, int(attempts_per_session))
        self._current_attempt = 1
        self._reset_after_success = reset_after_success
        self._reset_phase = ResetPhase.READY
        self._latest_successful_reset_step = -1

    @property
    def mode(self) -> str:
        """Return the configured toolkit mode."""
        return self._mode

    @property
    def is_exploration(self) -> bool:
        """Whether exploration-only lifecycle guards are active."""
        return self._mode == "exploration"

    @property
    def attempts_per_session(self) -> int:
        """Return the attempt budget, where zero means unlimited."""
        return self._attempts_per_session

    @property
    def current_attempt(self) -> int:
        """Return the one-based attempt number for this planner session."""
        return self._current_attempt

    @property
    def reset_in_progress(self) -> bool:
        """Whether a reset operation has begun and not yet returned."""
        return self._reset_phase is ResetPhase.IN_PROGRESS

    @property
    def reset_failed(self) -> bool:
        """Whether the most recent reset invocation raised an exception."""
        return self._reset_phase is ResetPhase.FAILED

    @property
    def latest_successful_reset_step(self) -> int:
        """Return the latest observed successful reset boundary, or ``-1``."""
        return self._latest_successful_reset_step

    def request_reset(self, *, solved: bool) -> ResetRefusal | None:
        """Decide whether a reset may start and consume its attempt if allowed."""
        if self.reset_in_progress:
            raise RuntimeError("an exploration reset is already in progress")
        if solved and self._reset_after_success is ResetAfterSuccess.REFUSE:
            return ResetRefusal.TASK_SOLVED
        budget = self._attempts_per_session
        if budget and self._current_attempt >= budget:
            return ResetRefusal.ATTEMPT_BUDGET_SPENT
        self._current_attempt += 1
        self._reset_phase = ResetPhase.IN_PROGRESS
        return None

    def perform_reset(
        self,
        operation: Callable[[], ResetResultT],
        *,
        successful_boundary_after_step: int | None = None,
    ) -> ResetResultT:
        """Run an approved adapter reset and retain increment-first semantics."""
        if not self.reset_in_progress:
            raise RuntimeError("exploration reset was not requested")
        try:
            result = operation()
        except BaseException:
            self._reset_phase = ResetPhase.FAILED
            raise
        self._reset_phase = ResetPhase.READY
        if successful_boundary_after_step is not None:
            self._latest_successful_reset_step = max(
                self._latest_successful_reset_step,
                successful_boundary_after_step,
            )
        return result

    def build_reset_result(
        self, result: Mapping[str, Any], **fields: Any
    ) -> dict[str, Any]:
        """Add the canonical attempt field plus adapter-specific result fields."""
        return {**result, "attempt": self._current_attempt, **fields}

    def finish_refusal(self, *, solved: bool) -> FinishRefusal | None:
        """Return remaining-attempt data when an exploration finish is premature."""
        budget = self._attempts_per_session
        if (
            self.is_exploration
            and budget
            and not solved
            and self._current_attempt < budget
        ):
            return FinishRefusal(
                remaining=budget - self._current_attempt,
                budget=budget,
            )
        return None

    def observe_record(self, record: Any) -> None:
        """Advance the trace boundary after a recorded successful reset."""
        if self.is_exploration and _is_successful_reset(record):
            self._latest_successful_reset_step = max(
                self._latest_successful_reset_step,
                int(record.step_idx),
            )
