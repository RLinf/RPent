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


"""Measure one evaluation attempt at the environment and executor boundaries."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from rpent.evaluation.result import write_json_atomic
from rpent.evaluation.usage import TokenLedger


@dataclass(frozen=True)
class _Verdict:
    value: bool | None
    source: str
    at: float
    confirmed_only: bool


class TrialRecorder:
    """Record one attempt under a condition ID shared by comparable methods.

    Clock values are seconds on one local monotonic clock. A simulator or
    remote executor must supply evidence relative to the same clock before
    attributing a success to an earlier instant. The recorder measures budgets;
    the caller enforces them and confirms when action execution has stopped.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        trial_id: str,
        method: str,
        task: str,
        scene: str | None = None,
        seed: int | None = None,
        instruction_version: str | None = None,
        budget: Mapping[str, Any] | None = None,
        checkpoint: str | None = None,
        environment: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        budget = dict(budget or {})
        wall_s = budget.get("wall_s")
        if wall_s is not None and (
            isinstance(wall_s, bool)
            or not isinstance(wall_s, (int, float))
            or not math.isfinite(wall_s)
            or wall_s < 0
        ):
            raise ValueError("budget.wall_s must be a finite nonnegative duration")
        self.run_id = uuid4().hex
        self.path = Path(output_dir) / "metrics" / self.run_id / "episode_metrics.json"
        self.tokens = TokenLedger()
        self._metadata = {
            "schema_version": 1,
            "trial_id": trial_id,
            "run_id": self.run_id,
            "method": method,
            "task": task,
            "scene": scene,
            "seed": seed,
            "instruction_version": instruction_version,
            "budget": budget,
            "checkpoint": checkpoint,
            "environment": environment,
        }
        self._clock = clock
        self._wall_s = wall_s
        self._start: float | None = None
        self._stop: float | None = None
        self._end: float | None = None
        self._reason: str | None = None
        self._verdicts: list[_Verdict] = []
        self._written = False
        self._lock = Lock()

    def _bind_before_start(self, field: str, value: str) -> None:
        with self._lock:
            if self._start is not None:
                raise ValueError(f"{field} must be bound before trial start")
            self._metadata[field] = value

    def set_instruction_version(self, version: str) -> None:
        """Bind the instruction returned by reset before starting this attempt."""
        self._bind_before_start("instruction_version", version)

    def set_checkpoint(self, checkpoint: str) -> None:
        """Bind the checkpoint identified by a model endpoint during setup."""
        self._bind_before_start("checkpoint", checkpoint)

    def start(self, *, at: float | None = None) -> None:
        """Start timing when the task and its first observation are available."""
        with self._lock:
            if self._start is None:
                self._start = self._clock() if at is None else at

    def record_native_verdict(
        self,
        value: bool | None,
        *,
        source: str,
        observed_at: float | None = None,
        confirmed_only: bool = False,
    ) -> None:
        """Record an official judgment, retaining missing evidence as None.

        Args:
            value: The environment's native result, or None if unavailable.
            source: Environment field or native evaluator providing the result.
            observed_at: Local monotonic evidence time. For a frozen snapshot,
                this is the freeze time. Omit this argument for a live sample.
            confirmed_only: The caller certifies this is a frozen verdict for
                the specified instant. When observed_at is omitted, the recorded
                stop is used for a frozen verdict, and the current clock for
                a live verdict. A frozen success alone has no time to success.
        """
        if value is not None and type(value) is not bool:
            raise ValueError("native verdict must be True, False, or None")
        with self._lock:
            if observed_at is None:
                if confirmed_only and self._stop is not None:
                    observed_at = self._stop
                else:
                    observed_at = self._clock()
            self._verdicts.append(_Verdict(value, source, observed_at, confirmed_only))

    def record_usage(self, **usage: Any) -> None:
        """Record a normalized model usage event as accepted by TokenLedger."""
        self.tokens.record_usage(**usage)

    def stop(self, reason: str, *, at: float | None = None) -> None:
        """Record the earliest real stop request or close of action admission."""
        with self._lock:
            when = self._clock() if at is None else at
            if self._start is not None and when < self._start:
                raise ValueError("stop precedes trial start")
            if self._end is not None and when > self._end and self._stop is None:
                raise ValueError("stop follows executor quiescence")
            if self._stop is None or when < self._stop:
                self._stop = when
                self._reason = reason

    def quiesce(self, *, at: float | None = None) -> None:
        """Confirm the executor has stopped and the attempt's queue is drained."""
        with self._lock:
            if self._end is None:
                when = self._clock() if at is None else at
                for prior in (self._start, self._stop):
                    if prior is not None and when < prior:
                        raise ValueError("quiescence precedes trial start or stop")
                self._end = when

    def _evidence(self, until: float) -> tuple[_Verdict | None, float | None, bool]:
        """Return admissible native evidence, first success time and budget expiry."""
        if self._start is None:
            return None, None, False
        cutoff = self._stop if self._stop is not None else until
        expired = False
        if self._wall_s is not None:
            deadline = self._start + self._wall_s
            expired = cutoff >= deadline
            cutoff = min(cutoff, deadline)
        observations = [
            verdict for verdict in self._verdicts if self._start <= verdict.at <= cutoff
        ]
        successes = [v for v in observations if v.value is True]
        if successes:
            first = min((v.at for v in successes if not v.confirmed_only), default=None)
            return min(successes, key=lambda v: v.at), first, expired
        last = max(reversed(observations), key=lambda v: v.at, default=None)
        return last, None, expired

    def _elapsed(self, moment: float | None) -> float | None:
        if moment is None or self._start is None:
            return None
        return moment - self._start

    def finish(
        self,
        *,
        failure: bool = False,
        infra_invalid: bool = False,
        manual_intervention: bool = False,
        error_type: str | None = None,
    ) -> Path:
        """Write this attempt atomically and return its unique artifact path.

        Args:
            failure: A known method failure or exhausted execution budget.
            infra_invalid: An independently established infrastructure fault
                that invalidates this attempt.
            manual_intervention: An operator altered the autonomous evaluation.
            error_type: Root exception class when the attempt raised an error.
        """
        with self._lock:
            if self._written:
                return self.path
            until = self._end if self._end is not None else self._clock()
            evidence, success_at, expired = self._evidence(until)
            native = evidence.value if evidence is not None else None
            if manual_intervention:
                status = "unresolved"
            elif infra_invalid:
                status = "infra_invalid"
            elif native is True:
                status = "success"
            elif failure or expired or native is False:
                status = "failure"
            else:
                status = "unresolved"
            tail = None
            if (
                self._start is not None
                and self._end is not None
                and self._stop is not None
            ):
                tail = self._end - self._stop
            last = max(reversed(self._verdicts), key=lambda v: v.at, default=None)
            latest = (
                {
                    "value": last.value,
                    "source": last.source,
                    "observed_s": self._elapsed(last.at),
                    "confirmed_only": last.confirmed_only,
                }
                if last is not None
                else None
            )
            record = {
                **self._metadata,
                "status": status,
                "manual_intervention": manual_intervention,
                "stop_reason": self._reason,
                "native_verdict": native,
                "verdict_source": evidence.source if evidence else None,
                "last_native_observation": latest,
                "timing": {
                    "run_s": self._elapsed(self._end),
                    "success_s": self._elapsed(success_at)
                    if status == "success"
                    else None,
                    "stop_s": self._elapsed(self._stop),
                    "tail_s": tail,
                    "observed_s": self._elapsed(until),
                },
                "token_ledger": self.tokens.to_dict(),
            }
            unconfirmed = self.tokens.unconfirmed_model_requests
            if unconfirmed:
                record["timing"]["unconfirmed_model_requests"] = unconfirmed
            if error_type is not None:
                record["error_type"] = error_type
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.path, record)
            self._written = True
            return self.path
