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

"""Measure one simulator episode across direct and planner-driven paths."""

from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Callable
from threading import RLock, Timer
from typing import Any

from rpent.evaluation import TrialRecorder
from rpent.robots.components.am_runtime import select_am
from rpent.robots.robot_spec import EvaluationTrial, RobotSpec, RunConfig
from rpent.utils.logging import get_logger

logger = get_logger(__name__)


def prepare_run_trial(
    args: argparse.Namespace, spec: RobotSpec, config: RunConfig, *, automatic: bool
) -> EvaluationTrial | None:
    """Prepare task memory and register the resolved planner for an automatic run."""
    from rpent.memory.loading import prepare_run_memory

    factory = getattr(spec, "create_evaluation_trial", None) if automatic else None
    try:
        prepare_run_memory(args, spec, config)
    except Exception as exc:
        if factory is not None:
            factory(args, config).finalize(
                error=True, reason="memory_initialization_failed", startup_error=exc
            )
        raise
    return factory(args, config) if factory is not None else None


class EpisodeTrial:
    """Control action admission, timing and accounting for one admitted episode."""

    native_source: str

    def __init__(
        self,
        args: argparse.Namespace,
        config: RunConfig,
        *,
        robot: str,
        task: str,
        scene: str,
        direct: bool = False,
        environment: str | None = None,
        seed: int | None = None,
        extra_budget: dict[str, Any] | None = None,
    ) -> None:
        am = select_am(args, robot)
        environment = environment or robot
        seed = args.seed if seed is None else seed
        self._legacy_endpoint = False
        self.direct = direct
        self.planner = None if direct else args.planner
        self.model = None if direct else args.model
        self.wall_s = getattr(args, "trial_wall_s", None)
        self.deadline: float | None = None
        method = (
            f"direct_{'wam' if am.is_wam else 'vla'}/{am.backend}"
            if direct
            else f"agentic/{self.planner}/{self.model or 'default'}/{am.backend}"
        )
        self.trial = TrialRecorder(
            config.output_dir,
            trial_id=f"{environment}/{scene}/{task}/{seed}",
            method=method,
            task=task,
            scene=scene,
            seed=seed,
            environment=environment,
            checkpoint=am.checkpoint,
            budget={
                **(extra_budget or {}),
                "wall_s": self.wall_s,
                "max_episode_steps": getattr(args, "max_episode_steps", None),
                **(
                    {}
                    if direct
                    else {
                        "max_turns": getattr(args, "max_turns", None),
                        "planner_timeout_s": getattr(args, "planner_timeout_s", None),
                    }
                ),
            },
        )
        self.env = None
        self.started = False
        self.finished = False
        self._gate = RLock()
        self._stop_at: float | None = None
        self._timer: Timer | None = None

    def reset_evidence(self, runtime: dict[str, Any]) -> tuple[str, bool | None]:
        """Return the initial instruction and native verdict from the active reset."""
        raise NotImplementedError

    def is_exhausted(self) -> bool:
        """Whether the executor reached the native step limit."""
        return False

    def seal(self) -> bool:
        """Confirm the environment has drained the admitted episode."""
        return self.env.seal_episode()

    def start(self, runtime: dict[str, Any]) -> dict[str, Any]:
        """Admit the reset observation before any per-trial model work."""
        self.env = runtime["env"] if "env" in runtime else runtime["env_client"]
        instruction, native = self.reset_evidence(runtime)
        self.trial.set_instruction_version(
            "sha256:" + hashlib.sha256(instruction.encode()).hexdigest()
        )
        wam = getattr(runtime.get("model"), "wam", None)
        checkpoint = wam.model_meta.get("checkpoint") if wam is not None else None
        if isinstance(checkpoint, str) and checkpoint:
            self.trial.set_checkpoint(checkpoint)
        start = time.monotonic()
        self.deadline = start + self.wall_s if self.wall_s is not None else None
        self.trial.start(at=start)
        self.started = True
        if self.direct:
            self.trial.record_usage(
                request_id="no-planner",
                model="none",
                module="planner",
                stage="online",
                granularity="session",
                input_tokens=0,
                output_tokens=0,
            )
        if self.deadline is not None:
            self._timer = Timer(
                max(0.0, self.deadline - time.monotonic()),
                self.close_admission,
                args=("wall_budget",),
            )
            self._timer.daemon = True
            self._timer.start()
        self.env.active_trial = self
        if native is True:
            self.trial.record_native_verdict(
                True, source=self.native_source, observed_at=start
            )
            self.close_admission("native_success")
        return {**runtime, "token_ledger": self.trial.tokens}

    def legacy_runtime(self, runtime: dict[str, Any]) -> dict[str, Any]:
        """Let older environment workers run without unverified episode metrics."""
        self._legacy_endpoint = True
        if self.wall_s is not None:
            raise ValueError(
                "A named evaluation episode is required for --trial-wall-s"
            )
        logger.warning(
            "Legacy environment endpoint: native scoring and timing are unavailable"
        )
        return runtime

    def prepare_planner(self, planner: Any) -> None:
        """Attach usage accounting and bound even a planner that issues no actions."""
        if self._legacy_endpoint:
            return
        planner.token_ledger = self.trial.tokens
        remaining = self.remaining()
        if remaining is not None:
            if getattr(planner, "timeout_s", None) is None:
                raise ValueError(
                    "This planner must expose a cancellation timeout to use --trial-wall-s"
                )
            planner.trial_deadline = self.deadline
            planner.timeout_s = min(remaining, planner.timeout_s)

    def remaining(self) -> float | None:
        """Refuse new physical actions at the evaluation cutoff."""
        if self.deadline is None:
            return None
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self.close_admission("wall_budget")
            raise TimeoutError("Episode evaluation wall budget exhausted")
        return remaining

    def close_admission(self, reason: str) -> None:
        """Close the local action gate and timestamp when this actually happens."""
        with self._gate:
            if self._stop_at is None:
                self._stop_at = time.monotonic()
                self.trial.stop(reason, at=self._stop_at)

    def admit_action(self) -> float | None:
        """Return the remaining budget for an action admitted before shutdown."""
        with self._gate:
            remaining = self.remaining()
            if self._stop_at is not None:
                raise RuntimeError("Episode evaluation stopped accepting actions")
        return remaining

    def native_update(
        self,
        value: bool | None,
        *,
        source: str,
        observed_at: float | None = None,
        exhausted: bool = False,
        sealed: bool = False,
    ) -> None:
        """Record the native decision at termination, budget exhaustion or sealing."""
        received = time.monotonic() if observed_at is None else observed_at
        if value is True:
            self.trial.record_native_verdict(True, source=source, observed_at=received)
            within_wall = self.deadline is None or received <= self.deadline
            self.close_admission("native_success" if within_wall else "wall_budget")
        elif sealed:
            # A latched false at drain proves that no earlier action succeeded,
            # including at the deadline if the timer callback ran late.
            cutoff = self._stop_at
            if value is False and cutoff is not None and self.deadline is not None:
                cutoff = min(cutoff, self.deadline)
            self.trial.record_native_verdict(
                value, source=source, observed_at=cutoff, confirmed_only=True
            )
        elif exhausted:
            self.trial.record_native_verdict(value, source=source, observed_at=received)
            self.close_admission("action_budget")

    def finalize(
        self,
        *,
        toolkit: Any = None,
        drain: Callable[[], None] | None = None,
        manual_intervention: Callable[[], bool] | None = None,
        result: Any = None,
        error: bool = False,
        reason: str | None = None,
        startup_error: BaseException | None = None,
    ) -> None:
        """Stop, drain and save this episode before unrelated artifact cleanup."""
        if self.finished:
            return
        if self._legacy_endpoint:
            try:
                if drain is not None:
                    drain()
                elif toolkit is not None:
                    toolkit.cancel_active_and_wait()
            finally:
                self.trial.stop("unsupported_environment_protocol")
                self.trial.finish(
                    manual_intervention=manual_intervention()
                    if manual_intervention
                    else False
                )
                self.finished = True
            return
        failed = error or bool(result and result.error)
        if self.started:
            now = time.monotonic()
            timed_out = self.deadline is not None and now >= self.deadline
            exhausted = self.is_exhausted()
            reason = reason or (
                "wall_budget"
                if timed_out
                else "method_error"
                if failed
                else "action_budget"
                if exhausted
                else "method_returned"
            )
            self.close_admission(reason)
            if self._timer is not None:
                self._timer.cancel()
                self._timer.join()
            try:
                if drain is not None:
                    drain()
                elif toolkit is not None:
                    toolkit.cancel_active_and_wait()
                if self.seal() and not self.trial.tokens.unconfirmed_model_requests:
                    self.trial.quiesce()
            except Exception as exc:
                logger.warning("Episode executor-stop evidence unavailable: %s", exc)
            if self.planner:
                self.trial.tokens.observe_planner(
                    self.planner,
                    self.model,
                    result.stats if result else {},
                    failed=failed or result is None,
                )
            self.trial.finish(
                failure=failed or timed_out or exhausted,
                manual_intervention=manual_intervention()
                if manual_intervention is not None
                else False,
            )
        else:
            component = str(startup_error or "")
            if reason is None:
                if component.startswith("[env] "):
                    reason = "environment_initialization_failed"
                elif component.startswith(("[vla] ", "[wam] ")):
                    reason = "method_initialization_failed"
                else:
                    reason = "runtime_initialization_failed"
            self.trial.stop(reason)
            # The runtime component identifies where setup stopped. The same
            # EnvServer exception can follow a bad scene or an infrastructure fault.
            cause = (
                (startup_error.__cause__ or startup_error) if startup_error else None
            )
            self.trial.finish(
                error_type=type(cause).__name__ if cause is not None else None,
                failure=reason == "method_initialization_failed",
                manual_intervention=manual_intervention()
                if manual_intervention is not None
                else False,
            )
        self.finished = True
