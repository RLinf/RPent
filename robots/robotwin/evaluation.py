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

"""Bind RoboTwin's native action stream to one evaluation trial."""

from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Callable
from threading import RLock, Timer
from typing import Any

from rpent.evaluation import TrialRecorder
from rpent.robots.components.am_runtime import select_am
from rpent.robots.robot_spec import RunConfig
from rpent.utils.logging import get_logger

logger = get_logger(__name__)
_NATIVE = "RoboTwin.TASK_ENV.eval_success"


class RoboTwinTrial:
    """Own the metrics of one admitted RoboTwin episode and its remote fence."""

    def __init__(
        self, args: argparse.Namespace, config: RunConfig, *, direct: bool = False
    ):
        am = select_am(args, "robotwin")
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
            trial_id=f"robotwin/{args.task_config}/{args.task_name}/{args.seed}",
            method=method,
            task=args.task_name,
            scene=args.task_config,
            seed=args.seed,
            environment="robotwin",
            checkpoint=am.checkpoint,
            budget={
                "wall_s": self.wall_s,
                "max_episode_steps": args.max_episode_steps,
                **(
                    {}
                    if direct
                    else {
                        "max_turns": args.max_turns,
                        "planner_timeout_s": args.planner_timeout_s,
                    }
                ),
            },
        )
        self.env = None
        self.started = False
        self.finished = False
        self._gate = RLock()
        self._stopped = False
        self._timer: Timer | None = None

    def start(self, runtime: dict[str, Any]) -> dict[str, Any]:
        """Admit the reset observation before any per-trial model work."""
        self.env = runtime["env"]
        instruction = self.env.last_reset_info["instruction"]
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
        if self.deadline is not None:
            self._timer = Timer(
                max(0.0, self.deadline - time.monotonic()),
                self.close_admission,
                args=("wall_budget",),
            )
            self._timer.daemon = True
            self._timer.start()
        self.env.active_trial = self
        if self.env.last_reset_info["episode_status"].get("eval_success") is True:
            self.trial.record_native_verdict(True, source=_NATIVE, observed_at=start)
            self.close_admission("native_success")
        return {**runtime, "token_ledger": self.trial.tokens}

    def prepare_planner(self, planner: Any) -> None:
        """Attach usage accounting and bound even a planner that issues no actions."""
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
            raise TimeoutError("RoboTwin evaluation wall budget exhausted")
        return remaining

    def close_admission(self, reason: str) -> None:
        """Close the local action gate and timestamp when this actually happens."""
        with self._gate:
            if not self._stopped:
                self._stopped = True
                self.trial.stop(reason)

    def admit_action(self) -> float | None:
        """Return the remaining budget for an action admitted before shutdown."""
        with self._gate:
            remaining = self.remaining()
            if self._stopped:
                raise RuntimeError("RoboTwin evaluation stopped accepting actions")
        return remaining

    def observe(self, info: dict[str, Any]) -> None:
        """Use native evidence and bound its timestamp in the runner clock."""
        received = time.monotonic()
        value = info.get("episode_status", {}).get("eval_success")
        age = info.get("native_success_age_s")
        if isinstance(age, (int, float)) and age >= 0:
            # The response transit time is nonnegative, making this an upper
            # bound on first success even if a chunk reply crosses the cutoff.
            received -= age
            value = True
        status = info.get("episode_status", {})
        limit = status.get("step_lim")
        exhausted = limit is not None and status.get("take_action_cnt", 0) >= limit
        if value is True:
            self.trial.record_native_verdict(True, source=_NATIVE, observed_at=received)
            within_wall = self.deadline is None or received <= self.deadline
            self.close_admission("native_success" if within_wall else "wall_budget")
        elif exhausted:
            self.trial.record_native_verdict(
                value, source=_NATIVE, observed_at=received
            )
            self.close_admission("action_budget")
        elif info.get("sealed") is True:
            # Native success is latched and the authority fences all old requests.
            # A false verdict after drain establishes that no prior action succeeded.
            self.trial.record_native_verdict(value, source=_NATIVE, confirmed_only=True)

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
        failed = error or bool(result and result.error)
        if self.started:
            now = time.monotonic()
            timed_out = self.deadline is not None and now >= self.deadline
            native = self.env.last_info.get("episode_status", {})
            steps = native.get("take_action_cnt", 0)
            limit = native.get("step_lim")
            exhausted = limit is not None and steps >= limit
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
                if self.env.seal_episode():
                    self.trial.quiesce()
            except Exception as exc:
                logger.warning("RoboTwin executor-stop evidence unavailable: %s", exc)
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
