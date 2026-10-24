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

"""Local execution feedback contract; owns no policy cache or model history."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any


def validate_receipt(feedback: dict, expected_step: int | None) -> tuple[int, int]:
    """Validate one server acknowledgement against its native counter."""
    before, after, count = (
        feedback[key] for key in ("step_before", "step_after", "executed_actions")
    )
    if type(count) is not int or count not in (0, 1):
        raise ValueError("one action RPC must report zero or one executed action")
    if (
        type(before) is not int
        or type(after) is not int
        or (expected_step is not None and before != expected_step)
        or after - before != count
        or feedback["episode_status"]["step"] != after
    ):
        raise ValueError("inconsistent native action acknowledgement")
    return after, count


class ActionRejected(ValueError):
    """A server-confirmed rejection before any action was executed."""


class ExecutionSyncError(RuntimeError):
    """Execution cannot continue until an explicit environment reset succeeds."""

    def __init__(self, message, *, feedback=None, execution_uncertain=False):
        super().__init__(message)
        self.feedback = feedback
        self.execution_uncertain = execution_uncertain


class ExecutionObserver:
    """Deliver synchronous events to one adapter without sampling its history.

    The callback must return only after accepting an event. It must not issue
    environment mutations or inference recursively. A failed delivery latches
    the boundary closed until an explicit reset and successful reset delivery.
    """

    def __init__(self, callback: Callable[[dict[str, Any]], None]):
        self.callback = callback
        self.trusted = False
        self.episode = 0
        self.step = 0
        self.instruction = None
        self.ended = False
        self.paused = False
        self.closed = False
        self.delivering = False

    def _emit(self, kind, **payload):
        event = dict(kind=kind, episode=self.episode, step=self.step, **payload)
        self.delivering = True
        try:
            # Adapters may retain events; they cannot mutate the client's frame.
            self.callback(deepcopy(event))
        finally:
            self.delivering = False

    def lost(self, reason, *, feedback=None, execution_uncertain=False):
        self.trusted = False
        try:
            self._emit("actions_invalidated", reason="synchronization_lost")
            self._emit("synchronization_lost", reason=reason)
        except Exception:
            pass  # Preserve the original failure even if the listener is broken.
        return ExecutionSyncError(
            reason, feedback=feedback, execution_uncertain=execution_uncertain
        )

    def notify(self, kind, **payload):
        try:
            self._emit(kind, **payload)
        except Exception as exc:
            raise self.lost(f"{kind} delivery failed: {exc}") from exc

    def initialize(self, observation, status, *, reset=False):
        self.trusted = False
        if reset:
            self.episode += 1
        self.step = status["step"]
        self.instruction = observation.get("instruction")
        self.ended = False
        self.notify(
            "reset" if reset else "attached", observation=observation, status=status
        )
        self.trusted = True
        self._end_if_needed(status)

    def _end_if_needed(self, status, *, terminated=False, truncated=False):
        terminated = terminated or status.get("success") is True
        truncated = truncated or status["step"] >= status["step_limit"]
        if (terminated or truncated) and not self.ended:
            self.ended = True
            self.notify("actions_invalidated", reason="episode_end")
            self.notify(
                "episode_end", status=status, terminated=terminated, truncated=truncated
            )

    def require_ready(self, status, instruction):
        if self.delivering:
            raise RuntimeError("execution callbacks cannot re-enter control")
        if self.closed or self.paused or not self.trusted:
            raise ExecutionSyncError(
                "execution unavailable; resume a pause or explicitly reset an untrusted episode"
            )
        if status["step"] != self.step:
            raise self.lost(
                "native action counter changed outside this execution boundary"
            )
        if instruction != self.instruction:
            self.notify("actions_invalidated", reason="instruction_changed")
            raise self.lost("instruction changed; explicitly reset before continuing")
        self._end_if_needed(status)
        if self.ended:
            raise ExecutionSyncError(
                "episode ended; explicitly reset before continuing"
            )

    def accept(self, feedback):
        """Deliver a receipt whose action counts were validated by the client."""
        self.step = feedback["step_after"]
        count = feedback["executed_actions"]
        if count and not isinstance(feedback.get("observation"), dict):
            raise self.lost("executed action has no observation", feedback=feedback)
        try:
            self._emit("action_feedback", **feedback)
        except Exception as exc:
            raise self.lost(
                f"action feedback delivery failed: {exc}", feedback=feedback
            ) from exc
        try:
            self._end_if_needed(
                feedback["episode_status"],
                terminated=feedback["terminated"],
                truncated=feedback["truncated"],
            )
        except ExecutionSyncError as exc:
            exc.feedback = feedback
            raise
