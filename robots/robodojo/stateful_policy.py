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

"""Awomo action chunks and session ownership on RoboDojo execution feedback."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from robots.robodojo.execution import ActionRejected, ExecutionSyncError
from robots.robodojo.tasks import validate_instruction
from rpent.tools import ToolResult, tool
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from robots.robodojo.awomo05 import Awomo05Client

logger = get_logger("stateful_policy")


@tool(readonly=False, exclude=("primitives",))
def run_policy(
    primitives: Any,
    max_steps: Annotated[int, Field(gt=0)] = 32,
) -> ToolResult:
    """Run the configured policy on the official task instruction for at most max_steps control steps.

    Inspect the resulting images before continuing. Episode end does not itself
    establish task success.
    """
    try:
        return ToolResult(data=primitives.env.run(max_steps))
    except Exception as exc:
        if not primitives.env.last_run:
            raise
        result = {**primitives.env.last_run, "error": str(exc)}
        if isinstance(exc, ToolCancelled):
            result.update(code="tool_cancelled", interrupted=True)
        else:
            logger.exception("Policy execution failed")
        return ToolResult(data=result)


class StatefulPolicyEnv:
    """Own one action queue and model session on a runtime-owned environment.

    Environment feedback owns confirmed progress, pause and synchronization state.
    Only acknowledged actions deliver observations; the official model owns history.
    A budget cut retains the chunk. Cancellation retains a confirmed inference only
    until its first action executes; after progress, resume requires fresh inference.
    """

    def __init__(self, env, policy: Awomo05Client, check_cancelled) -> None:
        self._env = env
        self._policy = policy
        self._check_cancelled = check_cancelled
        self._pending = deque()
        self._pending_step = None
        self._session_active = False
        self._released = False
        self.last_run: dict[str, Any] = {}
        self._recording_run = False
        try:
            self._env.register_execution_observer(self._execution_event)
        except Exception:
            try:
                self.release()
            except Exception:
                logger.warning(
                    "Failed to release policy after startup error", exc_info=True
                )
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def _step(self) -> int:
        return self._env.get_execution_state()["confirmed_step"] - self._origin_step

    def _execution_event(self, event: dict) -> None:
        """Consume each confirmed observation once, without maintaining model history."""
        kind = event["kind"]
        if kind in ("attached", "reset"):
            self._discard_actions()
            self._origin_step = event["status"]["step"]
            self._instruction = validate_instruction(self._env.get_task_language())
            # Even a failed reset response may have acquired the remote session.
            self._session_active = True
            self._policy.reset_session()
            self._observe(event["observation"])
        elif kind == "actions_invalidated":
            reason = event["reason"]
            # These are preflight notifications. A rejected zero-action RPC is
            # not a takeover; discard only on confirmed non-policy progress.
            if reason in ("external", "primitive", "hold"):
                return
            if not (reason == "cancelled" and self._pending_step == self._step):
                self._discard_actions()
        elif kind == "action_feedback" and event["executed_actions"]:
            if event["source"] != "policy":
                self._discard_actions()
            # Official Awomo deploy.py excludes the terminal observation.
            if not (event["terminated"] or event["truncated"]):
                self._observe(event["observation"])
        elif kind == "closed":
            self._discard_actions()
            self._end_episode()

    def _observe(self, obs: dict) -> None:
        started = time.perf_counter()
        try:
            self._policy.update_obs(
                {**obs, "instruction": self._instruction}, self._step
            )
        finally:
            if self._recording_run:
                self.last_run["observation_rpc_s"] += time.perf_counter() - started

    def _discard_actions(self) -> None:
        self._pending.clear()
        self._pending_step = None

    def _require_ready(self) -> None:
        if self._released:
            raise RuntimeError("Policy session is unavailable; execution was released")
        state = self._env.get_execution_state()
        if (
            state["episode_ended"]
            and state["synchronization_trusted"]
            and not state["paused"]
            and not state["closed"]
        ):
            return
        self._env.require_policy_ready()

    def reset(self) -> dict:
        """Reset native execution and model history together through reset feedback."""
        if self._released or self._env.get_execution_state()["closed"]:
            raise RuntimeError("Policy session is closed")
        return self._env.reset()

    def run(self, max_steps: int) -> dict[str, Any]:
        """Execute one continuing action stream, bounded by confirmed native steps."""
        self.last_run = {}
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self._require_ready()
        initial_step = self._step
        started = time.perf_counter()
        self.last_run = {
            "inferences": [],
            "policy_rpc_s": 0.0,
            "env_step_s": 0.0,
            "observation_rpc_s": 0.0,
            "stop_reason": "budget",
        }
        self._recording_run = True
        try:
            while (
                self._step - initial_step < max_steps
                and not self._env.get_execution_state()["episode_ended"]
            ):
                self._check_cancelled()
                if not self._pending:
                    self._env.require_policy_ready()
                    infer_started = time.perf_counter()
                    try:
                        actions = self._policy.get_action()
                    except ToolCancelled as exc:
                        # A communication failure is not a scheduler pause, even
                        # when the exception happens to use the same type.
                        raise ExecutionSyncError(
                            f"policy inference uncertain: {exc}"
                        ) from exc
                    finally:
                        self.last_run["policy_rpc_s"] += (
                            time.perf_counter() - infer_started
                        )
                    self.last_run["inferences"].append(
                        {
                            "observation_step": self._step,
                            "action_count": len(actions),
                            "info": self._policy.last_action_info,
                        }
                    )
                    self._pending.extend(actions)
                    self._pending_step = self._step
                # Check again after inference, before the first action RPC.
                self._check_cancelled()
                action_started = time.perf_counter()
                try:
                    self._env.step_policy(self._pending[0])
                finally:
                    self.last_run["env_step_s"] += time.perf_counter() - action_started
                if self._pending:  # Terminal feedback may already have cleared it.
                    self._pending.popleft()
        except Exception as exc:
            self.last_run["stop_reason"] = (
                "cancelled" if isinstance(exc, ToolCancelled) else "error"
            )
            if isinstance(exc, ToolCancelled):
                self._env.pause_execution()
            elif isinstance(exc, ActionRejected):
                self._discard_actions()
            else:
                self._env.invalidate_execution("policy state is uncertain")
            raise
        finally:
            self._recording_run = False
            ended = self._env.get_execution_state()["episode_ended"]
            if ended and self.last_run["stop_reason"] == "budget":
                self.last_run["stop_reason"] = "episode_end"
            self.last_run.update(
                executed_steps=self._step - initial_step,
                episode_ended=ended,
                observation_step=self._step,
                pending_actions=len(self._pending),
                elapsed_s=time.perf_counter() - started,
            )
        return self.last_run

    def _end_episode(self) -> None:
        if self._session_active:
            try:
                self._policy.end_episode()
            finally:
                self._session_active = False

    def pause_execution(self) -> None:
        if not self._released:
            self._env.pause_execution()

    def resume_execution(self) -> None:
        if not self._released:
            self._env.resume_execution()

    def end_execution(self) -> None:
        """End only this attachment, never a later Toolkit's execution."""
        if not self._released:
            self._env.end_execution()

    def release(self) -> None:
        """Release the episode and detach; Runtime retains both connections."""
        if self._released:
            return
        try:
            self._env.unregister_execution_observer(self._execution_event)
        finally:
            self._released = True
            self._end_episode()

    def close(self) -> None:
        """Explicitly release the policy session and close the environment."""
        try:
            self.release()
        finally:
            self._env.close()
