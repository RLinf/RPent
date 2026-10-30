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

"""RoboDojo env client — thin RPC layer over the Isaac Sim env server."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from threading import RLock
from typing import Any

import numpy as np

from robots.robodojo.access import CAMERAS, validate_rgb
from rpent.robots.components.env_client_base import BaseEnvClient


def _validate_receipt(feedback: dict, expected_step: int | None) -> None:
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


class ActionRejected(ValueError):
    """A server-confirmed rejection before any action was executed."""


class ExecutionSyncError(RuntimeError):
    """Execution is unavailable or its state is uncertain."""

    def __init__(self, message, *, feedback=None, execution_uncertain=False):
        super().__init__(message)
        self.feedback = feedback
        self.execution_uncertain = execution_uncertain


class _ExecutionState:
    """Track acknowledged actions and synchronously maintain attached policy history."""

    def __init__(self):
        self.policy_observer: Callable[[dict[str, Any]], None] | None = None
        self.trusted = False
        self.step: int | None = None
        self.ended = False
        self.paused = False
        self.closed = False

    def lost(self, reason, *, feedback=None, execution_uncertain=False):
        self.trusted = False
        return ExecutionSyncError(
            reason, feedback=feedback, execution_uncertain=execution_uncertain
        )

    def notify(self, kind, **payload):
        try:
            if self.policy_observer is not None:
                self.policy_observer(dict(kind=kind, **payload))
        except Exception as exc:
            raise self.lost(f"{kind} delivery failed: {exc}") from exc

    def initialize(self, observation, status, *, reset=False):
        self.trusted = False
        self.step = status["step"]
        if reset:
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

    def require_ready(self, *, allow_ended=False):
        if self.closed or self.paused or not self.trusted:
            raise ExecutionSyncError(
                "execution unavailable; resume a pause or explicitly reset an untrusted episode"
            )
        if self.ended and not allow_ended:
            raise ExecutionSyncError(
                "episode ended; explicitly reset before continuing"
            )

    def accept(self, feedback):
        """Deliver a receipt whose action counts were validated by the client."""
        self.step = feedback["step_after"]
        count = feedback["executed_actions"]
        if count and not isinstance(feedback.get("observation"), dict):
            raise self.lost("executed action has no observation", feedback=feedback)
        self.notify("action_feedback", **feedback)
        self._end_if_needed(
            feedback["episode_status"],
            terminated=feedback["terminated"],
            truncated=feedback["truncated"],
        )


class RoboDojoEnvClient(BaseEnvClient):
    """Unified env client for the RoboDojo Isaac Sim backend.

    Inherits :class:`BaseEnvClient` (``env.get_env_meta`` / ``env.reset`` /
    ``env.step`` / ``env.chunk_step``) and adds the RoboDojo-specific read
    surface used by the toolkit primitives (``env.get_obs``,
    ``env.get_reward_details``, ``env.solve_ik_position``, ...). The backend
    is single-environment, so no ``env_idx`` is exposed.

    ``record_vpp2_history`` retains the RGB frames needed by VPP2. Awomo05
    disables it because its model service owns the policy history.
    """

    def __init__(
        self,
        client,
        *,
        expected_meta: dict[str, Any],
        record_vpp2_history: bool = True,
    ):
        self._execution = _ExecutionState()
        self._execution_lock = RLock()
        self.eval_fair = expected_meta.get("mode") == "eval-fair"
        self._rgb_history = deque(maxlen=177) if record_vpp2_history else None
        self._history_valid = False
        self._inference_index = 0
        super().__init__(
            client,
            expected_meta=expected_meta,
            reset_on_connect=expected_meta.get("mode") != "eval-fair",
        )

    def _record_policy_frame(self, obs: dict) -> None:
        validate_rgb(obs)
        frame = [
            np.asarray(obs["vision"][camera]["color"]).copy() for camera in CAMERAS
        ]
        if not self._rgb_history:
            self._history_step = 0
            self._rgb_anchor = frame
        else:
            self._history_step += 1
        self._rgb_history.append((self._history_step, frame))

    def reset(self):
        """Reset the environment and synchronize a new model episode if attached."""
        with self._execution_lock:
            monitor = self._execution
            if monitor.closed and monitor.policy_observer is not None:
                raise RuntimeError("cannot reset a closed execution boundary")
            monitor.trusted = False
            monitor.closed = monitor.paused = False
            self._history_valid = False
            if self._rgb_history is not None:
                self._rgb_history.clear()
            self._rgb_anchor = None
            self._inference_index = 0
            try:
                obs = super().reset()
                status = self.get_status()
                if status["step"] != 0:
                    raise RuntimeError("environment advanced after reset")
                monitor.initialize(obs, status, reset=True)
                if self._rgb_history is not None:
                    self._record_policy_frame(obs)
                    self._history_valid = True
            except Exception as exc:
                raise monitor.lost(
                    f"reset failed: {exc}", execution_uncertain=True
                ) from exc
            return obs

    def _accept_feedback(self, obs: dict, count: int, was_valid: bool) -> None:
        if was_valid and count:
            self._record_policy_frame(obs)
        self._history_valid = was_valid
        self.last_obs = obs

    def step(self, flat_action):
        """Execute an external action and record its confirmed observation."""
        return self._call_action(
            "env.step",
            kwargs={"flat_action": flat_action},
            tuple_result=True,
        )

    def policy_observation(self) -> dict:
        """Reserve one inference index and sample anchor + stride-25 RGB history.

        Read-only observation refreshes do not advance this action timeline.
        Uncertain mutations invalidate it until an actual episode reset.
        """
        if self._rgb_history is None:
            raise RuntimeError(
                "VPP2 history is not enabled for this environment client"
            )
        if not self._history_valid or not self._execution.trusted:
            raise RuntimeError(
                "VPP2 history unavailable; reset the episode before inference"
            )
        frames = dict(self._rgb_history)
        indices = [
            0,
            *[max(0, self._history_step - offset) for offset in range(175, -1, -25)],
        ]
        state = self.last_obs["state"]
        ee16 = np.concatenate(
            [
                np.asarray(state[f"{arm}_{field}"], dtype=np.float32).reshape(size)
                for arm in ("left", "right")
                for field, size in (("ee_pose", 7), ("ee_joint_state", 1))
            ]
        )
        payload = {
            "history": [self._rgb_anchor if i == 0 else frames[i] for i in indices],
            "state": ee16,
            "inference_index": self._inference_index,
        }
        self._inference_index += 1
        return payload

    def get_obs(self) -> dict[str, Any]:
        return self._client.call("env.get_obs")

    def accept_observation_step(self, before: int, after: int) -> None:
        """Accept a stable capture, allowing analytic control without policy history."""
        with self._execution_lock:
            monitor = self._execution
            if (
                before == after
                and not monitor.trusted
                and monitor.policy_observer is None
            ):
                # Fresh inspection can resume analytic control. Policy history
                # remains invalid until reset because missed frames are unknown.
                monitor.step = after
                monitor.trusted = True
            if before != after or (monitor.step is not None and after != monitor.step):
                self.invalidate_execution()
                raise ExecutionSyncError(
                    "native control advanced during observation capture"
                )

    def get_status(self) -> dict[str, Any]:
        return self._client.call("env.get_status")

    def get_motion_state(self) -> dict[str, Any]:
        return self._client.call("env.get_motion_state")

    def plan_arm_path(self, arm: str, target_pose: list[float]) -> dict[str, Any]:
        return self._client.call(
            "env.plan_arm_path",
            kwargs={"arm": arm, "target_pose": target_pose},
            timeout_s=120,
        )

    def step_joint_update(
        self,
        arm: str,
        arm_qpos: list[float] | None = None,
        gripper: float | None = None,
    ) -> dict[str, Any]:
        return self._call_action(
            "env.step_joint_update",
            kwargs={"arm": arm, "arm_qpos": arm_qpos, "gripper": gripper},
        )

    def get_reward_details(self) -> dict[str, Any]:
        return self._client.call("env.get_reward_details")

    def solve_ik_position(self, arm: str, xyz: list) -> dict[str, Any]:
        return self._client.call(
            "env.solve_ik_position", kwargs={"arm": arm, "xyz": xyz}
        )

    def get_safety_status(self) -> dict[str, Any]:
        return self._client.call("env.get_safety_status")

    def is_success(self) -> bool:
        return bool(self._client.call("env.is_success"))

    def close(self) -> None:
        try:
            self.end_execution()
        finally:
            try:
                self._client.call("env.close", timeout_s=10)
            except Exception:  # noqa: BLE001
                pass

    def register_policy_observer(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Attach the policy history synchronizer and deliver the initial observation.

        The callback consumes reset, action_feedback and closed events. Delivery
        is synchronous: a failure invalidates execution until episode reset.
        While attached, a fresh observation cannot repair missing model history.
        """
        with self._execution_lock:
            if self._execution.policy_observer is not None:
                raise RuntimeError("a policy history observer is already registered")
            monitor = self._execution
            if monitor.step is not None:
                if not monitor.trusted:
                    raise ExecutionSyncError(
                        "execution unavailable; explicitly reset before attaching"
                    )
                monitor.closed = False
                self.require_policy_ready(allow_ended=True)
            monitor.policy_observer = callback
            self._initialize_execution()

    def _initialize_execution(self):
        monitor = self._execution
        try:
            before = self.get_status()
            obs = self.get_obs()
            after = self.get_status()
            self.accept_observation_step(before["step"], after["step"])
            self.last_obs = obs
            monitor.initialize(obs, after)
        except ExecutionSyncError:
            raise
        except Exception as exc:
            raise monitor.lost(f"initial feedback unavailable: {exc}") from exc

    def require_policy_ready(self, *, allow_ended: bool = False) -> None:
        """Check acknowledged environment state before inference or control."""
        with self._execution_lock:
            monitor = self._execution
            if monitor.step is None:
                self._initialize_execution()
            monitor.require_ready(allow_ended=allow_ended)

    def pause_execution(self) -> None:
        with self._execution_lock:
            monitor = self._execution
            if not monitor.closed:
                monitor.paused = True

    def resume_execution(self) -> None:
        with self._execution_lock:
            self._execution.paused = False

    def end_execution(self) -> None:
        with self._execution_lock:
            monitor = self._execution
            if not monitor.closed:
                monitor.closed = True
                monitor.notify("closed")

    def _call_action(self, method, *, kwargs, tuple_result=False):
        with self._execution_lock:
            monitor = self._execution
            if monitor.step is None:
                self._initialize_execution()
            monitor.require_ready()
            kwargs = {**kwargs, "expected_step": monitor.step}
            was_valid, self._history_valid = self._history_valid, False
            confirmed_feedback = None
            try:
                result = self._client.call(
                    method, kwargs=kwargs, timeout_s=self._TIMEOUT_S["env.step"]
                )
                if tuple_result:
                    obs, _, _, info = result
                    feedback = {
                        **info,
                        "episode_status": info["status"],
                        "observation": obs,
                    }
                else:
                    feedback = dict(result)
                    obs = feedback.get("observation")
                _validate_receipt(feedback, monitor.step)
                # Observation processing or delivery cannot erase confirmed progress.
                confirmed_feedback = feedback
                monitor.accept(feedback)
                count = feedback["executed_actions"]
                self._accept_feedback(obs if count else self.last_obs, count, was_valid)
            except ExecutionSyncError as exc:
                if confirmed_feedback is not None:
                    if exc.feedback is None:
                        exc.feedback = confirmed_feedback
                    exc.execution_uncertain = False
                raise
            except Exception as exc:
                raise monitor.lost(
                    f"action RPC/receipt failed: {exc}",
                    feedback=confirmed_feedback,
                    execution_uncertain=confirmed_feedback is None,
                ) from exc
            if feedback.get("rejection"):
                raise ActionRejected(feedback["rejection"])
            if tuple_result:
                return result
            # Large observations travel in the observer event, not tool-result text.
            result.pop("observation", None)
            return result

    @property
    def episode_ended(self) -> bool:
        """Whether execution has received an episode-end signal."""
        with self._execution_lock:
            return self._execution.ended

    def invalidate_execution(self) -> None:
        """Latch control closed after the adapter reports uncertain model state."""
        with self._execution_lock:
            self._history_valid = False
            self._execution.trusted = False

    def unregister_policy_observer(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Release the policy synchronizer, retaining runtime-owned resources."""
        with self._execution_lock:
            monitor = self._execution
            if monitor.policy_observer != callback:
                return
            try:
                self.end_execution()
            finally:
                self._history_valid = self._history_valid and monitor.trusted
                monitor.policy_observer = None
                monitor.closed = False
