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

"""Client for the native RLinf BEHAVIOR environment served by RPent."""

from __future__ import annotations

import math
from functools import cached_property
from typing import Any

import numpy as np

from robots.behavior.observation import valid_observation_token
from robots.behavior.tools import extract_public_proprio
from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient

BEHAVIOR_ACTION_DIM = 23


def env_runtime_contract(task_id: int) -> dict[str, Any]:
    """Describe one task on the native environment, including reset ownership."""
    return {
        "name": "behavior",
        "protocol": "rpent.behavior.rlinf.v1",
        "task_id": task_id,
        "action_dim": BEHAVIOR_ACTION_DIM,
        "reset_owner": "env_service",
    }


def validate_native_actions(values: Any, *, chunk: bool) -> np.ndarray:
    """Validate native 23D actions without changing units or normalization."""
    array = np.asarray(values)
    valid_shape = (
        array.ndim == 2 and array.shape[0] > 0 and array.shape[1] == BEHAVIOR_ACTION_DIM
        if chunk
        else array.shape == (BEHAVIOR_ACTION_DIM,)
    )
    if not valid_shape:
        expected = "[T, 23], T >= 1" if chunk else "[23]"
        raise ValueError(f"BEHAVIOR native actions must have shape {expected}")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("BEHAVIOR native actions must contain finite real numbers")
    return array


class BehaviorEnvClient(BaseEnvClient):
    """Connect to a native Env service without implicitly resetting its episode."""

    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "env.chunk_step": 360.0,
        "env.plan_arm_path": 360.0,
    }

    @cached_property
    def arm_joint_limits(self) -> dict[str, np.ndarray]:
        """Cache the native robot's limits; no client-side assets are required."""
        result = self._client.call(
            "env.get_arm_joint_limits", timeout_s=self._TIMEOUT_S["default"]
        )
        limits = {
            arm: np.asarray(result[arm], dtype=float) for arm in ("left", "right")
        }
        for values in limits.values():
            if (
                values.shape != (7, 2)
                or not np.isfinite(values).all()
                or np.any(values[:, 0] >= values[:, 1])
            ):
                raise ValueError("Env must provide finite seven-joint position limits")
            values.setflags(write=False)
        return limits

    def __init__(
        self,
        client: RpcClient,
        *,
        task_id: int,
        expected_task_language: str | None = None,
        timeout_s: float | None = None,
    ):
        self._TIMEOUT_S = dict(type(self)._TIMEOUT_S)
        if timeout_s is not None:
            if (
                isinstance(timeout_s, bool)
                or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s)
                or timeout_s <= 0
            ):
                raise ValueError("BEHAVIOR RPC timeout must be positive and finite")
            for method in (
                "default",
                "env.step",
                "env.reset",
                "env.chunk_step",
                "env.plan_arm_path",
            ):
                self._TIMEOUT_S[method] = float(timeout_s)
        self.last_obs: dict[str, Any] | None = None
        self.last_info: dict[str, Any] = {}
        super().__init__(
            client,
            expected_meta=env_runtime_contract(task_id),
            reset_on_connect=False,
        )
        if expected_task_language is not None:
            actual = self.get_task_language()
            if (
                not isinstance(actual, str)
                or actual.strip() != expected_task_language.strip()
            ):
                raise ValueError(
                    f"BEHAVIOR task {task_id} instruction mismatch: "
                    f"Planner={expected_task_language!r}, Env={actual!r}"
                )

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Explicitly reset the served episode and replace cached feedback."""
        observation, info = self._client.call(
            "env.reset", timeout_s=self._TIMEOUT_S["env.reset"]
        )
        self.last_obs = observation
        self.last_info = info
        self.__dict__.pop("arm_joint_limits", None)
        return observation, info

    def get_observation(self) -> dict[str, Any]:
        """Fetch the current raw observation for one native environment."""
        observation = self._client.call(
            "env.get_observation", timeout_s=self._TIMEOUT_S["default"]
        )
        if not isinstance(observation, dict):
            raise TypeError("BEHAVIOR observation must be a dictionary")
        self.last_obs = observation
        return observation

    def get_session_status(self) -> dict[str, Any]:
        """Read episode state without advancing or scoring the task."""
        status = self._client.call("env.get_session_status", timeout_s=30.0)
        if (
            not isinstance(status, dict)
            or "ended" not in status
            or not any(status["ended"] is value for value in (True, False, None))
        ):
            raise TypeError("BEHAVIOR session status must contain an ended flag")
        return status

    def get_motion_snapshot(self) -> dict[str, Any]:
        """Fetch atomically bound public observation and control-frame history."""
        snapshot = self._client.call(
            "env.get_motion_snapshot", timeout_s=self._TIMEOUT_S["default"]
        )
        if (
            not isinstance(snapshot, dict)
            or not isinstance(snapshot.get("observation"), dict)
            or not isinstance(snapshot.get("observation_token"), dict)
            or not isinstance(snapshot.get("motion_context"), dict)
        ):
            raise TypeError(
                "BEHAVIOR motion snapshot must bind observation and history"
            )
        self.last_obs = snapshot["observation"]
        return snapshot

    def plan_arm_path(
        self,
        arm: str,
        target_pose: Any,
        *,
        expected_observation_token: dict[str, int] | None = None,
        allow_orientation_relax: bool = False,
    ) -> dict[str, Any]:
        """Request a seven-joint arm path to robot-base xyz + xyzw pose.

        Planning uses the environment's current public observation and does not
        execute actions. The returned timing is the configured control period,
        independently of the video's encoding FPS.
        """
        pose = np.asarray(target_pose)
        if pose.dtype.kind not in "fiu":
            raise ValueError("target_pose must contain real numbers")
        kwargs = {"arm": arm, "target_pose": pose}
        if expected_observation_token is not None:
            kwargs["expected_observation_token"] = expected_observation_token
        if allow_orientation_relax:
            kwargs["allow_orientation_relax"] = True
        return self._client.call(
            "env.plan_arm_path",
            kwargs=kwargs,
            timeout_s=self._TIMEOUT_S["env.plan_arm_path"],
        )

    def _cache_execution(
        self, result: Any, *, expected_observation_token: dict[str, int] | None = None
    ) -> tuple:
        if not isinstance(result, (tuple, list)) or len(result) != 5:
            raise TypeError("BEHAVIOR execution must return a Gym 5-tuple")
        observation, _, _, _, info = result
        if not isinstance(info, dict):
            raise TypeError("BEHAVIOR final observation and info must be dictionaries")
        if expected_observation_token is not None:
            token = info.get("observation_token")
            confirmed = info.get("confirmed_actions")
            context = info.get("motion_context")
            if (
                not valid_observation_token(token)
                or type(confirmed) is not int
                or confirmed < 0
                or token["generation"] != expected_observation_token["generation"]
                or token["observation_index"]
                != expected_observation_token["observation_index"] + confirmed
                or not isinstance(context, dict)
                or context.get("current_token") != token
            ):
                info = {
                    **info,
                    "motion_feedback_error": "confirmation frame is not bound",
                }
                result = (*result[:4], info)
        if isinstance(observation, list):
            if not observation:
                if info.get("stop_reason") == "session_ended" or (
                    info.get("stop_reason") == "execution_outcome_unknown"
                    and info.get("confirmed_actions") == 0
                ):
                    # Preserve an unknown execution result even when no new
                    # frame exists. Never fabricate an acknowledgement.
                    self.last_info = info
                    return tuple(result)
                raise ValueError("BEHAVIOR execution returned no observation frames")
            final_observation = observation[-1]
        else:
            final_observation = observation
        if not isinstance(final_observation, dict) or not isinstance(info, dict):
            raise TypeError("BEHAVIOR final observation and info must be dictionaries")
        if (
            expected_observation_token is not None
            and "motion_feedback_error" not in info
        ):
            try:
                state = extract_public_proprio(final_observation)
            except (TypeError, ValueError, UnicodeError):
                state = None
            if state is None or not np.array_equal(
                state, np.asarray(info["motion_context"].get("current_proprio"))
            ):
                info = {
                    **info,
                    "motion_feedback_error": "public state differs from history",
                }
                result = (*result[:4], info)
        self.last_obs = final_observation
        self.last_info = info
        # Missing reward or termination feedback remains unknown (None).
        return tuple(result)

    def step(self, action: Any) -> tuple:
        """Request one native action and retain the backend's actual feedback."""
        array = validate_native_actions(action, chunk=False)
        result = self._client.call(
            "env.step",
            args=(array,),
            timeout_s=self._TIMEOUT_S["env.step"],
        )
        return self._cache_execution(result)

    def chunk_step(
        self,
        actions: Any,
        *,
        return_all_frames: bool = False,
        expected_observation_token: dict[str, int] | None = None,
    ) -> tuple:
        """Request a native chunk and cache its last returned observation."""
        array = validate_native_actions(actions, chunk=True)
        kwargs = {"return_all_frames": return_all_frames}
        if expected_observation_token is not None:
            kwargs["expected_observation_token"] = expected_observation_token
        result = self._client.call(
            "env.chunk_step",
            args=(array,),
            kwargs=kwargs,
            timeout_s=self._TIMEOUT_S["env.chunk_step"],
        )
        return self._cache_execution(
            result, expected_observation_token=expected_observation_token
        )
