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

"""Franka env client forwarding explicit methods over an RPC transport."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient


class FrankaEnvClient(BaseEnvClient):
    """Remote client for one RLinf-backed Franka environment."""

    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "default": 30.0,
        "env.reset": 180.0,
        "env.move_delta": 120.0,
        "env.rotate_delta": 120.0,
        "env.set_gripper": 120.0,
        "env.chunk_step": 300.0,
    }

    def __init__(self, client: RpcClient) -> None:
        self._client = client
        self._last_states: np.ndarray | None = None
        self.meta = self._client.call(
            "env.get_env_meta", timeout_s=self._TIMEOUT_S["default"]
        )
        self.reset()

    def reset(self) -> dict[str, Any]:
        result = self._client.call("env.reset", timeout_s=self._TIMEOUT_S["env.reset"])
        self.last_obs = result
        self._remember_states(result.get("states"))
        return result

    def _remember_states(self, states: Any) -> None:
        if states is not None:
            self._last_states = states

    def get_robot_state(self) -> dict[str, Any]:
        robot_state = self._client.call(
            "env.get_robot_state", timeout_s=self._TIMEOUT_S["default"]
        )
        if self._last_states is not None:
            robot_state["wrapped_state_vector"] = self._last_states
        return robot_state

    def get_observation(self) -> dict[str, Any]:
        observation = self._client.call(
            "env.get_observation", timeout_s=self._TIMEOUT_S["default"]
        )
        if self._last_states is not None:
            observation["states"] = self._last_states
        return observation

    def get_camera_meta(self) -> dict[str, Any] | None:
        return self._client.call(
            "env.get_camera_meta",
            timeout_s=self._TIMEOUT_S["default"],
        )

    def move_delta(self, delta_xyz: np.ndarray | list[float]) -> dict[str, Any]:
        result = self._client.call(
            "env.move_delta",
            kwargs={"delta_xyz": np.asarray(delta_xyz, dtype=np.float32)},
            timeout_s=self._TIMEOUT_S["env.move_delta"],
        )
        self._remember_states(result.get("states"))
        return result

    def rotate_delta(self, delta_rpy: np.ndarray | list[float]) -> dict[str, Any]:
        result = self._client.call(
            "env.rotate_delta",
            kwargs={"delta_rpy": np.asarray(delta_rpy, dtype=np.float32)},
            timeout_s=self._TIMEOUT_S["env.rotate_delta"],
        )
        self._remember_states(result.get("states"))
        return result

    def set_gripper(self, *, open: bool) -> dict[str, Any]:
        result = self._client.call(
            "env.set_gripper",
            kwargs={"open": bool(open)},
            timeout_s=self._TIMEOUT_S["env.set_gripper"],
        )
        self._remember_states(result.get("states"))
        return result

    def chunk_step(
        self,
        actions: np.ndarray,
        *,
        return_all_frames: bool = False,
    ) -> dict[str, Any]:
        result = self._client.call(
            "env.chunk_step",
            kwargs={
                "actions": np.asarray(actions, dtype=np.float32),
                "return_all_frames": bool(return_all_frames),
            },
            timeout_s=self._TIMEOUT_S["env.chunk_step"],
        )
        observation = result.get("observation")
        if isinstance(observation, list) and observation:
            self._remember_states(observation[-1].get("states"))
        elif isinstance(observation, dict):
            self._remember_states(observation.get("states"))
        return result
