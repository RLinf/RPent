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

"""LeKiwi env client forwarding explicit methods over an RPC transport."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient


class LeKiwiEnvClient(BaseEnvClient):
    """Remote client for one RLinf-backed LeKiwi environment."""

    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "default": 30.0,
        "env.reset": 60.0,
        # A drive holds for up to the server's max duration, then reports.
        "env.drive": 60.0,
        # A hold reports at once; the base keeps moving on the server.
        "env.hold": 10.0,
    }

    def __init__(self, client: RpcClient, *, reset_on_connect: bool = True) -> None:
        self._client = client
        self._last_states: np.ndarray | None = None
        self.meta = self._client.call(
            "env.get_env_meta", timeout_s=self._TIMEOUT_S["default"]
        )
        if reset_on_connect:
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
        if "states" not in observation and self._last_states is not None:
            observation["states"] = self._last_states
        self._remember_states(observation.get("states"))
        self.last_obs = observation
        return observation

    def get_camera_meta(self) -> dict[str, Any]:
        return self._client.call(
            "env.get_camera_meta",
            timeout_s=self._TIMEOUT_S["default"],
        )

    def drive(
        self, velocity: np.ndarray | list[float], *, duration_s: float = 1.0
    ) -> dict[str, Any]:
        result = self._client.call(
            "env.drive",
            kwargs={
                "velocity": np.asarray(velocity, dtype=np.float32),
                "duration_s": float(duration_s),
            },
            timeout_s=self._TIMEOUT_S["env.drive"],
        )
        self._remember_states(result.get("states"))
        return result

    def hold(
        self, velocity: np.ndarray | list[float], *, ttl_s: float = 1.0
    ) -> dict[str, Any]:
        """Drive ``velocity`` until the command expires or ``stop`` arrives."""
        return self._client.call(
            "env.hold",
            kwargs={
                "velocity": np.asarray(velocity, dtype=np.float32),
                "ttl_s": float(ttl_s),
            },
            timeout_s=self._TIMEOUT_S["env.hold"],
        )

    def stop(self) -> dict[str, Any]:
        return self._client.call(
            "env.stop",
            timeout_s=self._TIMEOUT_S["default"],
        )
