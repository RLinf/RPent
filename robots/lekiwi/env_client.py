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
        "env.get_navigation_state": 2.0,
    }

    def __init__(self, client: RpcClient) -> None:
        self._client = client
        self._last_states: np.ndarray | None = None
        self.meta = self._client.call(
            "env.get_env_meta", timeout_s=self._TIMEOUT_S["default"]
        )

    def close(self) -> None:
        self._client.close()

    def get_navigation_state(self) -> dict[str, Any]:
        return self._client.call(
            "env.get_navigation_state",
            timeout_s=self._TIMEOUT_S["env.get_navigation_state"],
        )

    def list_map_nodes(self, *, label=None) -> dict:
        """List existing map nodes without starting navigation."""
        return self._client.call(
            "env.list_map_nodes",
            kwargs={"label": label},
            timeout_s=self._TIMEOUT_S["default"],
        )

    def inspect_map_node(self, node_id: int) -> dict:
        """Read a node's historical reference image and map pose."""
        return self._client.call(
            "env.inspect_map_node",
            kwargs={"node_id": node_id},
            timeout_s=self._TIMEOUT_S["default"],
        )

    def view_map(self, *, center_node_id=None, radius_m=None) -> dict:
        """Read a static occupancy map with node and current-pose overlays."""
        return self._client.call(
            "env.view_map",
            kwargs={"center_node_id": center_node_id, "radius_m": radius_m},
            timeout_s=self._TIMEOUT_S["default"],
        )

    def navigate_to(self, target: int | str, *, timeout_s: float = 60) -> dict:
        """Navigate to an RTAB-Map node ID or label; poll state for its result."""
        return self._client.call(
            "env.navigate_to",
            kwargs={"target": target, "timeout_s": timeout_s},
            timeout_s=self._TIMEOUT_S["default"],
        )

    def recording(self, operation: str, **kwargs) -> dict:
        """Control the robot's stereo PNG recorder."""
        if operation not in {"start", "state", "stop"}:
            raise ValueError("Unknown recording operation")
        return self._client.call(
            "env.recording_" + operation,
            kwargs=kwargs,
            timeout_s=self._TIMEOUT_S["default"],
        )

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
        """Cancel motion without waiting for the active drive RPC to finish."""
        return self._client.call(
            "env.stop",
            timeout_s=self._TIMEOUT_S["default"],
        )
