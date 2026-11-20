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

from robots.lekiwi.motion import MAX_DURATION_S, MOTION_TIMEOUT_S, STOP_TIMEOUT_S
from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient


class LeKiwiEnvClient(BaseEnvClient):
    """Local client for one RLinf-backed LeKiwi environment.

    It runs on the agent machine and forwards each call over RPC to the env
    server that owns the hardware on the robot computer.
    """

    #: RPC timeout in seconds per method; ``default`` covers the rest.
    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "default": 30.0,
        "env.reset": 60.0,
        "env.drive": MAX_DURATION_S + STOP_TIMEOUT_S + 5.0,
        "env.rotate": MOTION_TIMEOUT_S + STOP_TIMEOUT_S + 5.0,
    }

    def __init__(self, client: RpcClient, *, reset_on_connect: bool = True) -> None:
        self._client = client
        self.meta = self._client.call(
            "env.get_env_meta", timeout_s=self._TIMEOUT_S["default"]
        )
        if reset_on_connect:
            self.reset()

    def get_robot_state(self) -> dict[str, Any]:
        return self._client.call(
            "env.get_robot_state", timeout_s=self._TIMEOUT_S["default"]
        )

    def get_observation(self) -> dict[str, Any]:
        self.last_obs = self._client.call(
            "env.get_observation", timeout_s=self._TIMEOUT_S["default"]
        )
        return self.last_obs

    def get_camera_meta(self) -> dict[str, Any]:
        return self._client.call(
            "env.get_camera_meta",
            timeout_s=self._TIMEOUT_S["default"],
        )

    def drive(
        self, velocity: np.ndarray | list[float], duration_s: float = 1.0
    ) -> dict[str, Any]:
        """Hold a body velocity and return after execution and stopping."""
        return self._client.call(
            "env.drive",
            kwargs={"velocity": list(velocity), "duration_s": duration_s},
            timeout_s=self._TIMEOUT_S["env.drive"],
        )

    def rotate(self, angle_deg: float) -> dict[str, Any]:
        """Rotate by a signed angle and return after execution and stopping."""
        return self._client.call(
            "env.rotate",
            kwargs={"angle_deg": angle_deg},
            timeout_s=self._TIMEOUT_S["env.rotate"],
        )

    def stop(self) -> dict[str, Any]:
        """Interrupt any active motion through a separate RPC request."""
        return self._client.call("env.stop", timeout_s=self._TIMEOUT_S["default"])
