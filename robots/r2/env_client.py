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

"""Agent-side client for the R2 env server."""

from __future__ import annotations

from typing import Any

from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient

#: The env server protocol this client speaks; see ``rlinf_env.PROTOCOL_VERSION``.
PROTOCOL_VERSION = 1


class R2EnvClient(BaseEnvClient):
    """Remote client for one R2 env server.

    The server reports its site bounds in ``env.get_env_meta``, so the client
    checks only that it speaks the same protocol to an R2 rather than
    comparing the whole metadata with a local copy.
    """

    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "default": 30.0,
        "env.reset": 120.0,
        "env.observe": 60.0,
        "env.motion": 120.0,
    }

    def __init__(self, client: RpcClient, *, reset_on_connect: bool = True) -> None:
        self._client = client
        self.meta = client.call(
            "env.get_env_meta", timeout_s=self._TIMEOUT_S["default"]
        )
        if (self.meta.get("robot"), self.meta.get("protocol")) != (
            "r2",
            PROTOCOL_VERSION,
        ):
            raise RuntimeError(
                f"expected an R2 env server speaking protocol {PROTOCOL_VERSION}, "
                f"got robot={self.meta.get('robot')!r} "
                f"protocol={self.meta.get('protocol')!r}; update RPent on the robot"
            )
        if reset_on_connect:
            self.reset()

    def observe(self) -> dict[str, Any]:
        """Read the robot state, camera images, and camera poses."""
        return self._client.call(
            "env.observe", timeout_s=self._TIMEOUT_S["env.observe"]
        )

    def get_robot_state(self) -> dict[str, Any]:
        """Read the measured robot state without images."""
        return self._client.call(
            "env.get_robot_state", timeout_s=self._TIMEOUT_S["default"]
        )

    def get_camera_meta(self) -> dict[str, Any]:
        """Read camera intrinsics and current ``base_link`` poses."""
        return self._client.call(
            "env.get_camera_meta", timeout_s=self._TIMEOUT_S["default"]
        )

    def request_stop(self) -> dict[str, Any]:
        """Stop the running motion, if any, where the robot is."""
        return self._client.call(
            "env.request_stop", timeout_s=self._TIMEOUT_S["default"]
        )

    def _motion(self, name: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.call(
            f"env.{name}", kwargs=kwargs, timeout_s=self._TIMEOUT_S["env.motion"]
        )

    def move_to(
        self, arm: str, xyz: list[float], rpy: list[float] | None = None
    ) -> dict[str, Any]:
        return self._motion("move_to", arm=arm, xyz=xyz, rpy=rpy)

    def move_delta(self, arm: str, delta_xyz: list[float]) -> dict[str, Any]:
        return self._motion("move_delta", arm=arm, delta_xyz=delta_xyz)

    def rotate_delta(self, arm: str, delta_rpy: list[float]) -> dict[str, Any]:
        return self._motion("rotate_delta", arm=arm, delta_rpy=delta_rpy)

    def set_gripper(self, arm: str, opening: float) -> dict[str, Any]:
        return self._motion("set_gripper", arm=arm, opening=opening)

    def look(self, yaw: float, pitch: float) -> dict[str, Any]:
        return self._motion("look", yaw=yaw, pitch=pitch)

    def go_home(self) -> dict[str, Any]:
        return self._motion("go_home")

    def move_base(
        self, vx: float, vy: float, wz: float, duration: float
    ) -> dict[str, Any]:
        return self._motion("move_base", vx=vx, vy=vy, wz=wz, duration=duration)
