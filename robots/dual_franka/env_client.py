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

"""Dual-Franka env client forwarding explicit methods over an RPC transport."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.franka.env_client import FrankaEnvClient
from rpent.utils.rpc import RpcClient

_MOTION_TIMEOUT_S = 120.0
_RECOVERY_TIMEOUT_S = 240.0


class DualFrankaEnvClient(FrankaEnvClient):
    """Remote client for one RLinf-backed dual-Franka environment."""

    def __init__(self, client: RpcClient, *, reset_on_connect: bool = True) -> None:
        super().__init__(client, reset_on_connect=False)
        if not reset_on_connect and self.meta.get("explicit_reset_only") is not True:
            raise RuntimeError(
                "dual-Franka exploration requires an env server advertising "
                "explicit_reset_only=True; upgrade/restart the external server"
            )
        if reset_on_connect:
            self.reset()

    def get_observation(self) -> dict[str, Any]:
        observation = self._client.call(
            "env.get_observation", timeout_s=self._TIMEOUT_S["default"]
        )
        if "states" not in observation:
            raise RuntimeError(
                "Dual-Franka server must return live states; restart the updated server"
            )
        self._remember_states(observation["states"])
        return observation

    def move_delta(
        self, arm: str, delta_xyz: np.ndarray | list[float]
    ) -> dict[str, Any]:
        result = self._client.call(
            "env.move_delta",
            kwargs={
                "arm": str(arm),
                "delta_xyz": np.asarray(delta_xyz, dtype=np.float32),
            },
            timeout_s=_MOTION_TIMEOUT_S,
        )
        self._remember_states(result.get("states"))
        return result

    def rotate_delta(
        self, arm: str, delta_rpy: np.ndarray | list[float]
    ) -> dict[str, Any]:
        result = self._client.call(
            "env.rotate_delta",
            kwargs={
                "arm": str(arm),
                "delta_rpy": np.asarray(delta_rpy, dtype=np.float32),
            },
            timeout_s=_MOTION_TIMEOUT_S,
        )
        self._remember_states(result.get("states"))
        return result

    def set_gripper(self, arm: str, *, open: bool) -> dict[str, Any]:
        result = self._client.call(
            "env.set_gripper",
            kwargs={"arm": str(arm), "open": bool(open)},
            timeout_s=_MOTION_TIMEOUT_S,
        )
        self._remember_states(result.get("states"))
        return result

    def recover_joint_posture(
        self, *, reason: str = "", return_to_start: bool = True
    ) -> dict[str, Any]:
        result = self._client.call(
            "env.recover_joint_posture",
            kwargs={
                "reason": str(reason),
                "return_to_start": bool(return_to_start),
            },
            timeout_s=_RECOVERY_TIMEOUT_S,
        )
        self._remember_states(result.get("states"))
        return result
