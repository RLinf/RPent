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

from rpent.utils.rpc import RpcClient

_DEFAULT_TIMEOUT_S = 30.0
_MOTION_TIMEOUT_S = 120.0
_RESET_TIMEOUT_S = 180.0
_VLA_CHUNK_TIMEOUT_S = 300.0


class FrankaEnvClient:
    """Remote client for one RLinf-backed Franka environment."""

    def __init__(self, client: RpcClient) -> None:
        self._client = client
        self.meta = self._client.call("env.ready", timeout_s=_DEFAULT_TIMEOUT_S)

    def reset(self) -> dict[str, Any]:
        return self._client.call("env.reset", timeout_s=_RESET_TIMEOUT_S)

    def get_robot_state(self) -> dict[str, Any]:
        return self._client.call("env.get_robot_state", timeout_s=_DEFAULT_TIMEOUT_S)

    def get_observation(self) -> dict[str, Any]:
        return self._client.call("env.get_observation", timeout_s=_DEFAULT_TIMEOUT_S)

    def get_camera_metadata(self) -> dict[str, Any] | None:
        return self._client.call(
            "env.get_camera_metadata",
            timeout_s=_DEFAULT_TIMEOUT_S,
        )

    def move_delta(self, delta_xyz: np.ndarray | list[float]) -> dict[str, Any]:
        return self._client.call(
            "env.move_delta",
            kwargs={"delta_xyz": np.asarray(delta_xyz, dtype=np.float32)},
            timeout_s=_MOTION_TIMEOUT_S,
        )

    def rotate_delta(self, delta_rpy: np.ndarray | list[float]) -> dict[str, Any]:
        return self._client.call(
            "env.rotate_delta",
            kwargs={"delta_rpy": np.asarray(delta_rpy, dtype=np.float32)},
            timeout_s=_MOTION_TIMEOUT_S,
        )

    def set_gripper(self, *, open: bool) -> dict[str, Any]:
        return self._client.call(
            "env.set_gripper",
            kwargs={"open": bool(open)},
            timeout_s=_MOTION_TIMEOUT_S,
        )

    def step_chunk(
        self,
        actions: np.ndarray,
        *,
        return_all_frames: bool = False,
    ) -> dict[str, Any]:
        return self._client.call(
            "env.step_chunk",
            kwargs={
                "actions": np.asarray(actions, dtype=np.float32),
                "return_all_frames": bool(return_all_frames),
            },
            timeout_s=_VLA_CHUNK_TIMEOUT_S,
        )
