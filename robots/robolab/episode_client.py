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

"""Client for RoboLab's single-environment measured-pose protocol."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.utils.rpc import RpcClient, make_rpc_client


class RoboLabEpisodeEnvClient:
    """Attach to a RoboLab episode already reset by the server."""

    def __init__(
        self, rpc_or_endpoint: RpcClient | str, *, expected_meta: dict | None = None
    ) -> None:
        self._client = (
            make_rpc_client(rpc_or_endpoint)
            if isinstance(rpc_or_endpoint, str)
            else rpc_or_endpoint
        )
        self._meta = self._client.call("env.get_env_meta")
        for key, expected in (expected_meta or {}).items():
            if self._meta.get(key) != expected:
                raise ValueError(
                    f"RoboLab {key} mismatch: expected {expected!r}, "
                    f"got {self._meta.get(key)!r}"
                )

    def get_env_meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def get_task_language(self) -> str:
        return self._client.call("env.get_task_language")

    def get_status(self) -> dict[str, Any]:
        return self._client.call("env.get_status")

    def get_motion_state(self) -> dict[str, Any]:
        return self._client.call("env.get_motion_state")

    def coordinate_calibration(self) -> dict[str, Any]:
        return self._client.call("env.coordinate_calibration")

    def get_obs(self) -> dict[str, Any]:
        return self._client.call("env.get_obs")

    def step_to(
        self,
        xyz: np.ndarray,
        quaternion_wxyz: np.ndarray,
        opening: float,
        *,
        expected_step: int,
    ) -> dict[str, Any]:
        """Execute one world-flange setpoint and verify the measured receipt."""
        feedback = self._client.call(
            "env.step",
            kwargs={
                "xyz": xyz,
                "quaternion_wxyz": quaternion_wxyz,
                "opening": opening,
                "expected_step": expected_step,
            },
            timeout_s=60,
        )
        if feedback["status"]["step"] != expected_step + 1:
            raise RuntimeError("RoboLab action acknowledgement has an unexpected step")
        return feedback

    def close(self) -> None:
        """Release the transport for this episode."""
        self._client.close()
