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

"""RPC client for one RoboLab episode."""

from __future__ import annotations

from typing import Any

from rpent.robots.components.env_client_base import BaseEnvClient


class RoboLabEnvClient(BaseEnvClient):
    """Use the shared environment RPC routes and expose native episode status."""

    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "env.reset": 300.0,
        "env.step": 120.0,
        "env.chunk_step": 600.0,
    }

    def __init__(self, client: Any, *, expected_meta: dict[str, Any]) -> None:
        self.terminated = False
        self.truncated = False
        self.last_info: dict[str, Any] = {}
        super().__init__(client, expected_meta=expected_meta)

    def reset(self) -> dict[str, Any]:
        self.terminated = False
        self.truncated = False
        self.last_info = {}
        return super().reset()

    def step(self, flat_action: Any) -> tuple:
        if self.terminated or self.truncated:
            raise RuntimeError("RoboLab episode has ended")
        result = super().step(flat_action)
        self.terminated = bool(result[2])
        self.truncated = bool(result[3])
        self.last_info = dict(result[4])
        return result

    def chunk_step(
        self, flat_actions: Any, *, return_all_frames: bool = False
    ) -> tuple:
        if self.terminated or self.truncated:
            raise RuntimeError("RoboLab episode has ended")
        result = super().chunk_step(flat_actions, return_all_frames=return_all_frames)
        self.terminated = bool(result[2])
        self.truncated = bool(result[3])
        self.last_info = dict(result[4])
        return result

    def get_status(self) -> dict[str, Any]:
        return self._client.call("env.get_status", timeout_s=self._TIMEOUT_S["default"])
