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

"""RoboDojo env client — thin RPC layer over the Isaac Sim env server."""

from __future__ import annotations

from typing import Any

from rpent.robots.components.env_client_base import BaseEnvClient


class RoboDojoEnvClient(BaseEnvClient):
    """Unified env client for the RoboDojo Isaac Sim backend.

    Inherits :class:`BaseEnvClient` (``env.get_env_meta`` / ``env.reset`` /
    ``env.step`` / ``env.chunk_step``) and adds the RoboDojo-specific read
    surface used by the toolkit primitives (``env.get_obs``,
    ``env.get_reward_details``, ``env.solve_ik_position``, ...). The backend
    is single-environment, so no ``env_idx`` is exposed.
    """

    def __init__(self, client, *, expected_meta: dict[str, Any]):
        super().__init__(client, expected_meta=expected_meta)

    def get_obs(self) -> dict[str, Any]:
        return self._client.call("env.get_obs")

    def get_status(self) -> dict[str, Any]:
        return self._client.call("env.get_status")

    def get_reward_details(self) -> dict[str, Any]:
        return self._client.call("env.get_reward_details")

    def solve_ik_position(self, arm: str, xyz: list) -> dict[str, Any]:
        return self._client.call(
            "env.solve_ik_position", kwargs={"arm": arm, "xyz": xyz}
        )

    def get_safety_status(self) -> dict[str, Any]:
        return self._client.call("env.get_safety_status")

    def is_success(self) -> bool:
        return bool(self._client.call("env.is_success"))

    def close(self) -> None:
        try:
            self._client.call("env.close", timeout_s=10)
        except Exception:  # noqa: BLE001
            pass
