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
    ``env.step`` / ``env.chunk_step``) and adds the RoboDojo-specific RPC
    surface used by the toolkit primitives (``env.get_obs``,
    ``env.apply_action``, ``env.solve_ik_position``, ...).
    """

    def __init__(self, client, *, expected_meta: dict[str, Any]):
        super().__init__(client, expected_meta=expected_meta)

    def healthz(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        return self._client.call("healthz", timeout_s=timeout_s)

    def get_obs(self, env_idx: int = 0) -> dict[str, Any]:
        return self._client.call("env.get_obs", args=(env_idx,))

    def get_status(self, env_idx: int = 0) -> dict[str, Any]:
        return self._client.call("env.get_status", args=(env_idx,))

    def get_reward_details(self, env_idx: int = 0) -> dict[str, Any]:
        return self._client.call("env.get_reward_details", args=(env_idx,))

    def solve_ik_position(
        self, arm: str, xyz: list, env_idx: int = 0
    ) -> dict[str, Any]:
        return self._client.call(
            "env.solve_ik_position",
            kwargs={"arm": arm, "xyz": xyz, "env_idx": env_idx},
        )

    def get_safety_status(self, env_idx: int = 0) -> dict[str, Any]:
        return self._client.call("env.get_safety_status", args=(env_idx,))

    def apply_action(
        self, action: dict, action_type: str, env_idx: int = 0
    ) -> dict[str, Any]:
        return self._client.call(
            "env.apply_action",
            kwargs={"action": action, "action_type": action_type, "env_idx": env_idx},
        )

    def apply_target(self, control_info: dict, env_idx: int = 0) -> dict[str, Any]:
        return self._client.call(
            "env.apply_target",
            kwargs={"control_info": control_info, "env_idx": env_idx},
        )

    def is_success(self, env_idx: int = 0) -> bool:
        return bool(self._client.call("env.is_success", args=(env_idx,)))

    def close(self) -> None:
        try:
            self._client.call("env.close", timeout_s=10)
        except Exception:  # noqa: BLE001
            pass
