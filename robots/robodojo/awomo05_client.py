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

"""RoboDojo Awomo05 observations and native joint actions."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.robodojo.access import ARX_X5_JOINT_FIELDS, CAMERAS
from rpent.robots.components.vla_client_base import BaseVLAClient

REPLAN_STEPS = 8


class Awomo05Client(BaseVLAClient):
    """Send per-step observations and receive native joint actions."""

    def update_obs(self, obs: dict[str, Any], step: int) -> None:
        """Send RGB, joint state and the task instruction."""
        payload = {
            "instruction": obs["instruction"],
            "state": {key: obs["state"][key] for key in ARX_X5_JOINT_FIELDS},
            "vision": {
                name: {"color": obs["vision"][name]["color"]} for name in CAMERAS
            },
        }
        self._client.call(
            "vla.update_obs", args=(payload, step), timeout_s=self._TIMEOUT_S["predict"]
        )

    def get_action(self) -> list[dict[str, Any]]:
        """Validate the native joint action chunk."""
        actions = self._client.call(
            "vla.get_action", timeout_s=self._TIMEOUT_S["predict"]
        )
        if not isinstance(actions, (list, tuple)) or len(actions) != REPLAN_STEPS:
            raise ValueError(f"Awomo05 must return {REPLAN_STEPS} native joint actions")
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("Awomo05 requires native joint action dictionaries")
            if set(action) != set(ARX_X5_JOINT_FIELDS):
                raise ValueError("Awomo05 returned unexpected action fields")
            for key, size in ARX_X5_JOINT_FIELDS.items():
                value = np.asarray(action[key])
                if (
                    value.shape != (size,)
                    or not np.issubdtype(value.dtype, np.number)
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"Awomo05 {key} must contain {size} finite values")
        return actions

    def reset_session(self) -> None:
        """Acquire the model and reset its native episode history."""
        self._client.call("vla.reset_session", timeout_s=self._TIMEOUT_S["predict"])

    def end_episode(self) -> None:
        """Release the model while keeping the runtime connection reusable."""
        self._client.call("vla.end_episode")

    def close(self) -> None:
        """Close this client's RPC session."""
        self._client.close()
