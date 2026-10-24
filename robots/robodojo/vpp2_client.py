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

"""Request-owned VPP2 observations and native RoboDojo EE16 actions."""

from __future__ import annotations

import numpy as np

from rpent.robots.components.vla_client_base import BaseVLAClient

ACTION_HORIZON = 32
REPLAN_STEPS = 24


class VPP2Client(BaseVLAClient):
    """Send sampled RGB history, proprioception and the agent's subtask."""

    _TIMEOUT_S = {**BaseVLAClient._TIMEOUT_S, "predict": 600.0}

    def infer(self, observation: dict, instruction: str) -> list[dict]:
        """Validate the full prediction, then return its executable prefix."""
        actions = np.asarray(
            self.predict({**observation, "instruction": instruction}), dtype=np.float32
        )
        if actions.shape != (ACTION_HORIZON, 16) or not np.isfinite(actions).all():
            raise ValueError("VPP2 must return finite actions with shape (32, 16)")
        return [
            {
                "left_ee_pose": action[:7],
                "left_ee_joint_state": action[7:8],
                "right_ee_pose": action[8:15],
                "right_ee_joint_state": action[15:16],
            }
            for action in actions[:REPLAN_STEPS]
        ]
