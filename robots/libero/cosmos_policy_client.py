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

"""LIBERO raw-observation client for the isolated Cosmos Policy service."""

from __future__ import annotations

import numpy as np

from rpent.robots.components.vla_client_base import BaseVLAClient


class CosmosPolicyClient(BaseVLAClient):
    """Send raw robosuite observations and receive native LIBERO actions."""

    _TIMEOUT_S = {**BaseVLAClient._TIMEOUT_S, "predict": 300.0}

    def predict(self, env_obs: dict, options: dict | None = None) -> np.ndarray:
        """Send only policy inputs; reject malformed actions before execution."""
        observation = {
            key: env_obs[key]
            for key in (
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot0_gripper_qpos",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "task_descriptions",
            )
        }
        actions = np.asarray(super().predict(observation, options), dtype=np.float32)
        if actions.shape != (16, 7) or not np.isfinite(actions).all():
            raise ValueError("Cosmos Policy must return finite actions shaped [16, 7]")
        return actions
