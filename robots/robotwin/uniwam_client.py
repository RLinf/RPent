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


"""RoboTwin input/output adapter for the UniWAM history-flow policy."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.robots.components.vla_client_base import BaseVLAClient

ACTION_CHUNK_SIZE = 16


class UniWAMClient(BaseVLAClient):
    """Send current observations and episode-owned history over RPent RPC."""

    _TIMEOUT_S = {**BaseVLAClient._TIMEOUT_S, "predict": 600.0}

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        """Return one native qpos14 action chunk."""
        views = observation["views"]
        payload = {
            "observation": {
                name: {"rgb": views[view]["rgb"]}
                for name, view in (
                    ("head_camera", "head"),
                    ("left_camera", "left_wrist"),
                    ("right_camera", "right_wrist"),
                )
            },
            "joint_action": {"vector": observation["robot_state"]["qpos_target14"]},
            "instruction": observation["task_language"],
            "history": observation["qpos_history"],
        }
        actions = np.asarray(self.predict(payload), dtype=np.float64)
        if actions.shape != (ACTION_CHUNK_SIZE, 14) or not np.isfinite(actions).all():
            raise ValueError("UniWAM must return finite actions with shape (16, 14)")
        return actions
