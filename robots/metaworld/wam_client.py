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

"""Connect a compatible external WAM to MetaWorld's native controller."""

import numpy as np

from robots.metaworld.control import METAWORLD_CONTROL
from robots.metaworld.observation import physical_observation
from rpent.robots.components.am_client_base import BaseWAMClient


class MetaWorldWAMClient(BaseWAMClient):
    def __init__(self, client):
        super().__init__(client)
        self.validate_contract(METAWORLD_CONTROL)
        if not set(self.model_meta["camera_roles"]) <= {"primary"}:
            raise ValueError("MetaWorld provides only the primary camera")
        available = {"eef_position": 3, "gripper_opening": 1}
        if any(
            available.get(k) != v for k, v in self.model_meta["state_schema"].items()
        ):
            raise ValueError("WAM requires unavailable MetaWorld state")
        if self.model_meta.get("joint_target_history_shape"):
            raise ValueError("MetaWorld does not provide joint target history")

    def predict(self, obs):
        request = {
            **physical_observation(obs),
            "embodiment": METAWORLD_CONTROL["embodiment"],
            "action_space": METAWORLD_CONTROL["action_space"],
        }
        result = super().predict(request)
        if np.any(np.abs(result["actions"]) > 1):
            raise ValueError("MetaWorld WAM actions must be in [-1, 1]")
        return result
