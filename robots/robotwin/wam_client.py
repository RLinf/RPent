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

"""RoboTwin qpos observation and controller adapter for compatible WAM workers."""

from functools import cached_property
from typing import Any

import numpy as np

from robots.robotwin.control import (
    ROBOTWIN_CONTROLLERS,
    ROBOTWIN_QPOS,
)
from robots.robotwin.observation import physical_observation
from rpent.robots.components.am_client_base import BaseWAMClient
from rpent.robots.components.am_contracts import (
    AMControlSpec,
    WAMPrediction,
    WAMRequest,
)
from rpent.utils.rpc import RpcClient

ACTION_SPACE = ROBOTWIN_QPOS["action_space"]
ACTION_SCHEMA = ROBOTWIN_QPOS["action_schema"]


def robotwin_request(
    observation: dict[str, Any], controller: AMControlSpec = ROBOTWIN_QPOS
) -> WAMRequest:
    """Attach the selected execution contract to native RoboTwin observations."""
    return {
        **physical_observation(observation),
        "embodiment": controller["embodiment"],
        "action_space": controller["action_space"],
    }


class RoboTwinWAMClient:
    """Return negotiated qpos14/EEF16 chunks for RoboTwin's native executor."""

    @cached_property
    def controller(self) -> AMControlSpec:
        caps = self.wam.model_meta
        try:
            controller = ROBOTWIN_CONTROLLERS[caps["control"]["action_space"]]
        except KeyError:
            raise ValueError(
                f"Unsupported RoboTwin action space: {caps['control']['action_space']}"
            ) from None
        self.wam.validate_contract(controller)
        return controller

    @property
    def action_type(self) -> str:
        return self.controller["action_type"]

    def __init__(self, client: RpcClient) -> None:
        self.wam = BaseWAMClient(client)

    def validate_robotwin(self) -> None:
        """Reject incompatible model controllers before executing actions."""
        self.controller

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        """Return executable actions in the selected platform control format."""
        return self.predict_result(observation)["actions"]

    def predict_result(self, observation: dict[str, Any]) -> WAMPrediction:
        """Return a validated action chunk and its model metadata."""
        return self.wam.predict(robotwin_request(observation, self.controller))

    def reset(self) -> None:
        """Clear model history before starting a new episode."""
        self.wam.reset()

    def close(self) -> None:
        """Release the RPC client and its session resources."""
        self.wam.close()
