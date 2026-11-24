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

"""LIBERO observation and controller adapter shared by compatible WAM workers."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np

from robots.libero.control import LIBERO_OSC
from robots.libero.observation import physical_observation
from rpent.robots.components.am_client_base import BaseWAMClient
from rpent.robots.components.am_contracts import (
    AMControlSpec,
    WAMPrediction,
    WAMRequest,
)
from rpent.utils.rpc import RpcClient

ACTION_SPACE = LIBERO_OSC["action_space"]


def libero_request(raw_obs: dict[str, Any]) -> WAMRequest:
    """Attach the WAM execution contract to model-independent observations."""
    return {
        **physical_observation(raw_obs),
        "embodiment": LIBERO_OSC["embodiment"],
        "action_space": ACTION_SPACE,
    }


class LiberoWAMClient:
    """Adapt raw LIBERO observations to the model-independent WAM client."""

    def __init__(self, client: RpcClient) -> None:
        self.wam = BaseWAMClient(client)

    def validate_libero(self) -> None:
        """Reject incompatible controllers before starting a control loop."""
        self.controller

    @cached_property
    def controller(self) -> AMControlSpec:
        """Connect once and retain the supported native control contract."""
        self.wam.validate_contract(LIBERO_OSC)
        return LIBERO_OSC

    def predict(self, env_obs: dict, options: dict | None = None) -> np.ndarray:
        """Return executable actions for the existing LIBERO chunk executor."""
        return self.predict_result(env_obs, options)["actions"]

    def predict_result(
        self, env_obs: dict, options: dict | None = None
    ) -> WAMPrediction:
        """Return actions and optional predictions separately from simulator state."""
        if options and options != {"mode": "eval"}:
            raise ValueError("LIBERO WAM supports only evaluation options")
        self.validate_libero()
        return self.wam.predict(libero_request(env_obs))

    def reset(self) -> None:
        """Clear model history when the caller begins a new episode."""
        self.wam.reset()

    def close(self) -> None:
        """Release model session resources."""
        self.wam.close()
