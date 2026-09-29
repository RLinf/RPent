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

"""Cosmos Policy RPC client for the supported LIBERO observation/action format."""

from __future__ import annotations

import numpy as np

from rpent.robots.components.action_model_client_base import BaseActionModelClient
from rpent.robots.components.action_model_protocol import (
    ActionModelCompatibilityError,
    ActionModelPrediction,
)
from rpent.robots.components.cosmos_policy_protocol import (
    ACTION_SCHEMA,
    PROPRIO_SCHEMA,
    libero_request,
)


class CosmosPolicyClient(BaseActionModelClient):
    """Send raw robosuite observations and receive native LIBERO actions."""

    def __init__(self, client):
        super().__init__(client, expected_backend="cosmos_policy")

    def validate_libero(self) -> None:
        """Reject incompatible checkpoints before starting a control loop."""
        capabilities = self.get_capabilities()
        capabilities.require_embodiment(
            "libero_7d", action_dim=7, action_schema=ACTION_SCHEMA
        )
        if (
            capabilities.proprio_schema != PROPRIO_SCHEMA
            or capabilities.camera_roles != ("primary", "wrist")
        ):
            raise ActionModelCompatibilityError(
                "Cosmos LIBERO requires primary/wrist cameras and the 9-field proprio schema"
            )

    def predict(self, env_obs: dict, options: dict | None = None) -> np.ndarray:
        """Send only policy inputs; reject malformed actions before execution."""
        return self.predict_result(env_obs, options).actions

    def predict_result(
        self, env_obs: dict, options: dict | None = None
    ) -> ActionModelPrediction:
        """Return optional model predictions separately from observed environment state."""
        if options and options != {"mode": "eval"}:
            raise ValueError("Cosmos Policy supports only evaluation options")
        self.validate_libero()
        result = super().predict(libero_request(env_obs))
        if result.actions.shape != (16, 7):
            raise ValueError("Cosmos Policy must return finite actions shaped [16, 7]")
        return result
