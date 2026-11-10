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

"""Thin client for BEHAVIOR VLA predictions through the standard RPC route."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.behavior.env_client import validate_native_actions
from rpent.robots.components.vla_client_base import BaseVLAClient


class BehaviorVLAClient(BaseVLAClient):
    """Request native 23D actions without changing policy preprocessing."""

    def reset(self) -> None:
        """Clear policy caches through ``vla.reset``; never reset simulation.

        The official evaluator's episode owner calls this before the first
        inference. The service has one policy and does not isolate clients.
        """
        result = self._client.call("vla.reset", timeout_s=self._TIMEOUT_S["default"])
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise RuntimeError("BEHAVIOR VLA reset was not acknowledged")

    def infer(
        self, observation: dict[str, Any], prompt: str | None = None
    ) -> np.ndarray:
        """Pass a raw observation and optional instruction to the policy backend."""
        result = self.predict(observation, options={"prompt": prompt})
        if not isinstance(result, dict) or "actions" not in result:
            raise TypeError("BEHAVIOR VLA response must contain 'actions'")
        return validate_native_actions(result["actions"], chunk=True)
