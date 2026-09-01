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

"""BEHAVIOR observation adapter for the common VLA RPC protocol."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.behavior.schemas import extract_policy_state, validate_action_chunk
from rpent.robots.components.vla_client_base import BaseVLAClient
from rpent.utils.rpc import RpcClient


def _instruction_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return ""
    return str(value or "")


class BehaviorVLAClient(BaseVLAClient):
    """Adapt three-camera R1Pro observations to ``vla.predict``."""

    _TIMEOUT_S = {"default": 30.0, "predict": 600.0}

    def __init__(self, client: RpcClient) -> None:
        super().__init__(client)

    def predict(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        **_kwargs: Any,
    ) -> np.ndarray:
        if mode != "eval":
            raise ValueError("BEHAVIOR VLA inference mode must be 'eval'")
        main = np.asarray(env_obs["main_images"])
        wrists = np.asarray(env_obs["wrist_images"])
        if main.ndim != 3 or main.shape[-1] != 3:
            raise ValueError(f"main_images must be [H,W,3], got {main.shape}")
        if wrists.ndim != 4 or wrists.shape[0] != 2 or wrists.shape[-1] != 3:
            raise ValueError(f"wrist_images must be [2,H,W,3], got {wrists.shape}")
        states = np.asarray(env_obs["states"], dtype=np.float32)
        if states.ndim != 1:
            raise ValueError(f"states must be [raw_proprio_dim], got {states.shape}")
        extract_policy_state(states)
        observation = {
            "main_images": main.astype(np.uint8, copy=False)[None],
            "wrist_images": wrists.astype(np.uint8, copy=False)[None],
            "states": states[None],
            "task_descriptions": [_instruction_text(env_obs.get("task_descriptions"))],
            "extra_view_images": None,
        }
        response = super().predict(observation, options={"mode": mode})
        action_batch = np.asarray(response, dtype=np.float32)
        if action_batch.ndim != 3 or action_batch.shape[0] != 1:
            raise ValueError(
                "BEHAVIOR VLA response actions must be [1,T,23], "
                f"got {action_batch.shape}"
            )
        return validate_action_chunk(action_batch[0])

    def close_transport(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


__all__ = ["BehaviorVLAClient"]
