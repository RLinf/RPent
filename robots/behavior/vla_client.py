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

"""BEHAVIOR observation adapter for the shared Pi0.5 client."""

from __future__ import annotations

import numpy as np

from robots.behavior.schemas import extract_policy_state, validate_action_chunk
from rpent.robots.components.pi05_vla_client import _ENCODE_OBS, Pi05VLAClient
from rpent.robots.components.vla_client_base import BaseVLAClient


def _encode_obs_behavior(env_obs: dict) -> dict:
    """BEHAVIOR/R1Pro single-env obs -> openpi batched wire obs."""

    if not isinstance(env_obs, dict):
        raise TypeError("BEHAVIOR observation must be a mapping")

    main = np.asarray(env_obs.get("main_images"))
    if main.ndim != 3 or main.shape[-1] != 3:
        raise ValueError(f"main_images must be [H,W,3], got {main.shape}")
    if main.dtype != np.uint8:
        raise TypeError(f"main_images must have dtype uint8, got {main.dtype}")

    wrists = np.asarray(env_obs.get("wrist_images"))
    if wrists.ndim != 4 or wrists.shape[0] != 2 or wrists.shape[-1] != 3:
        raise ValueError(f"wrist_images must be [2,H,W,3], got {wrists.shape}")
    if wrists.dtype != np.uint8:
        raise TypeError(f"wrist_images must have dtype uint8, got {wrists.dtype}")

    states = np.asarray(env_obs.get("states"), dtype=np.float32)
    if states.ndim != 1:
        raise ValueError(f"states must be [raw_proprio_dim], got {states.shape}")
    if not np.isfinite(states).all():
        raise ValueError("states contains NaN or infinity")
    extract_policy_state(states)

    task_description = env_obs.get("task_descriptions")
    if isinstance(task_description, (list, tuple)):
        instruction = next(
            (
                item.strip()
                for item in task_description
                if isinstance(item, str) and item.strip()
            ),
            "",
        )
    else:
        instruction = str(task_description or "")

    return {
        "main_images": np.ascontiguousarray(main)[None],
        "wrist_images": np.ascontiguousarray(wrists)[None],
        "extra_view_images": None,
        "states": np.ascontiguousarray(states)[None],
        "task_descriptions": [instruction],
    }


_ENCODE_OBS["behavior"] = _encode_obs_behavior


class BehaviorPi05VLAClient(Pi05VLAClient):
    _TIMEOUT_S = {**Pi05VLAClient._TIMEOUT_S, "predict": 600.0}

    def predict(self, env_obs: dict, options: dict | None = None) -> np.ndarray:
        actions = np.asarray(
            BaseVLAClient.predict(self, self.encode_obs(env_obs), options)
        )
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ValueError(f"BEHAVIOR actions must be [1,T,23], got {actions.shape}")
        return validate_action_chunk(actions[0])
