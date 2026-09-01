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

"""Unwired BEHAVIOR entries for the future shared Pi0.5 registries."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.behavior.schemas import extract_policy_state


def _encode_obs_behavior(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Encode one BEHAVIOR observation without changing the input mapping."""

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
    # Validate the R1Pro layout without replacing the raw proprio sent to RLinf.
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


PI05_BEHAVIOR_EMBODIMENT: dict[str, Any] = {
    "num_action_chunks": 32,
    "action_dim": 32,
    "use_proprio": True,
    "num_steps": 4,
    "add_value_head": False,
    "openpi_data": {
        "norm_stats_path": ("assets/behavior-1k/2025-challenge-demos/norm_stats.json"),
        "extra_delta_transform": False,
        "extract_state_from_proprio": True,
        "use_all_wrist_images": True,
        "use_quantile_norm": True,
    },
    "openpi": {
        "config_name": "pi05_behavior",
        "num_images_in_input": 3,
        "action_dim": 32,
        "action_horizon": 32,
        "action_chunk": 32,
        "action_env_dim": 23,
        "num_steps": 4,
        "add_value_head": False,
        "noise_level": 0.0,
        "noise_method": "flow_sde",
        "joint_logprob": False,
    },
}


__all__ = ["PI05_BEHAVIOR_EMBODIMENT", "_encode_obs_behavior"]
