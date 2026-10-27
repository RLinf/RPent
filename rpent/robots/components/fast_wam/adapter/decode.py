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

"""Fast-WAM action denormalization and platform controller conventions."""

from typing import Any

import numpy as np

from rpent.robots.components.am_contracts import WAMPrediction


def decode_libero(result: tuple[np.ndarray, dict]) -> WAMPrediction:
    """Unbatch executable actions already decoded and truncated by RLinf."""
    actions, metadata = result
    if actions.ndim != 3 or actions.shape[0] != 1:
        raise ValueError("RLinf Fast-WAM must return one batched action chunk")
    return WAMPrediction(actions=actions[0], metadata=metadata)


def decode_robotwin(
    result: dict,
    *,
    normalizer: Any,
    horizon: int,
    execute_steps: int,
) -> WAMPrediction:
    """Return the official 14D qpos vector unchanged after denormalization."""
    import torch

    action = result["action"]
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if tuple(action.shape) != (1, horizon, 14):
        raise ValueError(f"Fast-WAM actions must have [1,{horizon},14] shape")
    decoded = normalizer.backward(action.detach().to(dtype=torch.float32, device="cpu"))
    actions = decoded[0].numpy().astype(np.float32)
    if actions.shape != (horizon, 14):
        raise ValueError("Fast-WAM action normalizer changed the execution shape")
    # Check before prefix selection can hide invalid values.
    if not np.isfinite(actions).all():
        raise ValueError("Fast-WAM actions contain NaN or Inf")
    return WAMPrediction(actions=actions[:execute_steps])
