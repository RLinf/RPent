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


def _denormalize(
    result: dict, processor: Any, horizon: int, action_dim: int
) -> np.ndarray:
    import torch

    action = result["action"]
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if tuple(action.shape) != (1, horizon, action_dim):
        raise ValueError(f"Fast-WAM actions must have [1,{horizon},{action_dim}] shape")
    meta = processor.shape_meta["action"]
    if len(meta) != 1:
        raise ValueError("Fast-WAM requires one merged action key")
    normalizer = processor.normalizer.normalizers["action"][meta[0]["key"]]
    decoded = normalizer.backward(action.detach().to(dtype=torch.float32, device="cpu"))
    actions = decoded[0].numpy().astype(np.float32)
    if actions.shape != (horizon, action_dim):
        raise ValueError("Fast-WAM action normalizer changed the execution shape")
    # Check before gripper binarization or prefix selection can hide invalid values.
    if not np.isfinite(actions).all():
        raise ValueError("Fast-WAM actions contain NaN or Inf")
    return actions


def decode_libero(
    result: dict,
    *,
    processor: Any,
    horizon: int,
    execute_steps: int,
    binarize_gripper: bool = False,
) -> WAMPrediction:
    """Restore LIBERO gripper sign after denormalization, then select the prefix."""
    actions = _denormalize(result, processor, horizon, 7)
    actions[:, -1] = 1.0 - 2.0 * actions[:, -1]
    if binarize_gripper:
        actions[:, -1] = np.sign(actions[:, -1])
    return WAMPrediction(actions=actions[:execute_steps])


def decode_robotwin(
    result: dict,
    *,
    processor: Any,
    horizon: int,
    execute_steps: int,
    binarize_gripper: bool = False,
) -> WAMPrediction:
    """Return the official 14D qpos vector unchanged after denormalization."""
    if binarize_gripper:
        raise ValueError("LIBERO gripper binarization does not apply to RoboTwin qpos")
    return WAMPrediction(
        actions=_denormalize(result, processor, horizon, 14)[:execute_steps]
    )
