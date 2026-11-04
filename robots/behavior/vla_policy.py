# Copyright 2026 The RPent Authors.
# Copyright 2026 The RLinf Authors.
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

"""Thin official-observation adapter for RLinf's OpenPI PyTorch eval model.

The R1Pro state embedding and image layout preserve the BEHAVIOR evaluation
adapter's inputs. RLinf owns resizing, tokenization, normalization, sampling
and output unnormalization. This module neither executes nor queues actions.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

# (official public proprio slice, RLinf BehaviorInputs source slice).
_STATE_SLICES = (
    (slice(0, 3), slice(253, 256)),
    (slice(3, 10), slice(158, 165)),
    (slice(24, 26), slice(193, 195)),
    (slice(28, 35), slice(197, 204)),
    (slice(49, 51), slice(232, 234)),
    (slice(53, 57), slice(236, 240)),
)
MODEL_ARGUMENTS = {
    "state_token": "abs_joint",
    "config_name": "pi05_behavior",
    "action_chunk": 32,
    "action_dim": 23,
    "model_action_dim": 32,
    "num_steps": 5,
    "max_token_len": 200,
    "max_new_tokens": 24,
    "language_temperature": 0.0,
}


def make_model_config(checkpoint: str) -> Any:
    """Build the RLinf eval config without changing model or transform settings."""
    from omegaconf import OmegaConf

    args = MODEL_ARGUMENTS
    return OmegaConf.create(
        {
            "model_path": checkpoint,
            "precision": "bf16",
            "num_action_chunks": args["action_chunk"],
            "action_dim": args["action_dim"],
            "num_steps": args["num_steps"],
            "add_value_head": False,
            "openpi": {
                "task": "eval",
                "config_name": args["config_name"],
                "mode": "vla",
                "state_token": args["state_token"],
                "action_env_dim": args["action_dim"],
                "action_chunk": args["action_chunk"],
                "num_steps": args["num_steps"],
                "model_action_dim": args["model_action_dim"],
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": args["max_token_len"],
                "max_new_tokens": args["max_new_tokens"],
                "language_temperature": args["language_temperature"],
                "language_loss_weight": 1.0,
                "action_loss_weight": 1.0,
                "language_loss_probability": 1.0,
                "stop_gradient_to_vlm": False,
            },
        }
    )


def _rgb(value: Any, role: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{role} camera must be a 3D image array: {image.shape}")
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] < 3:
        raise ValueError(f"{role} camera needs at least 3 channels: {image.shape}")
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return np.ascontiguousarray(image)


def adapt_observation(observation: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Batch one official frame in the input layout consumed by RLinf.

    Keep images at their received resolution. Public R1Pro proprio is embedded
    in the legacy 256D layout that RLinf's ``BehaviorInputs`` consumes; existing
    256D inputs are passed through. No state normalization occurs here.
    """
    keys = {
        key.decode() if isinstance(key, bytes) else str(key): key for key in observation
    }

    def sensor(suffix: str) -> Any:
        matches = [key for key in keys if key.endswith(suffix)]
        if len(matches) != 1:
            raise KeyError(f"Expected one sensor ending in {suffix!r}: {matches}")
        return observation[keys[matches[0]]]

    proprio = np.asarray(sensor("::proprio"))
    if proprio.ndim != 1:
        raise ValueError(f"Expected one 1D proprio frame, got {proprio.shape}")
    proprio = np.asarray(proprio, dtype=np.float32)
    if proprio.size >= 256:
        state = np.ascontiguousarray(proprio[:256])
    else:
        if proprio.size < 57:
            raise ValueError("R1Pro proprio needs at least 57 values or a 256D state")
        state = np.zeros(256, dtype=np.float32)
        for source, target in _STATE_SLICES:
            state[target] = proprio[source]
    head = _rgb(sensor("zed_link:Camera:0::rgb"), "head")
    left = _rgb(sensor("left_realsense_link:Camera:0::rgb"), "left wrist")
    right = _rgb(sensor("right_realsense_link:Camera:0::rgb"), "right wrist")
    return {
        "states": state[None],
        "main_images": head[None],
        "wrist_images": np.stack((left, right), axis=0)[None],
        "task_descriptions": np.asarray([prompt], dtype=object),
    }


class RlinfBehaviorPolicy:
    """Infer fresh chunks through an already constructed RLinf VLA eval model.

    Args:
        backend: RLinf model implementing ``predict_action_batch(..., mode='eval')``.
        task_prompt: Default main-task instruction. The service enforces whether
            callers may override it; this adapter never changes its default.
    """

    def __init__(self, backend: Any, *, task_prompt: str) -> None:
        if getattr(getattr(backend, "model", None), "vlm_vla", False):
            raise ValueError("BEHAVIOR VLA service requires a VLA-only RLinf model")
        self.backend = backend
        self.task_prompt = task_prompt
        self._lock = threading.RLock()

    def reset(self) -> None:
        """Reset model-local state without advancing or resetting the environment."""
        with self._lock:
            reset = getattr(self.backend, "reset", None)
            if callable(reset):
                reset()

    def infer_chunk(
        self, observation: dict[str, Any], prompt: str | None = None
    ) -> np.ndarray:
        """Return the model's native unnormalized R1Pro action channels."""
        import torch

        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("prompt must be a nonempty string or None")
        with self._lock:
            env_obs = adapt_observation(
                observation, self.task_prompt if prompt is None else prompt
            )
            with torch.inference_mode():
                actions, _ = self.backend.predict_action_batch(env_obs, mode="eval")
            if isinstance(actions, dict):
                for key in ("actions", "action", "pred_actions"):
                    if key in actions:
                        actions = actions[key]
                        break
                else:
                    raise ValueError("Model output dictionary has no actions key")
            if hasattr(actions, "detach"):
                actions = actions.detach().float().cpu().numpy()
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim == 3 and actions.shape[0] >= 1:
                actions = actions[0]
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < 23:
                raise ValueError(
                    f"Model must return nonempty [T,A>=23]: {actions.shape}"
                )
            if not np.isfinite(actions).all():
                raise ValueError("Model returned NaN or infinite action values")
            return np.ascontiguousarray(actions[:, :23]).copy()
