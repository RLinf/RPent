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

"""R1Pro observations and checkpoint-specific transforms for the shared Pi0.5 worker."""

from __future__ import annotations

from pathlib import Path
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


def _behavior_rgb(value: Any, role: str) -> np.ndarray:
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


def encode_observation(env_obs: dict) -> dict:
    """Encode a BEHAVIOR frame in RLinf's three-view, padded R1Pro layout."""
    instruction = env_obs.get("task_descriptions")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("BEHAVIOR task_descriptions must be a nonempty instruction")

    def sensor(suffix: str) -> Any:
        matches = [
            value
            for key, value in env_obs.items()
            if isinstance(key, (str, bytes))
            and (key.decode() if isinstance(key, bytes) else key).endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one BEHAVIOR sensor ending in {suffix!r}")
        return matches[0]

    proprio = np.asarray(sensor("::proprio"), dtype=np.float32)
    if proprio.shape not in {(61,), (256,)} or not np.isfinite(proprio).all():
        raise ValueError(
            "expected a finite R1Pro proprio vector of shape (61,) or (256,), "
            f"got shape {proprio.shape}"
        )
    if proprio.size == 256:
        state = proprio
    else:
        state = np.zeros(256, dtype=np.float32)
        for source, target in _STATE_SLICES:
            state[target] = proprio[source]
    head = _behavior_rgb(sensor("zed_link:Camera:0::rgb"), "head")
    left = _behavior_rgb(sensor("left_realsense_link:Camera:0::rgb"), "left wrist")
    right = _behavior_rgb(sensor("right_realsense_link:Camera:0::rgb"), "right wrist")
    return {
        "main_images": head[None],
        "wrist_images": np.stack((left, right))[None],
        "extra_view_images": None,
        "states": state[None],
        "task_descriptions": [instruction],
    }


def native_action_batch(actions: Any, chunk_size: int) -> np.ndarray:
    """Extract a single finite R1Pro action chunk, discarding model padding."""
    import torch

    if isinstance(actions, dict):
        for key in ("actions", "action", "pred_actions"):
            if key in actions:
                actions = actions[key]
                break
        else:
            raise ValueError("BEHAVIOR model output has no actions")
    if torch.is_tensor(actions):
        actions = actions.detach().float().cpu().numpy()
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim == 1:
        array = array[None]
    if array.ndim == 2:
        array = array[None]
    if (
        array.ndim != 3
        or array.shape[0] != 1
        or array.shape[1] != chunk_size
        or array.shape[2] < 23
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"expected finite BEHAVIOR actions [1,{chunk_size},A>=23], got {array.shape}"
        )
    return np.ascontiguousarray(array[:, :, :23])


def _norm_stats_dir(cfg: Any, state_layout: str) -> Path | None:
    """Locate the training statistics for a local BEHAVIOR checkpoint."""
    options = cfg.get("openpi_data") or {}
    checkpoint = Path(cfg.model_path).expanduser()
    if options.get("norm_stats_path") is not None:
        path = Path(options["norm_stats_path"]).expanduser()
        stats = path if path.is_file() else path / "norm_stats.json"
        if stats.name != "norm_stats.json" or not stats.is_file():
            raise FileNotFoundError(f"expected BEHAVIOR norm_stats.json at {path}")
        return stats.parent.resolve()

    if options.get("repo_id") is not None:
        dataset = Path(options["repo_id"])
        if not dataset.parts or dataset.is_absolute() or ".." in dataset.parts:
            raise ValueError(
                f"expected a checkpoint-relative dataset ID, got {dataset}"
            )
        directories = (checkpoint / dataset, checkpoint / "assets" / dataset)
    elif state_layout == "abs_joint":
        # The PyTorch loader requires the statistics for its training preset.
        return None
    else:
        directories = (
            checkpoint / "assets/behavior-1k/2025-challenge-demos",
            # Dataset used by RLinf's pi05_behavior training preset.
            checkpoint / "physical-intelligence/behavior",
        )
    for directory in directories:
        if (directory / "norm_stats.json").is_file():
            return directory.resolve()
    raise FileNotFoundError(
        f"missing BEHAVIOR norm_stats.json in {directories[0]} or {directories[1]}; "
        "specify --norm-stats-path for a different location"
    )


def load_model(cfg: Any, state_layout: str) -> Any:
    """Load a BEHAVIOR policy with its training-time state order and statistics.

    Comet places the two grippers after both arms. The ABS-joint VLA export
    places each gripper next to its arm, using RLinf's OpenPI PyTorch eval
    transforms. Both consume the same public R1Pro observations.
    """
    if state_layout not in ("comet", "abs_joint"):
        raise ValueError(f"unknown BEHAVIOR state layout: {state_layout!r}")
    stats_dir = _norm_stats_dir(cfg, state_layout)
    if stats_dir is not None:
        if cfg.get("openpi_data") is None:
            cfg.openpi_data = {}
        cfg.openpi_data.norm_stats_path = str(stats_dir)
    if state_layout == "abs_joint":
        try:
            from rlinf.models.embodiment.openpi_pytorch import get_model
        except ModuleNotFoundError as exc:
            if exc.name != "rlinf.models.embodiment.openpi_pytorch":
                raise
            raise RuntimeError(
                "abs_joint requires an RLinf runtime with the "
                "openpi_pytorch BEHAVIOR eval loader"
            ) from exc
        cfg.openpi.state_token = "abs_joint"
        if stats_dir is not None:
            # This runtime reads the actual Normalize/Unnormalize assets from
            # openpi.assets_dir/asset_id even when openpi_data is provided.
            cfg.openpi.assets_dir = str(stats_dir.parent)
            cfg.openpi.asset_id = stats_dir.name
        return get_model(cfg, torch_dtype=None)

    from rlinf.models.embodiment.openpi import get_model
    from rlinf.models.embodiment.openpi.dataconfig.behavior_dataconfig import (
        LeRobotBehaviorDataConfig,
    )

    if hasattr(LeRobotBehaviorDataConfig, "state_token"):
        # The older two-loader RLinf distribution defaults to the newer state
        # order and loads its legacy weights non-strictly. Check the backbone
        # keys before invoking it to avoid running with uninitialized weights.
        from safetensors import safe_open

        with safe_open(
            Path(cfg.model_path) / "model.safetensors", framework="numpy"
        ) as f:
            if not any(k.startswith("paligemma_with_expert.") for k in f.keys()):
                raise ValueError(
                    "this RLinf Comet loader requires OpenPI-format weights "
                    "(paligemma_with_expert). For native Comet weights use the "
                    "unified RLinf loader or select abs_joint for an ABS-joint policy."
                )
        cfg.openpi_data.state_token = "abs_joint_old"
    return get_model(cfg, torch_dtype=None)
