# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Load a YAM robot YAML and combine it with task identity from CLI flags.

The YAML owns machine identity, calibration, operator and motion settings.
``control`` is flattened for the existing YAM runtime. Task identity belongs
to the runner and must be supplied to the env server with matching CLI flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from robots.yam.tasks import resolve_task_language

DEFAULT_CONFIG = Path(__file__).with_name("config") / "example.yaml"
EPISODE_STEPS = 1000

TASK_KEYS = (
    "task_name",
    "task_language",
    "task_description",
    "seed",
    "max_episode_steps",
)


def load_mapping(path: str | Path | None = None) -> dict[str, Any]:
    """Load a robot YAML as a plain mapping; None selects the example YAML."""
    config_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"YAM robot config must be a YAML file: {config_path}")
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"YAM robot config must be a mapping: {config_path}")
    return raw


def load_config(
    path: str | Path | None = None,
    *,
    task_name: str,
    task_language: str | None = None,
    seed: int = 0,
    max_episode_steps: int = EPISODE_STEPS,
) -> dict[str, Any]:
    """Return the flat dict consumed by YamAgentEnv and YamRgbdCameraRig."""
    if not str(task_name or "").strip():
        raise ValueError("task_name is required")
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(max_episode_steps, bool) or int(max_episode_steps) <= 0:
        raise ValueError("max_episode_steps must be a positive integer")

    raw = load_mapping(path)
    duplicated = sorted(set(raw) & set(TASK_KEYS))
    if duplicated:
        raise ValueError(
            f"task fields must come from CLI flags, not the robot YAML: {duplicated}"
        )
    control = raw.pop("control", {})
    if not isinstance(control, dict):
        raise ValueError("control must be a mapping of primitive-control knobs")
    collisions = sorted(set(raw) & set(control))
    if collisions:
        raise ValueError(
            f"control fields duplicate top-level robot fields: {collisions}"
        )
    return {
        **raw,
        **control,
        "task_name": str(task_name),
        "task_language": resolve_task_language(str(task_name), task_language),
        "seed": int(seed),
        "max_episode_steps": int(max_episode_steps),
    }


def validate_site_poses(config: dict[str, Any]) -> None:
    """Reject invalid reset/park poses before the env can open any hardware."""
    from robots.yam.geometry import as_qpos14, joint_limits_from_config

    lower, upper = joint_limits_from_config(config)
    for name in ("reset", "park_on_close"):
        pose = config.get(name)
        if not isinstance(pose, dict) or type(pose.get("enabled")) is not bool:
            raise ValueError(f"{name} must be a mapping with boolean enabled")
        if name == "park_on_close" and not pose["enabled"]:
            raise ValueError("site config must enable park_on_close")
        if not pose["enabled"]:
            continue
        left, right = pose.get("left_qpos"), pose.get("right_qpos")
        if not isinstance(left, list) or not isinstance(right, list):
            raise ValueError(f"{name} must specify left_qpos and right_qpos")
        target = as_qpos14([*left, *right], name=f"{name} pose")
        joints = target.reshape(2, 7)[:, :6]
        if np.any(joints < lower) or np.any(joints > upper):
            raise ValueError(f"{name} pose exceeds joint limits")
        for key in ("duration_s", "max_joint_delta", "tolerance", "timeout_s"):
            value = pose.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}.{key} must be a positive finite number")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name}.{key} must be a positive finite number")
