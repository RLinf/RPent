# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Execution schema of NVIDIA's Cosmos Policy LIBERO checkpoint."""

import numpy as np

ACTION_SCHEMA = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_axis_angle_x",
    "delta_axis_angle_y",
    "delta_axis_angle_z",
    "gripper",
)
PROPRIO_FIELDS = (
    ("robot0_gripper_qpos", 2),
    ("robot0_eef_pos", 3),
    ("robot0_eef_quat", 4),
)
PROPRIO_SCHEMA = (
    "gripper_qpos_0",
    "gripper_qpos_1",
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_quat_x",
    "eef_quat_y",
    "eef_quat_z",
    "eef_quat_w",
)


def libero_request(raw_obs: dict) -> dict:
    """Pack raw LIBERO fields without changing camera orientation."""
    parts = []
    for key, size in PROPRIO_FIELDS:
        part = np.asarray(raw_obs[key], dtype=np.float32)
        if part.shape != (size,) or not np.isfinite(part).all():
            raise ValueError(f"{key} must be a finite vector of length {size}")
        parts.append(part)
    return {
        "images": {
            "primary": raw_obs["agentview_image"],
            "wrist": raw_obs["robot0_eye_in_hand_image"],
        },
        "proprio": np.concatenate(parts),
        "instruction": raw_obs["task_descriptions"],
        "embodiment": "libero_7d",
        "metadata": {"proprio_schema": list(PROPRIO_SCHEMA)},
    }
