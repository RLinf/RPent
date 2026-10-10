# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Model-independent views and joint state from RPent's RoboTwin environment."""

from typing import Any

import numpy as np

ROBOTWIN_CAMERAS = ("head", "left_wrist", "right_wrist")


def physical_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Keep native RGB orientation and the native joint-action vector.

    ``qpos_target14`` matches RoboTwin's ``joint_action.vector``: left arm's
    six joint targets and gripper, followed by the right arm's targets and
    gripper. These are controller setpoints, not measured joint positions.
    Camera resizing and state normalization belong to the model adapter.
    """
    native = observation["robot_state"]
    fields = {
        "qpos_target14": "joint_targets",
        "arm_qpos_real12": "joint_positions",
        "left_eef_pose": "left_eef_pose",
        "right_eef_pose": "right_eef_pose",
        "left_tcp_pose": "left_tcp_pose",
        "right_tcp_pose": "right_tcp_pose",
        "left_gripper": "left_gripper",
        "right_gripper": "right_gripper",
    }
    return {
        "images": {name: view["rgb"] for name, view in observation["views"].items()},
        "state": {
            target: np.atleast_1d(native[source])
            for source, target in fields.items()
            if source in native
        },
        "instruction": observation["task_language"],
    }
