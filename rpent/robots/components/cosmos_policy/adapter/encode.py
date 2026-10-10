# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Physical observations to Cosmos Policy checkpoint inputs."""

from typing import Any

import numpy as np

STATE_SCHEMA = {"gripper_qpos": 2, "eef_position": 3, "eef_quaternion_xyzw": 4}


def encode_libero(request: dict[str, Any]) -> dict[str, Any]:
    """Pack the official two 256px images and gripper/position/quaternion state."""
    observation = {}
    for source, target in (("primary", "primary_image"), ("wrist", "wrist_image")):
        image = request["images"][source]
        if image.shape != (256, 256, 3):
            raise ValueError(f"{source} must be an RGB image shaped [256, 256, 3]")
        observation[target] = np.ascontiguousarray(image)
    observation["proprio"] = np.concatenate(
        [request["state"][name] for name in STATE_SCHEMA]
    )
    return {"observation": observation, "instruction": request["instruction"]}
