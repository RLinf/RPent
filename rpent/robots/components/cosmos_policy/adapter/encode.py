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
