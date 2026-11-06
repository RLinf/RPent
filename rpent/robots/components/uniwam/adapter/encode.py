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

"""Map RoboTwin physical observations to the official UniWAM policy."""

import numpy as np

from rpent.robots.components.am_contracts import WAMRequest
from rpent.robots.components.uniwam import ACTION_CHUNK_SIZE


def encode_robotwin(request: WAMRequest) -> dict:
    """Keep native images and rebuild history from executed controller targets."""
    history = np.asarray(request["joint_target_history"], dtype=np.float32)
    if (
        history.ndim != 2
        or history.shape[1] != 14
        or not 1 <= len(history) <= ACTION_CHUNK_SIZE
    ):
        raise ValueError("history must have shape [1..16, 14]")
    if not np.isfinite(history).all():
        raise ValueError("history must be finite")
    return {
        "observation": {
            camera: {"rgb": request["images"][role]}
            for camera, role in (
                ("head_camera", "head"),
                ("left_camera", "left_wrist"),
                ("right_camera", "right_wrist"),
            )
        },
        "joint_action": {"vector": request["state"]["joint_targets"]},
        "instruction": request["instruction"],
        "history": history,
    }
