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

"""Fast-WAM encoders matching the official LIBERO and RoboTwin evaluation paths."""

from typing import Any

import numpy as np
from PIL import Image


def _axis_angle(quaternion: np.ndarray) -> np.ndarray:
    # Match Fast-WAM/robosuite's quaternion branch, including negative w.
    # scipy's shortest-rotation convention differs for that branch.
    w = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - w * w)
    if denominator == 0:
        return np.zeros(3, dtype=np.float32)
    return quaternion[:3] * (2.0 * np.arccos(w) / denominator)


def encode_libero(
    request: dict,
    *,
    processor: Any,
    **_: Any,
) -> dict[str, Any]:
    """Map physical observations to RLinf's batched LIBERO policy input."""
    state = request["state"]
    proprio = np.concatenate(
        (
            state["eef_position"],
            _axis_angle(state["eef_quaternion_xyzw"]),
            state["gripper_qpos"],
        )
    ).astype(np.float32)
    # Physical views have the vertical flip; RLinf's policy expects the
    # official 180-degree orientation. RLinf owns resizing and normalization.
    return {
        "main_images": np.ascontiguousarray(request["images"]["primary"][:, ::-1])[
            None
        ],
        "wrist_images": np.ascontiguousarray(request["images"]["wrist"][:, ::-1])[None]
        if processor.num_output_cameras == 2
        else None,
        "states": proprio[None],
        "task_descriptions": [request["instruction"]],
    }


def encode_robotwin(
    request: dict,
    *,
    processor: Any,
    device: str,
    dtype: Any,
    prompt_template: str,
    video_size: tuple[int, int],
    concat: str = "robotwin",
) -> dict[str, Any]:
    """Match upstream's head-over-two-wrists layout and joint_action.vector."""
    import torch

    if (
        tuple(video_size) != (384, 320)
        or concat != "robotwin"
        or processor.num_output_cameras != 3
    ):
        raise ValueError(
            "RoboTwin requires three cameras in the [384,320] robotwin layout"
        )
    images = [
        np.asarray(
            Image.fromarray(request["images"][role]).resize(
                size, Image.Resampling.BILINEAR
            )
        )
        for role, size in (
            ("head", (320, 256)),
            ("left_wrist", (160, 128)),
            ("right_wrist", (160, 128)),
        )
    ]
    image = np.concatenate((images[0], np.concatenate(images[1:], axis=1)), axis=0)
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("Fast-WAM requires one merged state key")
    key = state_meta[0]["key"]
    state = torch.tensor(request["state"]["joint_targets"], dtype=torch.float32)
    batch = processor.action_state_transform({"state": {key: state[None]}})
    batch = processor.normalizer.forward(batch)
    pixels = torch.tensor(image).permute(2, 0, 1)[None].to(device=device, dtype=dtype)
    return {
        "prompt": prompt_template.format(task=request["instruction"]),
        "input_image": pixels * (2.0 / 255.0) - 1.0,
        "proprio": batch["state"][key],
    }
