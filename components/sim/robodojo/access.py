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

"""Public observation, episode-budget fields and RGB requirements."""

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ARX_X5_JOINT_FIELDS = {
    "left_arm_joint_state": 6,
    "left_ee_joint_state": 1,
    "right_arm_joint_state": 6,
    "right_ee_joint_state": 1,
}
STATE_KEYS = tuple(
    f"{arm}_{field}"
    for arm in ("left", "right")
    for field in ("ee_pose", "ee_joint_state", "arm_joint_state")
)
CAMERA_KEYS = (
    "color",
    "depth",
    "distance_to_image_plane",
    "intrinsic_matrix",
    "extrinsic_matrix",
    "shape",
)


def public_episode_budget(status: dict) -> dict[str, int]:
    """Extract native action counters without exposing scores or diagnostics."""
    step = int(status["step"])
    step_limit = int(status["step_limit"])
    return {
        "step": step,
        "step_limit": step_limit,
        "remaining_steps": max(0, step_limit - step),
    }


def public_observation(obs: dict) -> dict:
    """Keep RGB-D, calibration, proprioception and the public instruction only."""
    return {
        "instruction": obs.get("instruction", ""),
        "state": {
            key: obs["state"][key] for key in STATE_KEYS if key in obs.get("state", {})
        },
        "vision": {
            name: {key: cam[key] for key in CAMERA_KEYS if key in cam}
            for name, cam in obs.get("vision", {}).items()
            if name in CAMERAS
        },
    }


def validate_rgb(obs: dict) -> None:
    """Require nonempty HWC uint8 RGB images from the head and wrist cameras."""
    import numpy as np

    for name in CAMERAS:
        rgb = np.asarray(obs.get("vision", {}).get(name, {}).get("color"))
        if (
            rgb.ndim != 3
            or rgb.shape[-1] != 3
            or min(rgb.shape[:2]) == 0
            or rgb.dtype != np.uint8
        ):
            raise ValueError(f"RoboDojo observation requires HWC uint8 RGB: {name}")
