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

"""RPent-specific single-Franka environment configuration and reset behavior."""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from rlinf.envs.real.franka.base import FrankaEnv
from rlinf.envs.real.wrappers import build_stack
from rlinf.robotics.parts.cameras import CameraInfo, RealSenseCamera


def realsense_color_intrinsics(camera: RealSenseCamera) -> dict[str, Any]:
    """Return the color-stream intrinsics of a connected RealSense camera."""
    if not isinstance(camera, RealSenseCamera):
        raise TypeError(
            "camera projection metadata requires RLinf RealSenseCamera, "
            f"got {type(camera).__name__}"
        )

    import pyrealsense2 as rs

    intrinsics = (
        camera.profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


class RPentFrankaEnv(FrankaEnv):
    """FrankaEnv variant used as the RPent real-robot contract."""

    def get_live_camera_observation(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Refresh the robot once and return its current processed RGB-D frames."""
        if not self.config.is_dummy:
            self._franka_state = self._read_robot()
        observation = self._get_observation()
        return observation.get("frames", {}), observation.get("depths", {})

    @staticmethod
    def _crop_bounds(
        *,
        width: int,
        height: int,
        crop_region: tuple[float, float, float, float] | None,
    ) -> tuple[int, int, int, int]:
        """Return the crop that RLinf applies before its 128px resize."""
        if crop_region is not None:
            top, left, bottom, right = crop_region
            return (
                int(width * left),
                int(height * top),
                int(width * right),
                int(height * bottom),
            )
        crop_size = min(height, width)
        x1 = (width - crop_size) // 2
        y1 = (height - crop_size) // 2
        return x1, y1, x1 + crop_size, y1 + crop_size

    @classmethod
    def _camera_projection_metadata(
        cls,
        *,
        camera_info: CameraInfo,
        raw_intrinsics: dict[str, Any],
        output_size: tuple[int, int],
        depth_scale: float,
        depth_enabled: bool,
    ) -> dict[str, Any]:
        """Describe the image transform used by RLinf's camera observation."""
        raw_width = raw_intrinsics["width"]
        raw_height = raw_intrinsics["height"]
        x1, y1, x2, y2 = cls._crop_bounds(
            width=raw_width,
            height=raw_height,
            crop_region=camera_info.crop_region,
        )
        output_width, output_height = output_size
        scale_x = output_width / float(x2 - x1)
        scale_y = output_height / float(y2 - y1)
        metadata = {
            "name": camera_info.name,
            "serial_number": camera_info.serial_number,
            "camera_type": camera_info.camera_type,
            "raw_resolution": [raw_width, raw_height],
            "output_resolution": [output_width, output_height],
            "crop_bounds_xyxy": [x1, y1, x2, y2],
            "crop_region": (
                list(camera_info.crop_region)
                if camera_info.crop_region is not None
                else None
            ),
            "depth_scale": depth_scale,
            "depth_aligned_to_color": depth_enabled,
            "extrinsic_cam2base": None,
            "extrinsic_cam2ee": None,
            "raw_color_intrinsics": raw_intrinsics,
            "intrinsic_K": [
                [
                    raw_intrinsics["fx"] * scale_x,
                    0.0,
                    (raw_intrinsics["ppx"] - x1) * scale_x,
                ],
                [
                    0.0,
                    raw_intrinsics["fy"] * scale_y,
                    (raw_intrinsics["ppy"] - y1) * scale_y,
                ],
                [0.0, 0.0, 1.0],
            ],
        }
        return metadata

    def get_camera_metadata(self) -> dict[str, Any]:
        """Return projection metadata matching the emitted RGB-D observations."""
        cameras = {}
        for camera in self._cameras.values():
            info = camera.camera_info
            raw_intrinsics = realsense_color_intrinsics(camera)
            output_height, output_width = self.observation_space["frames"][
                info.name
            ].shape[:2]
            cameras[info.name] = self._camera_projection_metadata(
                camera_info=info,
                raw_intrinsics=raw_intrinsics,
                output_size=(int(output_width), int(output_height)),
                depth_scale=float(camera.depth_scale),
                depth_enabled=bool(info.enable_depth),
            )
        return {
            "source": "rlinf_franka_env",
            "image_coordinate_convention": "pixel [u, v] maps to array [v, u]",
            "depth_unit": "m",
            "depth_aligned_to_color": bool(self.config.enable_camera_depth),
            "cameras": cameras,
        }

    def go_to_rest(self, joint_reset: bool = False) -> None:
        """Lift away from the workspace before moving to the reset pose."""
        self._end_effector_action(np.array([-1.0]))
        self._franka_state = self._read_robot()
        self._move_action(self._franka_state.tcp_pose)
        self._franka_state = self._read_robot()

        reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
        reset_pose[2] += 0.10
        self._interpolate_move(reset_pose, timeout=1)
        super().go_to_rest(joint_reset)


def create_rpent_franka_env(
    override_cfg: dict,
    worker_info: object,
    robot_info: object,
    env_idx: int,
    env_cfg: dict,
) -> gym.Env:
    """Create the RPent-specific single-Franka environment."""
    env = RPentFrankaEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        robot_info=robot_info,
        env_idx=env_idx,
    )
    return build_stack(env, env_cfg)


def register_rpent_franka_env() -> None:
    """Register the RPent-specific Franka environment with Gymnasium."""
    if "RPentFrankaEnv-v1" not in gym.registry:
        register(
            id="RPentFrankaEnv-v1",
            entry_point=("robots.franka.rpent_env:create_rpent_franka_env"),
        )
