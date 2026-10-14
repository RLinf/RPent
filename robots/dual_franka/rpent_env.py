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

"""RPent observations on top of the official RLinf dual-Franka environment."""

from __future__ import annotations

import copy
import queue
from dataclasses import replace
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from rlinf.envs.real.franka.dual_franka_tcp import DualFrankaTCPEnv
from rlinf.envs.real.wrappers import build_stack
from rlinf.robotics.parts.cameras import Camera, CameraInfo, RealSenseCamera

from robots.franka.rpent_env import realsense_color_intrinsics

_CAMERA_FRAME_TIMEOUT_S = 0.5
_REALSENSE_BACKEND = "rpent_realsense"


@Camera.register(_REALSENSE_BACKEND)
class RPentRealSenseCamera(RealSenseCamera):
    """Expose projection metadata from the camera's existing SDK connection."""

    def get_color_intrinsics(self) -> dict[str, Any]:
        """Read serializable intrinsics on the node that owns the camera."""
        return realsense_color_intrinsics(self)


class RPentDualFrankaEnv(DualFrankaTCPEnv):
    """Preserve policy observations and retain their uncropped RGB-D for RPent."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._rpent_camera_snapshot: dict[str, dict[str, np.ndarray]] | None = None
        super().__init__(*args, **kwargs)

    def _camera_infos(self) -> list[CameraInfo]:
        # A public method on this driver is also forwarded by RLinf's remote
        # camera handle; SDK profile objects themselves cannot cross that boundary.
        return [
            replace(info, camera_type=_REALSENSE_BACKEND)
            if info.camera_type == "realsense"
            else info
            for info in super()._camera_infos()
        ]

    def _get_camera_observation(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Collect once for policy and perception, invalidating stale raw data."""
        self._rpent_camera_snapshot = None
        frames: dict[str, np.ndarray] = {}
        depths: dict[str, np.ndarray] = {}
        raw_frames: dict[str, np.ndarray] = {}
        raw_depths: dict[str, np.ndarray] = {}
        display_frames: dict[str, np.ndarray] = {}
        all_fresh = True

        for name, camera in self._cameras.items():
            try:
                reading = camera.get_observation(
                    timeout=_CAMERA_FRAME_TIMEOUT_S,
                    max_age=_CAMERA_FRAME_TIMEOUT_S,
                )
            except queue.Empty:
                # Preserve RLinf's policy fallback during camera recovery, but
                # never present its old image as a fresh projection observation.
                cached = self._last_camera_frame.get(name)
                if cached is None:
                    raise RuntimeError(
                        f"Camera {name} stalled with no cached frame to fall back to."
                    ) from None
                self._logger.error("Camera %s stalled; using the last frame.", name)
                reading = cached
                all_fresh = False

            raw_frames[name] = reading["frame"][..., ::-1].copy()
            reshape_size = self.observation_space["frames"][name].shape[:2][::-1]
            cropped, resized = self._crop_frame(reading["frame"], reshape_size)
            frames[name] = resized[..., ::-1]
            display_frames[name] = resized
            display_frames[f"{name}_full"] = cropped
            if "depth" in reading:
                raw_depths[name] = reading["depth"].copy()
                _, depths[name] = self._crop_frame(
                    reading["depth"], reshape_size, interpolation=cv2.INTER_NEAREST
                )
            self._last_camera_frame[name] = reading

        self.camera_player.put_frame(display_frames)
        if all_fresh:
            self._rpent_camera_snapshot = {
                "raw_frames": raw_frames,
                "raw_depths": raw_depths,
            }
        return frames, depths

    def get_raw_camera_metadata(self) -> dict[str, dict[str, Any]]:
        """Describe the complete, uncropped snapshot and supported calibration."""
        if self.config.is_dummy:
            return {}
        snapshot = self._rpent_camera_snapshot
        if snapshot is None:
            raise RuntimeError(
                "No fresh complete camera snapshot; acquire an observation first."
            )
        metadata = {}
        for name, camera in self._cameras.items():
            info = camera.camera_info
            depth = snapshot["raw_depths"].get(name)
            intrinsics = (
                camera.get_color_intrinsics()
                if isinstance(camera, RPentRealSenseCamera)
                else None
            )
            frame = snapshot["raw_frames"][name]
            height, width = frame.shape[:2]
            metadata[name] = {
                "name": name,
                "serial_number": info.serial_number,
                "camera_type": "realsense"
                if info.camera_type == _REALSENSE_BACKEND
                else info.camera_type,
                "raw_resolution": [width, height],
                "output_resolution": [width, height],
                "crop_bounds_xyxy": [0, 0, width, height],
                "crop_region": None,
                "depth_scale": float(camera.depth_scale),
                "depth_unit": "m" if depth is not None else None,
                "depth_aligned_to_color": bool(info.enable_depth),
                "extrinsic_cam2base": None,
                "extrinsic_cam2ee": None,
                "raw_color_intrinsics": intrinsics,
                "intrinsic_K": (
                    [
                        [intrinsics["fx"], 0.0, intrinsics["ppx"]],
                        [0.0, intrinsics["fy"], intrinsics["ppy"]],
                        [0.0, 0.0, 1.0],
                    ]
                    if intrinsics is not None
                    else None
                ),
                "depth_enabled": bool(info.enable_depth),
                "depth_available": depth is not None,
                "projection_supported": depth is not None and intrinsics is not None,
                "rgb_shape": list(frame.shape),
                "depth_shape": None if depth is None else list(depth.shape),
            }
        return metadata

    def get_raw_camera_snapshot(self) -> dict[str, Any]:
        """Copy the preceding acquisition, or return empty physical views in dummy mode."""
        if self.config.is_dummy:
            return {"raw_frames": {}, "raw_depths": {}, "camera_meta": {}}
        metadata = self.get_raw_camera_metadata()
        return {
            **copy.deepcopy(self._rpent_camera_snapshot),
            "camera_meta": metadata,
        }


def create_rpent_dual_franka_env(
    override_cfg: dict,
    worker_info: object,
    robot_info: object,
    env_idx: int,
    env_cfg: dict,
) -> gym.Env:
    """Create the dual-Franka adapter with RLinf's standard wrapper stack."""
    env = RPentDualFrankaEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        robot_info=robot_info,
        env_idx=env_idx,
    )
    try:
        return build_stack(env, env_cfg)
    except BaseException:
        env.close()
        raise


def register_rpent_dual_franka_env() -> None:
    """Register the RPent dual-Franka adapter with Gymnasium."""
    if "RPentDualFrankaTCPEnv-v1" not in gym.registry:
        register(
            id="RPentDualFrankaTCPEnv-v1",
            entry_point="robots.dual_franka.rpent_env:create_rpent_dual_franka_env",
        )
