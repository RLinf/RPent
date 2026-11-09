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

"""RPent observations on an RLinf-owned LeKiwi robot."""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from rlinf.envs.real.lekiwi.drive import LeKiwiDriveEnv
from rlinf.envs.real.wrappers import build_stack
from rlinf.robotics.parts.cameras import CameraInfo


class RPentLeKiwiEnv(LeKiwiDriveEnv):
    """Keep RLinf hardware ownership and expose RPent observation metadata."""

    def __init__(self, *args, mock_wheels=False, **kwargs):
        self._camera_metadata = {}
        self._velocity_sample = {}
        self.mock_wheels = mock_wheels
        super().__init__(*args, **kwargs)

    def _camera_infos(self) -> list[CameraInfo]:
        from robots.lekiwi.runtime_config import load_device_config

        infos = super()._camera_infos()
        if self.config.is_dummy:
            return infos
        devices = load_device_config()
        return [
            CameraInfo(
                name=infos[0].name,
                serial_number=devices["device"],
                camera_type="webcam",
                resolution=(devices["eye_size"][0] * 2, devices["eye_size"][1]),
                fps=15,
            )
        ]

    def _setup_hardware(self) -> None:
        if not self.mock_wheels:
            super()._setup_hardware()
            return
        from rlinf.robotics.parts.mobility.base import MobileBase
        from rlinf.robotics.robots.lekiwi import LeKiwiRobot

        from robots.lekiwi.control import MockBase

        class MockWheelPart(MockBase, MobileBase):
            @property
            def observation_features(self):
                return {"velocity": {"shape": (3,), "dtype": "float32"}}

            @property
            def action_features(self):
                return self.observation_features

        self._base = MockWheelPart()
        self.robot = LeKiwiRobot(
            base=self._base,
            **LeKiwiRobot.build_cameras({i.name: i for i in self._camera_infos()}),
        )
        self.robot.connect()

    def _get_observation(self) -> dict[str, Any]:
        sampled_at, sampled_monotonic = time.time(), time.monotonic()
        observation = super()._get_observation()
        self._velocity_sample = {
            "velocity": np.asarray(observation["state"]["velocity"]).tolist(),
            "sampled_at": sampled_at,
            "sampled_monotonic": sampled_monotonic,
        }
        return observation

    def get_velocity_sample(self) -> dict[str, Any]:
        return dict(self._velocity_sample)

    def stop(self) -> None:
        if not self.config.is_dummy:
            self._base.reset()

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """Use left-eye RGB for observations and keep raw stereo in RLinf's cache."""
        from rlinf.envs.real.utils.image import crop_and_resize

        frames = {}
        for camera in self._cameras:
            stamp, stereo = camera.get_frame(
                max_age=self.config.camera_max_age, return_timestamp=True
            )
            height, width = stereo.shape[:2]
            left_rgb = stereo[:, : width // 2, ::-1].copy()
            size = self.observation_space["frames"][camera.name].shape[:2][::-1]
            frames[camera.name] = crop_and_resize(left_rgb, size)
            self._camera_metadata[camera.name] = {
                "source_stamp": stamp,
                "clock_domain": "host_monotonic",
                "timestamp_kind": "uvc_read_complete",
                "encoding": "rgb8",
                "width": width // 2,
                "height": height,
                "eye": "left",
            }
        if hasattr(self, "camera_player"):
            self.camera_player.put_frame(frames)
        return frames

    def get_camera_metadata(self) -> dict[str, Any]:
        if self.config.is_dummy:
            return {
                "source": "rlinf_dummy",
                "cameras": {
                    info.name: {
                        "serial_number": info.serial_number,
                        "camera_type": info.camera_type,
                        "resolution": list(info.resolution),
                    }
                    for info in self._camera_infos()
                },
            }
        cameras = {}
        for camera in self._cameras:
            cameras[camera.name] = {
                "serial_number": camera.camera_info.serial_number,
                "camera_type": camera.camera_info.camera_type,
                "resolution": list(camera.camera_info.resolution),
                **self._camera_metadata.get(camera.name, {}),
                "observation_resolution": list(self.config.frame_size),
            }
        return {
            "source": "webcam",
            "cameras": cameras,
        }


def create_rpent_lekiwi_env(
    override_cfg: dict,
    worker_info: object,
    robot_info: object,
    env_idx: int,
    env_cfg: dict,
) -> gym.Env:
    cfg = dict(override_cfg)
    mock = cfg.pop("mock_wheels", False)
    env = RPentLeKiwiEnv(
        override_cfg=cfg,
        worker_info=worker_info,
        robot_info=robot_info,
        env_idx=env_idx,
        mock_wheels=mock,
    )
    return build_stack(env, env_cfg)


def register_rpent_lekiwi_env() -> None:
    if "RPentLeKiwiDriveEnv-v1" not in gym.registry:
        register(
            id="RPentLeKiwiDriveEnv-v1",
            entry_point="robots.lekiwi.rpent_env:create_rpent_lekiwi_env",
        )
