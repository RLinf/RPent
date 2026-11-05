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

"""RPent-specific LeKiwi environment: live reads, stop, and camera identity."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from rlinf.envs.real.lekiwi.drive import LeKiwiDriveEnv
from rlinf.envs.real.wrappers import build_stack


class RPentLeKiwiEnv(LeKiwiDriveEnv):
    """LeKiwiDriveEnv variant used as the RPent real-robot contract.

    ``LeKiwiDriveEnv`` converts the ``override_cfg`` dict into a
    ``LeKiwiEnvConfig`` and names the task; this subclass adds the live
    reads, camera identity, and immediate stop that RPent's RPC facade
    serves.
    """

    def get_live_observation(self) -> dict[str, Any]:
        """Refresh the sensors once without commanding motion."""
        return self._get_observation()

    def get_measured_velocity(self) -> np.ndarray:
        """Return the body velocity the wheels currently report."""
        if self.config.is_dummy:
            return np.zeros(3, dtype=np.float32)
        reading = self._state_parts.get_observation()["base"]
        return np.asarray(reading["velocity"], dtype=np.float32)

    def get_camera_metadata(self) -> dict[str, Any]:
        """Return the identity of every camera behind the emitted frames."""
        cameras = {
            camera.name: {
                "serial_number": camera.camera_info.serial_number,
                "camera_type": camera.camera_info.camera_type,
                "resolution": list(camera.camera_info.resolution),
            }
            for camera in getattr(self, "_cameras", [])
        }
        return {"source": "rlinf_lekiwi_env", "cameras": cameras}

    def stop(self) -> None:
        """Command zero wheel velocity immediately, without a control step."""
        if not self.config.is_dummy:
            self._base.reset()


def create_rpent_lekiwi_env(
    override_cfg: dict,
    worker_info: object,
    robot_info: object,
    env_idx: int,
    env_cfg: dict,
) -> gym.Env:
    """Create the RPent-specific LeKiwi drive environment."""
    env = RPentLeKiwiEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        robot_info=robot_info,
        env_idx=env_idx,
    )
    return build_stack(env, env_cfg)


def register_rpent_lekiwi_env() -> None:
    """Register the RPent-specific LeKiwi environment with Gymnasium."""
    if "RPentLeKiwiDriveEnv-v1" not in gym.registry:
        register(
            id="RPentLeKiwiDriveEnv-v1",
            entry_point="robots.lekiwi.rpent_env:create_rpent_lekiwi_env",
        )
