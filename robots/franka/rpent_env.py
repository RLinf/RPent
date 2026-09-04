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

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from rlinf.envs.realworld.common.wrappers import apply_single_arm_wrappers
from rlinf.envs.realworld.franka.franka_env import FrankaEnv


class RPentFrankaEnv(FrankaEnv):
    """FrankaEnv variant used as the RPent real-robot contract."""

    def go_to_rest(self, joint_reset: bool = False) -> None:
        """Lift away from the workspace before moving to the reset pose."""
        self._end_effector_action(np.array([-1.0]))
        self._franka_state = self._controller.get_state().wait()[0]
        self._move_action(self._franka_state.tcp_pose)
        self._franka_state = self._controller.get_state().wait()[0]

        reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
        reset_pose[2] += 0.10
        self._interpolate_move(reset_pose, timeout=1)
        super().go_to_rest(joint_reset)


def create_rpent_franka_env(
    override_cfg: dict,
    worker_info: object,
    hardware_info: object,
    env_idx: int,
    env_cfg: dict,
) -> gym.Env:
    """Create the RPent-specific single-Franka environment."""
    env = RPentFrankaEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_single_arm_wrappers(env, env_cfg)


def register_rpent_franka_env() -> None:
    """Register the RPent-specific Franka environment with Gymnasium."""
    if "RPentFrankaEnv-v1" not in gym.registry:
        register(
            id="RPentFrankaEnv-v1",
            entry_point=("robots.franka.rpent_env:create_rpent_franka_env"),
        )
