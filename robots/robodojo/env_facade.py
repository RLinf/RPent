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

"""Main-thread RPC facade for the RoboDojo agent environment."""

from __future__ import annotations

from typing import Any

from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin


class RoboDojoEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """Keep the existing RoboDojo RPC method names and wire payloads."""

    def __init__(self, env) -> None:
        self.env = env
        self.eval_fair = env.eval_fair
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc.update(
            {
                "env.get_obs": self.get_obs,
                "env.get_status": self.get_status,
                "env.get_reward_details": self.get_reward_details,
                "env.get_safety_status": self.get_safety_status,
                "env.solve_ik_position": self.solve_ik_position,
                "env.get_motion_state": self.get_motion_state,
                "env.plan_arm_path": self.plan_arm_path,
                "env.step_joint_update": self.step_joint_update,
                "env.hold_step": self.hold_step,
                "env.is_success": self.is_success,
                "env.close": self.request_close,
            }
        )
        self._readonly_methods.update(
            [
                "env.get_env_meta",
                "env.get_task_language",
                "env.get_camera_meta",
                "env.render_camera",
                "env.get_obs",
                "env.get_status",
                "env.get_reward_details",
                "env.get_safety_status",
                "env.solve_ik_position",
                "env.get_motion_state",
                "env.plan_arm_path",
            ]
        )
        if self.eval_fair:
            # Restrict RPC visibility for replay clients, independently of
            # planner tool registration (which never exposes diagnostics).
            for method in (
                "env.get_reward_details",
                "env.get_safety_status",
                "env.is_success",
                "env.reset",
            ):
                self._rpc.pop(method, None)

    def get_env_meta(self) -> dict:
        return self.env.get_env_meta()

    def get_task_language(self) -> str:
        return self.env.get_task_language()

    def get_camera_meta(self, camera_name: str, height=None, width=None) -> dict:
        return self.env.get_camera_meta(camera_name, height, width)

    def render_camera(
        self, camera_name: str, height=None, width=None, depth=False
    ) -> Any:
        return self.env.render_camera(camera_name, height, width, depth)

    def reset(self) -> dict:
        return self.env.reset_episode()

    def step(self, flat_action, *, expected_step=None):
        return self.env.step_native(flat_action, expected_step=expected_step)

    def chunk_step(self, flat_actions, *, return_all_frames: bool = False):
        raise NotImplementedError("RoboDojo uses step for action control")

    def get_obs(self) -> dict:
        return self.env.get_obs()

    def get_status(self) -> dict:
        return self.env.get_status()

    def get_reward_details(self) -> dict:
        return self.env.get_reward_details()

    def get_safety_status(self) -> dict:
        return self.env.get_safety_status()

    def solve_ik_position(self, arm: str, xyz: list) -> dict:
        return self.env.solve_ik_position(arm, xyz)

    def get_motion_state(self) -> dict:
        return self.env.get_motion_state()

    def plan_arm_path(self, arm: str, target_pose: list[float]) -> dict:
        return self.env.plan_arm_path(arm, target_pose)

    def step_joint_update(
        self,
        arm: str,
        arm_qpos: list[float] | None = None,
        gripper: float | None = None,
        *,
        expected_step: int | None = None,
    ) -> dict:
        return self.env.step_joint_update(
            arm, arm_qpos, gripper, expected_step=expected_step
        )

    def is_success(self) -> bool:
        return self.env.is_success()

    def request_close(self) -> None:
        self._shutdown_event.set()

    def close(self) -> None:
        self.env.close()

    def hold_step(self, *, expected_step=None) -> dict:
        return self.env.hold_step(expected_step=expected_step)
