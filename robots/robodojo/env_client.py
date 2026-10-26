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

"""RoboDojo env client — thin RPC layer over the Isaac Sim env server."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from rpent.robots.components.env_client_base import BaseEnvClient


class RoboDojoEnvClient(BaseEnvClient):
    """Unified env client for the RoboDojo Isaac Sim backend.

    Inherits :class:`BaseEnvClient` (``env.get_env_meta`` / ``env.reset`` /
    ``env.step`` / ``env.chunk_step``) and adds the RoboDojo-specific read
    surface used by the toolkit primitives (``env.get_obs``,
    ``env.get_reward_details``, ``env.solve_ik_position``, ...). The backend
    is single-environment, so no ``env_idx`` is exposed.
    """

    def __init__(self, client, *, expected_meta: dict[str, Any]):
        self.eval_fair = expected_meta.get("mode") == "eval-fair"
        self._rgb_history = deque(maxlen=177)
        self._history_valid = False
        self._inference_index = 0
        super().__init__(
            client,
            expected_meta=expected_meta,
            reset_on_connect=expected_meta.get("mode") != "eval-fair",
        )

    def _record_policy_frame(self, obs: dict) -> None:
        frame = []
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            rgb = np.asarray(obs["vision"][camera]["color"])
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f"{camera} must supply uint8 HWC RGB")
            frame.append(rgb.copy())
        if not self._rgb_history:
            self._history_step = 0
            self._rgb_anchor = frame
        else:
            self._history_step += 1
        self._rgb_history.append((self._history_step, frame))

    def reset(self):
        """Clear episode history before a potentially failing reset RPC."""
        self._history_valid = False
        self._rgb_history.clear()
        self._rgb_anchor = None
        self._inference_index = 0
        obs = super().reset()
        self._record_policy_frame(obs)
        self._history_valid = True
        return obs

    def _accept_feedback(self, obs: dict, count: int, was_valid: bool) -> None:
        if count not in (0, 1):
            raise ValueError("one action RPC must report zero or one executed action")
        if was_valid and count:
            self._record_policy_frame(obs)
        self._history_valid = was_valid
        self.last_obs = obs

    def step(self, flat_action):
        was_valid, self._history_valid = self._history_valid, False
        result = super().step(flat_action)
        self._accept_feedback(result[0], result[3]["executed_actions"], was_valid)
        return result

    def policy_observation(self) -> dict:
        """Reserve one inference index and sample anchor + stride-25 RGB history.

        Read-only observation refreshes do not advance this action timeline.
        Uncertain mutations invalidate it until an actual episode reset.
        """
        if not self._history_valid:
            raise RuntimeError(
                "VPP2 history unavailable; reset the episode before inference"
            )
        frames = dict(self._rgb_history)
        indices = [
            0,
            *[max(0, self._history_step - offset) for offset in range(175, -1, -25)],
        ]
        state = self.last_obs["state"]
        ee16 = np.concatenate(
            [
                np.asarray(state[f"{arm}_{field}"], dtype=np.float32).reshape(size)
                for arm in ("left", "right")
                for field, size in (("ee_pose", 7), ("ee_joint_state", 1))
            ]
        )
        payload = {
            "history": [self._rgb_anchor if i == 0 else frames[i] for i in indices],
            "state": ee16,
            "inference_index": self._inference_index,
        }
        self._inference_index += 1
        return payload

    def get_obs(self) -> dict[str, Any]:
        return self._client.call("env.get_obs")

    def get_status(self) -> dict[str, Any]:
        return self._client.call("env.get_status")

    def get_motion_state(self) -> dict[str, Any]:
        return self._client.call("env.get_motion_state")

    def plan_arm_path(self, arm: str, target_pose: list[float]) -> dict[str, Any]:
        return self._client.call(
            "env.plan_arm_path",
            kwargs={"arm": arm, "target_pose": target_pose},
            timeout_s=120,
        )

    def step_joint_update(
        self,
        arm: str,
        arm_qpos: list[float] | None = None,
        gripper: float | None = None,
    ) -> dict[str, Any]:
        was_valid, self._history_valid = self._history_valid, False
        result = self._client.call(
            "env.step_joint_update",
            kwargs={"arm": arm, "arm_qpos": arm_qpos, "gripper": gripper},
            timeout_s=60,
        )
        count = result["executed_actions"]
        obs = result.pop("observation") if count else self.last_obs
        self._accept_feedback(obs, count, was_valid)
        return result

    def get_reward_details(self) -> dict[str, Any]:
        return self._client.call("env.get_reward_details")

    def solve_ik_position(self, arm: str, xyz: list) -> dict[str, Any]:
        return self._client.call(
            "env.solve_ik_position", kwargs={"arm": arm, "xyz": xyz}
        )

    def get_safety_status(self) -> dict[str, Any]:
        return self._client.call("env.get_safety_status")

    def is_success(self) -> bool:
        return bool(self._client.call("env.is_success"))

    def close(self) -> None:
        try:
            self._client.call("env.close", timeout_s=10)
        except Exception:  # noqa: BLE001
            pass
