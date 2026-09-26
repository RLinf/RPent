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

"""RPent agent operations over the RLinf RoboDojo training adapter."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch
from rlinf.envs.sim.robodojo.robodojo_env import RoboDojoEnv

from robots.robodojo.agent_support import (
    _obs_dict,
    _reward_details,
    _SafetyMonitor,
    _solve_ik_position,
    _status,
    _VideoRecorder,
)
from rpent.utils.logging import get_logger

logger = get_logger(__name__)


def _infer_action_type(action: dict) -> str:
    return "ee" if any(key.endswith("_ee_pose") for key in action) else "joint"


def _policy_action(action: Any) -> dict:
    """Convert one ARX-X5 policy vector to the native four-joint-key action."""
    if isinstance(action, dict):
        return action
    array = np.asarray(action, dtype=np.float64)
    if array.shape != (14,) or not np.isfinite(array).all():
        raise ValueError(
            "RoboDojo policy action must have shape (14,) and finite values"
        )
    return dict(
        zip(
            (
                "left_arm_joint_state",
                "right_arm_joint_state",
                "left_ee_joint_state",
                "right_ee_joint_state",
            ),
            np.split(array, [6, 12, 13]),
        )
    )


class RoboDojoAgentEnv(RoboDojoEnv):
    """Single-scene agent adapter with native RPC observations and diagnostics."""

    def __init__(self, cfg, *, meta: dict, video_dir: str = "") -> None:
        self.meta = meta
        self.eval_fair = meta.get("mode", "dev") == "eval-fair"
        self.recorder = _VideoRecorder(video_dir)
        self.bottle_mon = _SafetyMonitor()
        super().__init__(cfg, 1, 0, 1, None, record_metrics=False)
        try:
            self.venv.reset(env_seeds=[meta["layout"]])
        except BaseException:
            self.close()
            raise

    def update_reset_state_ids(self, env_idx=None) -> None:
        """Agent episodes use the requested official layout ID."""
        self.reset_state_ids = torch.tensor([self.meta["layout"]], dtype=torch.long)

    def _sub_env(self, env_id: int = 0) -> Any:
        """Return the local runtime slot; RPC owns exactly one environment."""
        if env_id != 0:
            raise IndexError(f"invalid RoboDojo env_id: {env_id}")
        return self.venv.envs[env_id]

    @property
    def env(self) -> Any:
        return self._sub_env().env

    def get_native_obs(self) -> dict:
        """Record native RGB frames while preserving calibration and state."""
        return _obs_dict(self.env, self.recorder)

    def reset(self, env_idx=None, env_seeds=None):
        """Reset through RLinf and reinitialize agent episode diagnostics."""
        result = super().reset(env_idx=env_idx, env_seeds=env_seeds)
        self.bottle_mon = _SafetyMonitor()
        return result

    def _observation(self) -> dict:
        obs = self.get_native_obs()
        if self.eval_fair:
            from robots.robodojo.access import public_observation

            return public_observation(obs)
        return obs

    # ---- unified facade contract ----
    def get_env_meta(self) -> dict[str, Any]:
        """Return the meta info this server was launched with."""
        return dict(self.meta)

    def get_task_language(self) -> str:
        from robots.robodojo.language import resolve_instruction

        return resolve_instruction(self.env)

    def get_camera_meta(
        self,
        camera_name: str,
        height: int | None = None,
        width: int | None = None,
    ) -> dict[str, Any]:
        cam = (self.env.get_obs(env_idx=0).get("vision") or {}).get(camera_name)
        if cam is None:
            raise ValueError(f"unknown camera: {camera_name!r}")
        meta: dict[str, Any] = {
            "camera_name": camera_name,
            "intrinsic_matrix": cam.get("intrinsic_matrix"),
            "extrinsic_matrix": cam.get("extrinsic_matrix"),
        }
        color = cam.get("color")
        if color is not None:
            shape = np.asarray(color).shape
            meta["height"], meta["width"] = int(shape[0]), int(shape[1])
        return meta

    def render_camera(
        self,
        camera_name: str,
        height: int | None = None,
        width: int | None = None,
        depth: bool = False,
    ) -> Any:
        del height, width  # Isaac Sim cameras render at a fixed resolution
        cam = (self.env.get_obs(env_idx=0).get("vision") or {}).get(camera_name)
        if cam is None:
            raise ValueError(f"unknown camera: {camera_name!r}")
        rgb = np.asarray(cam.get("color"))
        if not depth:
            return rgb
        depth_map = cam.get("distance_to_image_plane")
        if depth_map is None:
            depth_map = cam.get("depth")
        return rgb, (np.asarray(depth_map) if depth_map is not None else None)

    def reset_episode(self) -> dict[str, Any]:
        if self.eval_fair:
            raise RuntimeError("eval-fair does not allow episode reset")
        self.venv.reset(env_seeds=[self.meta["layout"]])
        self.bottle_mon = _SafetyMonitor()
        return self.get_native_obs()

    def step_native(self, flat_action):
        """Apply one RoboDojo action and return ``(obs, reward, done, info)``.

        ``flat_action`` is a 14-value policy vector or a native action dict
        (joint or ee keys). The action type is inferred from the keys. Mirrors the unified
        ``BaseEnvClient.step`` contract used by the other robot backends.
        """
        flat_action = _policy_action(flat_action)
        action_type = _infer_action_type(flat_action)
        if self.eval_fair and int(self.env.take_action_cnt[0]) >= int(
            self.env.step_lim
        ):
            raise RuntimeError("Episode step budget exhausted")
        self._sub_env(0).apply_action(
            flat_action, action_type, eval_fair=self.eval_fair
        )
        if self.eval_fair:
            return self._observation(), 0.0, False, {"status": self.get_status()}
        alarms = self.bottle_mon.check(self.env)
        if alarms:
            logger.warning("SAFETY ALARM: %s", alarms)
        obs = self.get_native_obs()
        reward_details = _reward_details(self.env, self.bottle_mon)
        reward = float(reward_details.get("reward", 0.0) or 0.0)
        done = bool(self.env.is_success(env_idx=0))
        info: dict[str, Any] = {
            "status": _status(self.env, self.bottle_mon),
            "step_limit": int(self.env.step_lim),
            "safety": alarms,
        }
        return obs, reward, done, info

    # ---- RoboDojo-specific RPC ----
    def get_obs(self) -> dict[str, Any]:
        return self._observation()

    def get_status(self) -> dict[str, Any]:
        if self.eval_fair:
            return {
                "step": int(self.env.take_action_cnt[0]),
                "step_limit": int(self.env.step_lim),
            }
        return _status(self.env, self.bottle_mon)

    def get_reward_details(self) -> dict[str, Any]:
        return _reward_details(self.env, self.bottle_mon)

    def solve_ik_position(self, arm: str, xyz: list) -> dict[str, Any]:
        return _solve_ik_position(self.env, arm, xyz)

    def get_safety_status(self) -> dict[str, Any]:
        return self.bottle_mon.status()

    def is_success(self) -> bool:
        return bool(self.env.is_success(env_idx=0))

    def close(self, clear_cache: bool = True) -> None:
        """Flush video before the bridge releases cameras and the Isaac app."""
        if self._closed:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Isaac teardown must run on the main thread")
        logger.info("[robodojo-env] shutdown begin")
        try:
            paths = self.recorder.close()
            logger.info("[robodojo-env] videos written: %s", paths)
        finally:
            super().close(clear_cache)
            logger.info("[robodojo-env] shutdown complete")
