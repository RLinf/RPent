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

"""Single-thread MetaWorld RGB-D service over the common Env RPC contract."""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from robots.metaworld.config import CAMERAS, TASKS, validate
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin


class MetaWorldEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """Own one native MT1 environment and its renderer on the service thread."""

    def __init__(self, task: str, seed: int, max_episode_steps: int, camera: str):
        validate(task, seed, max_episode_steps, camera)
        import gymnasium as gym
        import metaworld
        import mujoco

        if "Meta-World/MT1" not in gym.registry:
            metaworld.register_mw_envs()

        self._meta = {
            "task": task,
            "seed": seed,
            "max_episode_steps": max_episode_steps,
            "camera": camera,
        }
        self._env = gym.make(
            "Meta-World/MT1",
            env_name=task,
            seed=seed,
            num_tasks=1,
            max_episode_steps=max_episode_steps,
            render_mode="rgb_array",
            camera_name=camera,
        )
        self._renderer = None
        self._closed = False
        try:
            self._env.reset(seed=seed)
            self._renderer = mujoco.Renderer(
                self._env.unwrapped.model, height=480, width=480
            )
            self._camera_id = mujoco.mj_name2id(
                self._env.unwrapped.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
            )
            self._steps = 0
            self._success = self._terminated = self._truncated = False
            super().__init__()
        except BaseException:
            self.close()
            raise

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc.update(
            {"env.get_obs": self.get_obs, "env.is_success": self.is_success}
        )
        self._readonly_methods.update({"env.get_obs", "env.is_success"})

    def get_env_meta(self) -> dict:
        return dict(self._meta)

    def get_task_language(self) -> str:
        return TASKS[self._meta["task"]]

    def get_camera_meta(self, camera_name: str, height=None, width=None) -> dict:
        if camera_name != self._meta["camera"]:
            raise ValueError(
                f"Only configured camera {self._meta['camera']!r} is available"
            )
        if height not in (None, 480) or width not in (None, 480):
            raise ValueError("MetaWorld service renders at 480 x 480")
        env = self._env.unwrapped
        focal = 240.0 / np.tan(np.deg2rad(env.model.cam_fovy[self._camera_id]) / 2)
        rotation = env.data.cam_xmat[self._camera_id].reshape(3, 3).copy()
        return {
            "intrinsics": [[focal, 0, 240.0], [0, focal, 240.0], [0, 0, 1]],
            "rotation_world_from_camera": rotation,
            "position_world": env.data.cam_xpos[self._camera_id].copy(),
            "axes": "camera x right, y up, z backward; depth is forward z in meters",
        }

    def render_camera(self, camera_name: str, height=None, width=None, depth=False):
        self.get_camera_meta(camera_name, height, width)
        self._renderer.update_scene(self._env.unwrapped.data, camera=camera_name)
        if depth:
            self._renderer.enable_depth_rendering()
        else:
            self._renderer.disable_depth_rendering()
        return self._renderer.render().copy()

    def get_obs(self, include_images: bool = True) -> dict[str, Any]:
        # Only robot proprioception and camera observations cross the agent boundary.
        # The benchmark's flat observation also contains privileged object/goal states.
        env = self._env.unwrapped
        camera = self._meta["camera"]
        raw = env._get_obs()
        obs = {
            "eef_position": np.asarray(raw[:3]).copy(),
            "gripper_opening": float(raw[3]),
            "instruction": self.get_task_language(),
            "steps": self._steps,
            "terminated": self._terminated,
            "truncated": self._truncated,
        }
        if include_images:
            obs.update(
                rgb=self.render_camera(camera),
                depth=self.render_camera(camera, depth=True),
                camera=self.get_camera_meta(camera),
            )
        return obs

    def reset(self) -> dict:
        self._env.reset(seed=self._meta["seed"])
        self._steps = 0
        self._success = self._terminated = self._truncated = False
        return self.get_obs()

    def step(self, flat_action):
        action = np.asarray(flat_action, dtype=np.float32)
        if (
            action.shape != (4,)
            or not np.isfinite(action).all()
            or np.any(np.abs(action) > 1)
        ):
            raise ValueError("action must contain four finite values in [-1, 1]")
        if self._terminated or self._truncated:
            raise RuntimeError("episode ended; no further actions are accepted")
        _, reward, terminated, truncated, info = self._env.step(action)
        self._steps += 1
        self._success |= bool(info.get("success", False))
        self._terminated = bool(terminated or self._success)
        self._truncated = bool(
            truncated or self._steps >= self._meta["max_episode_steps"]
        )
        return (
            self.get_obs(include_images=False),
            float(reward),
            self._terminated,
            self._truncated,
            {},
        )

    def chunk_step(self, flat_actions, *, return_all_frames=False):
        raise NotImplementedError("Use bounded single-step control for MetaWorld")

    def is_success(self) -> bool:
        return self._success

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._renderer is not None:
                self._renderer.close()
        finally:
            self._env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="reach-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--camera", choices=CAMERAS, default="corner2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = MetaWorldEnvFacade(args.task, args.seed, args.max_episode_steps, args.camera)
    try:
        env.serve(
            transport="http",
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
