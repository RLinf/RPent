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

"""Expose RLinf's RoboLab environment through RPent's single-env RPC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rpent.robots.components.action_spec import (  # noqa: E402
    box_action_spec,
    validate_action,
)
from rpent.robots.components.env_facade_base import BaseEnvFacade  # noqa: E402
from rpent.utils.logging import init_output_dir  # noqa: E402
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin  # noqa: E402
from rpent.utils.serialization import to_numpy_tree  # noqa: E402


def make_env(task: str, seed: int, device: str, output_dir: str) -> Any:
    """Create one RLinf environment governed by the native task's time limit."""
    from omegaconf import OmegaConf
    from rlinf.envs.sim.robolab.robolab_env import RoboLabEnv

    cfg = OmegaConf.create(
        {
            "auto_reset": False,
            "ignore_terminations": False,
            "max_episode_steps": None,
            "seed": seed,
            "init_params": {
                "task": task,
                "device": device,
                "output_dir": output_dir,
                "renderer": "realtime",
                "kit_args": (
                    "--/renderer/multiGpu/enabled=false "
                    "--/renderer/multiGpu/autoEnable=false"
                ),
                "enable_subtask": True,
            },
        }
    )
    return RoboLabEnv(
        cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None
    )


class RoboLabEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """Adapt batched RLinf observations and serialize access to its simulator socket."""

    def __init__(self, env: Any, meta: dict[str, Any]):
        self.env = env
        self.meta = meta
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.get_status"] = self.get_status
        self._readonly_methods.add("env.get_status")

    def get_env_meta(self) -> dict[str, Any]:
        return dict(self.meta)

    def get_action_spec(self) -> dict[str, dict[str, Any]]:
        return {
            "default": box_action_spec(
                self.env.action_bounds["low"],
                self.env.action_bounds["high"],
                "DROID joint positions: seven arm joints followed by the gripper command.",
            )
        }

    @staticmethod
    def _observation(raw: dict[str, Any]) -> dict[str, Any]:
        return {key: value[0] for key, value in to_numpy_tree(raw).items()}

    def reset(self) -> dict[str, Any]:
        obs, _ = self.env.reset()
        return self._observation(obs)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict]:
        status = self.get_status()
        if status["terminated"] or status["truncated"]:
            raise RuntimeError("RoboLab episode has ended")
        validated = validate_action(action, self.get_action_spec()["default"])
        obs, reward, terminated, truncated, _ = self.env.step(validated[None])
        return (
            self._observation(obs),
            float(reward[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            self.get_status(),
        )

    def chunk_step(
        self, actions: Any, *, return_all_frames: bool = False
    ) -> tuple[Any, float, bool, bool, dict]:
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 8 or not len(array):
            raise ValueError("RoboLab actions must have shape [steps, 8]")
        frames = []
        result = None
        for action in array:
            result = self.step(action)
            if return_all_frames:
                frames.append(result[0])
            if result[2] or result[3]:
                break
        assert result is not None
        return (frames if return_all_frames else result[0], *result[1:])

    def get_status(self) -> dict[str, Any]:
        return self.env.get_status()[0]

    def get_task_language(self) -> str:
        return self.env.task_descriptions[0]

    def render_camera(self, camera_name: str) -> np.ndarray:
        return self.env.render_camera(camera_name)

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        return self.env.get_camera_meta(camera_name)

    def close(self) -> None:
        self.env.close()


def main() -> None:
    """Start the RPC facade; RLinf owns the Isaac simulator child process."""
    parser = argparse.ArgumentParser(description="RoboLab RPent environment server")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    init_output_dir(args.output_dir)

    env = make_env(args.task, args.seed, args.device, args.output_dir)
    try:
        facade = RoboLabEnvFacade(env, {"task": args.task, "seed": args.seed})
        facade.serve(
            transport="http",
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
