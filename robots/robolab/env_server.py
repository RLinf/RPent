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

"""Serve one externally installed RoboLab task through RPent's environment RPC."""

from __future__ import annotations

import argparse
import os
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
from rpent.utils.logging import get_logger, init_output_dir  # noqa: E402
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin  # noqa: E402

logger = get_logger("robolab_env_server")


class RoboLabEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """Execute RoboLab's native DROID joint-position actions on the Isaac main thread."""

    def __init__(self, env: Any, env_cfg: Any, app: Any, meta: dict[str, Any]):
        self.env = env
        self.env_cfg = env_cfg
        self.app = app
        self.meta = meta
        self._last_obs: dict[str, Any] | None = None
        self._steps = 0
        self._terminated = False
        self._truncated = False
        self._closed = False
        from robolab.core.task.instruction import instruction_text

        self._instruction = instruction_text(env_cfg.instruction)
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.get_status"] = self.get_status
        self._readonly_methods.add("env.get_status")

    def get_env_meta(self) -> dict[str, Any]:
        return dict(self.meta)

    def get_action_spec(self) -> dict[str, dict[str, Any]]:
        space = getattr(self.env, "single_action_space", self.env.action_space)
        low = np.asarray(space.low).reshape(-1)
        high = np.asarray(space.high).reshape(-1)
        if low.size != 8:
            raise ValueError(
                f"DROID joint-position action must have 8 values, got {low.size}"
            )
        return {
            "default": box_action_spec(
                low,
                high,
                "DROID joint positions: seven arm joints followed by the gripper command.",
            )
        }

    def _observation(self, raw: dict[str, Any]) -> dict[str, Any]:
        from openpi_client import image_tools

        images = raw["image_obs"]
        state = raw["proprio_obs"]
        return {
            "main_images": image_tools.resize_with_pad(
                images["over_shoulder_left_camera"][0].detach().cpu().numpy(), 224, 224
            ),
            "wrist_images": image_tools.resize_with_pad(
                images["wrist_cam"][0].detach().cpu().numpy(), 224, 224
            ),
            "states": np.concatenate(
                [
                    state["arm_joint_pos"][0].detach().cpu().numpy(),
                    state["gripper_pos"][0].detach().cpu().numpy(),
                ]
            ).astype(np.float32),
            "task_descriptions": self._instruction,
        }

    def reset(self) -> dict[str, Any]:
        if self._last_obs is not None:
            from robolab.core.environments.runtime import end_episode

            end_episode(self.env)
        self.env.reset_eval_state()
        self.env.reset()
        self._last_obs, _ = self.env.reset()
        self._steps = 0
        self._terminated = False
        self._truncated = False
        return self._observation(self._last_obs)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict]:
        if self._terminated or self._truncated:
            raise RuntimeError("RoboLab episode has ended")
        validated = validate_action(action, self.get_action_spec()["default"])
        import torch

        tensor = torch.as_tensor(
            validated[None], dtype=torch.float32, device=self.env.device
        )
        self._last_obs, _, _, _, _ = self.env.step(tensor)
        self._steps += 1
        native_result = self.env.get_env_results()[0]
        success = native_result["success"]
        self._terminated = success is True
        self._truncated = not self._terminated and (
            success is False or self._steps >= int(self.env.max_episode_length)
        )
        info = self.get_status()
        return (
            self._observation(self._last_obs),
            float(self._terminated),
            self._terminated,
            self._truncated,
            info,
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
        native = self.env.get_env_results()[0]
        return {
            "steps": self._steps,
            "step_limit": int(self.env.max_episode_length),
            "success": native["success"],
            "terminated": self._terminated,
            "truncated": self._truncated,
        }

    def get_task_language(self) -> str:
        return self._instruction

    def render_camera(self, camera_name: str, **kwargs: Any) -> np.ndarray:
        del kwargs
        if camera_name not in {"over_shoulder_left_camera", "wrist_cam"}:
            raise ValueError(f"unknown RoboLab camera: {camera_name!r}")
        if self._last_obs is None:
            raise RuntimeError("RoboLab environment has not been reset")
        return self._last_obs["image_obs"][camera_name][0].detach().cpu().numpy()

    def get_camera_meta(self, camera_name: str, **kwargs: Any) -> dict[str, Any]:
        image = self.render_camera(camera_name, **kwargs)
        return {
            "camera_name": camera_name,
            "height": image.shape[0],
            "width": image.shape[1],
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.env.close()
        finally:
            self.app.close()


def main() -> None:
    """Launch Isaac before importing RoboLab environment modules."""
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

    device = args.device
    kit_args = ""
    if device.startswith("cuda"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        ordinal = int(device.split(":")[1]) if ":" in device else 0
        visible = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        if visible is not None:
            devices = visible.split(",")
            if ordinal >= len(devices) or not devices[ordinal].strip().isdigit():
                raise ValueError(
                    "RoboLab requires numeric CUDA_VISIBLE_DEVICES entries and a "
                    "--device index within that list"
                )
            ordinal = int(devices[ordinal])
        # Vulkan keeps physical GPU ordinals when CUDA masks remap CUDA devices.
        device = f"cuda:{ordinal}"
        kit_args = f"--/renderer/activeGpu={ordinal}"
    logger.info("RoboLab simulation device: %s (requested %s)", device, args.device)

    import cv2  # noqa: F401 -- Isaac Lab requires OpenCV to load first.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=True, enable_cameras=True, device=device,
        renderer="realtime", kit_args=kit_args,
    )
    app = app_launcher.app
    logger.info("Isaac app started for RoboLab task %s", args.task)
    env = None
    facade = None
    failure = None
    try:
        from robolab.constants import ASSET_DIR, set_output_dir
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_jointpos import (
            auto_register_droid_envs,
        )

        if not Path(ASSET_DIR).is_dir():
            raise FileNotFoundError(
                f"RoboLab assets not found at {ASSET_DIR}; install RoboLab from "
                "an external checkout containing its assets"
            )
        auto_register_droid_envs(task=args.task)
        logger.info("RoboLab task registered: %s", args.task)
        set_output_dir(args.output_dir)
        logger.info("Creating RoboLab environment for %s", args.task)
        env, env_cfg = create_env(
            args.task, device=device, seed=args.seed, num_envs=1,
            policy="rpent", renderer="realtime",
        )
        logger.info("RoboLab environment created for %s", args.task)
        facade = RoboLabEnvFacade(
            env,
            env_cfg,
            app,
            {"task": args.task, "seed": args.seed, "device": args.device},
        )
        facade.serve(
            transport="http",
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
        logger.info("RoboLab environment RPC stopped")
    except BaseException as exc:
        failure = exc
        logger.exception("RoboLab environment server failed")
        raise
    finally:
        try:
            if facade is not None:
                facade.close()
            else:
                try:
                    if env is not None:
                        env.close()
                finally:
                    app.close()
        except BaseException:
            if failure is None:
                raise
            logger.exception("RoboLab cleanup failed after server error")


if __name__ == "__main__":
    main()
