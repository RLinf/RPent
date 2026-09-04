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

"""Reusable RoboTwin GPU scenario phases for pytest."""

from __future__ import annotations

import os
from argparse import Namespace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from robots.robotwin.robot_spec import ROBOTWIN_CAMERA_NAMES, get_robot_spec
from tests.e2e_tests.common import (
    ScriptedToolCall,
    array_summary,
    parse_runtime_args,
    prepare_suite,
    require_array,
    require_camera_meta,
    require_depth,
    require_rgb,
    run_scripted_policy_chain,
    runtime_phase,
    selected_cuda_ordinal,
    synthetic_rgb,
)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _runtime_args() -> Namespace:
    return parse_runtime_args(
        get_robot_spec(),
        [
            "--task-name",
            "beat_block_hammer",
            "--seed",
            "100000",
            "--task-config",
            "demo_randomized",
            "--max-episode-steps",
            "32",
            "--robotwin-assets-path",
            str(Path(_required_env("ROBOTWIN_ASSETS_PATH")).resolve()),
            "--vla-model-path",
            str(Path(_required_env("LINGBOT_MODEL_PATH")).resolve()),
            "--cuda-device",
            str(selected_cuda_ordinal()),
        ],
    )


def _validate_reset(observation: Any, info: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        raise RuntimeError("RoboTwin observation must be a mapping")
    if not isinstance(info, dict) or not isinstance(info.get("robot_state"), dict):
        raise RuntimeError("RoboTwin reset info must contain robot_state")
    instruction = info.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise RuntimeError("RoboTwin reset info must contain an instruction")
    state = info["robot_state"]
    require_array(state.get("left_eef_pose"), "left_eef_pose", ndim=1, last_dim=7)
    require_array(state.get("right_eef_pose"), "right_eef_pose", ndim=1, last_dim=7)
    return {
        "observation_keys": sorted(observation),
        "robot_state_keys": sorted(state),
        "instruction": instruction,
    }


def _validate_step(observation: Any, info: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        raise RuntimeError("RoboTwin next observation must be a mapping")
    if not isinstance(info, dict) or not isinstance(info.get("robot_state"), dict):
        raise RuntimeError("RoboTwin step info must contain robot_state")
    status = info.get("episode_status")
    if not isinstance(status, dict):
        raise RuntimeError("RoboTwin step info must contain episode_status")
    return {
        "observation_keys": sorted(observation),
        "robot_state_keys": sorted(info["robot_state"]),
        "episode_status": status,
    }


def _capture_environment(
    output_dir: Path, args: Namespace
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = get_robot_spec()
    with runtime_phase(spec, args, output_dir / "env-capture", {"env"}) as runtime:
        env = runtime["env"]
        observation, info = env.reset()
        reset_check = _validate_reset(observation, info)
        views = {}
        camera_checks = {}
        for camera in ROBOTWIN_CAMERA_NAMES:
            rgb, depth = env.render_camera(camera, depth=True)
            rgb = require_rgb(rgb, f"{camera}.rgb")
            depth = require_depth(depth, f"{camera}.depth", rgb)
            meta = require_camera_meta(
                env.get_camera_meta(camera), f"{camera}.camera_meta"
            )
            views[camera] = {"rgb": rgb.copy()}
            camera_checks[camera] = {
                "rgb": array_summary(rgb),
                "depth": array_summary(depth),
                "camera_meta_keys": sorted(meta),
            }
        policy_observation = {
            "views": views,
            "robot_state": info["robot_state"],
            "task_language": env.get_task_language(),
        }
        return (
            policy_observation,
            info,
            {
                "status": "passed",
                "reset": reset_check,
                "cameras": camera_checks,
            },
        )


def _predict_lingbot(
    output_dir: Path,
    args: Namespace,
    observation: dict[str, Any],
) -> np.ndarray:
    spec = get_robot_spec()
    with runtime_phase(spec, args, output_dir, {"vla"}) as runtime:
        model = runtime["model"]
        return require_array(
            model.infer(observation),
            "LingBot chain actions",
            ndim=2,
            last_dim=16,
        )


def _lingbot_checks(
    output_dir: Path,
    args: Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = synthetic_rgb()
    eef_pose = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    observation = {
        "views": {camera: {"rgb": image.copy()} for camera in ROBOTWIN_CAMERA_NAMES},
        "robot_state": {
            "left_eef_pose": eef_pose,
            "left_gripper": 0.0,
            "right_eef_pose": eef_pose.copy(),
            "right_gripper": 0.0,
        },
        "task_language": "Use the hammer to beat the block.",
    }
    actions = _predict_lingbot(output_dir / "lingbot", args, observation)
    return actions, {
        "status": "passed",
        "input": {
            "cameras": sorted(observation["views"]),
            "task_language": observation["task_language"],
        },
        "action": array_summary(actions),
    }


def _policy_chain(output_dir: Path, args: Namespace) -> dict[str, Any]:
    return run_scripted_policy_chain(
        robot="robotwin",
        robot_argv=[
            "--task-name",
            args.task_name,
            "--seed",
            str(args.seed),
            "--task-config",
            args.task_config,
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--robotwin-assets-path",
            args.robotwin_assets_path,
            "--vla-model-path",
            args.vla_model_path,
            "--cuda-device",
            str(args.cuda_device),
        ],
        output_dir=output_dir / "policy-chain",
        action=ScriptedToolCall("lingbot_act", {"chunks": 1, "use_length": 50}),
        action_count_field="executed_steps",
    )


class RoboTwinScenario:
    """Cache staged component results across the RoboTwin pytest session."""

    def __init__(self, output_dir: Path) -> None:
        self.args = _runtime_args()
        self.output_dir = prepare_suite(
            output_dir=output_dir,
            stack="robotwin",
            extra="robotwin",
            details={
                "task": {
                    "task_name": self.args.task_name,
                    "task_config": self.args.task_config,
                    "seed": self.args.seed,
                },
                "resources": {
                    "lingbot_model": self.args.vla_model_path,
                    "robotwin_assets": self.args.robotwin_assets_path,
                },
            },
        )

    @cached_property
    def environment(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return _capture_environment(self.output_dir, self.args)

    @cached_property
    def lingbot(self) -> tuple[np.ndarray, dict[str, Any]]:
        return _lingbot_checks(self.output_dir, self.args)

    @cached_property
    def policy_chain(self) -> dict[str, Any]:
        return _policy_chain(self.output_dir, self.args)
