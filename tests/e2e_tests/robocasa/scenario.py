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

"""Reusable RoboCasa GPU scenario phases for pytest."""

from __future__ import annotations

import os
from argparse import Namespace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from robots.robocasa.robot_spec import get_robot_spec
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

RLDX_ACTION_DIMS = {
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
    "action.base_motion": 4,
    "action.control_mode": 1,
}


def _runtime_args() -> Namespace:
    return parse_runtime_args(
        get_robot_spec(),
        [
            "--task-name",
            "OpenDrawer",
            "--split",
            "target",
            "--seed",
            "0",
            "--vla-model-path",
            str(Path(_required_env("RLDX_MODEL_PATH")).resolve()),
            "--cuda-device",
            str(selected_cuda_ordinal()),
        ],
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _validate_raw_observation(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        raise RuntimeError("RoboCasa observation must be a mapping")
    required = (
        "robot0_gripper_qpos",
        "robot0_base_pos",
        "robot0_base_quat",
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
    )
    summaries = {}
    for key in required:
        summaries[key] = array_summary(require_array(observation.get(key), key))
    return summaries


def _capture_environment(output_dir: Path, args: Namespace) -> dict[str, Any]:
    spec = get_robot_spec()
    with runtime_phase(spec, args, output_dir / "env-capture", {"env"}) as runtime:
        env = runtime["env_client"]
        raw_observation = env.reset()
        raw_check = _validate_raw_observation(raw_observation)
        cameras = {}
        for camera in (
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ):
            cameras[camera] = require_rgb(
                env.render_camera(camera, height=256, width=256),
                f"{camera}.rgb",
            )
        rgb, depth = env.render_camera(
            "robot0_agentview_left", height=256, width=256, depth=True
        )
        rgb = require_rgb(rgb, "robot0_agentview_left.depth_rgb")
        depth = require_depth(depth, "robot0_agentview_left.depth", rgb)
        meta = require_camera_meta(
            env.get_camera_meta("robot0_agentview_left", height=256, width=256),
            "robot0_agentview_left.camera_meta",
        )
        return {
            "status": "passed",
            "observation": raw_check,
            "cameras": {name: array_summary(image) for name, image in cameras.items()},
            "depth": array_summary(depth),
            "camera_meta_keys": sorted(meta),
        }


def _rldx_checks(output_dir: Path, args: Namespace) -> dict[str, Any]:
    spec = get_robot_spec()
    with runtime_phase(spec, args, output_dir / "rldx", {"vla"}) as runtime:
        client = runtime["vla_client"]
        modality = client.get_modality_config()
        frame_count = len(modality["video_delta_indices"])
        image = synthetic_rgb()
        video = np.repeat(image[None, None], frame_count, axis=1)
        observation = {
            "state.gripper_qpos": np.zeros((1, 1, 2), dtype=np.float32),
            "state.base_position": np.zeros((1, 1, 3), dtype=np.float32),
            "state.base_rotation": np.asarray(
                [[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32
            ),
            "state.end_effector_position_relative": np.zeros(
                (1, 1, 3), dtype=np.float32
            ),
            "state.end_effector_rotation_relative": np.asarray(
                [[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32
            ),
            "video.robot0_agentview_left": video,
            "video.robot0_agentview_right": video.copy(),
            "video.robot0_eye_in_hand": video.copy(),
            "annotation.human.task_description": ["Open the drawer"],
        }
        actions = client.predict(
            observation,
            {
                "reset_memory": [True],
                "session_ids": ["rpent_gpu_ci_robocasa"],
            },
        )

    if not isinstance(actions, dict):
        raise RuntimeError("RLDX actions must be a mapping")
    action_checks = {
        name: array_summary(
            require_array(actions.get(name), name, ndim=3, last_dim=dimension)
        )
        for name, dimension in RLDX_ACTION_DIMS.items()
    }
    return {
        "status": "passed",
        "video_delta_indices": modality["video_delta_indices"],
        "actions": action_checks,
    }


def _policy_chain(output_dir: Path, args: Namespace) -> dict[str, Any]:
    return run_scripted_policy_chain(
        robot="robocasa",
        robot_argv=[
            "--task-name",
            args.task_name,
            "--split",
            args.split,
            "--seed",
            str(args.seed),
            "--vla-model-path",
            args.vla_model_path,
            "--cuda-device",
            str(args.cuda_device),
        ],
        output_dir=output_dir / "policy-chain",
        action=ScriptedToolCall(
            "rldx_skill",
            {
                "max_chunks": 1,
                "n_action_steps": 1,
                "use_prompt": True,
                "prompt": "Open the drawer",
                "force_reset": True,
            },
        ),
        action_count_field="steps_applied",
    )


class RoboCasaScenario:
    """Cache staged component results across the RoboCasa pytest session."""

    def __init__(self, output_dir: Path) -> None:
        self.args = _runtime_args()
        self.output_dir = prepare_suite(
            output_dir=output_dir,
            stack="robocasa",
            extra="robocasa",
            details={
                "task": {
                    "task_name": self.args.task_name,
                    "split": self.args.split,
                    "seed": self.args.seed,
                },
                "resources": {
                    "rldx_model": self.args.vla_model_path,
                    "robocasa_macros": str(
                        Path(_required_env("ROBOCASA_MACROS_PATH")).resolve()
                    ),
                    "robocasa_assets": str(
                        Path(_required_env("ROBOCASA_ASSETS_PATH")).resolve()
                    ),
                },
            },
        )

    @cached_property
    def environment(self) -> dict[str, Any]:
        return _capture_environment(self.output_dir, self.args)

    @cached_property
    def rldx(self) -> dict[str, Any]:
        return _rldx_checks(self.output_dir, self.args)

    @cached_property
    def policy_chain(self) -> dict[str, Any]:
        return _policy_chain(self.output_dir, self.args)
