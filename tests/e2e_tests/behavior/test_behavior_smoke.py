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

"""BEHAVIOR real components and scripted-planner GPU policy chain."""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from robots.behavior.robot_spec import get_robot_spec
from robots.behavior.schemas import ACTION_DIM, DEFAULT_ACTION_CHUNK, PHYSICAL_CAMERAS
from robots.behavior.task_specs import get_task_spec
from tests.e2e_tests.common import (
    ScriptedToolCall,
    array_summary,
    parse_runtime_args,
    prepare_suite,
    publish_check,
    require_array,
    require_camera_meta,
    require_rgb,
    run_scripted_policy_chain,
    runtime_phase,
)


@pytest.fixture(scope="module")
def behavior_suite() -> tuple[Namespace, Path, list[str]]:
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != 2 or any(not device.isdecimal() for device in devices):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name two physical GPU ordinals")
    env_device, model_device = devices
    if env_device == model_device:
        raise RuntimeError("BEHAVIOR ENV and model GPUs must be distinct")

    data_path = Path(os.environ["OMNIGIBSON_DATA_PATH"]).expanduser().resolve()
    robot_argv = [
        "--task-name",
        "turning_on_radio",
        "--public-seed",
        "1",
        "--max-episode-steps",
        str(DEFAULT_ACTION_CHUNK),
        "--behavior-repo",
        os.environ["RPENT_RLINF_ROOT"],
        "--behavior-python",
        os.environ["RPENT_BEHAVIOR_PYTHON"],
        "--activity-instance-dir",
        str(data_path / "2025-challenge-task-instances"),
        "--policy-checkpoint",
        os.environ["PI05_CHECKPOINT_PATH"],
        "--dino-source-archive",
        os.environ["DINOV2_SOURCE_ARCHIVE"],
        "--dino-weights",
        os.environ["DINOV2_WEIGHTS"],
        "--behavior-env-cuda-device",
        env_device,
        "--behavior-model-cuda-device",
        model_device,
    ]
    spec = get_robot_spec()
    args = parse_runtime_args(spec, robot_argv)
    spec.parse_config(args)
    output_dir = prepare_suite(
        output_dir=Path(os.environ["RPENT_E2E_OUTPUT_DIR"]),
        stack="behavior",
        extra="behavior",
        details={
            "task": {"task_name": args.task_name, "public_seed": args.public_seed},
            "resources": {
                "rlinf_repo": args.behavior_repo,
                "behavior_python": args.behavior_python,
                "omnigibson_data": str(data_path),
                "pi05_checkpoint": args.policy_checkpoint,
                "dino_source_archive": args.dino_source_archive,
                "dino_weights": args.dino_weights,
            },
            "gpu": {"env": env_device, "model_and_dino": model_device},
        },
    )
    return args, output_dir, robot_argv


def test_components(behavior_suite: tuple[Namespace, Path, list[str]]) -> None:
    args, output_dir, _ = behavior_suite
    spec = get_robot_spec()
    with runtime_phase(
        spec, args, output_dir / "components", {"env", "vla", "dino"}
    ) as runtime:
        observation = runtime["initial_observation"]
        info = runtime["initial_info"]
        if not isinstance(observation, dict) or not isinstance(info, dict):
            raise RuntimeError("BEHAVIOR runtime reset did not return observation/info")

        head = require_rgb(observation.get("main_images"), "main_images")
        wrists = require_array(
            observation.get("wrist_images"),
            "wrist_images",
            ndim=4,
            last_dim=3,
            dtype=np.uint8,
        )
        if wrists.shape[0] != 2:
            raise RuntimeError(f"BEHAVIOR needs two wrist views, got {wrists.shape}")
        proprio = require_array(observation.get("states"), "states", ndim=1)
        if proprio.size < 256:
            raise RuntimeError(f"R1Pro proprio is too short: {proprio.shape}")

        env = runtime["env"]
        cameras = {}
        for camera in PHYSICAL_CAMERAS:
            rgb = require_rgb(env.render_camera(camera), f"{camera}.rgb")
            meta = require_camera_meta(env.get_camera_meta(camera), camera)
            cameras[camera] = {
                "rgb": array_summary(rgb),
                "camera_meta_keys": sorted(meta),
            }

        actions = require_array(
            runtime["model"].predict(observation, options={"mode": "eval"}),
            "Pi0.5 actions",
            ndim=2,
            last_dim=ACTION_DIM,
        )
        if actions.shape[0] != DEFAULT_ACTION_CHUNK:
            raise RuntimeError(
                f"Pi0.5 returned {actions.shape[0]} actions, expected 32"
            )
        embedding = runtime["dino_component"].encode_batch([head])[0]
        embedding = require_array(embedding, "DINO embedding", ndim=1, last_dim=384)
        if not np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-3):
            raise RuntimeError("DINO embedding is not L2-normalized")

    publish_check(
        "behavior_components",
        {
            "status": "passed",
            "reset": "runtime-owned ENV reset returned observation/info",
            "head": array_summary(head),
            "wrists": array_summary(wrists),
            "proprio": array_summary(proprio),
            "cameras": cameras,
            "pi05_actions": array_summary(actions),
            "dino_embedding": array_summary(embedding),
        },
    )


def test_policy_chain(behavior_suite: tuple[Namespace, Path, list[str]]) -> None:
    _, output_dir, robot_argv = behavior_suite
    result = run_scripted_policy_chain(
        robot="behavior",
        robot_argv=robot_argv,
        output_dir=output_dir / "policy-chain",
        action=ScriptedToolCall(
            "pi0_nav_pick",
            {
                "instruction": get_task_spec("turning_on_radio").task_language,
                "chunks": 1,
            },
        ),
        action_count_field="env_steps_used",
    )
    publish_check("behavior_policy_chain", result)
