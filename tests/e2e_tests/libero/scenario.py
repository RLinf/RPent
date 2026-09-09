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

"""Reusable LIBERO GPU scenario phases for pytest."""

from __future__ import annotations

import os
from argparse import Namespace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from robots.libero.robot_spec import get_robot_spec
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

VARIANT_CONFIG = {
    "standard": {
        "extra": "libero",
        "suite": "libero_object",
        "task": 0,
        "asset_env": "LIBERO_ASSET_PATH",
    },
    "pro": {
        "extra": "libero-pro",
        "suite": "libero_object_swap",
        "task": 2,
        "asset_env": "LIBERO_PRO_ASSET_PATH",
    },
    "plus": {
        "extra": "libero-plus",
        "suite": "libero_object",
        "task": 0,
        "asset_env": "LIBERO_PLUS_ASSET_PATH",
    },
}


def _runtime_args(variant: str) -> Namespace:
    config = VARIANT_CONFIG[variant]
    return parse_runtime_args(
        get_robot_spec(),
        [
            "--suite",
            config["suite"],
            "--task",
            str(config["task"]),
            "--seed",
            "0",
            "--max-episode-steps",
            "32",
            "--libero-type",
            variant,
            "--cuda-device",
            str(selected_cuda_ordinal()),
        ],
    )


def _validate_observation(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        raise RuntimeError("LIBERO observation must be a mapping")
    main = require_rgb(observation.get("main_images"), "main_images")
    states = require_array(observation.get("states"), "states", ndim=1)
    task = observation.get("task_descriptions")
    if not isinstance(task, str) or not task.strip():
        raise RuntimeError("task_descriptions must be a non-empty string")
    return {
        "main_images": array_summary(main),
        "states": array_summary(states),
        "task_description": task,
    }


def _capture_environment(
    output_dir: Path, args: Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = get_robot_spec()
    with runtime_phase(spec, args, output_dir / "env-capture", {"env"}) as runtime:
        env = runtime["env"]
        observation, _ = env.reset()
        observation_check = _validate_observation(observation)
        rgb, depth = env.render_camera("agentview", height=256, width=256, depth=True)
        rgb = require_rgb(rgb, "agentview.rgb")
        depth = require_depth(depth, "agentview.depth", rgb)
        meta = require_camera_meta(
            env.get_camera_meta("agentview", height=256, width=256),
            "agentview.camera_meta",
        )
        return observation, {
            "status": "passed",
            "observation": observation_check,
            "rgb": array_summary(rgb),
            "depth": array_summary(depth),
            "camera_meta_keys": sorted(meta),
        }


def _environment_action(output_dir: Path, args: Namespace) -> dict[str, Any]:
    spec = get_robot_spec()
    with runtime_phase(spec, args, output_dir / "env-action", {"env"}) as runtime:
        env = runtime["env"]
        env.reset()
        next_observation, *_ = env.step(np.zeros(7, dtype=np.float32))
        next_check = _validate_observation(next_observation)
    return {
        "status": "passed",
        "action_shape": [7],
        "next_observation": next_check,
    }


def _predict_pi05(
    output_dir: Path,
    args: Namespace,
    observation: dict[str, Any],
) -> np.ndarray:
    spec = get_robot_spec()
    with runtime_phase(spec, args, output_dir, {"vla"}) as runtime:
        model = runtime["model"]
        actions, _ = model.predict_action_batch(observation)
    return require_array(actions, "Pi0.5 chain actions", ndim=2, last_dim=7)


def _pi05_checks(
    output_dir: Path,
    args: Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = synthetic_rgb()
    observation: dict[str, Any] = {
        "main_images": image,
        "wrist_images": image.copy(),
        "states": np.zeros(8, dtype=np.float32),
        "task_descriptions": ("pick up the alphabet soup and place it in the basket"),
    }
    actions = _predict_pi05(output_dir / "pi05", args, observation)
    return actions, {
        "status": "passed",
        "input": {
            "main_images": array_summary(observation["main_images"]),
            "wrist_images": array_summary(observation["wrist_images"]),
            "states": array_summary(observation["states"]),
            "task_description": observation["task_descriptions"],
        },
        "action": array_summary(actions),
    }


def _sam3_check(output_dir: Path, args: Namespace) -> dict[str, Any]:
    spec = get_robot_spec()
    with runtime_phase(spec, args, output_dir / "sam3", {"sam3"}) as runtime:
        image = synthetic_rgb()
        result = runtime["sam3_client"].segment(
            image,
            point=[128, 128],
            min_score=0.0,
        )
        if not result.found or result.mask is None or not result.mask.any():
            raise RuntimeError(f"SAM3 returned no usable mask: {result.reason}")
        mask = require_array(result.mask, "SAM3 mask", ndim=2)
        if mask.shape != image.shape[:2]:
            raise RuntimeError(
                f"SAM3 mask shape {mask.shape} does not match {image.shape[:2]}"
            )
        return {
            "status": "passed",
            "mask": array_summary(mask),
            "mask_pixels": int(mask.sum()),
            "score": result.score,
        }


def _policy_chain(output_dir: Path, args: Namespace) -> dict[str, Any]:
    return run_scripted_policy_chain(
        robot="libero",
        robot_argv=[
            "--suite",
            args.suite,
            "--task",
            str(args.task),
            "--seed",
            str(args.seed),
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--libero-type",
            args.libero_type,
            "--cuda-device",
            str(args.cuda_device),
        ],
        output_dir=output_dir / "policy-chain",
        action=ScriptedToolCall(
            "pi0_doubled",
            {
                "prompt": "perform one bounded action chunk for the current task",
                "max_chunks": 1,
            },
        ),
        action_count_field="chunks_used",
    )


class LiberoScenario:
    """Cache staged component results across the LIBERO pytest session."""

    def __init__(self, variant: str, output_dir: Path) -> None:
        if variant not in VARIANT_CONFIG:
            raise ValueError(f"unknown LIBERO variant: {variant}")
        self.variant = variant
        self.args = _runtime_args(variant)
        config = VARIANT_CONFIG[variant]
        self.output_dir = prepare_suite(
            output_dir=output_dir,
            stack=f"libero-{variant}",
            extra=config["extra"],
            details={
                "task": {
                    "suite": self.args.suite,
                    "task": self.args.task,
                    "seed": self.args.seed,
                },
                "resources": {
                    "simulator_assets": str(
                        Path(os.environ[config["asset_env"]]).resolve()
                    ),
                    "pi05_checkpoint": str(
                        Path(os.environ["PI05_CHECKPOINT_PATH"]).resolve()
                    ),
                    "sam3_checkpoint": str(
                        Path(os.environ["SAM3_CHECKPOINT_PATH"]).resolve()
                    ),
                },
            },
        )

    @cached_property
    def environment(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return _capture_environment(self.output_dir, self.args)

    @cached_property
    def environment_action(self) -> dict[str, Any]:
        return _environment_action(self.output_dir, self.args)

    @cached_property
    def pi05(self) -> tuple[np.ndarray, dict[str, Any]]:
        return _pi05_checks(self.output_dir, self.args)

    @cached_property
    def sam3(self) -> dict[str, Any]:
        return _sam3_check(self.output_dir, self.args)

    @cached_property
    def policy_chain(self) -> dict[str, Any]:
        return _policy_chain(self.output_dir, self.args)
