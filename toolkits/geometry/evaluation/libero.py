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

"""Evaluator-owned LIBERO collection without changing the production runtime."""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import import_module
from typing import Any

import numpy as np

from rpent.utils.serialization import to_numpy_tree
from toolkits.geometry.evaluation.checklist import frame_id
from toolkits.geometry.evaluation.evaluation_scene import evaluation_env_class
from toolkits.geometry.observation import GeometryObservation


def make_environment(case: dict[str, Any]) -> Any:
    """Build one opt-in reference-reading environment with the existing config."""
    os.environ.setdefault("LIBERO_TYPE", "pro")
    if os.environ["LIBERO_TYPE"] != "pro":
        raise ValueError("the fixed checklist requires LIBERO_TYPE=pro")
    # Import the production configuration before RLinf imports MuJoCo. This
    # retains its GL initialization order without changing or patching its code.
    configuration = import_module("robots.libero.env_server")
    libero = import_module("rlinf.envs.sim.libero.libero_env")
    benchmark = import_module("rlinf.envs.sim.libero.utils").benchmark

    suite = benchmark.get_benchmark(case["suite"])()
    task = case["task"]
    if not 0 <= task < suite.n_tasks:
        raise ValueError("checklist task is absent from the installed LIBERO suite")
    trials = len(suite.get_task_init_states(task))
    if not trials:
        raise ValueError("the selected task has no initial states")
    reset_id = sum(len(suite.get_task_init_states(i)) for i in range(task))
    reset_id += case["seed"] % trials
    cfg = configuration.build_env_cfg(
        task_suite_name=case["suite"],
        specific_reset_id=reset_id,
        seed=case["seed"],
        max_episode_steps=1,
    )
    return evaluation_env_class(libero.LiberoEnv)(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def capture_case(
    case: dict[str, Any],
    *,
    height: int = 1024,
    width: int = 1024,
    environment_factory: Callable[[dict[str, Any]], Any] = make_environment,
) -> tuple[GeometryObservation, dict[str, Any]]:
    """Capture sensors and independent model geometry at the same settled reset.

    No action or planner is constructed. Only the environment created by this
    call is closed, including when rendering or reference collection fails.
    """
    from robots.libero.tools import _metric_depth, _world_from_depth

    env = environment_factory(case)
    try:
        env.reset()
        rgb, depth = to_numpy_tree(
            env.render_camera("agentview", height=height, width=width, depth=True)
        )
        meta = to_numpy_tree(
            env.get_camera_meta("agentview", height=height, width=width)
        )
        if not isinstance(meta, dict):
            raise ValueError("LIBERO did not return camera calibration")
        # Match the existing LIBERO dump convention: flip RGB and depth
        # together, linearize depth, then project with that camera's metadata.
        rgb = np.ascontiguousarray(np.asarray(rgb)[::-1])
        depth = np.ascontiguousarray(_metric_depth(depth, meta)[::-1])
        observation = GeometryObservation(
            camera="agentview",
            step=0,
            world_frame_id=frame_id(case["case_id"]),
            rgb=rgb,
            depth_m=depth,
            world_xyz=_world_from_depth(depth, meta),
            intrinsics=meta["intrinsic_K"],
            camera_to_world=meta["extrinsic_cam2world"],
            up_direction=np.array([0.0, 0.0, 1.0]),
        )
        snapshot = to_numpy_tree(
            env.env.workers[0].get_env_attr("rpent_reference_snapshot")
        )
        snapshot.update(
            case_id=case["case_id"],
            frame_id=observation.world_frame_id,
            step=0,
            camera="agentview",
        )
        return observation, snapshot
    finally:
        env.env.close()
