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

"""RoboTwin environment identity, task choices and native action layouts."""

from __future__ import annotations

from typing import Literal

from components.sim.robotwin.observation import ROBOTWIN_CAMERAS

# Native RoboTwin task YAMLs exposed to the dashboard, the CLI, and the env
# server -- defined once so every consumer shows the same choices.
ROBOTWIN_TASK_CONFIGS = (
    "demo_clean",
    "demo_randomized",
)

#: Env-side camera names exposed by the RoboTwin EnvServer, in fixed order.
#: Shared across the env client, primitives, and toolkit.
ROBOTWIN_CAMERA_NAMES = ROBOTWIN_CAMERAS

#: Supported native action representations for the RoboTwin agent runtime.
RoboTwinActionType = Literal["qpos", "ee"]

#: End-effector pose layout the environment steps, sized for two 7-wide poses,
#: two grippers and two finger positions. LingBot's own spec reuses the name.
ROBOTWIN_EEF_LAYOUT = "eef16"


def env_runtime_contract(
    *,
    task_name: str,
    task_config: str,
    seed: int,
    max_episode_steps: int = 10000,
) -> dict[str, object]:
    """Return the identity required from a RoboTwin EnvServer."""
    return {
        "runtime": "rlinf_robotwin_env",
        "task_name": task_name,
        "task_config": task_config,
        "seed": int(seed),
        "seed_mode": "exact",
        "action_layouts": ["qpos14", ROBOTWIN_EEF_LAYOUT],
        "execution": {
            "reset": True,
            "step": True,
            "chunk_step": True,
            "action_layouts": ["qpos14", ROBOTWIN_EEF_LAYOUT],
            "chunk_step_all_frames": True,
            "chunk_step_frame_format": "robotwin.observation",
            "step_limit": int(max_episode_steps),
        },
        "extensions": {
            "render_camera": {
                "camera_names": list(ROBOTWIN_CAMERA_NAMES),
                "metric_depth": True,
            },
            "get_camera_meta": True,
            "get_task_language": True,
            "plan_arm_path": True,
        },
    }
