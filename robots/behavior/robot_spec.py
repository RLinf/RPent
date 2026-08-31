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

"""BEHAVIOR robot extension: RobotSpec factory and toolkit bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robots.behavior.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig

BEHAVIOR_DASHBOARD_SPEC = {
    "classes": {
        "server": "robots.behavior.dashboard:BehaviorDashboardServer",
        "state": "robots.behavior.dashboard:BehaviorDashboardState",
    },
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task_name> <public_seed>",
        "fields": (
            {
                "name": "task_name",
                "suggestions": ("turning_on_radio", "picking_up_trash"),
            },
            {"name": "public_seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{task_name} / s{public_seed}",
        "output_slug": "{task_name}_s{public_seed}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "unique"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
        {"name": "dino", "label": "DINO", "scope": "shared"},
        {"name": "memory", "label": "MEM", "scope": "unique"},
    ),
    "frame_channels": (
        {"name": "head", "label": "head"},
        {"name": "left_wrist", "label": "left wrist"},
        {"name": "right_wrist", "label": "right wrist"},
    ),
    "behavior_control": {
        "targets": ("chassis", "left_arm", "right_arm"),
        "actions": (
            "forward",
            "backward",
            "turn_left",
            "turn_right",
            "up",
            "down",
            "rotate_left",
            "rotate_right",
            "open",
            "close",
            "observe",
        ),
        "cameras": ("head", "left_wrist", "right_wrist"),
        "pipeline": ("prepare", "execute", "discard", "capture", "stop"),
        "official_success_source": (
            'backend raw info["done"]["success"] or info_done.success only'
        ),
    },
}


def get_robot_spec() -> RobotSpec:
    from robots.behavior import runtime

    return RobotSpec(
        name="behavior",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=runtime.add_cli_args,
        parse_config=runtime.parse_config,
        init_runtime=runtime.init_runtime,
        dashboard=BEHAVIOR_DASHBOARD_SPEC,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
):
    """Return the BEHAVIOR toolkit through the standard main contract."""

    from robots.behavior.toolkit import BehaviorToolkit

    mode = str(config.prompt_vars.get("behavior_mode", "eval"))
    memory_dir = config.prompt_vars.get("memory_dir")
    if not memory_dir:
        memory_dir = Path(config.output_dir) / "behavior_memory_empty"
    memory = MemoryManager(
        root=Path(memory_dir),
        memory_access="inbox_write" if mode == "explore" else "read_only",
        inbox_cell_tag=config.recipe_tag if mode == "explore" else None,
    )
    video_path = Path(config.output_dir) / "episode.mp4"
    return BehaviorToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
        config=config,
        video_path=video_path,
    )


__all__ = ["BEHAVIOR_DASHBOARD_SPEC", "get_robot_spec", "get_toolkit"]
