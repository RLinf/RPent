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
from rpent.dashboard.events import DashboardEventSink, RuntimeStatusEvent
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig

BEHAVIOR_DASHBOARD_SPEC = {
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
        {
            "name": "camera",
            "label": "head camera",
            "legacy_path_key": "image_cam_path",
        },
        {
            "name": "wrist",
            "label": "wrist cameras",
            "legacy_path_key": "image_wrist_path",
        },
    ),
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
    mode: str | None = None,
    attempts_per_session: int = 0,
    state_output_dir: Path | str | None = None,
):
    """Return the BEHAVIOR toolkit through the standard main contract."""

    from robots.behavior.toolkit import BehaviorToolkit

    toolkit_kwargs = dict(primitives_kwargs)
    if attempts_per_session > 0:
        raise ValueError(
            "BEHAVIOR explore runs one attempt per session; use --explore-sessions"
        )
    memory_selected = bool(toolkit_kwargs.pop("_memory_component_selected", False))
    if memory_selected:
        dashboard_events.emit(RuntimeStatusEvent("memory", "starting"))
    try:
        if mode is None:
            behavior_mode = str(config.prompt_vars.get("behavior_mode", "eval"))
        elif mode == "exploration":
            behavior_mode = "explore"
        elif mode == "evaluation":
            behavior_mode = "eval"
        else:
            raise ValueError(f"unsupported BEHAVIOR toolkit mode: {mode!r}")
        if behavior_mode not in {"eval", "explore"}:
            raise ValueError(f"unsupported BEHAVIOR toolkit mode: {behavior_mode!r}")
        toolkit_kwargs["behavior_phase"] = behavior_mode
        memory_dir = config.prompt_vars.get("memory_dir")
        if not memory_dir:
            raise ValueError("BEHAVIOR RunConfig is missing memory_dir")
        memory = MemoryManager(
            root=Path(memory_dir),
            memory_access=(
                "inbox_write" if behavior_mode == "explore" else "read_only"
            ),
            inbox_cell_tag=(config.recipe_tag if behavior_mode == "explore" else None),
        )
    except Exception as exc:
        if memory_selected:
            dashboard_events.emit(RuntimeStatusEvent("memory", "failed", error=exc))
        raise
    if memory_selected:
        dashboard_events.emit(RuntimeStatusEvent("memory", "ready"))
    return BehaviorToolkit(
        primitives_kwargs=toolkit_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
        config=config,
        video_path=toolkit_kwargs.get("video_path"),
        state_output_dir=state_output_dir,
    )


__all__ = ["BEHAVIOR_DASHBOARD_SPEC", "get_robot_spec", "get_toolkit"]
