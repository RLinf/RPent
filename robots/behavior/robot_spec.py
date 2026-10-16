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

from robots.behavior.memory import BehaviorMemoryManager
from robots.behavior.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink
from rpent.dashboard.spec import DashboardSpec
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import stop_owned_daemons

BEHAVIOR_DASHBOARD_SPEC: DashboardSpec = {
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
    ),
    "primitives": (),
}


def get_robot_spec() -> RobotSpec:
    from robots.behavior import runtime

    return RobotSpec(
        name="behavior",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=runtime.add_cli_args,
        parse_config=runtime.parse_config,
        init_runtime=runtime.init_runtime,
        supports_exploration=True,
        dashboard=BEHAVIOR_DASHBOARD_SPEC,
    )


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
    mode: str = "evaluation",
    attempts_per_session: int = 0,
    state_output_dir: Path | str | None = None,
):
    """Return the BEHAVIOR toolkit through the standard main contract."""

    from robots.behavior.toolkit import BehaviorToolkit

    toolkit_kwargs = dict(runtime_kwargs)
    if attempts_per_session > 0:
        raise ValueError(
            "BEHAVIOR explore runs one attempt per session; use --explore-sessions"
        )
    if mode == "exploration":
        behavior_mode = "explore"
    elif mode == "evaluation":
        behavior_mode = "eval"
    else:
        raise ValueError(f"unsupported BEHAVIOR toolkit mode: {mode!r}")
    toolkit_kwargs["behavior_phase"] = behavior_mode
    memory_dir = config.prompt_vars.get("memory_dir")
    if not memory_dir:
        raise ValueError("BEHAVIOR RunConfig is missing memory_dir")
    memory = BehaviorMemoryManager(
        root=Path(memory_dir),
        memory_access="inbox_write" if behavior_mode == "explore" else "read_only",
        inbox_cell_tag=config.recipe_tag if behavior_mode == "explore" else None,
    )
    env_args = toolkit_kwargs.pop("env_args", None)
    owned_daemons = {}
    if env_args is not None:
        from robots.behavior.runtime import init_runtime

        daemons, env_kwargs = init_runtime(
            env_args,
            Path(state_output_dir or config.output_dir),
            dashboard_events,
            {"env"},
        )
        owned_daemons = {"env": daemons[0]}
        toolkit_kwargs.update(env_kwargs)
    try:
        return BehaviorToolkit(
            runtime_kwargs=toolkit_kwargs,
            dashboard_events=dashboard_events,
            memory=memory,
            config=config,
            state_output_dir=state_output_dir,
            owned_daemons=owned_daemons,
        )
    except BaseException:
        stop_owned_daemons(owned_daemons, dashboard_events)
        raise


__all__ = ["BEHAVIOR_DASHBOARD_SPEC", "get_robot_spec", "get_toolkit"]
