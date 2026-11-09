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

"""LeKiwi mobile-base robot extension — RobotSpec, toolkit factory, and runtime hooks."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from robots.lekiwi.prompt_bundle import system_prompt, user_prompt
from robots.lekiwi.runtime_config import set_robot_config_path
from robots.lekiwi.tasks import LEKIWI_TASKS, get_lekiwi_task
from rpent.dashboard.events import DashboardEventSink
from rpent.dashboard.spec import DashboardSpec
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import try_wait_server
from rpent.utils.config import get_memory_dir, get_repo_root
from rpent.utils.daemon import ProcessDaemon
from rpent.utils.rpc import make_rpc_client

LEKIWI_DASHBOARD_SPEC: DashboardSpec = {
    # The env runs on the robot's onboard computer and is attached, never
    # spawned per TaskRun, so dashboard control requires --env-endpoint.
    "external_env": True,
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task_id>",
        "fields": (
            {
                "name": "task_id",
                "kind": "integer",
                "minimum": 0,
                "suggestions": tuple(sorted(LEKIWI_TASKS)),
            },
        ),
        "display": "LeKiwi task {task_id}",
        "output_slug": "lekiwi_t{task_id}",
    },
    "runtime_components": ({"name": "env", "label": "LEKIWI", "scope": "unique"},),
    "primitives": (
        "drive",
        "stop",
    ),
}


def get_robot_spec() -> RobotSpec:
    """Return the LeKiwi identity, prompts, runtime hooks, and dashboard spec."""
    return RobotSpec(
        name="lekiwi",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=LEKIWI_DASHBOARD_SPEC,
        is_real_robot=True,
        supports_exploration=False,
    )


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
):
    """Return the LeKiwi toolkit."""
    from robots.lekiwi.toolkit import LeKiwiToolkit

    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("lekiwi"),
    )
    return LeKiwiToolkit(
        runtime_kwargs=runtime_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    parser.add_argument(
        "--task-id",
        type=int,
        default=None if use_dashboard else 0,
        choices=sorted(LEKIWI_TASKS),
    )
    parser.add_argument(
        "--env-endpoint",
        default=None,
        help="LeKiwi env_server URL (http://host:port) running on the robot's computer",
    )
    parser.add_argument(
        "--robot-config",
        default=None,
        help="LeKiwi site YAML (port, camera serials, velocity limits)",
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    set_robot_config_path(args.robot_config)
    if args.task_id is None:
        raise ValueError("--task-id is required")
    task = get_lekiwi_task(args.task_id)
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    output_dir = Path(
        args.output_dir
        or get_repo_root() / "logs" / f"{timestamp}_lekiwi_t{args.task_id}"
    )
    constraints = "\n".join(
        f"{index}. {constraint}" for index, constraint in enumerate(task.constraints, 1)
    )
    return RunConfig(
        recipe_tag=f"lekiwi_t{args.task_id}",
        output_dir=output_dir,
        prompt_vars={
            "task_name": task.name,
            "instruction": task.instruction,
            "success_criteria": task.success_criteria,
            "constraints": constraints,
        },
        task_desc={"task_id": args.task_id, "task_name": task.name},
    )


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Attach to the Env started on the robot, following RPent's real-robot contract."""
    from robots.lekiwi.env_client import LeKiwiEnvClient

    selected = {"env"} if components is None else set(components)
    if selected - {"env"}:
        raise ValueError(
            f"unknown LeKiwi runtime components: {sorted(selected - {'env'})}"
        )
    if "env" not in selected:
        return [], {}
    if not args.env_endpoint:
        raise ValueError(
            "--env-endpoint is required; start robots.lekiwi.env_server on the robot first"
        )
    rpc = make_rpc_client(args.env_endpoint)

    def connect_env():
        env = LeKiwiEnvClient(rpc)
        if env.meta.get("control_mode", "manual") == "manual":
            env.reset()
        return {"env": env}

    runtime_kwargs = try_wait_server(
        {},
        dashboard_events,
        "env",
        rpc,
        None,
        5.0 if getattr(args, "dashboard", False) else 60.0,
        post_fn=connect_env,
    )
    return [], runtime_kwargs
