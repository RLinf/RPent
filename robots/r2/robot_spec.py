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

"""R2 robot extension: RobotSpec factory, toolkit factory, and runtime hooks."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.r2.prompt_bundle import system_prompt, user_prompt
from robots.r2.tasks import R2_TASKS, get_r2_task
from rpent.dashboard.events import DashboardEventSink
from rpent.dashboard.spec import DashboardSpec
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import try_wait_server
from rpent.utils.config import get_memory_dir, get_repo_root
from rpent.utils.rpc import make_rpc_client

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon


R2_DASHBOARD_SPEC: DashboardSpec = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task_id>",
        "fields": (
            {
                "name": "task_id",
                "kind": "integer",
                "minimum": 0,
                "choices": tuple(str(key) for key in R2_TASKS),
            },
        ),
        "display": "R2 task {task_id}",
        "output_slug": "r2_t{task_id}",
    },
    "runtime_components": ({"name": "env", "label": "R2", "scope": "unique"},),
    "primitives": (
        "move_to",
        "move_delta",
        "rotate_delta",
        "set_gripper",
        "look",
        "go_home",
    ),
}


def get_robot_spec() -> RobotSpec:
    """Return the R2 identity, prompts, runtime hooks, and dashboard spec."""
    return RobotSpec(
        name="r2",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=R2_DASHBOARD_SPEC,
        is_real_robot=True,
        supports_exploration=False,
    )


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
):
    """Return the R2 toolkit."""
    from robots.r2.toolkit import R2Toolkit

    return R2Toolkit(
        runtime_kwargs=runtime_kwargs,
        dashboard_events=dashboard_events,
        memory=MemoryManager(
            root=config.prompt_vars.get("memory_dir") or get_memory_dir("r2"),
        ),
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    parser.add_argument(
        "--task-id",
        type=int,
        default=None if use_dashboard else 0,
        choices=sorted(R2_TASKS),
    )
    parser.add_argument(
        "--task-language",
        default=None,
        help="Replace the task's instruction, e.g. to name the object to pick",
    )
    parser.add_argument(
        "--env-endpoint",
        required=not use_dashboard,
        help="R2 env_server endpoint running on the robot, e.g. http://ROBOT_IP:8120",
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if args.task_id is None:
        raise ValueError("--task-id is required")
    task = get_r2_task(args.task_id)
    instruction = args.task_language or task.instruction
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    output_dir = Path(
        args.output_dir or get_repo_root() / "logs" / f"{timestamp}_r2_t{args.task_id}"
    )
    constraints = "\n".join(
        f"{index}. {constraint}" for index, constraint in enumerate(task.constraints, 1)
    )
    return RunConfig(
        recipe_tag=f"r2_t{args.task_id}",
        output_dir=output_dir,
        prompt_vars={
            "task_name": task.name,
            "instruction": instruction,
            "success_criteria": task.success_criteria,
            "constraints": constraints,
        },
        task_desc={
            "task_id": args.task_id,
            "task_name": task.name,
            "instruction": instruction,
        },
    )


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Attach to the env server the operator started on the robot.

    RPent never starts the R2 env server: it owns the hardware and runs on the
    robot under the operator's control.
    """
    del output_dir
    selected = {"env"} if components is None else set(components)
    unknown = selected - {"env"}
    if unknown:
        raise ValueError(f"unknown R2 runtime components: {sorted(unknown)}")
    if "env" not in selected:
        return [], {}
    if not args.env_endpoint:
        raise ValueError(
            "--env-endpoint is required for R2; start robots.r2.env_server on the robot first"
        )
    from robots.r2.env_client import R2EnvClient

    rpc = make_rpc_client(args.env_endpoint)
    owned_daemons: dict[str, ProcessDaemon] = {}
    runtime_kwargs = try_wait_server(
        owned_daemons,
        dashboard_events,
        "env",
        rpc,
        None,
        5.0 if getattr(args, "dashboard", False) else 60.0,
        post_fn=lambda: {"env": R2EnvClient(rpc)},
    )
    return [], runtime_kwargs
