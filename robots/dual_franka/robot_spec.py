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

"""Dual-Franka robot extension — RobotSpec factory, toolkit factory, and runtime hooks."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.dual_franka.prompt_bundle import system_prompt, user_prompt
from robots.dual_franka.tasks import DUAL_FRANKA_TASKS, get_dual_franka_task
from robots.franka.runtime_config import DEFAULT_CALIBRATION_PATH
from rpent.dashboard.events import DashboardEventSink, RuntimeStatusEvent
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import try_spawn_server, try_wait_server
from rpent.utils.config import get_memory_dir, get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc import make_rpc_client
from rpent.utils.rpc.http_rpc import HttpRpcClient

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient


DUAL_FRANKA_DASHBOARD_SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task_id>",
        "fields": (
            {
                "name": "task_id",
                "kind": "integer",
                "minimum": 0,
                "suggestions": (0, 1),
            },
        ),
        "display": "Dual Franka task {task_id}",
        "output_slug": "dual_franka_t{task_id}",
    },
    "runtime_components": (
        {"name": "env", "label": "DUAL FRANKA", "scope": "unique"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
    ),
    "frame_channels": (
        {
            "name": "left_wrist",
            "label": "left wrist camera",
            "legacy_path_key": "image_left_wrist_path",
        },
        {
            "name": "base",
            "label": "base camera",
            "legacy_path_key": "image_base_path",
        },
        {
            "name": "right_wrist",
            "label": "right wrist camera",
            "legacy_path_key": "image_right_wrist_path",
        },
    ),
}


def get_robot_spec() -> RobotSpec:
    """Return the dual-Franka identity, prompts, runtime hooks, and dashboard spec."""
    return RobotSpec(
        name="dual_franka",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=DUAL_FRANKA_DASHBOARD_SPEC,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
):
    """Return the dual-Franka toolkit."""
    from robots.dual_franka.toolkit import DualFrankaToolkit

    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("dual_franka"),
    )
    return DualFrankaToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    parser.add_argument(
        "--task-id",
        type=int,
        default=None if use_dashboard else 0,
        choices=sorted(DUAL_FRANKA_TASKS),
    )
    parser.add_argument("--env-endpoint", default=None)
    parser.add_argument("--vla-endpoint", default=None)
    parser.add_argument("--robot-config", default=None)
    parser.add_argument(
        "--vla-model-path",
        default=os.environ.get("PI05_CHECKPOINT_PATH"),
    )
    parser.add_argument(
        "--vla-repo-id",
        default=os.environ.get("DUAL_FRANKA_REPO_ID"),
        help="SFT dataset repo ID used to locate norm_stats.json",
    )
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument(
        "--calibration-path",
        default=str(DEFAULT_CALIBRATION_PATH),
        help="Path to hand_eye_calibration.json (defaults to easy_handeye's "
        "~/.ros/easy_handeye directory).",
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if args.task_id is None:
        raise ValueError("--task-id is required")
    task = get_dual_franka_task(args.task_id)
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    output_dir = Path(
        args.output_dir
        or get_repo_root() / "logs" / f"{timestamp}_dual_franka_t{args.task_id}"
    )
    constraints = "\n".join(
        f"{index}. {constraint}" for index, constraint in enumerate(task.constraints, 1)
    )
    return RunConfig(
        recipe_tag=f"dual_franka_t{args.task_id}",
        output_dir=output_dir,
        prompt_vars={
            "task_id": args.task_id,
            "task_name": task.name,
            "instruction": task.instruction,
            "success_criteria": task.success_criteria,
            "constraints": constraints,
            "output_dir": str(output_dir),
        },
        task_desc={"task_id": args.task_id, "task_name": task.name},
    )


def _env_server_command(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
) -> list[str]:
    task = get_dual_franka_task(args.task_id)
    command = [
        sys.executable,
        "-m",
        "robots.dual_franka.env_server",
        "--transport",
        "http",
        "--host",
        host,
        "--port",
        str(port),
        "--task-description",
        task.instruction,
        "--parent-watch",
    ]
    if args.robot_config:
        command.extend(["--robot-config", args.robot_config])
    return command


def _vla_server_command(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "robots.dual_franka.vla_server",
        "--transport",
        "http",
        "--host",
        host,
        "--port",
        str(port),
        "--model-path",
        args.vla_model_path,
        "--repo-id",
        args.vla_repo_id,
        "--parent-watch",
    ]
    if args.cuda_device is not None:
        command.extend(["--cuda-device", str(args.cuda_device)])
    return command


def _spawn_env_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Spawn (or attach to) the RLinf-backed dual-Franka environment server.

    Returns ``(daemon, rpc)`` — the daemon is ``None`` when an external
    endpoint was attached (the caller must not own it).
    """
    if args.env_endpoint is not None:
        return None, make_rpc_client(args.env_endpoint)
    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        name="dual_franka_env_server",
        cmd=_env_server_command(args, host=host, port=port),
        # Both Ray nodes already use this project's pre-provisioned environment.
        # Disable Ray's automatic uv hook: it otherwise runs uv from a temporary
        # uploaded directory and can repeatedly miss the worker startup timeout.
        env_overrides={"RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"},
        log_path=str(output_dir / "dual_franka_env_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _spawn_vla_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Spawn (or attach to) the dual-Franka VLA service."""
    if args.vla_endpoint is not None:
        return None, make_rpc_client(args.vla_endpoint)
    if not args.vla_model_path or not args.vla_repo_id:
        raise ValueError(
            "dual-Franka VLA auto-start requires --vla-model-path and "
            "--vla-repo-id (or PI05_CHECKPOINT_PATH and "
            "DUAL_FRANKA_REPO_ID)"
        )
    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        name="dual_franka_vla_server",
        cmd=_vla_server_command(args, host=host, port=port),
        log_path=str(output_dir / "dual_franka_vla_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize every dual-Franka component, or only ``components`` when given.

    Each server can be spawned or attached-to independently: pass an endpoint
    to attach, or leave it unset to spawn a local subprocess. A VLA is only
    started for the ``vla_grasp`` task (or an explicit ``--vla-endpoint``).
    """
    from robots.dual_franka.env_client import DualFrankaEnvClient
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient

    available = {"env", "vla"}
    selected = available if components is None else components
    unknown = selected.difference(available)
    if unknown:
        raise ValueError(f"unknown dual-Franka runtime components: {sorted(unknown)}")

    needs_vla = args.vla_endpoint is not None
    if args.task_id is not None:
        needs_vla = needs_vla or (
            get_dual_franka_task(args.task_id).name == "vla_grasp"
        )

    starters = {
        "env": lambda: _spawn_env_server(args, output_dir),
        "vla": lambda: _spawn_vla_server(args, output_dir),
    }
    connectors = {
        "env": lambda rpc: {
            "env": DualFrankaEnvClient(rpc),
            "task_description": get_dual_franka_task(args.task_id).instruction,
        },
        "vla": lambda rpc: {"model": Pi05VLAClient(rpc, embodiment="dual_franka")},
    }

    owned_daemons: dict[str, ProcessDaemon] = {}
    pending: dict[str, tuple[ProcessDaemon | None, RpcClient]] = {}
    for component, starter in starters.items():
        if component not in selected:
            continue
        if component == "vla" and not needs_vla:
            continue
        pending[component] = try_spawn_server(
            owned_daemons, dashboard_events, component, starter
        )

    primitives_kwargs: dict[str, Any] = {}
    for component, (daemon, rpc) in pending.items():
        component_kwargs = try_wait_server(
            owned_daemons,
            dashboard_events,
            component,
            rpc,
            daemon,
            300.0,
            post_fn=partial(connectors[component], rpc),
        )
        primitives_kwargs.update(component_kwargs)

    if "vla" in selected and not needs_vla:
        dashboard_events.emit(RuntimeStatusEvent("vla", "ready"))
        primitives_kwargs["model"] = None

    primitives_kwargs["calibration_path"] = args.calibration_path

    return list(owned_daemons.values()), primitives_kwargs
