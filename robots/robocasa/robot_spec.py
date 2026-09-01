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

"""RoboCasa robot extension — RobotSpec factory, toolkit factory, and runtime hooks."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.robocasa.prompt_bundle import (
    system_prompt,
    user_prompt,
)
from rpent.dashboard.events import DashboardEventSink
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


ROBOCASA_DASHBOARD_SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task_name> <split> <seed>",
        "fields": (
            {"name": "task_name"},
            {"name": "split", "suggestions": ("target", "pretrain", "all")},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{task_name} / {split} / seed {seed}",
        "output_slug": "{task_name}_{split}_s{seed}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "unique"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
    ),
    "frame_channels": (
        {
            "name": "camera",
            "label": "fixed camera",
            "legacy_path_key": "image_cam_path",
        },
        {
            "name": "wrist",
            "label": "wrist camera",
            "legacy_path_key": "image_wrist_path",
        },
    ),
}


def get_robot_spec() -> RobotSpec:
    """Return the RoboCasa robot identity, prompt bundle, and runner hooks.

    Tool schemas, handlers, server lifecycle, and the MCP allowlist live on
    the RoboCasa toolkit (see :func:`get_toolkit`).
    """
    return RobotSpec(
        name="robocasa",
        prompts=PromptBundle(
            system=system_prompt,
            user=user_prompt,
        ),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=ROBOCASA_DASHBOARD_SPEC,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
):
    """Return the RoboCasa toolkit for the current session."""
    from robots.robocasa.toolkit import RoboCasaToolkit

    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("robocasa"),
    )
    return RoboCasaToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """Register RoboCasa CLI flags on the shared ``parser``."""
    required = not use_dashboard
    parser.add_argument(
        "--task-name",
        default=None,
        required=required,
        help="RoboCasa task name, e.g. OpenDrawer",
    )
    parser.add_argument(
        "--split",
        default="target",
        choices=["target", "pretrain", "all"],
        help="RoboCasa data split (default: target)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hi-res", type=int, default=0, help="Hi-res agentview resolution (0=off)"
    )
    parser.add_argument(
        "--env-endpoint",
        default=None,
        help="[protocol://]host:port of an existing env_server",
    )
    parser.add_argument(
        "--vla-endpoint",
        default=None,
        help="[protocol://]host:port of an existing vla_server",
    )
    parser.add_argument(
        "--vla-model-path",
        default=None,
        help="RLDX checkpoint path for locally spawned vla_server",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device to pin MuJoCo and torch(CUDA ordinal).",
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    """Validate final ``args`` and derive per-run identifiers."""
    if not args.task_name:
        raise ValueError("--task-name is required")

    recipe_tag = f"{args.task_name}_{args.split}_s{args.seed}"
    prompt_vars = {
        "task_name": args.task_name,
        "split": args.split,
        "seed": args.seed,
        "recipe_tag": recipe_tag,
    }

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = (
            get_repo_root()
            / "logs"
            / f"{timestamp}_{args.task_name}_{args.split}_s{args.seed}"
        )
    output_dir = Path(output_dir)

    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars=prompt_vars,
        task_desc={"task_name": args.task_name, "split": args.split, "seed": args.seed},
    )


def _spawn_env_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Spawn (or attach to) the RoboCasa env_server.

    Returns ``(daemon, rpc)`` — the daemon is ``None`` when an external
    endpoint was attached (the caller must not own it).
    """
    if args.env_endpoint is None:
        host, port = "127.0.0.1", pick_free_port()
        daemon = ProcessDaemon(
            name="env_server",
            cmd=[
                sys.executable,
                str(get_repo_root() / "robots" / "robocasa" / "env_server.py"),
                "--task-name",
                args.task_name,
                "--split",
                args.split,
                "--seed",
                str(args.seed),
                "--transport",
                "http",
                "--host",
                host,
                "--port",
                str(port),
                "--parent-watch",
                *(
                    ["--cuda-device", str(args.cuda_device)]
                    if args.cuda_device is not None
                    else []
                ),
            ],
            env_overrides={
                "MUJOCO_GL": "egl",
                "ROBOT_PLATFORM": "ROBOCASA",
            },
            log_path=str(Path(output_dir) / "env_server.log"),
        )
        daemon.start()
        return daemon, HttpRpcClient(f"http://{host}:{port}")
    return None, make_rpc_client(args.env_endpoint)


def _spawn_vla_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Spawn (or attach to) the RoboCasa vla_server.

    Returns ``(daemon, rpc)`` — the daemon is ``None`` when an external
    endpoint was attached (the caller must not own it).
    """
    if args.vla_endpoint is None:
        if not args.vla_model_path:
            raise ValueError(
                "--vla-model-path is required when spawning a local vla_server"
            )
        host, port = "127.0.0.1", pick_free_port()
        daemon = ProcessDaemon(
            name="vla_server",
            cmd=[
                sys.executable,
                str(get_repo_root() / "robots" / "robocasa" / "vla_server.py"),
                "--model-path",
                args.vla_model_path,
                "--transport",
                "http",
                "--host",
                host,
                "--port",
                str(port),
                "--parent-watch",
                *(
                    ["--cuda-device", str(args.cuda_device)]
                    if args.cuda_device is not None
                    else []
                ),
            ],
            log_path=str(Path(output_dir) / "vla_server.log"),
        )
        daemon.start()
        return daemon, HttpRpcClient(f"http://{host}:{port}")
    return None, make_rpc_client(args.vla_endpoint)


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize every RoboCasa component, or only ``components`` when given.

    Each server can be spawned or attached-to independently: pass an
    endpoint to attach, or leave it unset to spawn a local subprocess.
    """
    from robots.robocasa.env_client import RoboCasaEnvClient
    from robots.robocasa.vla_client import RoboCasaVLAClient

    starters = {
        "env": lambda: _spawn_env_server(args, output_dir),
        "vla": lambda: _spawn_vla_server(args, output_dir),
    }
    connectors = {
        "env": lambda rpc: {
            "env_client": RoboCasaEnvClient(
                rpc,
                expected_meta={
                    "task_name": args.task_name,
                    "split": args.split,
                    "seed": args.seed,
                    "camera_h": 256,
                    "camera_w": 256,
                },
            ),
            "workdir": str(output_dir),
            "hi_res": args.hi_res or None,
        },
        "vla": lambda rpc: {"vla_client": RoboCasaVLAClient(rpc)},
    }
    timeouts = {"env": 120.0, "vla": 300.0}
    selected = set(starters) if components is None else components
    unknown = selected.difference(starters)
    if unknown:
        raise ValueError(f"unknown RoboCasa runtime components: {sorted(unknown)}")

    pending: dict[str, tuple[ProcessDaemon | None, RpcClient]] = {}
    owned_daemons: dict[str, ProcessDaemon] = {}
    for component, starter in starters.items():
        if component in selected:
            pending[component] = try_spawn_server(
                owned_daemons,
                dashboard_events,
                component,
                starter,
            )

    primitives_kwargs: dict[str, Any] = {}
    for component, (daemon, rpc) in pending.items():
        component_kwargs = try_wait_server(
            owned_daemons,
            dashboard_events,
            component,
            rpc,
            daemon,
            timeouts[component],
            post_fn=partial(connectors[component], rpc),
        )
        primitives_kwargs.update(component_kwargs)

    return list(owned_daemons.values()), primitives_kwargs
