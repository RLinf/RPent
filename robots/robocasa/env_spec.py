"""RoboCasa environment extension — EnvSpec factory, toolkit factory, and runtime hooks."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rpent.dashboard.events import DashboardEventSink
from rpent.envs.env_spec import EnvSpec, RunConfig
from rpent.envs.prompt_bundle import PromptBundle
from rpent.envs.runtime import try_spawn_server, try_wait_server
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.http_rpc import HttpRpcClient
from rpent.utils.rpc import parse_endpoint
from rpent.utils.socket_rpc import SocketRpcClient

from robots.robocasa.prompt_bundle import (
    system_prompt,
    user_prompt,
)

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient


ROBOCASA_DASHBOARD_SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <env> <split> <seed>",
        "fields": (
            {"name": "task_name"},
            {"name": "split", "suggestions": ("target", "pretrain", "all")},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{task_name} / {split} / seed {seed}",
        "output_slug": "{task_name}_{split}_s{seed}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "task"},
        {"name": "vla", "label": "VLA"},
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


def get_env_spec() -> EnvSpec:
    """Return the RoboCasa env identity, prompt bundle, and runner hooks.

    Tool schemas, handlers, server lifecycle, and the MCP allowlist live on
    the RoboCasa toolkit (see :func:`get_toolkit`).
    """
    return EnvSpec(
        name="robocasa",
        prompts=PromptBundle(
            system=system_prompt,
            user=user_prompt,
        ),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_shared_runtime=init_shared_runtime,
        init_task_runtime=init_task_runtime,
        init_runtime=_init_runtime,
        dashboard=ROBOCASA_DASHBOARD_SPEC,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
):
    """Return the RoboCasa toolkit (common tools + RoboCasa primitives)."""
    from robots.robocasa.toolkit import RoboCasaToolkit

    return RoboCasaToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """Register RoboCasa CLI flags on the shared ``parser``."""
    required = not use_dashboard
    parser.add_argument("--task-name", default=None, required=required,
                        help="RoboCasa task name, e.g. OpenDrawer")
    parser.add_argument("--split", default="target",
                        choices=["target", "pretrain", "all"],
                        help="RoboCasa data split (default: target)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hi-res", type=int, default=0,
                        help="Hi-res agentview resolution (0=off)")
    parser.add_argument("--env-endpoint", default=None,
                        help="[protocol://]host:port of an existing env_server")
    parser.add_argument("--vla-endpoint", default=None,
                        help="[protocol://]host:port of an existing vla_server")
    parser.add_argument("--vla-model-path", default=None,
                        help="RLDX checkpoint path for locally spawned vla_server")
    parser.add_argument("--cuda-device", type=int, default=None,
                        help="GPU device to pin MuJoCo and torch(CUDA ordinal).")


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
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{args.task_name}_{args.split}_s{args.seed}"
    output_dir = Path(output_dir)

    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars=prompt_vars,
        task_desc={"task_name": args.task_name, "split": args.split, "seed": args.seed},
    )


def _subprocess_env(**extra: str) -> dict[str, str]:
    """Build the env dict for a subprocess: inherit from parent, layer extras on top.

    CUDA device selection is passed via ``--cuda-device`` on the server command
    line — the server itself handles ``CUDA_VISIBLE_DEVICES`` and EGL alignment.
    """
    env = os.environ.copy()
    env.update(extra)
    return env


def _cuda_args(args: argparse.Namespace) -> list[str]:
    """Return the ``--cuda-device`` CLI args for spawned servers."""
    return (
        ["--cuda-device", str(args.cuda_device)]
        if args.cuda_device is not None
        else []
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
                "--task-name", args.task_name,
                "--split", args.split,
                "--seed", str(args.seed),
                "--transport", "http",
                "--host", host,
                "--port", str(port),
                "--parent-watch",
                *_cuda_args(args),
            ],
            env=_subprocess_env(
                MUJOCO_GL="egl",
                ROBOT_PLATFORM="ROBOCASA",
            ),
            log_path=str(Path(output_dir) / "env_server.log"),
        )
        daemon.start()
        return daemon, HttpRpcClient(f"http://{host}:{port}")
    protocol, host, port = parse_endpoint(args.env_endpoint)
    if protocol == "socket":
        return None, SocketRpcClient(host, port)
    if protocol == "http":
        return None, HttpRpcClient(f"http://{host}:{port}")
    raise ValueError(
        f"--env-endpoint protocol must be socket or http, got {protocol!r}"
    )


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
                "--model-path", args.vla_model_path,
                "--transport", "http",
                "--host", host,
                "--port", str(port),
                "--parent-watch",
                *_cuda_args(args),
            ],
            env=_subprocess_env(),
            log_path=str(Path(output_dir) / "vla_server.log"),
        )
        daemon.start()
        return daemon, HttpRpcClient(f"http://{host}:{port}")
    protocol, host, port = parse_endpoint(args.vla_endpoint)
    if protocol == "socket":
        return None, SocketRpcClient(host, port)
    if protocol == "http":
        return None, HttpRpcClient(f"http://{host}:{port}")
    raise ValueError(
        f"--vla-endpoint protocol must be socket or http, got {protocol!r}"
    )


def init_task_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize one TaskRun-owned RoboCasa environment.

    A local env server is fresh for every call. When ``--env-endpoint`` is
    supplied, the returned daemon list is empty so the external service stays
    running. The VLA service is Session-owned and comes from
    :func:`init_shared_runtime`.
    """
    from robots.robocasa.env_client import RoboCasaEnvClient

    owned_daemons: dict[str, ProcessDaemon] = {}

    env_daemon, env_rpc = try_spawn_server(
        owned_daemons,
        dashboard_events,
        "env",
        lambda: _spawn_env_server(args, output_dir),
    )

    env_post_fn = lambda: RoboCasaEnvClient(
        env_rpc,
        expected_meta={
            "task_name": args.task_name,
            "split": args.split,
            "seed": args.seed,
            "camera_h": 256,
            "camera_w": 256,
        },
    )
    env_client = try_wait_server(
        owned_daemons, dashboard_events, "env", env_rpc, env_daemon, 120.0,
        post_fn = env_post_fn,
    )
    return list(owned_daemons.values()), {
        "env_client": env_client,
        "workdir": str(output_dir),
        "hi_res": args.hi_res or None,
    }


def init_shared_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize the Session-owned RoboCasa VLA service.

    The returned list contains only locally started services. External
    endpoints are connected to but never become owned.
    """
    from robots.robocasa.vla_client import RoboCasaVLAClient

    owned_daemons: dict[str, ProcessDaemon] = {}

    vla_daemon, vla_rpc = try_spawn_server(
        owned_daemons,
        dashboard_events,
        "vla",
        lambda: _spawn_vla_server(args, output_dir),
    )

    vla_post_fn = lambda: RoboCasaVLAClient(vla_rpc)
    vla_client = try_wait_server(
        owned_daemons, dashboard_events, "vla", vla_rpc, vla_daemon, 300.0,
        post_fn = vla_post_fn,
    )
    return list(owned_daemons.values()), {"vla_client": vla_client}


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Spawn env + vla daemons and build clients for RoboCasa.

    Each server can be spawned or attached-to independently: pass an
    endpoint to attach, or leave it unset to spawn a local subprocess.

    Heavy deps (vla / env_client) are imported lazily so that a bare
    ``import robots.robocasa`` (for ``get_env_spec`` / ``get_toolkit``)
    doesn't drag them in. ``rpent.utils`` helpers are imported at module
    top level.
    """
    from robots.robocasa.env_client import RoboCasaEnvClient
    from robots.robocasa.vla_client import RoboCasaVLAClient

    owned_daemons: dict[str, ProcessDaemon] = {}

    env_daemon, env_rpc = try_spawn_server(
        owned_daemons,
        dashboard_events,
        "env",
        lambda: _spawn_env_server(args, output_dir),
    )

    vla_daemon, vla_rpc = try_spawn_server(
        owned_daemons,
        dashboard_events,
        "vla",
        lambda: _spawn_vla_server(args, output_dir),
    )

    # All local daemons are running, so they initialize concurrently while
    # readiness is checked in a deterministic order.
    env_post_fn = lambda: RoboCasaEnvClient(
        env_rpc,
        expected_meta={
            "task_name": args.task_name,
            "split": args.split,
            "seed": args.seed,
            "camera_h": 256,
            "camera_w": 256,
        },
    )
    vla_post_fn = lambda: RoboCasaVLAClient(vla_rpc)
    results = {}
    for component, rpc, daemon, post_fn, timeout_s in (
        ("env", env_rpc, env_daemon, env_post_fn, 120.0),
        ("vla", vla_rpc, vla_daemon, vla_post_fn, 300.0),
    ):
        results[component] = try_wait_server(
            owned_daemons, dashboard_events, component, rpc, daemon, timeout_s,
            post_fn = post_fn,
        )

    return list(owned_daemons.values()), {
        "env_client": results["env"],
        "vla_client": results["vla"],
        "workdir": str(output_dir),
        "hi_res": args.hi_res or None,
    }