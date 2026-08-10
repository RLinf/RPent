"""RoboCasa environment extension."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.robocasa.prompt_bundle import (
    system_prompt,
    user_prompt,
)
from rpent.dashboard.events import DashboardEventSink, RuntimeStatusEvent
from rpent.envs.env_spec import EnvSpec, RunConfig
from rpent.envs.prompt_bundle import PromptBundle
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.http_rpc import HttpRpcClient
from rpent.utils.rpc import parse_endpoint, wait_for_ready
from rpent.utils.socket_rpc import SocketRpcClient

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient


ROBOCASA_DASHBOARD_SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <env> <split> <seed>",
        "fields": (
            {"name": "robocasa_env"},
            {"name": "robocasa_split", "suggestions": ("target", "pretrain", "all")},
            {"name": "robocasa_seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{robocasa_env} / {robocasa_split} / seed {robocasa_seed}",
        "output_slug": "{robocasa_env}_{robocasa_split}_s{robocasa_seed}",
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
    parser.add_argument("--robocasa-env", default=None, required=required,
                        help="RoboCasa task name, e.g. OpenDrawer")
    parser.add_argument("--robocasa-split", default="target",
                        choices=["target", "pretrain", "all"],
                        help="RoboCasa data split (default: target)")
    parser.add_argument("--robocasa-seed", type=int, default=0)
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
    if not args.robocasa_env:
        raise ValueError("--robocasa-env is required")

    recipe_tag = f"{args.robocasa_env}_{args.robocasa_split}_s{args.robocasa_seed}"
    prompt_vars = {
        "suite": args.robocasa_split,
        "task": args.robocasa_env,
        "env_name": args.robocasa_env,
        "split": args.robocasa_split,
        "seed": args.robocasa_seed,
        "recipe_tag": recipe_tag,
    }

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{args.robocasa_env}_{args.robocasa_split}_s{args.robocasa_seed}"
    output_dir = Path(output_dir)

    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars=prompt_vars,
        task_desc={"env_name": args.robocasa_env, "split": args.robocasa_split, "seed": args.robocasa_seed},
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
                "--env", args.robocasa_env,
                "--split", args.robocasa_split,
                "--seed", str(args.robocasa_seed),
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


def _stop_owned_daemons(daemons: list[ProcessDaemon]) -> None:
    """Stop owned daemons in reverse order without masking startup errors."""
    for daemon in reversed(daemons):
        try:
            daemon.stop()
        except Exception:
            pass


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

    owned_daemons: list[ProcessDaemon] = []

    dashboard_events.emit(RuntimeStatusEvent("env", "starting"))
    try:
        env_daemon, env_rpc = _spawn_env_server(args, output_dir)
        if env_daemon is not None:
            owned_daemons.append(env_daemon)
        wait_for_ready(env_rpc, daemon=env_daemon, timeout_s=120.0)
        env_client = RoboCasaEnvClient(
            env_rpc,
            expected_meta={
                "env_name": args.robocasa_env,
                "split": args.robocasa_split,
                "seed": args.robocasa_seed,
                "camera_h": 256,
                "camera_w": 256,
            },
        )
    except Exception as exc:
        _stop_owned_daemons(owned_daemons)
        dashboard_events.emit(RuntimeStatusEvent("env", "failed", error=exc))
        raise
    dashboard_events.emit(RuntimeStatusEvent("env", "ready"))
    return owned_daemons, {
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

    owned_daemons: list[ProcessDaemon] = []

    dashboard_events.emit(RuntimeStatusEvent("vla", "starting"))
    try:
        vla_daemon, vla_rpc = _spawn_vla_server(args, output_dir)
        if vla_daemon is not None:
            owned_daemons.append(vla_daemon)
        wait_for_ready(vla_rpc, daemon=vla_daemon, timeout_s=300.0)
        vla_client = RoboCasaVLAClient(vla_rpc)
    except Exception as exc:
        _stop_owned_daemons(owned_daemons)
        dashboard_events.emit(RuntimeStatusEvent("vla", "failed", error=exc))
        raise
    dashboard_events.emit(RuntimeStatusEvent("vla", "ready"))
    return owned_daemons, {"vla_client": vla_client}


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

    daemons: list[ProcessDaemon] = []

    # --- env_server --------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("env", "starting"))
    try:
        env_daemon, env_rpc = _spawn_env_server(args, output_dir)
        if env_daemon is not None:
            daemons.append(env_daemon)
    except Exception as exc:
        _stop_owned_daemons(daemons)
        dashboard_events.emit(RuntimeStatusEvent("env", "failed", error=exc))
        raise

    # --- vla_server --------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("vla", "starting"))
    try:
        vla_daemon, vla_rpc = _spawn_vla_server(args, output_dir)
        if vla_daemon is not None:
            daemons.append(vla_daemon)
    except Exception as exc:
        _stop_owned_daemons(daemons)
        dashboard_events.emit(RuntimeStatusEvent("vla", "failed", error=exc))
        raise

    # All local daemons are running, so they initialize concurrently while
    # readiness is checked in a deterministic order.
    for component, client, daemon, timeout_s in (
        ("env", env_rpc, env_daemon, 120.0),
        ("vla", vla_rpc, vla_daemon, 300.0),
    ):
        try:
            wait_for_ready(client, daemon=daemon, timeout_s=timeout_s)
        except Exception as exc:
            _stop_owned_daemons(daemons)
            dashboard_events.emit(RuntimeStatusEvent(component, "failed", error=exc))
            raise
        dashboard_events.emit(RuntimeStatusEvent(component, "ready"))

    primitives_kwargs = {
        "env_client": RoboCasaEnvClient(
            env_rpc,
            expected_meta={
                "env_name": args.robocasa_env,
                "split": args.robocasa_split,
                "seed": args.robocasa_seed,
                "camera_h": 256,
                "camera_w": 256,
            },
        ),
        "workdir": str(output_dir),
        "hi_res": args.hi_res or None,
        "vla_client": RoboCasaVLAClient(vla_rpc),
    }
    return daemons, primitives_kwargs

