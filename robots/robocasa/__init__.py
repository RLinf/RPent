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

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon
    from rpent.utils.rpc import RpcClient


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
        init_runtime=_init_runtime,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    video_path: str | None = None,
    save_action_videos: bool = False,
):
    """Return the RoboCasa toolkit (common tools + RoboCasa primitives)."""
    from robots.robocasa.toolkit import RoboCasaToolkit

    return RoboCasaToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
        video_path=video_path,
        save_action_videos=save_action_videos,
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


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Spawn env + vla daemons and build clients for RoboCasa.

    Each server can be spawned or attached-to independently: pass an
    endpoint to attach, or leave it unset to spawn a local subprocess.

    Heavy deps (rpc / vla / daemon / env_client) are imported lazily so
    that a bare ``import robots.robocasa`` (for ``get_env_spec`` /
    ``get_toolkit``) doesn't drag them in.
    """
    from robots.robocasa.env_client import RoboCasaEnvClient
    from robots.robocasa.vla_client import RoboCasaVLAClient
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import parse_endpoint, wait_for_ready
    from rpent.utils.socket_rpc import SocketRpcClient

    daemons: list[ProcessDaemon] = []
    cuda_args = ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []

    # --- env_server --------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("env", "starting"))
    try:
        env_daemon: ProcessDaemon | None = None
        if args.env_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            env_daemon = ProcessDaemon(
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
                    *cuda_args,
                ],
                env=_subprocess_env(
                    MUJOCO_GL="egl",
                ),
                log_path=str(Path(output_dir) / "env_server.log"),
            )
            env_daemon.start()
            daemons.append(env_daemon)
            env_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.env_endpoint)
            if protocol == "socket":
                env_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                env_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--env-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
        dashboard_events.emit(RuntimeStatusEvent("env", "failed", error=exc))
        raise

    # --- vla_server --------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("vla", "starting"))
    try:
        vla_daemon: ProcessDaemon | None = None
        if args.vla_endpoint is None:
            if not args.vla_model_path:
                raise ValueError(
                    "--vla-model-path is required when spawning a local vla_server"
                )
            host, port = "127.0.0.1", pick_free_port()
            vla_daemon = ProcessDaemon(
                name="vla_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "robocasa" / "vla_server.py"),
                    "--model-path", args.vla_model_path,
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(),
                log_path=str(Path(output_dir) / "vla_server.log"),
            )
            vla_daemon.start()
            daemons.append(vla_daemon)
            vla_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.vla_endpoint)
            if protocol == "socket":
                vla_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                vla_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--vla-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
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
            for started_daemon in reversed(daemons):
                started_daemon.stop()
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
