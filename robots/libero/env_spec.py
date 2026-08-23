"""LIBERO environment extension — EnvSpec factory, toolkit factory, and runtime hooks."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.libero.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink
from rpent.envs.env_spec import EnvSpec, RunConfig
from rpent.envs.prompt_bundle import PromptBundle
from rpent.envs.runtime import try_spawn_server, try_wait_server
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.http_rpc import HttpRpcClient
from rpent.utils.rpc import parse_endpoint
from rpent.utils.socket_rpc import SocketRpcClient

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient


LIBERO_SUITE_NAMES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_object_task",
    "libero_object_swap",
    "libero_object_lan",
    "libero_goal_task",
    "libero_goal_swap",
    "libero_goal_lan",
    "libero_spatial_task",
    "libero_spatial_swap",
    "libero_spatial_lan",
    "libero_10",
    "libero_10_task",
    "libero_10_swap",
    "libero_10_lan",
)

LIBERO_DASHBOARD_SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <suite> <task> <seed>",
        "fields": (
            {"name": "suite", "suggestions": LIBERO_SUITE_NAMES},
            {"name": "task", "kind": "integer", "minimum": 0},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{suite} / task {task} / seed {seed}",
        "output_slug": "{suite}_t{task}_s{seed}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "task"},
        {"name": "vla", "label": "VLA"},
        {"name": "sam3", "label": "SAM3"},
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
    """Return the LIBERO env identity, prompt bundle, and runner hooks.

    Tool schemas, handlers, server lifecycle, and the MCP allowlist live on
    the LIBERO toolkit (see :func:`get_toolkit`).
    """
    return EnvSpec(
        name="libero",
        prompts=PromptBundle(
            system=system_prompt,
            user=user_prompt,
        ),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_shared_runtime=init_shared_runtime,
        init_task_runtime=init_task_runtime,
        init_runtime=_init_runtime,
        dashboard=LIBERO_DASHBOARD_SPEC,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
):
    """Return the LIBERO toolkit (common tools + LIBERO primitives)."""
    from robots.libero.toolkit import LiberoToolkit

    return LiberoToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """Register LIBERO CLI flags on the shared ``parser``.

    When ``use_dashboard`` is True, ``--suite`` / ``--task`` are made optional
    because the dashboard launcher will fill them in before ``_parse_config``
    validates. Under CLI-only, they are required — argparse errors out early
    with the usual usage message.
    """
    required = not use_dashboard
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--libero-type", default=None,
                        choices=["standard", "pro", "plus"],
                        help="LIBERO variant (auto-routed from suite suffix if not set).")
    parser.add_argument("--suite", default=None, required=required,
                        help="e.g. libero_object_task, libero_spatial_swap")
    parser.add_argument("--task", type=int, default=None, required=required)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-endpoint", default=None,
                        help="[protocol://]host:port of an existing env_server "
                             "(protocol=http|socket, defaults to http). "
                             "If unset, a local env_server is spawned.")
    parser.add_argument("--vla-endpoint", default=None,
                        help="[protocol://]host:port of an existing vla_server "
                             "(protocol=http|socket, defaults to http). "
                             "If unset, a local vla_server is spawned.")
    parser.add_argument("--sam3-endpoint", default=None,
                        help="[protocol://]host:port of an existing SAM3 server "
                             "(protocol=http|socket, defaults to http). "
                             "If unset, a local SAM3 server is spawned.")
    parser.add_argument("--cuda-device", type=int, default=None,
                        help="GPU device to expose via CUDA_VISIBLE_DEVICES.")


def _parse_config(args: argparse.Namespace) -> RunConfig:
    """Validate final ``args`` and derive per-run identifiers.

    Under ``--dashboard``, ``_add_cli_args`` left ``--suite`` / ``--task``
    optional so the dashboard could fill them; this is where we enforce
    they're set now that any overrides have been applied.
    """
    if not args.suite:
        raise ValueError("--suite is required")
    if args.task is None:
        raise ValueError("--task is required")

    recipe_tag = f"{args.suite.replace('libero_', '')}_t{args.task}_s{args.seed}"
    prompt_vars = {
        "suite": args.suite,
        "task": args.task,
        "seed": args.seed,
        "recipe_tag": recipe_tag,
    }

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{args.suite}_t{args.task}_s{args.seed}"
    output_dir = Path(output_dir)

    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars=prompt_vars,
        task_desc={"suite": args.suite, "task": args.task, "seed": args.seed},
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
    return (
        ["--cuda-device", str(args.cuda_device)]
        if args.cuda_device is not None
        else []
    )


def _external_rpc(endpoint: str, option: str) -> RpcClient:
    protocol, host, port = parse_endpoint(endpoint)
    if protocol == "socket":
        return SocketRpcClient(host, port)
    if protocol == "http":
        return HttpRpcClient(f"http://{host}:{port}")
    raise ValueError(
        f"{option} protocol must be socket or http, got {protocol!r}"
    )


def _spawn_env_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    if args.env_endpoint is not None:
        return None, _external_rpc(args.env_endpoint, "--env-endpoint")

    from rpent.utils.config import get_libero_type

    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        name="env_server",
        cmd=[
            sys.executable,
            str(get_repo_root() / "robots" / "libero" / "env_server.py"),
            "--suite", args.suite,
            "--task", str(args.task),
            "--seed", str(args.seed),
            "--max-episode-steps", str(args.max_episode_steps),
            "--transport", "http",
            "--host", host,
            "--port", str(port),
            "--parent-watch",
            *_cuda_args(args),
        ],
        env=_subprocess_env(
            LIBERO_TYPE=args.libero_type or get_libero_type(),
            MUJOCO_GL="egl",
            ROBOT_PLATFORM="LIBERO",
        ),
        log_path=str(output_dir / "env_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _spawn_vla_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    if args.vla_endpoint is not None:
        return None, _external_rpc(args.vla_endpoint, "--vla-endpoint")

    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        name="vla_server",
        cmd=[
            sys.executable,
            str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
            "--transport", "http",
            "--host", host,
            "--port", str(port),
            "--parent-watch",
            *_cuda_args(args),
        ],
        env=_subprocess_env(),
        log_path=str(output_dir / "vla_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _spawn_sam3_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    if args.sam3_endpoint is not None:
        return None, _external_rpc(args.sam3_endpoint, "--sam3-endpoint")

    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        name="sam3_server",
        cmd=[
            sys.executable,
            str(get_repo_root() / "robots" / "libero" / "sam3_server.py"),
            "--transport", "http",
            "--host", host,
            "--port", str(port),
            "--parent-watch",
            *_cuda_args(args),
        ],
        env=_subprocess_env(),
        log_path=str(output_dir / "sam3_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _env_client(args: argparse.Namespace, rpc: RpcClient):
    from robots.libero.env_client import LiberoEnvClient

    return LiberoEnvClient(
        rpc,
        expected_meta={
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
        },
    )


def init_task_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize one TaskRun-owned LIBERO environment.

    A local env server is fresh for every call. When ``--env-endpoint`` is
    supplied, the returned daemon list is empty so the external service stays
    running.

    Simulator and model dependencies stay lazy so importing
    :mod:`robots.libero` for its descriptor or toolkit remains lightweight.
    """
    owned_daemons: dict[str, ProcessDaemon] = {}
    env_daemon, env_rpc = try_spawn_server(
        owned_daemons,
        dashboard_events,
        "env",
        lambda: _spawn_env_server(args, output_dir),
    )
    env = try_wait_server(
        owned_daemons,
        dashboard_events,
        "env",
        env_rpc,
        env_daemon,
        300.0,
        post_fn=lambda: _env_client(args, env_rpc),
    )
    return list(owned_daemons.values()), {"env": env}


def init_shared_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize Session-owned VLA and SAM3 services.

    The returned list contains only locally started services. External
    endpoints are connected to but never become owned.
    """
    from robots.libero.sam3_client import Sam3Client
    from robots.libero.vla_client import LiberoVLAClient

    owned_daemons: dict[str, ProcessDaemon] = {}
    vla_daemon, vla_rpc = try_spawn_server(
        owned_daemons,
        dashboard_events,
        "vla",
        lambda: _spawn_vla_server(args, output_dir),
    )
    sam3_daemon, sam3_rpc = try_spawn_server(
        owned_daemons,
        dashboard_events,
        "sam3",
        lambda: _spawn_sam3_server(args, output_dir),
    )

    sam3_client = try_wait_server(
        owned_daemons,
        dashboard_events,
        "sam3",
        sam3_rpc,
        sam3_daemon,
        300.0,
        post_fn=lambda: Sam3Client(sam3_rpc),
    )
    model = try_wait_server(
        owned_daemons,
        dashboard_events,
        "vla",
        vla_rpc,
        vla_daemon,
        300.0,
        post_fn=lambda: LiberoVLAClient(vla_rpc),
    )

    return list(owned_daemons.values()), {
        "model": model,
        "sam3_client": sam3_client,
    }


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Spawn env + vla + SAM3 daemons and build clients for LIBERO.

    Each server can be spawned or attached-to independently: pass an
    endpoint to attach, or leave it unset to spawn a local subprocess.

    Simulator, model, and adapter dependencies are imported lazily so a bare
    ``import robots.libero`` does not load them.
    """
    from robots.libero.sam3_client import Sam3Client
    from robots.libero.vla_client import LiberoVLAClient

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
    sam3_daemon, sam3_rpc = try_spawn_server(
        owned_daemons,
        dashboard_events,
        "sam3",
        lambda: _spawn_sam3_server(args, output_dir),
    )

    clients: dict[str, Any] = {}
    for component, rpc, daemon, post_fn in (
        ("env", env_rpc, env_daemon, lambda: _env_client(args, env_rpc)),
        ("sam3", sam3_rpc, sam3_daemon, lambda: Sam3Client(sam3_rpc)),
        ("vla", vla_rpc, vla_daemon, lambda: LiberoVLAClient(vla_rpc)),
    ):
        clients[component] = try_wait_server(
            owned_daemons,
            dashboard_events,
            component,
            rpc,
            daemon,
            300.0,
            post_fn=post_fn,
        )

    return list(owned_daemons.values()), {
        "env": clients["env"],
        "model": clients["vla"],
        "sam3_client": clients["sam3"],
    }
