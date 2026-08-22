"""LIBERO environment extension."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.libero.prompt_bundle import system_prompt, user_prompt
from robots.libero.spec import LIBERO_DASHBOARD_SPEC
from rpent.dashboard.events import DashboardEventSink
from rpent.envs.env_spec import EnvSpec, RunConfig
from rpent.envs.prompt_bundle import PromptBundle
from rpent.envs.runtime import try_spawn_server, try_wait_server
from rpent.utils.config import get_repo_root
from rpent.utils.rpc import make_rpc_client

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon
    from rpent.utils.rpc import RpcClient


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


def _spawn_env_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Spawn or attach to the LIBERO environment server."""
    from rpent.utils.config import get_libero_type
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient

    libero_type = args.libero_type or get_libero_type()
    cuda_args = (
        ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []
    )
    if args.env_endpoint is not None:
        return None, make_rpc_client(args.env_endpoint)

    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        name="env_server",
        cmd=[
            sys.executable,
            str(get_repo_root() / "robots" / "libero" / "env_server.py"),
            "--suite",
            args.suite,
            "--task",
            str(args.task),
            "--seed",
            str(args.seed),
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--transport",
            "http",
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
            *cuda_args,
        ],
        env_overrides={
            "LIBERO_TYPE": libero_type,
            "MUJOCO_GL": "egl",
            "ROBOT_PLATFORM": "LIBERO",
        },
        log_path=str(output_dir / "env_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _spawn_vla_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Spawn or attach to the LIBERO VLA server."""
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient

    if args.vla_endpoint is not None:
        return None, make_rpc_client(args.vla_endpoint)

    host, port = "127.0.0.1", pick_free_port()
    cuda_args = (
        ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []
    )
    daemon = ProcessDaemon(
        name="vla_server",
        cmd=[
            sys.executable,
            str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
            "--transport",
            "http",
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
            *cuda_args,
        ],
        log_path=str(output_dir / "vla_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _spawn_sam3_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Spawn or attach to the LIBERO SAM3 server."""
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient

    if args.sam3_endpoint is not None:
        return None, make_rpc_client(args.sam3_endpoint)

    host, port = "127.0.0.1", pick_free_port()
    cuda_args = (
        ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []
    )
    daemon = ProcessDaemon(
        name="sam3_server",
        cmd=[
            sys.executable,
            str(get_repo_root() / "robots" / "libero" / "sam3_server.py"),
            "--transport",
            "http",
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
            *cuda_args,
        ],
        log_path=str(output_dir / "sam3_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize every LIBERO component, or only ``components`` when given."""
    from robots.libero.env_client import LiberoEnvClient
    from rpent.utils.sam3_client import Sam3Client
    from rpent.utils.vla_client import VLAClient

    starters = {
        "env": lambda: _spawn_env_server(args, output_dir),
        "vla": lambda: _spawn_vla_server(args, output_dir),
        "sam3": lambda: _spawn_sam3_server(args, output_dir),
    }
    connectors = {
        "env": lambda rpc: {
            "env": LiberoEnvClient(
                rpc,
                expected_meta={
                    "suite": args.suite,
                    "task": args.task,
                    "seed": args.seed,
                    "max_episode_steps": args.max_episode_steps,
                },
            )
        },
        "vla": lambda rpc: {"model": VLAClient(rpc)},
        "sam3": lambda rpc: {"sam3_client": Sam3Client(rpc)},
    }
    selected = set(starters) if components is None else components
    unknown = selected.difference(starters)
    if unknown:
        raise ValueError(f"unknown LIBERO runtime components: {sorted(unknown)}")

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
    wait_order = ("env", "sam3", "vla")
    for component in (name for name in wait_order if name in pending):
        daemon, rpc = pending[component]
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

    return list(owned_daemons.values()), primitives_kwargs
