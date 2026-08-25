"""LIBERO robot extension."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.libero.prompt_bundle import system_prompt, user_prompt
from robots.libero.spec import LIBERO_DASHBOARD_SPEC
from rpent.dashboard.events import DashboardEventSink, RuntimeStatusEvent
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.prompt_bundle import PromptBundle
from rpent.utils.config import get_memory_dir, get_repo_root

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon
    from rpent.utils.rpc import RpcClient


def get_robot_spec() -> RobotSpec:
    """Return the LIBERO robot identity, prompt bundle, and runner hooks.

    Tool schemas, handlers, server lifecycle, and the MCP allowlist live on
    the LIBERO toolkit (see :func:`get_toolkit`).
    """
    return RobotSpec(
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
        finalize_run=_finalize_run,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    mode: str = "evaluation",
    attempts_per_session: int = 0,
    state_output_dir: Path | str | None = None,
):
    """Return the LIBERO toolkit (common tools + LIBERO primitives)."""
    from robots.libero.toolkit import LiberoToolkit

    return LiberoToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
        mode=mode,
        attempts_per_session=attempts_per_session,
        state_output_dir=state_output_dir,
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
    parser.add_argument("--auto-merge-memory",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Merge exploration output into layered memory (default: enabled).")
    parser.add_argument("--explore-attempts-per-session", type=int, default=5,
                        help="Attempts per exploration session (default: 5; 0 disables limit).")
    parser.add_argument("--explore-sessions", type=int, default=3,
                        help="Independent planner sessions per exploration run (default: 3).")
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
    explore = bool(getattr(args, "explore", False))
    requested_profile = getattr(args, "memory_profile", None)
    if explore and requested_profile == "hf":
        raise ValueError("--explore cannot be used with --memory-profile hf")
    if explore and args.explore_sessions <= 0:
        raise ValueError("--explore-sessions must be greater than 0")
    memory_profile = requested_profile or ("local" if explore else "hf")
    if memory_profile == "hf" and args.memory_dir is not None:
        raise ValueError("--memory-dir requires --memory-profile local or --explore")
    args.memory_profile = memory_profile
    memory_dir = (
        Path(args.memory_dir).expanduser().resolve()
        if args.memory_dir
        else get_memory_dir("libero")
    )
    local_eval = not explore and memory_profile == "local"
    if local_eval and not (memory_dir / "MEMORY.md").is_file():
        raise ValueError(
            f"local memory corpus not found at {memory_dir}; "
            "run exploration first or use --memory-profile hf"
        )
    prompt_vars = {
        "suite": args.suite,
        "task": args.task,
        "seed": args.seed,
        "recipe_tag": recipe_tag,
        "mode": "explore" if explore else "eval",
        "memory_profile": memory_profile,
        "memory_dir": str(memory_dir),
        "reference_tag": f"{args.suite.replace('libero_', '')}_t{args.task}_s0",
        # Per-cell inbox: parallel explore runs must not append to a shared file.
        "memory_inbox": str(memory_dir / "_inbox" / recipe_tag),
        "session_number": 1,
        "session_max": max(1, args.explore_sessions) if explore else 1,
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


def _finalize_run(args: argparse.Namespace, config: RunConfig) -> dict[str, Any] | None:
    """Publish completed exploration artifacts into the local layered corpus."""
    if not args.explore or not args.auto_merge_memory:
        return None
    from rpent.memory import merge_cell

    return merge_cell(
        memory_dir=config.prompt_vars["memory_dir"],
        cell_tag=config.recipe_tag,
        output_dir=config.output_dir,
    )


def _subprocess_env(**extra: str) -> dict[str, str]:
    """Build the env dict for a subprocess: inherit from parent, layer extras on top.

    CUDA device selection is passed via ``--cuda-device`` on the server command
    line — the server itself handles ``CUDA_VISIBLE_DEVICES`` and EGL alignment.
    """
    env = os.environ.copy()
    env.update(extra)
    return env


def init_task_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize one TaskRun-owned LIBERO environment.

    A local env server is fresh for every call. When ``--env-endpoint`` is
    supplied, the returned daemon list is empty so the external service stays
    running.

    Heavy runtime dependencies stay lazy so importing :mod:`robots.libero`
    for its descriptor or toolkit does not load RPC/model packages.
    """
    from robots.libero.env_client import LiberoEnvClient
    from rpent.utils.config import get_libero_type
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import parse_endpoint, wait_for_ready
    from rpent.utils.socket_rpc import SocketRpcClient

    owned_daemons: list[ProcessDaemon] = []
    libero_type = args.libero_type or get_libero_type()
    cuda_args = ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []

    dashboard_events.emit(RuntimeStatusEvent("env", "starting"))
    try:
        env_daemon: ProcessDaemon | None = None
        if args.env_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            env_daemon = ProcessDaemon(
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
                    *cuda_args,
                ],
                env=_subprocess_env(
                    LIBERO_TYPE=libero_type,
                    MUJOCO_GL="egl",
                    ROBOT_PLATFORM="LIBERO",
                ),
                log_path=str(Path(output_dir) / "env_server.log"),
            )
            env_daemon.start()
            owned_daemons.append(env_daemon)
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
        wait_for_ready(env_rpc, daemon=env_daemon)
        env = LiberoEnvClient(
            env_rpc,
            expected_meta={
                "suite": args.suite,
                "task": args.task,
                "seed": args.seed,
                "max_episode_steps": args.max_episode_steps,
            },
        )
    except Exception as exc:
        _stop_owned_daemons(owned_daemons)
        dashboard_events.emit(RuntimeStatusEvent("env", "failed", error=exc))
        raise
    dashboard_events.emit(RuntimeStatusEvent("env", "ready"))
    return owned_daemons, {"env": env}


def init_shared_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize Session-owned VLA and SAM3 services.

    The returned list contains only locally started services. External
    endpoints are connected to but never become owned.
    """
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import parse_endpoint, wait_for_ready
    from rpent.utils.sam3_client import Sam3Client
    from rpent.utils.socket_rpc import SocketRpcClient
    from rpent.utils.vla_client import VLAClient

    owned_daemons: list[ProcessDaemon] = []
    cuda_args = (
        ["--cuda-device", str(args.cuda_device)]
        if args.cuda_device is not None
        else []
    )

    # --- vla_server --------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("vla", "starting"))
    try:
        vla_daemon: ProcessDaemon | None = None
        if args.vla_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            vla_daemon = ProcessDaemon(
                name="vla_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
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
            owned_daemons.append(vla_daemon)
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
        _stop_owned_daemons(owned_daemons)
        dashboard_events.emit(RuntimeStatusEvent("vla", "failed", error=exc))
        raise

    # --- sam3_server -------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("sam3", "starting"))
    try:
        sam3_daemon: ProcessDaemon | None = None
        if args.sam3_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            sam3_daemon = ProcessDaemon(
                name="sam3_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "sam3_server.py"),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(),
                log_path=str(Path(output_dir) / "sam3_server.log"),
            )
            sam3_daemon.start()
            owned_daemons.append(sam3_daemon)
            sam3_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.sam3_endpoint)
            if protocol == "socket":
                sam3_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                sam3_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--sam3-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
        _stop_owned_daemons(owned_daemons)
        dashboard_events.emit(RuntimeStatusEvent("sam3", "failed", error=exc))
        raise

    # Start both local services before waiting so heavyweight initialization
    # continues concurrently, matching the one-shot runtime behavior.
    for component, client, daemon in (
        ("sam3", sam3_rpc, sam3_daemon),
        ("vla", vla_rpc, vla_daemon),
    ):
        try:
            wait_for_ready(client, daemon=daemon)
        except Exception as exc:
            _stop_owned_daemons(owned_daemons)
            dashboard_events.emit(RuntimeStatusEvent(component, "failed", error=exc))
            raise
        dashboard_events.emit(RuntimeStatusEvent(component, "ready"))

    model = VLAClient(vla_rpc)
    sam3_client = Sam3Client(sam3_rpc)

    return owned_daemons, {
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

    Heavy deps (rpc / vla / daemon / env_client) are imported lazily so
    that a bare ``import robots.libero`` (for ``get_robot_spec`` /
    ``get_toolkit``) doesn't drag them in.
    """
    from robots.libero.env_client import LiberoEnvClient
    from rpent.utils.config import get_libero_type
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import parse_endpoint, wait_for_ready
    from rpent.utils.sam3_client import Sam3Client
    from rpent.utils.socket_rpc import SocketRpcClient
    from rpent.utils.vla_client import VLAClient

    daemons: list[ProcessDaemon] = []
    libero_type = args.libero_type or get_libero_type()
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
                    str(get_repo_root() / "robots" / "libero" / "env_server.py"),
                    "--suite", args.suite,
                    "--task", str(args.task),
                    "--seed", str(args.seed),
                    "--max-episode-steps", str(args.max_episode_steps),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(
                    LIBERO_TYPE=libero_type,
                    MUJOCO_GL="egl",
                    ROBOT_PLATFORM="LIBERO",
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
            host, port = "127.0.0.1", pick_free_port()
            vla_daemon = ProcessDaemon(
                name="vla_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
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

    # --- sam3_server -------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("sam3", "starting"))
    try:
        sam3_daemon: ProcessDaemon | None = None
        if args.sam3_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            sam3_daemon = ProcessDaemon(
                name="sam3_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "sam3_server.py"),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(),
                log_path=str(Path(output_dir) / "sam3_server.log"),
            )
            sam3_daemon.start()
            daemons.append(sam3_daemon)
            sam3_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.sam3_endpoint)
            if protocol == "socket":
                sam3_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                sam3_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--sam3-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
        dashboard_events.emit(RuntimeStatusEvent("sam3", "failed", error=exc))
        raise

    # All local daemons are running now, so they initialize concurrently while
    # readiness is checked in a deterministic order.
    for component, client, daemon in (
        ("env", env_rpc, env_daemon),
        ("sam3", sam3_rpc, sam3_daemon),
        ("vla", vla_rpc, vla_daemon),
    ):
        try:
            wait_for_ready(client, daemon=daemon)
        except Exception as exc:
            for started_daemon in reversed(daemons):
                started_daemon.stop()
            dashboard_events.emit(RuntimeStatusEvent(component, "failed", error=exc))
            raise
        dashboard_events.emit(RuntimeStatusEvent(component, "ready"))

    primitives_kwargs = {
        "env": LiberoEnvClient(
            env_rpc,
            expected_meta={
                "suite": args.suite,
                "task": args.task,
                "seed": args.seed,
                "max_episode_steps": args.max_episode_steps,
            },
        ),
        "model": VLAClient(vla_rpc),
        "sam3_client": Sam3Client(sam3_rpc),
    }
    return daemons, primitives_kwargs


def _stop_owned_daemons(daemons: list[ProcessDaemon]) -> None:
    """Stop owned daemons in reverse order without masking startup errors."""
    for daemon in reversed(daemons):
        try:
            daemon.stop()
        except Exception:
            pass
