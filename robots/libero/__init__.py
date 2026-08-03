"""LIBERO environment extension."""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.libero.prompt_bundle import (
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


@dataclass(frozen=True, slots=True)
class LiberoSharedRuntime:
    """Session-owned VLA/SAM3 clients and their local daemons.

    ``owned_daemons`` contains only subprocesses started by this process.
    Clients connected through external endpoints therefore remain usable by
    their external owner after :meth:`close`.
    """

    model: Any
    sam3_client: Any
    owned_daemons: tuple["ProcessDaemon", ...] = ()

    def close(self) -> None:
        """Stop every locally owned shared-service daemon, best effort."""
        _stop_owned_daemons(self.owned_daemons)


@dataclass(frozen=True, slots=True)
class LiberoTaskRuntime:
    """TaskRun-owned LIBERO env client and optional local env daemon."""

    env: Any
    owned_daemons: tuple["ProcessDaemon", ...] = ()

    def close(self) -> None:
        """Stop the locally owned env daemon, if any."""
        _stop_owned_daemons(self.owned_daemons)


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
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    video_path: str | None = None,
    save_action_videos: bool = False,
):
    """Return the LIBERO toolkit (common tools + LIBERO primitives)."""
    from robots.libero.toolkit import LiberoToolkit

    return LiberoToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
        video_path=video_path,
        save_action_videos=save_action_videos,
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


def init_task_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> LiberoTaskRuntime:
    """Initialize one TaskRun-owned LIBERO environment.

    A local env server is fresh for every call. When ``--env-endpoint`` is
    supplied, the returned handle owns no daemon and closing it leaves the
    external service running.

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
            env_rpc = _external_rpc_client(
                args.env_endpoint,
                option="--env-endpoint",
                parse_endpoint=parse_endpoint,
                http_client_factory=HttpRpcClient,
                socket_client_factory=SocketRpcClient,
            )
            wait_for_ready(env_rpc)
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
        _stop_owned_daemons(owned_daemons, suppress_errors=True)
        dashboard_events.emit(RuntimeStatusEvent("env", "failed", error=exc))
        raise
    dashboard_events.emit(RuntimeStatusEvent("env", "ready"))
    return LiberoTaskRuntime(
        env=env,
        owned_daemons=tuple(owned_daemons),
    )


def init_shared_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> LiberoSharedRuntime:
    """Initialize Session-owned VLA and SAM3 services.

    Local services are started once per call and recorded in the returned
    handle. External endpoints are connected to but never become owned.
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
            vla_rpc = _external_rpc_client(
                args.vla_endpoint,
                option="--vla-endpoint",
                parse_endpoint=parse_endpoint,
                http_client_factory=HttpRpcClient,
                socket_client_factory=SocketRpcClient,
            )
            wait_for_ready(vla_rpc)
    except Exception as exc:
        _stop_owned_daemons(owned_daemons, suppress_errors=True)
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
            sam3_rpc = _external_rpc_client(
                args.sam3_endpoint,
                option="--sam3-endpoint",
                parse_endpoint=parse_endpoint,
                http_client_factory=HttpRpcClient,
                socket_client_factory=SocketRpcClient,
            )
            wait_for_ready(sam3_rpc)
        model = VLAClient(vla_rpc)
        sam3_client = Sam3Client(sam3_rpc)
    except Exception as exc:
        _stop_owned_daemons(owned_daemons, suppress_errors=True)
        dashboard_events.emit(RuntimeStatusEvent("sam3", "failed", error=exc))
        raise

    return LiberoSharedRuntime(
        model=model,
        sam3_client=sam3_client,
        owned_daemons=tuple(owned_daemons),
    )


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Compose task and shared runtimes for the existing one-shot runner.

    The env is initialized before VLA and SAM3, preserving the established
    one-shot startup order and return value. If a later shared service fails,
    the already-started task env is stopped before the exception escapes.
    """
    task_runtime: LiberoTaskRuntime | None = None
    try:
        task_runtime = init_task_runtime(args, output_dir, dashboard_events)
        shared_runtime = init_shared_runtime(args, output_dir, dashboard_events)
    except Exception:
        if task_runtime is not None:
            # Preserve the shared-runtime startup error while still cleaning
            # the env. Cleanup errors must not replace the actionable cause.
            _stop_owned_daemons(
                task_runtime.owned_daemons,
                suppress_errors=True,
            )
        raise

    daemons = [
        *task_runtime.owned_daemons,
        *shared_runtime.owned_daemons,
    ]
    primitives_kwargs = {
        "env": task_runtime.env,
        "model": shared_runtime.model,
        "sam3_client": shared_runtime.sam3_client,
    }
    return daemons, primitives_kwargs


def _external_rpc_client(
    endpoint: str,
    *,
    option: str,
    parse_endpoint: Callable[[str], tuple[str, str, int]],
    http_client_factory: Callable[[str], RpcClient],
    socket_client_factory: Callable[[str, int], RpcClient],
) -> RpcClient:
    """Build a non-owned RPC transport for one configured endpoint."""
    protocol, host, port = parse_endpoint(endpoint)
    if protocol == "socket":
        return socket_client_factory(host, port)
    if protocol == "http":
        return http_client_factory(f"http://{host}:{port}")
    raise ValueError(f"{option} protocol must be socket or http, got {protocol!r}")


def _stop_owned_daemons(
    daemons: Iterable[ProcessDaemon],
    *,
    suppress_errors: bool = False,
) -> None:
    """Stop owned daemons in reverse order while attempting every stop."""
    errors: list[Exception] = []
    for daemon in reversed(tuple(daemons)):
        try:
            daemon.stop()
        except Exception as exc:
            errors.append(exc)
    if errors and not suppress_errors:
        raise RuntimeError(
            f"failed to stop {len(errors)} LIBERO runtime daemon(s)"
        ) from errors[0]
