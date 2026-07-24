"""LIBERO environment extension."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rpent.envs.prompt_bundle import PromptBundle
from rpent.envs.env_spec import EnvSpec, RunConfig
from rpent.utils.config import get_repo_root
from robots.libero.prompt_bundle import (
    system_prompt,
    user_prompt,
)

if TYPE_CHECKING:
    from rpent.dashboard.state import State
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
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    video_path: str | None = None,
    dashboard: Any = None,
):
    """Return the LIBERO toolkit (common tools + LIBERO primitives)."""
    from robots.libero.toolkit import LiberoToolkit

    return LiberoToolkit(
        primitives_kwargs=primitives_kwargs,
        video_path=video_path,
        dashboard=dashboard,
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
    parser.add_argument("--cuda-device", default=None,
                        help="GPU device(s) to expose via CUDA_VISIBLE_DEVICES.")


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

    dashboard_state = None
    if getattr(args, "dashboard", False):
        from rpent.dashboard.state import State
        dashboard_state = State(
            run_id=f"{args.suite}/{output_dir.name}",
            name=recipe_tag,
            suite=args.suite,
            task=args.task,
            seed=args.seed,
            output_dir=str(output_dir),
            video_path=str(output_dir / "episode.mp4"),
        )
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars=prompt_vars,
        dashboard_state=dashboard_state,
        task_desc={"suite": args.suite, "task": args.task, "seed": args.seed},
    )


def _parse_endpoint(endpoint: str) -> tuple[str, str, int]:
    """Parse ``[protocol://]host:port`` into ``(protocol, host, port)``.

    Protocol defaults to ``http`` when the prefix is omitted.
    """
    if "://" in endpoint:
        protocol, _, rest = endpoint.partition("://")
    else:
        protocol, rest = "http", endpoint
    host, _, port = rest.partition(":")
    if not host or not port:
        raise ValueError(f"endpoint must be [protocol://]host:port, got {endpoint!r}")
    return protocol, host, int(port)


def _subprocess_env(cuda_device: str | None, **extra: str) -> dict[str, str]:
    """Build the env dict for a subprocess: inherit from parent, apply
    ``--cuda-device`` uniformly, layer optional extras on top.

    If ``cuda_device`` is None, ``CUDA_VISIBLE_DEVICES`` is left as inherited
    (respecting whatever the parent shell set). If given, it wins.
    """
    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env.update(extra)
    return env


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Spawn env + vla daemons and build primitives_kwargs for LIBERO.

    Each server can be spawned or attached-to independently: pass an
    endpoint to attach, or leave it unset to spawn a local subprocess.

    Heavy deps (rpc / vla / daemon / env_client) are imported lazily so
    that a bare ``import robots.libero`` (for ``get_env_spec`` /
    ``get_toolkit``) doesn't drag them in.
    """
    from rpent.utils.config import get_libero_type
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import wait_for_ready
    from rpent.utils.socket_rpc import SocketRpcClient
    from rpent.utils.vla_client import VLAClient
    from robots.libero.env_client import LiberoEnvClient

    daemons: list[ProcessDaemon] = []
    libero_type = args.libero_type or get_libero_type()

    # --- env_server --------------------------------------------------------
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
            ],
            env=_subprocess_env(
                args.cuda_device,
                LIBERO_TYPE=libero_type,
                MUJOCO_GL="egl",
                ROBOT_PLATFORM="LIBERO",
            ),
            log_path=str(Path(output_dir) / "env_server.log"),
        )
        env_daemon.start()
        daemons.append(env_daemon)
        env_client: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        wait_for_ready(env_client)
    else:
        protocol, host, port = _parse_endpoint(args.env_endpoint)
        if protocol == "socket":
            env_client = SocketRpcClient(host, port)
        elif protocol == "http":
            env_client = HttpRpcClient(f"http://{host}:{port}")
        else:
            raise ValueError(
                f"--env-endpoint protocol must be socket or http, got {protocol!r}"
            )

    # --- vla_server --------------------------------------------------------
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
            ],
            env=_subprocess_env(args.cuda_device),
            log_path=str(Path(output_dir) / "vla_server.log"),
        )
        vla_daemon.start()
        daemons.append(vla_daemon)
        vla_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        wait_for_ready(vla_rpc)
    else:
        protocol, host, port = _parse_endpoint(args.vla_endpoint)
        if protocol == "socket":
            vla_rpc = SocketRpcClient(host, port)
        elif protocol == "http":
            vla_rpc = HttpRpcClient(f"http://{host}:{port}")
        else:
            raise ValueError(
                f"--vla-endpoint protocol must be socket or http, got {protocol!r}"
            )

    primitives_kwargs = {
        "env": LiberoEnvClient(
            env_client,
            expected_meta={
                "suite": args.suite,
                "task": args.task,
                "seed": args.seed,
                "max_episode_steps": args.max_episode_steps,
            },
        ),
        "model": VLAClient(vla_rpc),
    }
    return daemons, primitives_kwargs
