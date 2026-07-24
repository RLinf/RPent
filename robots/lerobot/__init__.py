"""LeRobot SO101 environment extension.

Entry point for the env registry: :func:`get_env_spec` and
:func:`get_toolkit` are discovered by
:func:`rpent.envs.base._resolve_env` via
``importlib.import_module("robots.lerobot")`` — dropping this
package on disk is the entire registration step.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.lerobot.prompt import (
    system_prompt,
    user_prompt,
)
from rpent.envs.env_spec import EnvSpec, RunConfig
from rpent.envs.prompt_bundle import PromptBundle
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon
    from rpent.utils.rpc import RpcClient

logger = get_logger("lerobot")


def get_env_spec() -> EnvSpec:
    """Return the SO101 env identity, prompt bundle, and runner hooks.

    Tool schemas, handlers, and the MCP allowlist live on the toolkit (see
    :func:`get_toolkit`). The three runner hooks (:func:`_add_cli_args` /
    :func:`_parse_config` / :func:`_init_runtime`) keep ``rpent/cli/main.py``
    env-agnostic, mirroring :mod:`robots.libero`.
    """
    return EnvSpec(
        name="lerobot",
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
    """Return the SO101 toolkit (common tools + SO101 primitives).

    ``primitives_kwargs`` is assembled by :func:`_init_runtime` and carries the
    env RPC stub (``{"env": LerobotEnvClient(...)}``, optionally plus
    ``"model"``).
    """
    from robots.lerobot.toolkit import LerobotToolkit

    return LerobotToolkit(
        video_path=video_path,
        dashboard=dashboard,
        **primitives_kwargs,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """Register SO101 CLI flags on the shared ``parser``.

    ``use_dashboard`` is unused: the SO101 is a real robot with no
    suite/task/seed, so there is nothing for the (libero-shaped) dashboard
    launcher to fill in.
    """
    del use_dashboard
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument(
        "--env-endpoint", default=None,
        help="[protocol://]host:port of an existing lerobot env_server "
             "(protocol=http|socket, defaults to http). If unset, a local "
             "env_server is spawned (requires the 'lerobot' extra in this venv).",
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    """Derive per-run identifiers for a SO101 run.

    Real robots have no suite/task/seed, so the run is identified by the env
    name. The dashboard is currently libero-shaped, so it is not wired here.
    """
    if getattr(args, "dashboard", False):
        logger.warning(
            "--dashboard is only supported for the libero env; "
            "continuing without the live dashboard."
        )

    recipe_tag = "lerobot"
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_lerobot"
    output_dir = Path(output_dir)

    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars={"env_name": "lerobot", "recipe_tag": recipe_tag},
        dashboard_state=None,
        task_desc={"env": "lerobot"},
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
        raise ValueError(
            f"--env-endpoint must be [protocol://]host:port, got {endpoint!r}"
        )
    return protocol, host, int(port)


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Spawn (or attach to) the SO101 env_server; build primitives_kwargs.

    The SO101 driver needs the ``lerobot`` stack. It is spawned with this
    interpreter (works when the venv has the ``lerobot`` extra); otherwise run
    the driver in its own ``lerobot`` conda env and attach via
    ``--env-endpoint``. No VLA server — the SO101 primitives are scripted.

    Heavy deps are imported lazily so a bare ``import robots.lerobot`` (for
    ``get_env_spec`` / ``get_toolkit``) doesn't drag them in.
    """
    from robots.lerobot.env_client import LerobotEnvClient
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import wait_for_ready
    from rpent.utils.socket_rpc import SocketRpcClient

    daemons: list[ProcessDaemon] = []
    if args.env_endpoint is None:
        host, port = "127.0.0.1", pick_free_port()
        env_daemon = ProcessDaemon(
            name="env_server",
            cmd=[
                sys.executable,
                str(get_repo_root() / "robots" / "lerobot" / "env_server.py"),
                "--output-dir", str(output_dir),
                "--max-episode-steps", str(args.max_episode_steps),
                "--transport", "http",
                "--host", host,
                "--port", str(port),
            ],
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

    primitives_kwargs = {"env": LerobotEnvClient(env_client)}
    return daemons, primitives_kwargs
