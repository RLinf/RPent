"""Single-Franka real-robot environment extension."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.franka.prompt_bundle import system_prompt, user_prompt
from robots.franka.spec import FRANKA_DASHBOARD_SPEC
from robots.franka.tasks import FRANKA_TASKS, get_franka_task
from rpent.dashboard.events import DashboardEventSink, RuntimeStatusEvent
from rpent.envs.env_spec import EnvSpec, RunConfig
from rpent.envs.prompt_bundle import PromptBundle
from rpent.utils.config import get_repo_root, get_rlinf_repo_path

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon


def get_env_spec() -> EnvSpec:
    """Return the Franka identity, prompts, runtime hooks, and dashboard spec."""
    return EnvSpec(
        name="franka",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_shared_runtime=init_shared_runtime,
        init_task_runtime=init_task_runtime,
        init_runtime=_init_runtime,
        dashboard=FRANKA_DASHBOARD_SPEC,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
):
    """Return the Franka toolkit."""
    from robots.franka.toolkit import FrankaToolkit

    return FrankaToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    parser.add_argument(
        "--task-id",
        type=int,
        default=None if use_dashboard else 0,
        choices=sorted(FRANKA_TASKS),
    )
    parser.add_argument("--env-endpoint", default=None)
    parser.add_argument("--vla-endpoint", default=None)
    parser.add_argument("--rlinf-python", default=os.environ.get("RPENT_RLINF_PYTHON"))
    parser.add_argument("--rlinf-config-name", default="realworld_physical_agent_eval")
    parser.add_argument("--rlinf-override", action="append", default=[])
    parser.add_argument("--controller-config", default=None)


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if args.task_id is None:
        raise ValueError("--task-id is required")
    task = get_franka_task(args.task_id)
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    output_dir = Path(
        args.output_dir
        or get_repo_root() / "logs" / f"{timestamp}_franka_t{args.task_id}"
    )
    constraints = "\n".join(
        f"{index}. {constraint}" for index, constraint in enumerate(task.constraints, 1)
    )
    return RunConfig(
        recipe_tag=f"franka_t{args.task_id}",
        output_dir=output_dir,
        prompt_vars={
            "task_id": args.task_id,
            "task_name": task.name,
            "instruction": task.instruction,
            "success_criteria": task.success_criteria,
            "constraints": constraints,
            "output_dir": str(output_dir),
        },
        task_desc={"task_id": args.task_id, "task_name": task.name},
    )


def _rpc_client(endpoint: str):
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import parse_endpoint
    from rpent.utils.socket_rpc import SocketRpcClient

    protocol, host, port = parse_endpoint(endpoint)
    if protocol == "http":
        return HttpRpcClient(f"http://{host}:{port}")
    if protocol == "socket":
        return SocketRpcClient(host, port)
    raise ValueError(f"unsupported RPC protocol: {protocol!r}")


def _rlinf_python(args: argparse.Namespace) -> str:
    if args.rlinf_python:
        return str(Path(args.rlinf_python).expanduser())
    rlinf_root = get_rlinf_repo_path() or (get_repo_root().parent / "rlinf")
    candidate = rlinf_root / "requirements" / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    raise ValueError(
        "spawning the Franka server requires --rlinf-python or RPENT_RLINF_PYTHON"
    )


def init_shared_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Connect to an optional externally managed real-Franka VLA service."""
    del output_dir
    if args.vla_endpoint is None:
        dashboard_events.emit(RuntimeStatusEvent("vla", "ready"))
        return [], {"model": None}
    from rpent.utils.rpc import wait_for_ready
    from rpent.utils.vla_client import VLAClient

    dashboard_events.emit(RuntimeStatusEvent("vla", "starting"))
    client = _rpc_client(args.vla_endpoint)
    wait_for_ready(client)
    dashboard_events.emit(RuntimeStatusEvent("vla", "ready"))
    return [], {"model": VLAClient(client)}


def init_task_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Spawn or attach to the RLinf-backed Franka environment server."""
    from robots.franka.env_client import FrankaEnvClient
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.rpc import wait_for_ready

    task = get_franka_task(args.task_id)
    owned_daemons: list[ProcessDaemon] = []
    dashboard_events.emit(RuntimeStatusEvent("env", "starting"))
    try:
        daemon: ProcessDaemon | None = None
        if args.env_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            rlinf_root = get_rlinf_repo_path() or (get_repo_root().parent / "rlinf")
            command = [
                _rlinf_python(args),
                str(get_repo_root() / "robots" / "franka" / "env_server.py"),
                "--transport",
                "http",
                "--host",
                host,
                "--port",
                str(port),
                "--config-name",
                args.rlinf_config_name,
                "--output-dir",
                str(output_dir),
                "--parent-watch",
            ]
            for override in args.rlinf_override:
                command.extend(["--override", override])
            if args.controller_config:
                command.extend(["--controller-config", args.controller_config])
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [str(get_repo_root()), str(rlinf_root), env.get("PYTHONPATH", "")]
            )
            daemon = ProcessDaemon(
                name="franka_env_server",
                cmd=command,
                env=env,
                log_path=str(output_dir / "franka_env_server.log"),
            )
            daemon.start()
            owned_daemons.append(daemon)
            endpoint = f"http://{host}:{port}"
        else:
            endpoint = args.env_endpoint
        client = _rpc_client(endpoint)
        wait_for_ready(client, daemon=daemon)
        franka_env = FrankaEnvClient(client)
    except Exception as exc:
        for started in reversed(owned_daemons):
            started.stop()
        dashboard_events.emit(RuntimeStatusEvent("env", "failed", error=exc))
        raise
    dashboard_events.emit(RuntimeStatusEvent("env", "ready"))
    return owned_daemons, {
        "env": franka_env,
        "task_description": task.instruction,
    }


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    shared_daemons, shared_kwargs = init_shared_runtime(
        args, output_dir, dashboard_events
    )
    task_daemons, task_kwargs = init_task_runtime(args, output_dir, dashboard_events)
    return shared_daemons + task_daemons, {**shared_kwargs, **task_kwargs}
