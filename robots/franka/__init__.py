"""Single-Franka real-robot environment extension."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.franka.config import DEFAULT_CALIBRATION_PATH
from robots.franka.prompt_bundle import system_prompt, user_prompt
from robots.franka.spec import FRANKA_DASHBOARD_SPEC
from robots.franka.tasks import FRANKA_TASKS, get_franka_task
from rpent.dashboard.events import DashboardEventSink, RuntimeStatusEvent
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import try_spawn_server, try_wait_server
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.http_rpc import HttpRpcClient
from rpent.utils.rpc import make_rpc_client

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient


def get_robot_spec() -> RobotSpec:
    """Return the Franka identity, prompts, runtime hooks, and dashboard spec."""
    return RobotSpec(
        name="franka",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
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
    parser.add_argument("--robot-config", default=None)
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--camera-serial-wrist", default=None)
    parser.add_argument("--camera-serial-external", default=None)
    parser.add_argument("--gripper-connection", default=None)
    parser.add_argument(
        "--calibration-path",
        default=str(DEFAULT_CALIBRATION_PATH),
        help="Path to hand_eye_calibration.json (defaults to easy_handeye's "
        "~/.ros/easy_handeye directory).",
    )


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


def _env_server_command(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
) -> list[str]:
    task = get_franka_task(args.task_id)
    command = [
        sys.executable,
        str(get_repo_root() / "robots" / "franka" / "env_server.py"),
        "--transport",
        "http",
        "--host",
        host,
        "--port",
        str(port),
        "--task-description",
        task.instruction,
        "--parent-watch",
    ]
    if args.robot_config:
        command.extend(["--robot-config", args.robot_config])
    for flag, value in (
        ("--robot-ip", args.robot_ip),
        ("--camera-serial-wrist", args.camera_serial_wrist),
        ("--camera-serial-external", args.camera_serial_external),
        ("--gripper-connection", args.gripper_connection),
    ):
        if value is not None:
            command.extend([flag, value])
    return command


def _spawn_env_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Spawn (or attach to) the RLinf-backed Franka environment server.

    Returns ``(daemon, rpc)`` — the daemon is ``None`` when an external
    endpoint was attached (the caller must not own it).
    """
    if args.env_endpoint is not None:
        return None, make_rpc_client(args.env_endpoint)
    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        name="franka_env_server",
        cmd=_env_server_command(args, host=host, port=port),
        log_path=str(output_dir / "franka_env_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _spawn_vla_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Attach to an externally managed real-Franka VLA service."""
    del output_dir
    return None, make_rpc_client(args.vla_endpoint)


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize every Franka component, or only ``components`` when given.

    The Franka VLA is attach-only; it is skipped unless ``--vla-endpoint`` is
    provided.
    """
    from robots.franka.env_client import FrankaEnvClient
    from rpent.utils.vla_client import VLAClient

    available = {"env", "vla"}
    selected = available if components is None else components
    unknown = selected.difference(available)
    if unknown:
        raise ValueError(f"unknown Franka runtime components: {sorted(unknown)}")

    needs_vla = args.vla_endpoint is not None

    starters = {
        "env": lambda: _spawn_env_server(args, output_dir),
        "vla": lambda: _spawn_vla_server(args, output_dir),
    }
    connectors = {
        "env": lambda rpc: {
            "env": FrankaEnvClient(rpc),
            "task_description": get_franka_task(args.task_id).instruction,
        },
        "vla": lambda rpc: {"model": VLAClient(rpc)},
    }

    owned_daemons: dict[str, ProcessDaemon] = {}
    pending: dict[str, tuple[ProcessDaemon | None, RpcClient]] = {}
    for component, starter in starters.items():
        if component not in selected:
            continue
        if component == "vla" and not needs_vla:
            continue
        pending[component] = try_spawn_server(
            owned_daemons, dashboard_events, component, starter
        )

    primitives_kwargs: dict[str, Any] = {}
    for component, (daemon, rpc) in pending.items():
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

    if "vla" in selected and not needs_vla:
        dashboard_events.emit(RuntimeStatusEvent("vla", "ready"))
        primitives_kwargs["model"] = None

    primitives_kwargs["calibration_path"] = args.calibration_path

    return list(owned_daemons.values()), primitives_kwargs
