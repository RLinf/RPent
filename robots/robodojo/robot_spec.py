# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboDojo robot extension — RobotSpec factory, toolkit factory, runtime hooks."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.robodojo.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import try_spawn_server, try_wait_server
from rpent.utils.config import get_repo_root

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon


DEFAULT_WORKSPACE = "/home/admin/robodojo_pro6000_ws"
DEFAULT_TASK = "put_bottles_into_dustbin"
DEFAULT_ENV_CFG = "arx_x5"


def _read_runtime_env(workspace: str) -> dict[str, str]:
    """Parse ``config/runtime.env`` exports into an env dict (no shell)."""
    path = Path(workspace) / "config" / "runtime.env"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        out[key.strip()] = value
    return out


def _subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    runtime = _read_runtime_env(os.environ.get("ROBODOJO_WORKSPACE", DEFAULT_WORKSPACE))
    env.update(runtime)
    conda_bin = runtime.get("ROBODOJO_CONDA_ROOT", "")
    if conda_bin:
        env["PATH"] = f"{conda_bin}/bin" + (
            f":{env.get('PATH', '')}" if env.get("PATH") else ""
        )
    source_root = runtime.get(
        "ROBODOJO_SOURCE_ROOT", DEFAULT_WORKSPACE + "/src/RoboDojo"
    )
    xpolicy_root = runtime.get(
        "ROBODOJO_XPOLICYLAB_ROOT",
        DEFAULT_WORKSPACE + "/src/RoboDojo/XPolicyLab",
    )
    env["PYTHONPATH"] = ":".join(
        [
            str(get_repo_root()),
            source_root,
            xpolicy_root,
            env.get("PYTHONPATH", ""),
        ]
    )
    if extra:
        env.update(extra)
    return env


def get_robot_spec() -> RobotSpec:
    """Return the RoboDojo robot identity, prompt bundle, and runner hooks."""
    return RobotSpec(
        name="robodojo",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=None,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
):
    """Return the RoboDojo toolkit (common tools + RoboDojo primitives)."""
    from robots.robodojo.toolkit import RoboDojoToolkit

    return RoboDojoToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    required = not use_dashboard
    parser.add_argument(
        "--workspace",
        default=None,
        help="RoboDojo workspace root "
        "(default: env or /home/admin/robodojo_pro6000_ws)",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        required=required,
        help="RoboDojo task name, e.g. put_bottles_into_dustbin",
    )
    parser.add_argument("--layout", type=int, default=0, help="Layout id (== env seed)")
    parser.add_argument(
        "--env-cfg-type", default=DEFAULT_ENV_CFG, help="RoboDojo env config (arx_x5)"
    )
    parser.add_argument(
        "--action-type",
        choices=["joint", "ee"],
        default="joint",
        help="Policy action representation for VLA primitives",
    )
    parser.add_argument("--max-episode-steps", type=int, default=700)
    parser.add_argument(
        "--sim-device",
        type=int,
        default=0,
        help="GPU device for the Isaac Sim env server",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample a fresh random scene per episode (ignores --layout)",
    )
    parser.add_argument(
        "--env-endpoint",
        default=None,
        help="[protocol://]host:port of an existing env_server",
    )
    parser.add_argument(
        "--vla-endpoint",
        default=None,
        help="[protocol://]host:port of an existing Pi_05 policy server",
    )
    parser.add_argument(
        "--sam3-endpoint",
        default=None,
        help="[protocol://]host:port of an existing SAM3 server",
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if getattr(args, "workspace", None):
        os.environ.setdefault("ROBODOJO_WORKSPACE", args.workspace)
    from robots.robodojo import tasks as robodojo_tasks

    err = robodojo_tasks.validate_task(args.task)
    if err:
        raise ValueError(err)
    summary = robodojo_tasks.task_summary(args.task)
    recipe_tag = f"{args.task}_l{args.layout}"
    if getattr(args, "random", False):
        recipe_tag = f"{args.task}_random"
    output_dir = args.output_dir
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = (
            get_repo_root() / "logs" / f"{ts}_robodojo_{args.task}_l{args.layout}"
        )
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=Path(output_dir),
        prompt_vars={
            "task": args.task,
            "layout": args.layout,
            "env_cfg_type": args.env_cfg_type,
            "action_type": args.action_type,
            "recipe_tag": recipe_tag,
            "task_summary": summary,
        },
        task_desc={"task": args.task, "layout": args.layout},
    )


def _spawn_env_server(
    args: argparse.Namespace, output_dir: Path
) -> tuple["ProcessDaemon | None", Any]:
    """Spawn (or attach to) the RoboDojo Isaac Sim env_server."""
    from rpent.utils.config import get_repo_root
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.rpc.http_rpc import HttpRpcClient

    runtime = _read_runtime_env(os.environ.get("ROBODOJO_WORKSPACE", DEFAULT_WORKSPACE))
    sim_python = str(Path(runtime.get("ROBODOJO_SIM_ENV", "")) / "bin" / "python")
    if not Path(sim_python).exists():
        raise RuntimeError(
            f"RoboDojo sim env python not found: {sim_python}; "
            "is ROBODOJO_SIM_ENV configured?"
        )

    if args.env_endpoint is None:
        host, port = "127.0.0.1", pick_free_port()
        daemon = ProcessDaemon(
            name="robodojo_env_server",
            cmd=[
                sim_python,
                "-u",
                str(get_repo_root() / "robots" / "robodojo" / "env_server.py"),
                "--task",
                args.task,
                "--layout",
                str(args.layout),
                "--env-cfg-type",
                args.env_cfg_type,
                "--device-id",
                str(args.sim_device),
                "--num-envs",
                "1",
                "--max-episode-steps",
                str(args.max_episode_steps),
                "--host",
                host,
                "--port",
                str(port),
                "--parent-pid",
                str(os.getpid()),
                "--video-dir",
                str(Path(output_dir) / "videos"),
                "--headless",
                "--enable_cameras",
                "--transport",
                "http",
                "--parent-watch",
                "--kit_args",
                "--enable isaacsim.replicator.behavior "
                "--enable isaacsim.sensors.camera",
            ]
            + (["--random"] if getattr(args, "random", False) else []),
            env=_subprocess_env({"CUDA_VISIBLE_DEVICES": str(args.sim_device)}),
            log_path=str(Path(output_dir) / "robodojo_env_server.log"),
        )
        daemon.start()
        rpc: Any = HttpRpcClient(f"http://{host}:{port}")
    else:
        from rpent.utils.rpc import parse_endpoint

        _, host, port = parse_endpoint(args.env_endpoint)
        rpc = HttpRpcClient(f"http://{host}:{port}")
        daemon = None
    return daemon, rpc


def _spawn_sam3_server(
    args: argparse.Namespace, output_dir: Path
) -> tuple["ProcessDaemon | None", Any]:
    """Spawn (or attach to) the env-agnostic SAM3 server."""
    from rpent.utils.config import get_repo_root
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.rpc import parse_endpoint
    from rpent.utils.rpc.http_rpc import HttpRpcClient

    if args.sam3_endpoint is None:
        host, port = "127.0.0.1", pick_free_port()
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
                "--cuda-device",
                str(args.sim_device),
            ],
            env=os.environ.copy(),
            log_path=str(Path(output_dir) / "sam3_server.log"),
        )
        daemon.start()
        rpc: Any = HttpRpcClient(f"http://{host}:{port}")
    else:
        _, host, port = parse_endpoint(args.sam3_endpoint)
        rpc = HttpRpcClient(f"http://{host}:{port}")
        daemon = None
    return daemon, rpc


def _spawn_vla_server(
    args: argparse.Namespace, output_dir: Path
) -> tuple["ProcessDaemon | None", Any]:
    """Spawn (or attach to) the RoboDojo Pi_05 VLA server."""
    from rpent.utils.config import get_repo_root
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.rpc import parse_endpoint
    from rpent.utils.rpc.http_rpc import HttpRpcClient

    if args.vla_endpoint is None:
        runtime = _read_runtime_env(
            os.environ.get("ROBODOJO_WORKSPACE", DEFAULT_WORKSPACE)
        )
        pi05_python = str(Path(runtime.get("ROBODOJO_PI05_ENV", "")) / "bin" / "python")
        if not Path(pi05_python).exists():
            raise RuntimeError(f"RoboDojo Pi_05 env python not found: {pi05_python}")
        policy_port = pick_free_port()
        host, port = "127.0.0.1", pick_free_port()
        daemon = ProcessDaemon(
            name="robodojo_vla_server",
            cmd=[
                pi05_python,
                "-u",
                str(get_repo_root() / "robots" / "robodojo" / "vla_server.py"),
                "--task",
                args.task,
                "--env-cfg-type",
                args.env_cfg_type,
                "--action-type",
                args.action_type,
                "--ckpt",
                "RoboDojo-sim-arx_x5-joint-0",
                "--policy-gpu",
                str(args.sim_device),
                "--policy-port",
                str(policy_port),
                "--host",
                host,
                "--port",
                str(port),
                "--parent-pid",
                str(os.getpid()),
                "--transport",
                "http",
                "--parent-watch",
            ],
            env=_subprocess_env({"CUDA_VISIBLE_DEVICES": str(args.sim_device)}),
            log_path=str(Path(output_dir) / "robodojo_vla_server.log"),
        )
        daemon.start()
        rpc: Any = HttpRpcClient(f"http://{host}:{port}")
    else:
        _, host, port = parse_endpoint(args.vla_endpoint)
        rpc = HttpRpcClient(f"http://{host}:{port}")
        daemon = None
    return daemon, rpc


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list["ProcessDaemon"], dict[str, Any]]:
    """Initialize every RoboDojo component, or only ``components`` when given."""
    from robots.robodojo.env_client import RoboDojoEnvClient
    from robots.robodojo.vla_client import RoboDojoVLAClient
    from rpent.utils.sam3_client import Sam3Client

    available = {"env", "vla", "sam3"}
    selected = set(components or available)
    unknown = selected.difference(available)
    if unknown:
        raise ValueError(f"unknown RoboDojo runtime components: {sorted(unknown)}")

    owned_daemons: dict[str, ProcessDaemon] = {}
    primitives_kwargs: dict[str, Any] = {}

    if "env" in selected:
        env_daemon, env_rpc = try_spawn_server(
            owned_daemons,
            dashboard_events,
            "env",
            lambda: _spawn_env_server(args, output_dir),
        )
        env_kwargs = try_wait_server(
            owned_daemons,
            dashboard_events,
            "env",
            env_rpc,
            env_daemon,
            900.0 if env_daemon is not None else 300.0,
            post_fn=lambda: {
                "env": RoboDojoEnvClient(
                    env_rpc,
                    expected_meta={
                        "task": args.task,
                        "layout": args.layout,
                        "env_cfg_type": args.env_cfg_type,
                        "device_id": args.sim_device,
                        "num_envs": 1,
                        "max_episode_steps": args.max_episode_steps,
                        "random": getattr(args, "random", False),
                    },
                )
            },
        )
        primitives_kwargs.update(env_kwargs)

    if "sam3" in selected:
        sam3_daemon, sam3_rpc = try_spawn_server(
            owned_daemons,
            dashboard_events,
            "sam3",
            lambda: _spawn_sam3_server(args, output_dir),
        )
        sam3_kwargs = try_wait_server(
            owned_daemons,
            dashboard_events,
            "sam3",
            sam3_rpc,
            sam3_daemon,
            300.0 if sam3_daemon is not None else 60.0,
            post_fn=lambda: {"sam3_client": Sam3Client(sam3_rpc)},
        )
        primitives_kwargs.update(sam3_kwargs)

    if "vla" in selected:
        vla_daemon, vla_rpc = try_spawn_server(
            owned_daemons,
            dashboard_events,
            "vla",
            lambda: _spawn_vla_server(args, output_dir),
        )
        vla_kwargs = try_wait_server(
            owned_daemons,
            dashboard_events,
            "vla",
            vla_rpc,
            vla_daemon,
            1800.0 if vla_daemon is not None else 300.0,
            post_fn=lambda: {"vla_client": RoboDojoVLAClient(vla_rpc)},
        )
        primitives_kwargs.update(vla_kwargs)

    primitives_kwargs["action_type"] = args.action_type
    primitives_kwargs["task"] = args.task
    return list(owned_daemons.values()), primitives_kwargs
