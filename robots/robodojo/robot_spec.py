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
import subprocess
import sys
from contextlib import suppress
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.robodojo.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink
from rpent.dashboard.spec import DashboardSpec
from rpent.evaluation import RunFinalizationContext, write_json_atomic
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import stop_owned_daemons, try_spawn_server, try_wait_server
from rpent.utils.config import get_memory_dir, get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc import make_rpc_client
from rpent.utils.rpc.http_rpc import HttpRpcClient

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient


DEFAULT_TASK = "put_bottles_into_dustbin"
DEFAULT_ENV_CFG = "arx_x5"

ROBODOJO_TASK_SUGGESTIONS = (
    "put_bottles_into_dustbin",
    "fill_pen_holder",
    "stack_bowls_random",
)

ROBODOJO_DASHBOARD_SPEC: DashboardSpec = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task> <layout>",
        "fields": (
            {"name": "task", "suggestions": ROBODOJO_TASK_SUGGESTIONS},
            {"name": "layout", "kind": "integer", "minimum": 0},
        ),
        "display": "{task} / layout {layout}",
        "output_slug": "{task}_l{layout}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "unique"},
        {"name": "sam3", "label": "SAM3", "scope": "shared"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
    ),
    "frame_channels": (
        {
            "name": "head",
            "label": "head camera",
            "artifact": "cam_head.png",
        },
        {
            "name": "wrist_left",
            "label": "left wrist camera",
            "artifact": "cam_left_wrist.png",
        },
        {
            "name": "wrist_right",
            "label": "right wrist camera",
            "artifact": "cam_right_wrist.png",
        },
    ),
    "primitives": (
        "view_env_state",
        "back_project",
        "segment",
        "render",
        "sample_world_xyz",
        "query_world_map",
        "move_to",
        "rotate_wrist",
        "set_gripper",
        "release",
        "vpp2_act",
        "awomo05_act",
    ),
}


def _runtime_overrides(args: argparse.Namespace) -> dict[str, str]:
    """Resolve packaged code in the simulator interpreter, preserving overrides."""
    if args.source_root:
        source = Path(args.source_root).expanduser().resolve()
    else:
        source = Path(
            subprocess.check_output(
                [
                    str(Path(args.sim_python).expanduser()),
                    "-c",
                    "from robodojo_runtime import source_root; print(source_root())",
                ],
                text=True,
            ).strip()
        )
    xpolicy = (
        Path(args.xpolicylab_root).expanduser().resolve()
        if args.xpolicylab_root
        else source / "XPolicyLab"
    )
    overrides = {
        "ROBODOJO_SOURCE_ROOT": str(source),
        "ROBODOJO_ROOT": str(source),
        "PYTHONPATH": os.pathsep.join(
            [str(get_repo_root()), str(source), str(xpolicy)]
            + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
        ),
    }
    if os.environ.get("ROBODOJO_ASSETS_ROOT"):
        overrides["ROBODOJO_ASSETS_ROOT"] = os.environ["ROBODOJO_ASSETS_ROOT"]
    if args.cuda_device is not None:
        overrides["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    return overrides


def get_robot_spec() -> RobotSpec:
    """Return the RoboDojo robot identity, prompt bundle, and runner hooks."""
    return RobotSpec(
        name="robodojo",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=ROBODOJO_DASHBOARD_SPEC,
        validate_args=_validate_args,
        finalize_run=finalize_run,
    )


def finalize_run(context: RunFinalizationContext) -> Path:
    """Write RoboDojo task identity and environment-authoritative run results."""
    record = {
        "task": context.task_desc["task"],
        "layout": context.task_desc["layout"],
        "seed": context.task_desc.get("seed", 0),
        "mode": context.task_desc.get("mode", "dev"),
        "success": context.environment_success,
        "environment_result_available": context.environment_success is not None,
        "agent_error_present": context.agent_error is not None,
        "elapsed_s": round(context.elapsed_s, 1),
        "planner": {
            "backend": context.planner,
            "model": context.model,
            "reasoning_effort": context.reasoning_effort,
            "max_turns": context.max_turns,
        },
        "planner_timeout_s": context.planner_timeout_s,
    }
    return write_json_atomic(context.output_dir / "result.json", record)


def _vla_model_path(args: argparse.Namespace) -> str | None:
    """Select the resource root for the requested VLA backend."""
    variable = f"{args.vla_backend.upper()}_MODEL_PATH"
    return getattr(args, "vla_model_path", None) or os.environ.get(variable)


def _validate_args(args: argparse.Namespace) -> None:
    """Reject unsupported replay and invalid scene identity before startup."""
    if getattr(args, "planner", None) == "flash":
        raise ValueError(
            "RoboDojo legacy Flash replay is unavailable with the new motion tools"
        )
    for name in ("seed", "layout"):
        if getattr(args, name, 0) < 0:
            raise ValueError(f"{name} must be nonnegative")

    backend = getattr(args, "vla_backend", "vpp2")
    if backend == "awomo05":
        if args.env_cfg_type != "arx_x5" or args.action_type != "joint":
            raise ValueError("awomo05 requires arx_x5 with absolute joint actions")


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
    state_output_dir: Path | str | None = None,
):
    """Return the RoboDojo toolkit (common tools + RoboDojo primitives)."""
    from robots.robodojo.toolkit import RoboDojoToolkit

    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("robodojo"),
        memory_access="read_only",
    )
    if config.prompt_vars.get("mode") == "eval-fair":
        from robots.robodojo.flash.replay import load_plan

        load_plan(memory.root, config.prompt_vars["task"])
    return RoboDojoToolkit(
        primitives_kwargs=runtime_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
        eval_fair=config.prompt_vars.get("mode") == "eval-fair",
        state_output_dir=state_output_dir,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    required = not use_dashboard
    parser.add_argument(
        "--source-root",
        default=None,
        help="Override the source tree shipped by rlinf-robodojo-runtime",
    )
    parser.add_argument(
        "--xpolicylab-root",
        default=None,
        help="XPolicyLab checkout (default: SOURCE_ROOT/XPolicyLab)",
    )
    parser.add_argument(
        "--sim-python",
        default=sys.executable,
        help="Python executable with Isaac Sim and RoboDojo installed",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        required=required,
        help="RoboDojo task name, e.g. put_bottles_into_dustbin",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Official layout group (Eval_Layout seed directory)",
    )
    parser.add_argument(
        "--layout",
        type=int,
        default=0,
        help="Layout index within the selected seed group",
    )
    parser.add_argument(
        "--env-cfg-type", default=DEFAULT_ENV_CFG, help="RoboDojo env config (arx_x5)"
    )
    parser.add_argument(
        "--action-type",
        choices=["joint", "ee"],
        default="joint",
        help="Legacy policy action representation; scripted primitives use joint targets",
    )
    parser.add_argument("--max-episode-steps", type=int, default=700)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device for local services (default: inherit CUDA_VISIBLE_DEVICES)",
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
        "--vla-backend",
        choices=("vpp2", "awomo05"),
        default="vpp2",
        help="VLA policy backend",
    )
    parser.add_argument(
        "--vla-model-path",
        default=None,
        help="VLA resource root containing policy/ and checkpoints/ "
        "(or VPP2_MODEL_PATH / AWOMO05_MODEL_PATH for the selected backend)",
    )
    parser.add_argument(
        "--vla-endpoint",
        default=None,
        help="[protocol://]host:port of an existing RPent VLA RPC server",
    )
    parser.add_argument(
        "--sam3-endpoint",
        default=None,
        help="[protocol://]host:port of an existing SAM3 server",
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    from robots.robodojo import tasks as robodojo_tasks

    source_root = getattr(args, "source_root", None)
    err = robodojo_tasks.validate_task(args.task, source_root)
    if err:
        raise ValueError(err)
    eval_fair = getattr(args, "planner", None) == "flash"
    summary = {} if eval_fair else robodojo_tasks.task_summary(args.task, source_root)
    seed = getattr(args, "seed", 0)
    recipe_tag = f"{args.task}_s{seed}_l{args.layout}"
    if getattr(args, "random", False):
        recipe_tag = f"{args.task}_random"
    memory_arg = getattr(args, "memory_dir", None)
    memory_dir = (
        Path(memory_arg).expanduser().resolve()
        if memory_arg
        else get_memory_dir("robodojo")
    )
    output_dir = args.output_dir
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{ts}_robodojo_{recipe_tag}"
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=Path(output_dir),
        prompt_vars={
            "task": args.task,
            "layout": args.layout,
            "seed": seed,
            "env_cfg_type": args.env_cfg_type,
            "action_type": args.action_type,
            "recipe_tag": recipe_tag,
            "task_summary": summary,
            "memory_dir": str(memory_dir),
            "mode": "eval-fair" if eval_fair else "dev",
            "vla_backend": args.vla_backend,
        },
        task_desc={
            "task": args.task,
            "layout": args.layout,
            "seed": seed,
            **({"mode": "eval-fair"} if eval_fair else {}),
        },
    )


def _cuda_args(args: argparse.Namespace) -> list[str]:
    return (
        ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []
    )


def _spawn_env_server(
    args: argparse.Namespace, output_dir: Path
) -> tuple["ProcessDaemon | None", RpcClient]:
    """Spawn (or attach to) the RoboDojo Isaac Sim env_server."""
    if args.env_endpoint is not None:
        return None, make_rpc_client(args.env_endpoint)

    sim_python = str(Path(args.sim_python).expanduser())
    if not Path(sim_python).exists():
        raise RuntimeError(
            f"RoboDojo sim env python not found: {sim_python}; configure --sim-python"
        )

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
            "--seed",
            str(getattr(args, "seed", 0)),
            "--env-cfg-type",
            args.env_cfg_type,
            "--num-envs",
            "1",
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--host",
            host,
            "--port",
            str(port),
            "--video-dir",
            str(Path(output_dir) / "videos"),
            "--save-dir",
            str(output_dir),
            "--headless",
            "--enable_cameras",
            "--transport",
            "http",
            "--parent-watch",
            "--kit_args",
            "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
        ]
        + _cuda_args(args)
        + (["--mode", "eval-fair"] if getattr(args, "planner", None) == "flash" else [])
        + (["--random"] if getattr(args, "random", False) else []),
        env_overrides=_runtime_overrides(args),
        log_path=str(Path(output_dir) / "robodojo_env_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _spawn_sam3_server(
    args: argparse.Namespace, output_dir: Path
) -> tuple["ProcessDaemon | None", RpcClient]:
    """Spawn (or attach to) the env-agnostic SAM3 server."""
    if args.sam3_endpoint is not None:
        return None, make_rpc_client(args.sam3_endpoint)

    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        name="sam3_server",
        cmd=[
            sys.executable,
            str(get_repo_root() / "rpent" / "robots" / "components" / "sam3_server.py"),
            "--transport",
            "http",
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
        ]
        + _cuda_args(args),
        log_path=str(Path(output_dir) / "sam3_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _spawn_model_server(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    model_path: str | None,
    rpc_options: dict[str, Any],
) -> tuple["ProcessDaemon | None", RpcClient]:
    """Start a local model or borrow its RPC endpoint with the required session mode."""
    if args.vla_endpoint is not None:
        return None, make_rpc_client(args.vla_endpoint, **rpc_options)
    host, port = "127.0.0.1", pick_free_port()
    name = f"robodojo_{args.vla_backend}_server"
    daemon = ProcessDaemon(
        name=name,
        cmd=[
            sys.executable,
            "-u",
            "-m",
            f"robots.robodojo.{args.vla_backend}_server",
            "--model-path",
            str(Path(model_path).expanduser().resolve()),
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
        ],
        env_overrides=(
            {"CUDA_VISIBLE_DEVICES": str(args.cuda_device)}
            if args.cuda_device is not None
            else {}
        ),
        log_path=str(Path(output_dir) / f"{name}.log"),
    )
    try:
        daemon.start()
        rpc = make_rpc_client(f"socket://{host}:{port}", **rpc_options)
    except BaseException:
        daemon.stop()
        raise
    return daemon, rpc


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Start default backend services or an explicitly requested component set."""
    _validate_args(args)
    from robots.robodojo.env_client import RoboDojoEnvClient
    from rpent.robots.components.sam3_client import Sam3Client

    if args.vla_backend == "awomo05":
        from robots.robodojo.awomo05_client import Awomo05Client as VLAClient
        from robots.robodojo.awomo05_server import model_paths

        rpc_options = {"enable_sessions": True}
        record_vpp2_history = False
    elif args.vla_backend == "vpp2":
        from robots.robodojo.vpp2_client import VPP2Client as VLAClient
        from robots.robodojo.vpp2_server import model_paths

        rpc_options = {}
        record_vpp2_history = True
    else:
        raise ValueError(f"unknown RoboDojo VLA backend: {args.vla_backend}")
    model_path = _vla_model_path(args)
    starters = {
        "env": lambda: _spawn_env_server(args, output_dir),
        "vla": lambda: _spawn_model_server(
            args, output_dir, model_path=model_path, rpc_options=rpc_options
        ),
        "sam3": lambda: _spawn_sam3_server(args, output_dir),
    }
    connectors = {
        "env": lambda rpc: {
            "env": RoboDojoEnvClient(
                rpc,
                record_vpp2_history=record_vpp2_history,
                expected_meta={
                    "task": args.task,
                    "layout": args.layout,
                    "seed": getattr(args, "seed", 0),
                    "env_cfg_type": args.env_cfg_type,
                    "device_id": args.cuda_device
                    if args.cuda_device is not None
                    else 0,
                    "num_envs": 1,
                    "max_episode_steps": args.max_episode_steps,
                    "random": getattr(args, "random", False),
                    **(
                        {"mode": "eval-fair"}
                        if getattr(args, "planner", None) == "flash"
                        else {}
                    ),
                },
            )
        },
        "vla": lambda rpc: {"vla_client": VLAClient(rpc)},
        "sam3": lambda rpc: {"sam3_client": Sam3Client(rpc)},
    }
    timeouts = {"env": 900.0, "sam3": 300.0, "vla": 1800.0}
    wait_order = ("env", "sam3", "vla")
    selected = set(starters) if components is None else set(components)
    unknown = selected.difference(starters)
    if unknown:
        raise ValueError(f"unknown RoboDojo runtime components: {sorted(unknown)}")
    if "vla" in selected and args.vla_endpoint is None:
        model_paths(model_path)

    pending: dict[str, tuple[ProcessDaemon | None, RpcClient]] = {}
    owned_daemons: dict[str, ProcessDaemon] = {}
    try:
        for component, starter in starters.items():
            if component in selected:
                pending[component] = try_spawn_server(
                    owned_daemons,
                    dashboard_events,
                    component,
                    starter,
                )

        primitives_kwargs: dict[str, Any] = {}
        for component in (name for name in wait_order if name in pending):
            daemon, rpc = pending[component]
            component_kwargs = try_wait_server(
                owned_daemons,
                dashboard_events,
                component,
                rpc,
                daemon,
                timeouts[component],
                post_fn=partial(connectors[component], rpc),
            )
            primitives_kwargs.update(component_kwargs)
    except BaseException as exc:
        # Shared helpers clean up ordinary failures; interruption is owned here.
        if not isinstance(exc, Exception):
            stop_owned_daemons(owned_daemons, dashboard_events)
        for _, rpc in pending.values():
            with suppress(Exception):
                rpc.close()
        raise

    primitives_kwargs["action_type"] = args.action_type
    primitives_kwargs["task"] = args.task
    primitives_kwargs["vla_backend"] = args.vla_backend
    return list(owned_daemons.values()), primitives_kwargs
