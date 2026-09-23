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
from rpent.robots.runtime import try_spawn_server, try_wait_server
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
        {"name": "vla", "label": "VLA", "scope": "shared"},
        {"name": "sam3", "label": "SAM3", "scope": "shared"},
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
        "move_to",
        "set_gripper",
        "pi0_pick",
        "stabilize",
        "place_in_bin",
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
        "ROBODOJO_PI05_POLICY_ROOT": str(xpolicy / "policy" / "Pi_05"),
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
        run_flash=_run_flash,
        finalize_run=finalize_run,
    )


def finalize_run(context: RunFinalizationContext) -> Path:
    """Write RoboDojo task identity and environment-authoritative run results."""
    record = {
        "task": context.task_desc["task"],
        "layout": context.task_desc["layout"],
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


def _run_flash(toolkit, cell_tag: str, note) -> dict:
    from robots.robodojo.flash.replay import run_flash

    return run_flash(toolkit, cell_tag, note)


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
    allowed_tool_groups: frozenset[str] | None = None,
):
    """Return the RoboDojo toolkit (common tools + RoboDojo primitives)."""
    from robots.robodojo.toolkit import RoboDojoToolkit

    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("robodojo"),
    )
    if config.prompt_vars.get("mode") == "eval-fair":
        from robots.robodojo.flash.replay import load_plan

        load_plan(memory.root, config.prompt_vars["task"])
    return RoboDojoToolkit(
        primitives_kwargs=runtime_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
        allowed_tool_groups=allowed_tool_groups,
        eval_fair=config.prompt_vars.get("mode") == "eval-fair",
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
        "--policy-backend",
        choices=["rlinf", "xpolicylab"],
        default="rlinf",
        help="Pi0.5 policy backend (default: rlinf)",
    )
    parser.add_argument(
        "--pi05-python",
        default=sys.executable,
        help="Python executable with Pi_05 dependencies installed",
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
    from robots.robodojo import tasks as robodojo_tasks

    source_root = getattr(args, "source_root", None)
    err = robodojo_tasks.validate_task(args.task, source_root)
    if err:
        raise ValueError(err)
    eval_fair = getattr(args, "planner", None) == "flash"
    summary = {} if eval_fair else robodojo_tasks.task_summary(args.task, source_root)
    recipe_tag = f"{args.task}_l{args.layout}"
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
            "memory_dir": str(memory_dir),
            "mode": "eval-fair" if eval_fair else "dev",
        },
        task_desc={
            "task": args.task,
            "layout": args.layout,
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


def _spawn_vla_server(
    args: argparse.Namespace, output_dir: Path
) -> tuple["ProcessDaemon | None", RpcClient]:
    """Spawn (or attach to) the RoboDojo Pi_05 VLA server."""
    if args.vla_endpoint is not None:
        return None, make_rpc_client(args.vla_endpoint)

    pi05_python = str(Path(args.pi05_python).expanduser())
    if not Path(pi05_python).exists():
        raise RuntimeError(f"RoboDojo Pi_05 env python not found: {pi05_python}")
    host, port = "127.0.0.1", pick_free_port()
    overrides = _runtime_overrides(args)
    if args.policy_backend == "rlinf":
        # End-to-end policy-chain smoke only (real weights, one prediction);
        # simulation task success rates have not yet been established.
        policy_args = [
            "rpent.robots.components.pi05_vla_server",
            "--embodiment",
            "robodojo",
        ]
    else:
        policy_args = [
            "rpent.robots.components.xpolicylab_vla_server",
            "--output-dir",
            str(output_dir),
            "--policy-root",
            overrides["ROBODOJO_PI05_POLICY_ROOT"],
            "--bench",
            "RoboDojo",
            "--evaluation-id",
            os.environ.get(
                "ROBODOJO_RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ),
            "--task",
            args.task,
            "--env-cfg-type",
            args.env_cfg_type,
            "--action-type",
            args.action_type,
            "--ckpt",
            "RoboDojo-sim-arx_x5-joint-0",
            "--policy-port",
            str(pick_free_port()),
        ]
        if args.cuda_device is not None:
            policy_args += ["--policy-gpu", str(args.cuda_device)]
    daemon = ProcessDaemon(
        name="robodojo_vla_server",
        cmd=[
            pi05_python,
            "-u",
            "-m",
            *policy_args,
            "--host",
            host,
            "--port",
            str(port),
            "--transport",
            "http",
            "--parent-watch",
        ],
        env_overrides=overrides,
        log_path=str(Path(output_dir) / "robodojo_vla_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize every RoboDojo component, or only ``components`` when given."""
    from robots.robodojo.env_client import RoboDojoEnvClient
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient
    from rpent.robots.components.sam3_client import Sam3Client
    from rpent.robots.components.xpolicylab_vla_client import XPolicyLabVLAClient

    starters = {
        "env": lambda: _spawn_env_server(args, output_dir),
        "vla": lambda: _spawn_vla_server(args, output_dir),
        "sam3": lambda: _spawn_sam3_server(args, output_dir),
    }
    connectors = {
        "env": lambda rpc: {
            "env": RoboDojoEnvClient(
                rpc,
                expected_meta={
                    "task": args.task,
                    "layout": args.layout,
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
        "vla": lambda rpc: {
            "vla_client": (
                Pi05VLAClient(rpc, embodiment="robodojo")
                if args.policy_backend == "rlinf"
                else XPolicyLabVLAClient(rpc)
            )
        },
        "sam3": lambda rpc: {"sam3_client": Sam3Client(rpc)},
    }
    timeouts = {"env": 900.0, "sam3": 300.0, "vla": 1800.0}
    wait_order = ("env", "sam3", "vla")
    selected = set(starters) if components is None else set(components)
    unknown = selected.difference(starters)
    if unknown:
        raise ValueError(f"unknown RoboDojo runtime components: {sorted(unknown)}")

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

    primitives_kwargs["action_type"] = args.action_type
    primitives_kwargs["task"] = args.task
    return list(owned_daemons.values()), primitives_kwargs
