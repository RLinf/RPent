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

"""RPent registration and process ownership for RoboLab."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from robots.robolab.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink
from rpent.dashboard.spec import DashboardSpec
from rpent.evaluation import RunFinalizationContext, write_json_atomic
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import try_spawn_server, try_wait_server
from rpent.utils.config import get_memory_dir, get_pi05_checkpoint_path, get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc import make_rpc_client
from rpent.utils.rpc.http_rpc import HttpRpcClient

ROBOLAB_DASHBOARD_SPEC: DashboardSpec = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task> <seed>",
        "fields": (
            {"name": "task", "suggestions": ("BananaInBowlTask",)},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{task} / seed {seed}",
        "output_slug": "{task}_s{seed}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "unique"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
    ),
    "frame_channels": (
        {"name": "main", "label": "external camera", "artifact": "main.png"},
        {"name": "wrist", "label": "wrist camera", "artifact": "wrist.png"},
    ),
    "primitives": ("view_env_state", "pi05_act", "execute_action"),
}


def get_robot_spec() -> RobotSpec:
    """Register RoboLab as a simulated RPent robot."""
    return RobotSpec(
        name="robolab",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        prepare_memory=_prepare_memory,
        validate_args=_validate_args,
        dashboard=ROBOLAB_DASHBOARD_SPEC,
        finalize_run=finalize_run,
    )


def finalize_run(context: RunFinalizationContext) -> Path:
    """Record benchmark success separately from the planner's finish status."""
    # The shared runner coerces solved() to bool; the trace retains the native None.
    state_path = context.output_dir / "states.json"
    native_success = None
    if state_path.is_file():
        steps = json.loads(state_path.read_text(encoding="utf-8"))["steps"]
        if steps:
            native_success = steps[-1]["state"]["success"]
            if native_success is not None and type(native_success) is not bool:
                raise ValueError("RoboLab success must be true, false, or null")
    return write_json_atomic(
        context.output_dir / "result.json",
        {
            "task": context.task_desc["task"],
            "seed": context.task_desc["seed"],
            "success": native_success,
            "environment_result_available": native_success is not None,
            "agent_error_present": context.agent_error is not None,
            "elapsed_s": round(context.elapsed_s, 1),
            "planner": {"backend": context.planner, "model": context.model},
        },
    )


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
):
    """Build the RoboLab toolkit for one episode."""
    from robots.robolab.toolkit import RoboLabToolkit

    return RoboLabToolkit(
        **runtime_kwargs,
        dashboard_events=dashboard_events,
        memory=MemoryManager(config.prompt_vars["memory_dir"]),
        output_dir=config.output_dir,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    parser.add_argument(
        "--task", required=not use_dashboard, help="RoboLab benchmark task class"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sim-python",
        default=sys.executable,
        help="Python with Isaac Lab and an external RoboLab source checkout installed",
    )
    parser.add_argument(
        "--policy-python",
        default=sys.executable,
        help="Python with RLinf and the Pi0.5 policy dependencies installed",
    )
    parser.add_argument("--device", default="cuda:0", help="Isaac Lab device")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument(
        "--policy-backend",
        choices=("pi05", "none"),
        default="pi05",
        help="Use Pi0.5 or native direct actions only",
    )
    parser.add_argument(
        "--pi05-checkpoint", default=None, help="External Pi0.5 checkpoint directory"
    )
    parser.add_argument("--env-endpoint", default=None)
    parser.add_argument("--vla-endpoint", default=None)


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if not args.task or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", args.task):
        raise ValueError("--task must be a RoboLab benchmark task class name")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.policy_backend == "none" and args.vla_endpoint:
        raise ValueError("--vla-endpoint requires --policy-backend pi05")
    if (
        args.policy_backend == "pi05"
        and not args.vla_endpoint
        and not (args.pi05_checkpoint or get_pi05_checkpoint_path())
    ):
        raise ValueError("Pi0.5 requires --pi05-checkpoint or PI05_CHECKPOINT_PATH")

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = (
            get_repo_root() / "logs" / f"{timestamp}_robolab_{args.task}_s{args.seed}"
        )
    memory_dir = (
        Path(args.memory_dir).expanduser().resolve()
        if getattr(args, "memory_dir", None)
        else get_memory_dir("robolab")
    )
    return RunConfig(
        recipe_tag=f"{args.task}_s{args.seed}",
        output_dir=Path(output_dir),
        prompt_vars={
            "task": args.task,
            "seed": args.seed,
            "policy_backend": args.policy_backend,
            "memory_dir": str(memory_dir),
            "output_dir": str(output_dir),
        },
        task_desc={"task": args.task, "seed": args.seed},
        enable_direct_action=True,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.dashboard and args.policy_backend == "none":
        raise ValueError("RoboLab Dashboard requires --policy-backend pi05")
    if (
        args.dashboard
        and not args.vla_endpoint
        and not (args.pi05_checkpoint or get_pi05_checkpoint_path())
    ):
        raise ValueError("RoboLab Dashboard requires a Pi0.5 checkpoint")


def _prepare_memory(args: argparse.Namespace, config: RunConfig) -> None:
    """Use local RoboLab notes until a benchmark memory corpus is available."""
    del args
    Path(config.prompt_vars["memory_dir"]).mkdir(parents=True, exist_ok=True)


def _process_env(args: argparse.Namespace) -> dict[str, str]:
    paths = [str(get_repo_root())]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    result = {
        "PYTHONPATH": os.pathsep.join(paths),
        "OMNI_KIT_ACCEPT_EULA": "Y",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    if args.cuda_device is not None:
        result["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    return result


def _spawn_env(args: argparse.Namespace, output_dir: Path):
    if args.env_endpoint:
        return None, make_rpc_client(args.env_endpoint)
    sim_python = Path(os.path.abspath(Path(args.sim_python).expanduser()))
    if not sim_python.is_file():
        raise FileNotFoundError(f"RoboLab simulator Python not found: {sim_python}")
    env_overrides = _process_env(args)
    libraries = sim_python.parent.parent / "system-libs" / "usr/lib/x86_64-linux-gnu"
    if libraries.is_dir():
        env_overrides["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, (str(libraries), os.environ.get("LD_LIBRARY_PATH")))
        )
        nvidia_icd = Path("/etc/vulkan/icd.d/nvidia_icd.json")
        if nvidia_icd.is_file() and not os.environ.get("VK_DRIVER_FILES"):
            env_overrides["VK_DRIVER_FILES"] = str(nvidia_icd)
    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        name="env_server",
        cmd=[
            str(sim_python),
            str(get_repo_root() / "robots" / "robolab" / "env_server.py"),
            "--task",
            args.task,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--output-dir",
            str(output_dir / "native"),
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
        ],
        env_overrides=env_overrides,
        log_path=str(output_dir / "env_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _spawn_vla(args: argparse.Namespace, output_dir: Path):
    if args.vla_endpoint:
        return None, make_rpc_client(args.vla_endpoint)
    policy_python = Path(os.path.abspath(Path(args.policy_python).expanduser()))
    if not policy_python.is_file():
        raise FileNotFoundError(f"Pi0.5 Python not found: {policy_python}")
    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        name="vla_server",
        cmd=[
            str(policy_python),
            str(
                get_repo_root()
                / "rpent"
                / "robots"
                / "components"
                / "pi05_vla_server.py"
            ),
            "--embodiment",
            "robolab",
            "--model-path",
            args.pi05_checkpoint or get_pi05_checkpoint_path(),
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
        ],
        env_overrides=_process_env(args),
        log_path=str(output_dir / "vla_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    from robots.robolab.env_client import RoboLabEnvClient
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient

    selected = (
        ({"env", "vla"} if args.policy_backend == "pi05" else {"env"})
        if components is None
        else set(components)
    )
    if selected - {"env", "vla"}:
        raise ValueError(
            f"unknown RoboLab runtime components: {sorted(selected - {'env', 'vla'})}"
        )
    if "vla" in selected and args.policy_backend == "none":
        raise ValueError("Pi0.5 is disabled")
    starters = {"env": _spawn_env, "vla": _spawn_vla}
    pending = {}
    owned: dict[str, ProcessDaemon] = {}
    for name in ("env", "vla"):
        if name in selected:
            pending[name] = try_spawn_server(
                owned,
                dashboard_events,
                name,
                lambda name=name: starters[name](args, output_dir),
            )
    runtime: dict[str, Any] = {"model": None}
    for name in ("env", "vla"):
        if name not in pending:
            continue
        daemon, rpc = pending[name]
        if name == "env":
            post = partial(
                RoboLabEnvClient,
                rpc,
                expected_meta={
                    "task": args.task,
                    "seed": args.seed,
                    "device": args.device,
                },
            )
        else:
            post = partial(Pi05VLAClient, rpc, embodiment="robolab")
        connected = try_wait_server(
            owned,
            dashboard_events,
            name,
            rpc,
            daemon,
            900.0 if name == "env" else 1800.0,
            post_fn=post,
        )
        runtime["env" if name == "env" else "model"] = connected
    return list(owned.values()), runtime
