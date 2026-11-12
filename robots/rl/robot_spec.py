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

"""Shared adaptive simulator spec, toolkit factory, and runtime hooks."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.rl.prompt_bundle import (
    OBSERVATION_MODES,
    PROMPT_VARIANTS,
    WORKFLOWS,
    system_prompt,
    user_prompt,
    validate_workflow,
)
from rpent.dashboard.events import DashboardEventSink
from rpent.evaluation import RunFinalizationContext, write_json_atomic
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import try_spawn_server, try_wait_server
from rpent.utils.config import get_memory_dir, get_repo_root
from rpent.utils.daemon import ProcessDaemon
from rpent.utils.rpc import make_rpc_client, wait_for_endpoint

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient

SUPPORTED_PLANNERS = ("api",)


def get_robot_spec() -> RobotSpec:
    """Return the shared robot identity, prompts, and runner hooks."""
    return RobotSpec(
        name="rl",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        # Calls share one physical scene, including paired-arm actions.
        sequential_tools=True,
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        validate_args=_validate_args,
        prepare_memory=_skip_memory,
        finalize_run=_finalize_run,
        configure_planner=configure_planner,
    )


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
):
    """Return the adaptive toolkit for the selected simulator and workflow."""
    if "remote_endpoint" in runtime_kwargs:
        from robots.rl.remote import RemoteToolkit

        return RemoteToolkit(
            endpoint=runtime_kwargs["remote_endpoint"],
            request=runtime_kwargs["remote_request"],
            config=config,
            dashboard_events=dashboard_events,
            memory=MemoryManager(get_memory_dir("rl")),
        )
    from robots.rl.toolkit import RLToolkit

    return RLToolkit(
        runtime_kwargs=runtime_kwargs,
        dashboard_events=dashboard_events,
        # Required by Toolkit; this toolkit registers no file tools to read it.
        memory=MemoryManager(get_memory_dir("rl")),
        observation_mode=config.prompt_vars["observation_mode"],
        output_dir=config.output_dir,
        environment=config.prompt_vars.get("environment", "libero"),
        workflow=config.prompt_vars.get("workflow", "episode"),
        image_size=config.prompt_vars.get("image_size", 1024),
        action_mode=config.prompt_vars.get("action_mode", "single"),
        max_actions_per_call=config.prompt_vars.get("max_actions_per_call", 3),
        reasoning_history=config.prompt_vars.get("reasoning_history", -1),
        image_history=config.prompt_vars.get("image_history", -1),
        perception_mode=config.prompt_vars.get("perception_mode", "single"),
        max_observe_calls=config.prompt_vars.get("max_observe_calls", 8),
        observe_concurrency=config.prompt_vars.get("observe_concurrency", 8),
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """Register simulator, task, observation, and runtime flags."""
    parser.add_argument(
        "--simulator",
        dest="env",
        choices=("libero", "robocasa", "robotwin"),
        default="libero",
    )
    parser.add_argument("--workflow", choices=WORKFLOWS, default="episode")
    parser.add_argument("--action-mode", choices=("single", "batch"), default="single")
    parser.add_argument("--max-actions-per-call", type=int, default=3)
    parser.add_argument(
        "--perception-mode",
        choices=("single", "batch"),
        default="single",
        help="single: view_env/segment/back_project; batch: observe(calls=[...]), episode only.",
    )
    parser.add_argument(
        "--max-observe-calls",
        type=int,
        default=8,
        help="Maximum total subcalls in observe; >=1, requires perception-mode batch.",
    )
    parser.add_argument(
        "--observe-concurrency",
        type=int,
        default=8,
        help="Maximum independent observe subcalls in flight; >=1. Does not change SAM3 worker limits.",
    )
    parser.add_argument(
        "--reasoning-history",
        type=int,
        default=-1,
        help="-1 all surviving replies, 0 none, N latest model replies.",
    )
    parser.add_argument(
        "--image-history",
        type=int,
        default=-1,
        help="-1 all surviving observations, 0 no images, N latest image-return groups.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        choices=(1024, 512, 256),
        default=1024,
        help="VLM image width/height; native RGB-D and SAM3 remain 1024.",
    )
    parser.add_argument("--task-name", help="RoboCasa or RoboTwin task identifier.")
    parser.add_argument(
        "--split", default="target", choices=("target", "pretrain", "all")
    )
    parser.add_argument(
        "--task-config",
        default="demo_randomized",
        choices=("demo_clean", "demo_randomized"),
    )
    parser.add_argument(
        "--assets-path",
        type=Path,
        default=os.environ.get("ROBOTWIN_ASSETS_PATH"),
        help="RoboTwin asset snapshot root; defaults to ROBOTWIN_ASSETS_PATH.",
    )
    parser.add_argument(
        "--sim-python", help="Interpreter containing the selected simulator."
    )
    parser.add_argument(
        "--observation-mode",
        choices=OBSERVATION_MODES,
        default="adaptive",
        help="Adaptive observations: explicit in episode, automatic in zap_* modes.",
    )
    parser.add_argument(
        "--prompt",
        choices=PROMPT_VARIANTS,
        default="simple",
        help="System prompt variant: simple is the bare tool contract; base adds "
        "the robot, frames and execution; base_gripper adds finger geometry; "
        "base_gripper_camera adds camera geometry.",
    )
    parser.add_argument("--suite", help="LIBERO suite, e.g. libero_object")
    parser.add_argument("--task", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument(
        "--libero-type",
        default=None,
        choices=["standard", "pro", "plus"],
        help="LIBERO variant (LIBERO_TYPE when unset).",
    )
    parser.add_argument(
        "--env-endpoint",
        default=None,
        help="[protocol://]host:port of an existing env_server; spawned when unset.",
    )
    parser.add_argument(
        "--remote-toolkit-endpoint",
        default=os.environ.get("RPENT_REMOTE_TOOLKIT_ENDPOINT"),
        help="Remote RL backend; motion, perception and simulator run on its GPUs.",
    )
    parser.add_argument(
        "--sam3-endpoint",
        default=None,
        help="[protocol://]host:port of an existing SAM3 server; spawned when unset.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device for the spawned servers.",
    )


def _validate_args(args: argparse.Namespace) -> None:
    """Reject options that would change the fixed rollout setup unnoticed."""
    validate_workflow(
        getattr(args, "env", "libero"), getattr(args, "workflow", "episode")
    )
    if (
        getattr(args, "image_size", 1024) != 1024
        and getattr(args, "env", "libero") != "libero"
    ):
        raise ValueError("--image_size currently supports LIBERO only")
    from robots.rl.contracts import validate_options, validate_perception_options

    validate_perception_options(
        getattr(args, "workflow", "episode"),
        getattr(args, "perception_mode", "single"),
        getattr(args, "max_observe_calls", 8),
        getattr(args, "observe_concurrency", 8),
    )
    validate_options(
        getattr(args, "env", "libero"),
        getattr(args, "workflow", "episode"),
        getattr(args, "action_mode", "single"),
        getattr(args, "max_actions_per_call", 3),
        getattr(args, "reasoning_history", -1),
        getattr(args, "image_history", -1),
    )
    if args.planner not in SUPPORTED_PLANNERS:
        raise ValueError(
            f"rl supports --planner {' or '.join(SUPPORTED_PLANNERS)}; "
            f"{args.planner} cannot restrict its built-in tools"
        )
    if args.dashboard or args.tui:
        raise ValueError("rl runs non-interactive rollouts")
    if args.memory_dir is not None or args.memory_repo is not None:
        raise ValueError("rl loads no memory; remove --memory-dir/--memory-repo")
    if args.observation_mode != "adaptive":
        raise ValueError("rl supports only adaptive observations")
    if args.no_images:
        raise ValueError("rl requires camera images; remove --no-images")
    if getattr(args, "remote_toolkit_endpoint", None) is not None:
        if any(
            getattr(args, name, default) != default
            for name, default in (
                ("perception_mode", "single"),
                ("max_observe_calls", 8),
                ("observe_concurrency", 8),
                ("action_mode", "single"),
                ("max_actions_per_call", 3),
                ("reasoning_history", -1),
                ("image_history", -1),
            )
        ):
            raise ValueError(
                "remote toolkit does not yet support action/history/observe options"
            )
        if getattr(args, "workflow", "episode") != "episode":
            raise ValueError("remote toolkit currently supports --workflow episode")
        if args.env_endpoint is not None or args.sam3_endpoint is not None:
            raise ValueError(
                "remote toolkit owns its env/SAM3; remove separate endpoints"
            )


def _skip_memory(args: argparse.Namespace, config: RunConfig) -> None:
    """Rollouts start without guides, memory, or recipes."""


def _parse_config(args: argparse.Namespace) -> RunConfig:
    """Derive the run identifiers and the planner-visible prompt variables."""
    if args.observation_mode != "adaptive":
        raise ValueError("rl supports only adaptive observations")
    environment = getattr(args, "env", "libero")
    workflow = getattr(args, "workflow", "episode")
    validate_workflow(environment, workflow)
    from robots.rl.contracts import validate_options, validate_perception_options

    validate_perception_options(
        getattr(args, "workflow", "episode"),
        getattr(args, "perception_mode", "single"),
        getattr(args, "max_observe_calls", 8),
        getattr(args, "observe_concurrency", 8),
    )
    validate_options(
        environment,
        workflow,
        getattr(args, "action_mode", "single"),
        getattr(args, "max_actions_per_call", 3),
        getattr(args, "reasoning_history", -1),
        getattr(args, "image_history", -1),
    )
    if environment == "libero" and not args.suite:
        raise ValueError("--suite is required")
    if environment == "libero" and args.task is None:
        raise ValueError("--task is required")
    if environment != "libero" and not args.task_name:
        raise ValueError("--task-name is required for RoboCasa and RoboTwin")
    if environment != "libero" and not re.fullmatch(r"[A-Za-z0-9_]+", args.task_name):
        raise ValueError("task names must contain only letters, digits and underscores")
    cell = (
        f"{args.suite}_t{args.task}"
        if environment == "libero"
        else f"{environment}_{args.task_name}"
    )
    cell += f"_s{args.seed}_{args.observation_mode}"
    output_dir = args.output_dir or (
        get_repo_root() / "logs" / f"{datetime.now():%Y%m%d-%H:%M:%S}_{cell}"
    )
    return RunConfig(
        recipe_tag=cell.removeprefix("libero_"),
        output_dir=Path(output_dir),
        prompt_vars={
            "observation_mode": args.observation_mode,
            "prompt": args.prompt,
            "environment": environment,
            "workflow": workflow,
            "image_size": getattr(args, "image_size", 1024),
            "action_mode": getattr(args, "action_mode", "single"),
            "perception_mode": getattr(args, "perception_mode", "single"),
            "max_observe_calls": getattr(args, "max_observe_calls", 8),
            "observe_concurrency": getattr(args, "observe_concurrency", 8),
            "max_actions_per_call": getattr(args, "max_actions_per_call", 3),
            "reasoning_history": getattr(args, "reasoning_history", -1),
            "image_history": getattr(args, "image_history", -1),
        },
        task_desc={
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "observation_mode": args.observation_mode,
            "prompt": args.prompt,
            "environment": environment,
            "workflow": workflow,
            "image_size": getattr(args, "image_size", 1024),
            "action_mode": getattr(args, "action_mode", "single"),
            "perception_mode": getattr(args, "perception_mode", "single"),
            "max_observe_calls": getattr(args, "max_observe_calls", 8),
            "observe_concurrency": getattr(args, "observe_concurrency", 8),
            "max_actions_per_call": getattr(args, "max_actions_per_call", 3),
            "reasoning_history": getattr(args, "reasoning_history", -1),
            "image_history": getattr(args, "image_history", -1),
            **({"task_name": args.task_name} if environment != "libero" else {}),
        },
    )


def _spawn_server(
    args: argparse.Namespace,
    output_dir: Path,
    name: str,
    script: Path,
    script_args: list[str],
    env_overrides: dict[str, str] | None = None,
    python: str | None = None,
) -> tuple[ProcessDaemon, RpcClient]:
    log = output_dir / f"{name}.log"
    log.write_text("")
    cuda = (
        ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []
    )
    daemon = ProcessDaemon(
        name=name,
        cmd=[
            python or sys.executable,
            str(script),
            *script_args,
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--parent-watch",
            *cuda,
        ],
        env_overrides=env_overrides,
        log_path=str(log),
    )
    daemon.start()
    try:
        endpoint = wait_for_endpoint(daemon)
    except BaseException:
        daemon.stop()
        raise
    return daemon, make_rpc_client(endpoint)


def _start_env(
    args: argparse.Namespace, output_dir: Path
) -> tuple[ProcessDaemon | None, RpcClient]:
    if args.env_endpoint is not None:
        return None, make_rpc_client(args.env_endpoint)
    environment = getattr(args, "env", "libero")
    if environment != "libero":
        simulator_python = args.sim_python or os.environ.get(
            f"RPENT_{environment.upper()}_PYTHON"
        )
        overrides = {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "ROBOT_PLATFORM": environment.upper(),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": str(get_repo_root()),
        }
        if simulator_python:
            overrides["LD_LIBRARY_PATH"] = _simulator_library_path(simulator_python)
        return _spawn_server(
            args,
            output_dir,
            "env_server",
            get_repo_root() / "robots/rl/server.py",
            [
                "--env",
                environment,
                "--task-name",
                args.task_name,
                "--seed",
                str(args.seed),
                "--split",
                args.split,
                "--task-config",
                args.task_config,
                "--max-episode-steps",
                str(args.max_episode_steps),
                *(["--assets-path", str(args.assets_path)] if args.assets_path else []),
            ],
            overrides,
            python=simulator_python,
        )
    from rpent.utils.config import get_libero_type

    # The LIBERO env server and client are shared with robots.libero.
    return _spawn_server(
        args,
        output_dir,
        "env_server",
        get_repo_root() / "robots" / "libero" / "env_server.py",
        [
            "--suite",
            args.suite,
            "--task",
            str(args.task),
            "--seed",
            str(args.seed),
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--rl-control-state",
        ],
        {
            "LIBERO_TYPE": args.libero_type or get_libero_type(),
            "MUJOCO_GL": "egl",
            "ROBOT_PLATFORM": "LIBERO",
        },
    )


def _simulator_library_path(python: str) -> str:
    """Use the simulator's CUDA libraries while retaining host driver paths."""
    purelib = Path(
        subprocess.check_output(
            [python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
            timeout=20,
        ).strip()
    )
    paths = [str(path) for path in sorted(purelib.glob("nvidia/*/lib"))]
    if (purelib / "torch/lib").is_dir():
        paths.append(str(purelib / "torch/lib"))
    paths.extend(
        path
        for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if path
        and "/site-packages/nvidia/" not in path
        and "/site-packages/torch/" not in path
    )
    return os.pathsep.join(dict.fromkeys(paths))


def _start_sam3(
    args: argparse.Namespace, output_dir: Path
) -> tuple[ProcessDaemon | None, RpcClient]:
    if args.sam3_endpoint is not None:
        return None, make_rpc_client(args.sam3_endpoint)
    return _spawn_server(
        args,
        output_dir,
        "sam3_server",
        get_repo_root() / "rpent" / "robots" / "components" / "sam3_server.py",
        [],
    )


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Start or connect the environment and SAM3 services."""
    if getattr(args, "remote_toolkit_endpoint", None) is not None:
        config = _parse_config(args)
        task = dict(config.task_desc)
        task.update(split=args.split, task_config=args.task_config)
        if task["environment"] == "libero":
            from rpent.utils.config import get_libero_type

            task["libero_type"] = args.libero_type or get_libero_type()
        return [], {
            "remote_endpoint": args.remote_toolkit_endpoint,
            "remote_request": {
                "task": task,
                "gpu": args.cuda_device,
                "max_episode_steps": args.max_episode_steps,
            },
        }
    from rpent.robots.components.sam3_client import Sam3Client

    starters = {"env": _start_env, "sam3": _start_sam3}
    connectors = {
        "env": partial(_connect_env, args),
        "sam3": lambda rpc: {"sam3_client": Sam3Client(rpc)},
    }
    selected = set(starters) if components is None else set(components)
    if unknown := selected.difference(starters):
        raise ValueError(f"unknown rl runtime components: {sorted(unknown)}")

    owned: dict[str, ProcessDaemon] = {}
    pending = {
        name: try_spawn_server(
            owned, dashboard_events, name, partial(starter, args, output_dir)
        )
        for name, starter in starters.items()
        if name in selected
    }
    runtime_kwargs: dict[str, Any] = {}
    for name, (daemon, rpc) in pending.items():
        runtime_kwargs.update(
            try_wait_server(
                owned,
                dashboard_events,
                name,
                rpc,
                daemon,
                300.0,
                post_fn=partial(connectors[name], rpc),
            )
        )
    return list(owned.values()), runtime_kwargs


def _connect_env(args: argparse.Namespace, rpc) -> dict[str, Any]:
    environment = getattr(args, "env", "libero")
    if environment == "libero":
        from robots.libero.env_client import LiberoEnvClient

        env = LiberoEnvClient(
            rpc,
            expected_meta={
                "suite": args.suite,
                "task": args.task,
                "seed": args.seed,
                "max_episode_steps": args.max_episode_steps,
            },
        )
    elif environment == "robocasa":
        from robots.robocasa.env_client import RoboCasaEnvClient

        env = RoboCasaEnvClient(
            rpc,
            expected_meta={
                "task_name": args.task_name,
                "split": args.split,
                "seed": args.seed,
                "camera_h": 256,
                "camera_w": 256,
            },
        )
    else:
        from robots.robotwin.env_client import RoboTwinEnvClient
        from robots.robotwin.robot_spec import env_runtime_contract

        env = RoboTwinEnvClient(
            rpc,
            expected_meta=env_runtime_contract(
                task_name=args.task_name,
                task_config=args.task_config,
                seed=args.seed,
                max_episode_steps=args.max_episode_steps,
            ),
        )
    return {
        "env": env,
        **(
            {"max_episode_steps": args.max_episode_steps}
            if environment != "libero"
            else {}
        ),
    }


def configure_planner(planner, config: RunConfig):
    """Install RL request policy while retaining the provider and native agent."""
    from robots.rl.workflow import WorkflowCapability
    from rpent.planner.api_loop import ApiAgentLoop

    if not isinstance(planner, ApiAgentLoop):
        raise ValueError("rl workflows require --planner api")
    from robots.rl.history import HistoryRetention

    planner.capabilities_factory = lambda toolkit, system, user, turns: [
        WorkflowCapability(toolkit, system, user, turns, timeout_s=planner.timeout_s),
        *(
            [HistoryRetention(toolkit)]
            if config.prompt_vars.get("environment", "libero") == "libero"
            and config.prompt_vars.get("workflow", "episode")
            in {"episode", "episode_image"}
            else []
        ),
    ]
    if config.prompt_vars.get("workflow", "episode") == "zap_turn":
        planner.compaction = None
    return planner


def _finalize_run(context: RunFinalizationContext) -> Path:
    """Write the native success flag beside the rollout's observation counts."""
    rollout_path = context.output_dir / "rollout.json"
    rollout = json.loads(rollout_path.read_text()) if rollout_path.is_file() else {}
    return write_json_atomic(
        context.output_dir / "result.json",
        {
            "robot": context.robot_name,
            "task": dict(context.task_desc),
            "success": context.environment_success,
            "agent_error": context.agent_error,
            "elapsed_s": round(context.elapsed_s, 1),
            "planner": context.planner,
            "model": context.model,
            "max_turns": context.max_turns,
            "stats": dict(context.stats),
            **rollout,
        },
    )
