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

"""MetaWorld factories using common runtime, RPC, memory and result contracts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from robots.metaworld.config import CAMERAS, TASKS, validate
from robots.metaworld.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink
from rpent.evaluation import write_json_atomic
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import try_spawn_server, try_wait_server
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_memory_dir, get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc import make_rpc_client


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool = False) -> None:
    parser.add_argument("--task", choices=TASKS, default="reach-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--camera", choices=CAMERAS, default="corner2")
    parser.add_argument("--env-endpoint", default=None)
    parser.add_argument("--sim-python", default=sys.executable)


def _validate(args) -> None:
    validate(args.task, args.seed, args.max_episode_steps, args.camera)


def _parse_config(args) -> RunConfig:
    tag = f"{args.task}_s{args.seed}"
    memory = Path(args.memory_dir or get_memory_dir("metaworld")).expanduser().resolve()
    output = Path(
        args.output_dir
        or get_repo_root() / "runs" / f"metaworld-{tag}-{datetime.now():%Y%m%d-%H%M%S}"
    )
    return RunConfig(
        recipe_tag=tag,
        output_dir=output,
        prompt_vars={
            "instruction": TASKS[args.task],
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
            "memory_dir": str(memory),
        },
        task_desc={
            "task": args.task,
            "seed": args.seed,
            "camera": args.camera,
            "max_episode_steps": args.max_episode_steps,
            "benchmark": "MT1",
            "num_tasks": 1,
        },
    )


def _prepare_memory(args, config) -> None:
    if args.memory_profile != "local":
        raise ValueError(
            "MetaWorld currently uses local memory; pass --memory-profile local"
        )
    Path(config.prompt_vars["memory_dir"]).mkdir(parents=True, exist_ok=True)


def _spawn(args, output_dir):
    if args.env_endpoint:
        return None, make_rpc_client(args.env_endpoint)
    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        "metaworld_env_server",
        [
            args.sim_python,
            "-m",
            "robots.metaworld.env_server",
            "--task",
            args.task,
            "--seed",
            str(args.seed),
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--camera",
            args.camera,
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
            "--video-dir",
            str(output_dir / "videos"),
        ],
        env_overrides={"MUJOCO_GL": "egl"},
        cwd=str(get_repo_root()),
        log_path=str(output_dir / "env_server.log"),
    )
    daemon.start()
    return daemon, make_rpc_client(f"http://{host}:{port}")


def _init_runtime(args, output_dir, dashboard_events, components=None):
    from robots.metaworld.env_client import MetaWorldEnvClient

    if components == set():
        return [], {}
    if components is not None and components != {"env"}:
        raise ValueError("MetaWorld only has the env runtime component")
    daemons = {}
    daemon, rpc = try_spawn_server(
        daemons, dashboard_events, "env", lambda: _spawn(args, output_dir)
    )
    expected = {
        "task": args.task,
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
        "camera": args.camera,
    }

    def connect() -> MetaWorldEnvClient:
        client = MetaWorldEnvClient(rpc, expected_meta=expected)
        write_json_atomic(output_dir / "environment.json", client.get_runtime_info())
        return client

    client = try_wait_server(
        daemons,
        dashboard_events,
        "env",
        rpc,
        daemon,
        120,
        post_fn=connect,
    )
    return list(daemons.values()), {"env_client": client}


def _finalize(context) -> Path:
    return write_json_atomic(
        context.output_dir / "result.json",
        {
            **context.task_desc,
            "success": context.environment_success,
            "environment_result_available": context.environment_success is not None,
            "agent_error_present": context.agent_error is not None,
            "elapsed_s": round(context.elapsed_s, 2),
            "planner": context.planner,
            "model": context.model,
            "reasoning_effort": context.reasoning_effort,
            "max_turns": context.max_turns,
            "planner_timeout_s": context.planner_timeout_s,
            "planner_finish": context.finish_result,
        },
    )


def get_robot_spec() -> RobotSpec:
    """Describe the MetaWorld CLI and environment-only Dashboard runtime."""
    return RobotSpec(
        name="metaworld",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        validate_args=_validate,
        prepare_memory=_prepare_memory,
        finalize_run=_finalize,
        dashboard={
            "task": {
                "command": "/rpent-task",
                "usage": "/rpent-task <task> <seed>",
                "fields": (
                    {"name": "task", "suggestions": tuple(TASKS)},
                    {"name": "seed", "kind": "integer", "minimum": 0},
                ),
                "display": "{task} / seed {seed}",
                "output_slug": "{task}_s{seed}",
            },
            "runtime_components": ({"name": "env", "label": "ENV", "scope": "unique"},),
            "primitives": ("move_to", "set_gripper"),
        },
    )


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
    mode: str = "evaluation",
    attempts_per_session: int = 0,
    state_output_dir: Path | str | None = None,
    **kwargs: Any,
) -> Toolkit:
    """Build a visual toolkit for one fixed episode."""
    from robots.metaworld.toolkit import MetaWorldToolkit

    if mode != "evaluation":
        raise ValueError("MetaWorld currently supports fixed evaluation episodes only")
    return MetaWorldToolkit(
        runtime_kwargs=runtime_kwargs,
        dashboard_events=dashboard_events,
        memory=MemoryManager(root=Path(config.prompt_vars["memory_dir"])),
        state_output_dir=state_output_dir or config.output_dir,
    )
