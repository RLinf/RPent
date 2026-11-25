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

"""VLABench command-line options and run configuration."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from robots.vlabench.sim.contracts import (
    DEFAULT_EPISODE_STEPS,
    DEFAULT_TRACK,
    VLABENCH_ROBOT,
)
from rpent.evaluation import RunFinalizationContext, write_json_atomic
from rpent.robots.robot_spec import RunConfig
from rpent.utils.config import add_cuda_device_arg, get_memory_dir, get_repo_root


def add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """Add the VLABench task, resource and service options."""
    required = not use_dashboard
    parser.add_argument(
        "--task-name",
        required=required,
        help="VLABench task, such as select_painting.",
    )
    parser.add_argument(
        "--track",
        default=DEFAULT_TRACK,
        help="Official evaluation track that lists the task's episodes.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode of the task within the selected track.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Run seed.")
    parser.add_argument(
        "--eval-unseen",
        action="store_true",
        help="Evaluate on the unseen-object split.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help=(
            f"Native control-step budget (default: {DEFAULT_EPISODE_STEPS}). "
            "This is a custom run budget; the task's official limit is still "
            "reported for provenance."
        ),
    )
    parser.add_argument(
        "--vlabench-root",
        default=os.environ.get("VLABENCH_ROOT"),
        help=(
            "Inner VLABench package directory that contains configs/. "
            "Defaults to VLABENCH_ROOT."
        ),
    )
    parser.add_argument(
        "--vlabench-asset-root",
        default=os.environ.get("VLABENCH_ASSET_ROOT"),
        help=(
            "External asset directory that contains obj/ and scenes/. "
            "Defaults to VLABENCH_ASSET_ROOT."
        ),
    )
    parser.add_argument(
        "--vlabench-python",
        default=os.environ.get("VLABENCH_PYTHON"),
        help=(
            "Python interpreter of the VLABench environment that runs the "
            "environment server. Defaults to VLABENCH_PYTHON, then the "
            "interpreter running RPent."
        ),
    )
    parser.add_argument(
        "--env-endpoint",
        default=None,
        help="Connect to a running VLABench environment server instead of launching one.",
    )
    parser.add_argument(
        "--env-cuda-device",
        default=None,
        help=(
            "CUDA_VISIBLE_DEVICES for the local environment server. "
            "Defaults to --cuda-device, then inherits CUDA_VISIBLE_DEVICES."
        ),
    )
    add_cuda_device_arg(parser, device_type=str)


def validate_args(args: argparse.Namespace) -> None:
    """Reject incomplete options before any service starts.

    Raises:
        ValueError: If a launched server lacks its resource roots, numeric
            bounds are invalid, or an unsupported memory source is requested.
    """
    if getattr(args, "memory_repo", None) is not None:
        raise ValueError(
            "--memory-repo is not supported for VLABench; use --memory-profile "
            "local with --memory-dir"
        )
    episode_index = getattr(args, "episode_index", None)
    if episode_index is not None and episode_index < 0:
        raise ValueError("--episode-index must be nonnegative")
    seed = getattr(args, "seed", None)
    if seed is not None and seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.max_episode_steps is not None and args.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps must be a positive integer")
    if args.env_endpoint is not None:
        return
    missing = [
        flag
        for flag, value in (
            ("--vlabench-root", args.vlabench_root),
            ("--vlabench-asset-root", args.vlabench_asset_root),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"{' and '.join(missing)} required to launch the environment server"
        )


def parse_config(args: argparse.Namespace) -> RunConfig:
    """Derive the run identifiers, prompt variables and task description."""
    if not args.task_name:
        raise ValueError("--task-name is required")
    recipe_tag = f"vlabench_{args.task_name}_e{args.episode_index}_s{args.seed}"
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{recipe_tag}"
    memory_dir = (
        Path(args.memory_dir).expanduser().resolve()
        if args.memory_dir
        else get_memory_dir("vlabench")
    )
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=Path(output_dir),
        prompt_vars={
            "task_name": args.task_name,
            "track": args.track,
            "episode_index": args.episode_index,
            "seed": args.seed,
            "mode": "eval",
            "memory_dir": str(memory_dir),
            "recipe_tag": recipe_tag,
        },
        task_desc={
            "env": "vlabench",
            "robot": VLABENCH_ROBOT,
            "task_name": args.task_name,
            "track": args.track,
            "episode_index": args.episode_index,
            "seed": args.seed,
            "eval_unseen": args.eval_unseen,
            "instruction": None,
            "max_episode_steps": args.max_episode_steps or DEFAULT_EPISODE_STEPS,
        },
    )


def prepare_memory(args: argparse.Namespace, config: RunConfig) -> None:
    """Create the run's memory root; VLABench has no published corpus to sync."""
    Path(config.prompt_vars["memory_dir"]).mkdir(parents=True, exist_ok=True)


def finalize_run(context: RunFinalizationContext) -> Path:
    """Write VLABench task identity and environment-authoritative run results."""
    record = {
        "robot": context.robot_name,
        "task": dict(context.task_desc),
        "environment_success": context.environment_success,
        "environment_result_available": context.environment_success is not None,
        "agent_error": context.agent_error,
        "elapsed_s": round(context.elapsed_s, 1),
        "planner": {
            "backend": context.planner,
            "model": context.model,
            "reasoning_effort": context.reasoning_effort,
            "max_turns": context.max_turns,
        },
        "planner_timeout_s": context.planner_timeout_s,
        "finish": (
            dict(context.finish_result) if context.finish_result is not None else None
        ),
        "stats": dict(context.stats),
        "metrics": dict(context.environment_metrics),
    }
    return write_json_atomic(context.output_dir / "result.json", record)
