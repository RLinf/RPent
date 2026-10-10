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

"""Measure Cosmos RPC latency and native success on a selected LIBERO suite."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from robots.libero.robot_spec import get_robot_spec
from robots.libero.suites import LIBERO_SUITE_NAMES, suite_variant
from rpent.utils.logging import get_logger, init_output_dir
from tests.e2e_tests.common import parse_runtime_args, runtime_phase

logger = get_logger("benchmark_cosmos_policy")
HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def runtime_args(
    endpoint: str, suite: str, task: int, seed: int, horizon: int
) -> list[str]:
    """Select the matching simulator and an operator-owned Cosmos service."""
    return [
        "--suite",
        suite,
        "--task",
        str(task),
        "--seed",
        str(seed),
        "--libero-type",
        suite_variant(suite),
        "--max-episode-steps",
        str(horizon),
        "--wam-backend",
        "cosmos-policy",
        "--wam-endpoint",
        endpoint,
    ]


def save_scene(runtime: dict[str, Any], path: Path) -> None:
    """Save the policy's external and wrist views in their upright orientation."""
    from PIL import Image

    raw = runtime["env"].raw_obs()
    views = [
        np.flipud(raw[key]) for key in ("agentview_image", "robot0_eye_in_hand_image")
    ]
    Image.fromarray(np.concatenate(views, axis=1)).save(path)


def predict(runtime: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Time a complete RPC, excluding simulator observation acquisition."""
    env = runtime["env"]
    raw = {**env.raw_obs(), "task_descriptions": env.get_task_language()}
    start = time.perf_counter()
    actions = runtime["model"].predict(raw)
    return actions, time.perf_counter() - start


def latency_summary(seconds: list[float]) -> dict[str, float]:
    """Summarize client round trips for sequential 16-action predictions."""
    values = np.asarray(seconds)
    return {
        "mean_ms": float(values.mean() * 1000),
        "p50_ms": float(np.percentile(values, 50) * 1000),
        "p95_ms": float(np.percentile(values, 95) * 1000),
        "p99_ms": float(np.percentile(values, 99) * 1000),
        "chunks_per_second": float(1 / values.mean()),
    }


def main() -> None:
    """Write raw timings and every episode outcome to a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.getenv("RPENT_COSMOS_ENDPOINT"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--suite",
        choices=tuple(suite for suite in LIBERO_SUITE_NAMES if suite != "libero_90"),
        default="libero_spatial",
    )
    parser.add_argument("--tasks", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if not args.endpoint:
        parser.error("--endpoint or RPENT_COSMOS_ENDPOINT is required")
    if args.warmup < 0 or args.samples < 0:
        parser.error("warmup and samples must be nonnegative; zero skips latency calls")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("seeds must be unique nonnegative initial-state indices")
    if len(set(args.tasks)) != len(args.tasks) or any(
        t not in range(10) for t in args.tasks
    ):
        parser.error("tasks must be unique indices between 0 and 9")
    base_suite = (
        args.suite.rsplit("_", 1)[0]
        if suite_variant(args.suite) == "pro"
        else args.suite
    )
    horizon = args.horizon if args.horizon is not None else HORIZONS[base_suite]
    if horizon <= 0:
        parser.error("horizon must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    init_output_dir(args.output_dir)
    spec = get_robot_spec()
    report: dict[str, Any] = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("torch", "robosuite", "mujoco")},
        "suite": args.suite,
        "libero_type": suite_variant(args.suite),
        "tasks": args.tasks,
        "seeds": args.seeds,
        "horizon": horizon,
        "total_output_tokens": 0,
        "endpoint": args.endpoint,
        "warmup_calls": args.warmup,
        "measured_calls": args.samples,
        "latency_definition": "client RPC wall time; excludes observation acquisition and env steps",
        "episodes": [],
    }

    def save() -> None:
        (args.output_dir / "results.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    save()
    if args.warmup or args.samples:
        config = parse_runtime_args(
            spec,
            runtime_args(
                args.endpoint, args.suite, args.tasks[0], args.seeds[0], horizon
            ),
        )
        with runtime_phase(
            spec, config, args.output_dir / "latency", {"env", "wam"}
        ) as runtime:
            for _ in range(args.warmup):
                predict(runtime)
            seconds = [predict(runtime)[1] for _ in range(args.samples)]
        report["latency_seconds"] = seconds
        if seconds:
            report["latency"] = latency_summary(seconds)
            logger.info("Warm RPC latency: %s", report["latency"])
        save()

    for task in args.tasks:
        for seed in args.seeds:
            episode: dict[str, Any] = {
                "task": task,
                "seed": seed,
                "success": False,
                "steps": 0,
                "rpc_seconds": [],
            }
            started = time.perf_counter()
            config = parse_runtime_args(
                spec, runtime_args(args.endpoint, args.suite, task, seed, horizon)
            )
            episode_dir = args.output_dir / f"task-{task}-seed-{seed}"
            try:
                with runtime_phase(
                    spec,
                    config,
                    episode_dir,
                    {"env", "wam"},
                ) as runtime:
                    env = runtime["env"]
                    episode["instruction"] = env.get_task_language()
                    save_scene(runtime, episode_dir / "initial.png")
                    active_started = time.perf_counter()
                    while episode["steps"] < horizon and not (
                        env.terminated or env.truncated
                    ):
                        actions, elapsed = predict(runtime)
                        episode["rpc_seconds"].append(elapsed)
                        # Stop on the exact native success step, even within a chunk.
                        for action in actions[: horizon - episode["steps"]]:
                            env.step(action)
                            episode["steps"] += 1
                            if env.terminated or env.truncated:
                                break
                    episode["success"] = env.terminated
                    episode["truncated"] = env.truncated
                    episode["control_seconds"] = time.perf_counter() - active_started
                    save_scene(runtime, episode_dir / "final.png")
            except Exception as exc:
                episode["success"] = False
                episode["error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("Episode failed: task=%s seed=%s", task, seed)
            episode["wall_seconds_including_startup"] = time.perf_counter() - started
            report["episodes"].append(episode)
            save()
            logger.info(
                "task=%s seed=%s success=%s steps=%s",
                task,
                seed,
                episode["success"],
                episode["steps"],
            )

    episodes = report["episodes"]
    report["successes"] = sum(e["success"] for e in episodes)
    report["errors"] = sum("error" in e for e in episodes)
    report["success_rate"] = report["successes"] / len(episodes)
    save()
    logger.info(
        "Success: %s/%s; errors: %s",
        report["successes"],
        len(episodes),
        report["errors"],
    )
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
