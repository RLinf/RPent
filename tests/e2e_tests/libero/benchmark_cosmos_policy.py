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

"""Measure warm Cosmos RPC latency and all ten standard LIBERO Spatial tasks."""

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
from rpent.utils.logging import get_logger, init_output_dir
from tests.e2e_tests.common import parse_runtime_args, runtime_phase

logger = get_logger("benchmark_cosmos_policy")
HORIZON = 220


def runtime_args(endpoint: str, task: int, seed: int) -> list[str]:
    """Select the standard simulator and an operator-owned Cosmos service."""
    return [
        "--suite",
        "libero_spatial",
        "--task",
        str(task),
        "--seed",
        str(seed),
        "--libero-type",
        "standard",
        "--max-episode-steps",
        str(HORIZON),
        "--vla-backend",
        "cosmos-policy",
        "--vla-endpoint",
        endpoint,
    ]


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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if not args.endpoint:
        parser.error("--endpoint or RPENT_COSMOS_ENDPOINT is required")
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup must be nonnegative and samples must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("seeds must be unique nonnegative initial-state indices")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    init_output_dir(args.output_dir)
    spec = get_robot_spec()
    report: dict[str, Any] = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("torch", "robosuite", "mujoco")},
        "suite": "libero_spatial",
        "tasks": list(range(10)),
        "seeds": args.seeds,
        "horizon": HORIZON,
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
    config = parse_runtime_args(spec, runtime_args(args.endpoint, 0, args.seeds[0]))
    with runtime_phase(
        spec, config, args.output_dir / "latency", {"env", "vla"}
    ) as runtime:
        for _ in range(args.warmup):
            predict(runtime)
        seconds = [predict(runtime)[1] for _ in range(args.samples)]
    report["latency_seconds"] = seconds
    report["latency"] = latency_summary(seconds)
    save()
    logger.info("Warm RPC latency: %s", report["latency"])

    for task in range(10):
        for seed in args.seeds:
            episode: dict[str, Any] = {
                "task": task,
                "seed": seed,
                "success": False,
                "steps": 0,
                "rpc_seconds": [],
            }
            started = time.perf_counter()
            config = parse_runtime_args(spec, runtime_args(args.endpoint, task, seed))
            try:
                with runtime_phase(
                    spec,
                    config,
                    args.output_dir / f"task-{task}-seed-{seed}",
                    {"env", "vla"},
                ) as runtime:
                    env = runtime["env"]
                    episode["instruction"] = env.get_task_language()
                    while episode["steps"] < HORIZON and not (
                        env.terminated or env.truncated
                    ):
                        actions, elapsed = predict(runtime)
                        episode["rpc_seconds"].append(elapsed)
                        # Stop on the exact native success step, even within a chunk.
                        for action in actions[: HORIZON - episode["steps"]]:
                            env.step(action)
                            episode["steps"] += 1
                            if env.terminated or env.truncated:
                                break
                    episode["success"] = env.terminated
                    episode["truncated"] = env.truncated
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
