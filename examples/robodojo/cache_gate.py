#!/usr/bin/env python3
# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Gate a full RoboDojo run on cache usage from recorded multimodal decisions."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

from rpent.benchmarks.robodojo import RoboDojoEmbodiedClient

DEFAULT_TASKS = ("align_blocks", "classify_objects", "general_pickup")
MIN_TASKS = 3
MIN_DECISIONS_PER_TASK = 4


class ReplayError(RuntimeError):
    """Record the provider status without writing its response body."""

    def __init__(self, message: str, *, status_code: int | None = None, **_: Any):
        super().__init__(message)
        self.status_code = status_code


def recorded_messages(
    record_path: Path, demo_messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rebuild a decision's original system, demo, and three JPEG inputs."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    stem = record_path.stem
    current = []
    for camera in record["cameras"]:
        jpeg = record_path.with_name(f"{stem}_{camera}.jpg").read_bytes()
        current.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
                },
            }
        )
    current.append({"type": "text", "text": record["prompt"]})
    return [
        {"role": "system", "content": record["system_prompt"]},
        *demo_messages,
        {"role": "user", "content": current},
    ]


def decide_gate(samples: list[dict[str, Any]], expected: int, threshold: float) -> dict:
    """Use token-weighted provider usage and require every replay to finish."""
    inp = sum(int(row.get("input_tokens") or 0) for row in samples)
    cached = sum(int(row.get("cached_input_tokens") or 0) for row in samples)
    written = sum(int(row.get("cache_write_tokens") or 0) for row in samples)
    if cached > inp:
        raise ValueError("cached input exceeds total input")
    rate = cached / inp if inp else None
    tasks = sorted({row["task"] for row in samples})
    task_usage = {}
    for task in tasks:
        rows = [row for row in samples if row["task"] == task]
        task_input = sum(row["input_tokens"] for row in rows)
        task_cached = sum(row["cached_input_tokens"] for row in rows)
        task_usage[task] = {
            "decisions": len(rows),
            "input_tokens": task_input,
            "cached_input_tokens": task_cached,
            "cache_hit_rate": task_cached / task_input if task_input else None,
        }
    representative = len(tasks) >= MIN_TASKS and all(
        entry["decisions"] >= MIN_DECISIONS_PER_TASK for entry in task_usage.values()
    )
    return {
        "expected_decisions": expected,
        "completed_decisions": len(samples),
        "input_tokens": inp,
        "cached_input_tokens": cached,
        "cache_write_tokens": written,
        "cache_hit_rate": rate,
        "threshold": threshold,
        "representative": representative,
        "qualified": bool(
            representative
            and len(samples) == expected
            and rate is not None
            and rate > threshold
        ),
        "task_usage": task_usage,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--decisions-per-task", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()
    if args.decisions_per_task < 1 or not 0 < args.threshold < 1:
        parser.error(
            "decisions-per-task must be positive and threshold between 0 and 1"
        )
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("tasks must be distinct")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty to avoid mixing cache probes")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = args.source_experiment.resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    policy_root = source / "workspace/XPolicyLab/policy"
    sys.path.insert(0, str(policy_root))
    from RoboDojo_EmbodiedAgent.icl_demos import build_demo, demo_messages

    os.environ["OPENAI_API_KEY"] = args.key_file.read_text(encoding="utf-8").strip()
    expected = len(args.tasks) * args.decisions_per_task
    samples: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    for task in args.tasks:
        episode_roots = list(
            (source / "runtime/vlm_logs").glob(
                f"{args.run_prefix}*-{task}-a1/episode_001"
            )
        )
        if len(episode_roots) != 1:
            raise RuntimeError(f"expected one recorded episode for {task}")
        records = sorted(episode_roots[0].glob("decision_*.json"))
        if len(records) < args.decisions_per_task:
            raise RuntimeError(f"not enough recorded decisions for {task}")
        demo = build_demo(source / "workspace/RoboDawn-demos" / task)
        client = RoboDojoEmbodiedClient(
            model=manifest["model"],
            base_url=manifest["api_base_url"],
            output_dir=args.output_dir / task,
            error_type=ReplayError,
        )
        for record_path in records[: args.decisions_per_task]:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            try:
                response = client.complete(
                    recorded_messages(record_path, demo_messages(demo)),
                    max_tokens=record["config"]["max_tokens"],
                    reasoning_effort=record["config"]["reasoning_effort"],
                )
            except ReplayError as error:
                failure = {
                    "task": task,
                    "decision": record_path.stem,
                    "error_type": type(error).__name__,
                    "status_code": error.status_code,
                }
                break
            usage = response["usage"]
            details = usage.get("prompt_tokens_details") or {}
            samples.append(
                {
                    "task": task,
                    "decision": record_path.stem,
                    "input_tokens": int(usage["prompt_tokens"]),
                    "cached_input_tokens": int(details.get("cached_tokens") or 0),
                    "cache_write_tokens": int(details.get("cache_write_tokens") or 0),
                    "output_tokens": int(usage["completion_tokens"]),
                    "model_requests": int(response["attempts"]),
                }
            )
            current = decide_gate(samples, expected, args.threshold)
            print(
                f"{task} {record_path.stem}: "
                f"cached={current['cached_input_tokens']}/"
                f"{current['input_tokens']} ({current['cache_hit_rate']:.1%})",
                flush=True,
            )
        if failure is not None:
            break

    result = decide_gate(samples, expected, args.threshold)
    result["tasks"] = args.tasks
    result["decisions_per_task"] = args.decisions_per_task
    result["failure"] = failure
    (args.output_dir / "cache-gate.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("QUALIFIED" if result["qualified"] else "NOT QUALIFIED")
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
