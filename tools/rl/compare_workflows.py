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

"""Run four fixed LIBERO panels and winner-only VLM image-size ablations."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from rpent.evaluation import write_json_atomic as atomic
from tools.rl.run_batch import source_identity, stop_process

WORKFLOWS = ("episode", "zap_turn", "zap_episode", "zap_subtask")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def jobs(seed: int = 0) -> list[dict]:
    """Return the fixed forty task rows, independent of scheduling order."""
    return [
        {"environment": "libero", "suite": suite, "task": task, "seed": seed}
        for suite in SUITES
        for task in range(10)
    ]


def wire_statistics(attempt: Path) -> dict:
    """Count actual HTTP requests and reported usage, including truncated replies."""
    requests = sorted((attempt / "api_trace/requests").glob("*/metadata.json"))
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    with_usage = 0
    for metadata in requests:
        usage = None
        try:
            meta = json.loads(metadata.read_text())
            raw = metadata.with_name("response.body").read_text()
            if "text/event-stream" in meta.get("content_type", ""):
                for line in raw.splitlines():
                    if line.startswith("data:") and line[5:].strip() != "[DONE]":
                        usage = json.loads(line[5:]).get("usage") or usage
            else:
                usage = json.loads(raw).get("usage")
        except (OSError, ValueError):
            pass
        if not usage or not all(
            name in usage for name in ("prompt_tokens", "completion_tokens")
        ):
            continue
        with_usage += 1
        totals["input_tokens"] += usage["prompt_tokens"]
        totals["output_tokens"] += usage["completion_tokens"]
        totals["cached_input_tokens"] += (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0
        )
    return {
        "requests": len(requests),
        "requests_with_usage": with_usage,
        "requests_missing_usage": len(requests) - with_usage,
        "observed_usage": totals,
    }


def score(batch: Path, metrics_path: Path | None = None) -> dict:
    """Summarize fixed-denominator outcomes while retaining missing/error rows."""
    state = json.loads((batch / "state.json").read_text())
    rows = json.loads((batch / "outcomes-index.json").read_text())["episodes"]
    errors_path = batch / "infrastructure-errors.json"
    errors = (
        json.loads(errors_path.read_text())["episodes"] if errors_path.exists() else []
    )
    result_rows = []
    for row in rows:
        attempt = Path(row["directory"])
        result_path = attempt / "result.json"
        result = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else {
                "elapsed_s": json.loads((attempt / "EVALUATED.json").read_text())[
                    "status"
                ]["elapsed_seconds"]
            }
        )
        result_rows.append(
            {
                **row,
                "elapsed_s": result["elapsed_s"],
                "native_success_seconds": result.get("native_success_seconds"),
                "context_resets": result.get("context_resets", 0),
                "perception_seconds": result.get("perception_seconds", 0),
                "model_seconds": result.get("model_seconds"),
                "images_sent": result.get("images_sent"),
                "invalid_tool_responses": result.get("invalid_tool_responses", 0),
                "wire": wire_statistics(attempt),
            }
        )
    sam_rows = []
    if metrics_path is not None and metrics_path.exists():
        for line in metrics_path.open():
            row = json.loads(line)
            if (
                state.get("started_at", 0)
                <= row["time"]
                <= state.get("finished_at", float("inf"))
            ):
                sam_rows.append(row)
    sam = {
        "requests": len(sam_rows),
        "errors": sum(row["error"] is not None for row in sam_rows),
        "queue_seconds": sum(row["queue_seconds"] for row in sam_rows),
        "inference_seconds": sum(row["inference_seconds"] for row in sam_rows),
    }
    if metrics_path is None or not metrics_path.exists():
        sam = None

    return {
        "all_attempts_wire": {
            str(attempt.relative_to(batch)): wire_statistics(attempt)
            for attempt in sorted(batch.glob("*/attempt-*"))
            if attempt.is_dir()
        },
        "sam3": sam,
        "expected": 40,
        "evaluated": len(rows),
        "infrastructure_errors": errors,
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / 40,
        "wall_seconds": state["elapsed_seconds"],
        "episodes": result_rows,
    }


def select_winner(reports: dict[str, dict]) -> str:
    """Rank by native successes, batch wall time, then declared workflow order."""
    return min(
        WORKFLOWS,
        key=lambda name: (
            -reports[name]["successes"],
            reports[name]["wall_seconds"],
            WORKFLOWS.index(name),
        ),
    )


def paired_successes(left: dict, right: dict) -> list[dict]:
    """Compare only identical task/seed rows that succeeded in both panels."""

    def key(row):
        task = row["task"]
        return task["suite"], task["task"], task["seed"]

    successful = {key(row): row for row in left["episodes"] if row["success"]}
    return [
        {
            "task": key(row),
            "left_seconds": successful[key(row)]["elapsed_s"],
            "right_seconds": row["elapsed_s"],
        }
        for row in right["episodes"]
        if row["success"] and key(row) in successful
    ]


def run(args) -> None:
    """Run stages serially; resume sealed groups without changing source/settings."""
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    qualification = json.loads(args.qualification.read_text())
    if (
        not qualification.get("passed")
        or qualification["source"] != identity
        or not all(
            qualification.get("checks", {}).get(name)
            for name in ("cpu", "capacity", "native_chain", "image_sizes")
        )
    ):
        raise ValueError("capacity/policy qualification does not match running source")
    settings = {
        "RPENT_MODEL": "openai-chat:qwen3.8-27b",
        "RPENT_MAX_TURNS": "100",
        "RPENT_MAX_EPISODE_STEPS": "10000",
        "RPENT_MAX_TOKENS": "32768",
        "RPENT_CONTEXT_WINDOW": "262144",
        "RPENT_COMPACTION_FRACTION": "0.7",
        "RPENT_PLANNER_TIMEOUT_S": "7200",
        "RPENT_REASONING_EFFORT": "xhigh",
        "RPENT_PRESENCE_PENALTY": "0",
        "RPENT_REPETITION_PENALTY": "1",
        "RPENT_SAM3_ENDPOINT": args.sam3_endpoint,
        "LIBERO_TYPE": "pro",
        "RPENT_RECORD_MODEL_REQUESTS": "1",
    }
    manifest = {
        "source": identity,
        "jobs": jobs(),
        "settings": settings,
        "endpoints": args.endpoints,
        "gpus": args.gpus,
        "per_gpu": args.per_gpu,
    }
    path = root / "comparison-manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("comparison source or settings changed; use a fresh output")
    atomic(path, manifest)
    atomic(root / "jobs.json", jobs())
    reports = {}
    active = None

    def stop(signum, frame):
        if active is not None and active.poll() is None:
            stop_process(active)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    stages = [(name, 1024) for name in WORKFLOWS]
    position = 0
    while position < len(stages):
        if (root / "STOP_REQUESTED.json").exists():
            atomic(root / "state.json", {"status": "stopped", "reports": list(reports)})
            return
        if source_identity() != identity:
            raise ValueError("frozen production source changed during comparison")
        workflow, size = stages[position]
        label = f"{workflow}-{size}"
        batch = root / "formal" / label
        terminal = batch / "state.json"
        if (
            not terminal.exists()
            or json.loads(terminal.read_text()).get("status") != "finished"
        ):
            atomic(
                root / "state.json",
                {
                    "status": "running",
                    "phase": label,
                    "stage": position + 1,
                    "started_at": time.time(),
                    "completed_groups": list(reports),
                },
            )
            flushes = []
            for endpoint in args.endpoints.split(","):
                url = endpoint.removesuffix("/v1").rstrip("/") + "/flush_cache"
                request = urllib.request.Request(
                    url,
                    data=b"{}",
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    flushes.append(
                        {
                            "url": url,
                            "status": response.status,
                            "body": response.read().decode(),
                        }
                    )
            atomic(root / f"cache-flush-{label}.json", flushes)
            command = [
                sys.executable,
                "-m",
                "tools.rl.run_batch",
                "run",
                str(batch),
                "--jobs",
                str(root / "jobs.json"),
                "--gpus",
                args.gpus,
                "--per-gpu",
                str(args.per_gpu),
                "--max-initializing-per-gpu",
                "2",
                "--workflow",
                workflow,
                "--image_size",
                str(size),
                "--start-barrier",
                "--start-interval",
                "0",
                "--endpoints",
                args.endpoints,
                "--infrastructure-retries",
                "1",
            ]
            with (root / f"{label}.log").open("a") as log:
                active = subprocess.Popen(
                    command,
                    env={**os.environ, **settings},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                atomic(
                    root / "collector-process.json",
                    {
                        "pid": active.pid,
                        "boot_id": Path("/proc/sys/kernel/random/boot_id")
                        .read_text()
                        .strip(),
                        "command": command,
                        "phase": label,
                        "started_at": time.time(),
                    },
                )
                while active.poll() is None:
                    if (root / "STOP_REQUESTED.json").exists():
                        stop_process(active)
                        return
                    time.sleep(2)
                if active.returncode:
                    raise RuntimeError(
                        f"{label} collector exited {active.returncode}; see stage log"
                    )
        report = score(batch, args.sam3_metrics)
        if report["evaluated"] + len(report["infrastructure_errors"]) != 40:
            raise ValueError(f"{label} has incomplete fixed-panel coverage")
        reports[label] = report
        atomic(root / f"report-{label}.json", report)
        if position == 3:
            winner = select_winner(
                {name: reports[f"{name}-1024"] for name in WORKFLOWS}
            )
            atomic(
                root / "winner.json",
                {
                    "workflow": winner,
                    "rule": "successes, batch wall time, declared order",
                },
            )
            stages.extend([(winner, 512), (winner, 256)])
        position += 1
    paired = {
        label: paired_successes(reports["episode-1024"], report)
        for label, report in reports.items()
        if label != "episode-1024"
    }
    winner = json.loads((root / "winner.json").read_text())["workflow"]
    paired.update(
        {
            f"{winner}-1024-vs-{size}": paired_successes(
                reports[f"{winner}-1024"], reports[f"{winner}-{size}"]
            )
            for size in (512, 256)
        }
    )
    atomic(
        root / "final-report.json",
        {
            "winner": winner,
            "groups": reports,
            "paired_success_times": paired,
            "formal_episodes": 240,
            "sampling": settings,
            "completed_at": time.time(),
        },
    )
    atomic(
        root / "state.json",
        {
            "status": "complete",
            "formal_episodes": 240,
            "winner": winner,
            "completed_at": time.time(),
        },
    )


def main() -> None:
    """Parse portable service locations and the exact-source qualification proof."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--endpoints", required=True)
    parser.add_argument("--sam3-endpoint", required=True)
    parser.add_argument("--sam3-metrics", type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--per-gpu", type=int, default=20)
    args = parser.parse_args()
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / ".comparison.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)
    except Exception as exc:
        atomic(
            args.output / "controller-error.json",
            {"error": f"{type(exc).__name__}: {exc}", "time": time.time()},
        )
        raise


if __name__ == "__main__":
    main()
