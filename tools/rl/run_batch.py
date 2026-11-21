#!/usr/bin/env python3
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

"""Collect audited episodes with bounded concurrency and resumable attempts."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from robots.rl.prompt_bundle import PROMPT_VARIANTS, WORKFLOWS, validate_workflow
from rpent.evaluation import write_json_atomic as atomic
from rpent.evaluation.trajectory import audit_episode

EVAL_DIR = Path(__file__).resolve().parent
CONFIGS = ("adaptive",)
LOCK = threading.Lock()
STOP = threading.Event()


# These flags are declared once in run.sh; batch CLI values override that argv.
RUN_FLAGS = (
    "--sam3-timeout-s",
    "--workflow",
    "--world-cord",
    "--perception-mode",
    "--max-observe-calls",
    "--observe-concurrency",
    "--action-mode",
    "--max-actions-per-call",
    "--reasoning-history",
    "--image-history",
    "--image_size",
    "--max-turns",
    "--max-tokens",
    "--context-window",
    "--compaction-fraction",
    "--compaction-max-messages",
    "--compaction-keep-messages",
    "--reasoning-effort",
    "--temperature",
    "--top-p",
    "--top-k",
    "--min-p",
    "--presence-penalty",
    "--repetition-penalty",
    "--max-episode-steps",
    "--planner-timeout-s",
    "--observation-mode",
    "--prompt",
)


def resolve_run_options(args: argparse.Namespace) -> None:
    """Resolve editable launcher defaults before pinning or starting a batch."""
    defaults = subprocess.run(
        ["bash", str(EVAL_DIR / "run.sh"), "--print-defaults"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    _, collection = build_parser()
    base = collection.parse_args(
        ["unused", "--jobs", "unused", "--gpus", "0", *defaults]
    )
    for flag in RUN_FLAGS:
        name = flag[2:].replace("-", "_")
        if getattr(args, name, None) is None:
            setattr(args, name, getattr(base, name))
    if getattr(args, "no_thinking", None) is None:
        args.no_thinking = base.no_thinking
    from rpent.planner.api_loop import sliding_window_compaction
    from rpent.planner.qwen import qwen_model_settings

    sliding_window_compaction(
        context_window=args.context_window,
        max_fraction=args.compaction_fraction,
        max_messages=args.compaction_max_messages,
        keep_messages=args.compaction_keep_messages,
    )()
    qwen_model_settings(
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        thinking=not args.no_thinking,
        **{
            name: getattr(args, name)
            for name in (
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "repetition_penalty",
            )
        },
    )
    from robots.rl.contracts import validate_perception_options

    validate_perception_options(
        args.workflow,
        args.perception_mode,
        args.max_observe_calls,
        args.observe_concurrency,
    )
    import math

    if not math.isfinite(args.sam3_timeout_s) or args.sam3_timeout_s <= 0:
        raise ValueError("sam3-timeout-s must be finite and positive")
    if min(args.max_turns, args.max_episode_steps, args.planner_timeout_s) < 1:
        raise ValueError("Request/step budgets and planner timeout must be positive")


def episode_options(args: argparse.Namespace) -> list[str]:
    """Pass every resolved value explicitly so a batch cannot drift mid-run."""
    result = []
    for flag in RUN_FLAGS:
        result += [flag, str(getattr(args, flag[2:].replace("-", "_")))]
    result.append("--no-thinking" if args.no_thinking else "--thinking")
    return result


def source_identity() -> dict:
    """Pin production code and prompts, including uncommitted development changes."""
    root = EVAL_DIR.parent.parent
    files = [
        *root.joinpath("rpent").rglob("*.py"),
        *root.joinpath("robots").rglob("*.py"),
        *root.joinpath("robots/rl/prompts").rglob("*.md"),
        *root.joinpath("tools/rl").glob("*.py"),
        *root.joinpath("tools/rl").glob("*.sh"),
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }


def dependency_identity() -> dict:
    """Record the planner-side versions without importing heavyweight runtimes."""
    result = {}
    for name in (
        "rpent",
        "pydantic-ai-slim",
        "pydantic-ai-harness",
        "openai",
        "httpx",
        "numpy",
        "pillow",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def read_jobs(path: Path, seeds: list[int] | None) -> list[dict]:
    """Read portable task JSON or the existing four-column LIBERO panel."""
    if path.suffix == ".json":
        jobs = json.loads(path.read_text())
        if not isinstance(jobs, list):
            raise ValueError("task manifest must be a JSON list")
    else:
        jobs = []
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            mode, suite, task, seed = line.split()
            jobs.append(
                {
                    "environment": "libero",
                    "observation_mode": mode,
                    "suite": suite,
                    "task": int(task),
                    "seed": int(seed),
                }
            )
    normalized = []
    for original in jobs:
        row = {"environment": "libero", "observation_mode": "adaptive", **original}
        if (
            row["environment"] not in ("libero", "robocasa", "robotwin")
            or row["observation_mode"] != "adaptive"
        ):
            raise ValueError(f"invalid environment/observation mode: {row}")
        identifiers = (
            [row.get("suite")]
            if row["environment"] == "libero"
            else [row.get("task_name")]
        )
        if any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name)
            for name in identifiers
        ):
            raise ValueError(f"invalid task identifier: {row}")
        if row["environment"] == "libero" and (
            not isinstance(row.get("task"), int) or row["task"] < 0
        ):
            raise ValueError(f"invalid task index: {row}")
        for seed in seeds if seeds is not None else [row.get("seed")]:
            if not isinstance(seed, int) or seed < 0:
                raise ValueError(f"invalid seed: {row}")
            item = {**row, "seed": seed}
            if item not in normalized:
                normalized.append(item)
    return normalized


def job_name(job: dict, prompt: str, workflow: str) -> str:
    """Name one native task/seed without collisions between workflows."""
    task = (
        f"{job['suite']}_t{job['task']}"
        if job["environment"] == "libero"
        else f"{job['environment']}_{job['task_name']}"
    )
    return f"{task}_s{job['seed']}_adaptive-{prompt}-{workflow}"


def job_arguments(job: dict) -> list[str]:
    """Translate the manifest fields into the shared CLI's task interface."""
    result = ["--simulator", job["environment"], "--seed", str(job["seed"])]
    if job["environment"] == "libero":
        result += ["--suite", job["suite"], "--task", str(job["task"])]
    else:
        result += ["--task-name", job["task_name"]]
    for field in ("split", "task_config", "assets_path", "sim_python"):
        if field in job:
            result += ["--" + field.replace("_", "-"), str(job[field])]
    return result


def seed_values(value: str) -> list[int]:
    """Parse distinct nonnegative seeds and inclusive ranges."""
    result = []
    for part in value.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            raise argparse.ArgumentTypeError("Expected seeds such as 0,3-66")
        bounds = part.split("-")
        start, end = int(bounds[0]), int(bounds[-1])
        if start < 0 or end < start:
            raise argparse.ArgumentTypeError(
                "Seeds must be nonnegative, ordered ranges"
            )
        result.extend(range(start, end + 1))
    return list(dict.fromkeys(result))


def completed(output: Path) -> dict | None:
    """Return an audited sealed attempt, ignoring partial or damaged attempts."""
    for path in sorted(output.glob("attempt-*/COMPLETED.json")):
        try:
            result, _ = audit_episode(path.parent, verify_assets=True)
            return {
                "directory": str(path.parent.resolve()),
                "success": result["success"],
                "task": result["task"],
                "stats": result["stats"],
            }
        except (OSError, ValueError, KeyError, StopIteration):
            continue
    return None


def terminal_failure(
    attempt: Path, *, deadline: bool = False, job: dict | None = None
) -> dict | None:
    """Keep normal model/budget failures as evaluations, outside training data."""
    try:
        result_path = attempt / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        transcripts = sorted(attempt.glob("transcript_*.json"))
        transcript = json.loads(transcripts[-1].read_text()) if transcripts else {}
        kind = transcript.get("error_kind")
        if deadline:
            kind = "episode_timeout"
        elif (
            result.get("invalid_tool_responses", 0) > 0
            and not result.get("agent_error")
            and not transcript.get("error")
            and not kind
        ):
            # Rejected replies remain in the lossless trace. A malformed model
            # argument can make that trace unsuitable for training, but is not
            # a reason to rerun an otherwise completed benchmark episode.
            kind = "invalid_tool_reply"
        if kind not in {
            "response_truncated",
            "episode_timeout",
            "context_limit",
            "tool_arguments",
            "tool_name",
            "invalid_tool_reply",
        }:
            return None
        state_path = attempt / "states.json"
        states = (
            json.loads(state_path.read_text())["steps"] if state_path.exists() else []
        )
        if (
            not states
            or deadline
            and not any((attempt / "api_trace/native").glob("*.json"))
        ):
            # A stuck simulator startup is infrastructure failure, not model budget exhaustion.
            return None
        success = bool(states and states[-1]["terminated"])
        if result and bool(result.get("success")) != success:
            return None
        return {
            "directory": str(attempt.resolve()),
            "task": result.get("task", job),
            "success": success,
            "stats": result.get("stats", {}),
            "training_usable": False,
            "termination_reason": kind,
            "error": result.get("agent_error") or transcript.get("error") or kind,
        }
    except (OSError, ValueError, KeyError):
        return None


def evaluated(output: Path) -> dict | None:
    """Resume a completed normal failure without silently retrying its task."""
    for marker in sorted(output.glob("attempt-*/EVALUATED.json")):
        try:
            value = json.loads(marker.read_text())
            if all(
                hashlib.sha256((marker.parent / name).read_bytes()).hexdigest()
                == digest
                for name, digest in value["hashes"].items()
            ):
                return value["outcome"]
        except (OSError, ValueError, KeyError):
            continue
    return None


def interrupt(signum: int, frame: object) -> None:
    """Ask workers to stop their own process groups."""
    STOP.set()


def stop_process(process: subprocess.Popen) -> None:
    """Terminate the owned process group, escalating if graceful close stalls."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run(args: argparse.Namespace) -> None:
    """Collect the manifest without mixing configurations or repeating sealed jobs."""
    batch = Path(args.batch_dir).resolve()
    batch.mkdir(parents=True, exist_ok=True)
    with (batch / ".collector.lock").open("a") as ownership:
        try:
            fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"A collector already owns {batch}") from error
        collect(args, batch)


def collect(args: argparse.Namespace, batch: Path) -> None:
    """Run one exclusively owned batch."""
    STOP.clear()
    resolve_run_options(args)
    if (
        args.infrastructure_retries < 0
        or args.episode_timeout <= 0
        or args.start_interval < 0
        or args.max_initializing_per_gpu < 1
    ):
        raise ValueError(
            "Retry count and start interval must be nonnegative; timeout must be positive"
        )
    jobs = read_jobs(Path(args.jobs), args.seeds)
    for job in jobs:
        validate_workflow(job["environment"], args.workflow)
        from robots.rl.contracts import (
            validate_coordinates,
            validate_image_size,
            validate_options,
        )

        world_cord = getattr(args, "world_cord", "base")
        validate_coordinates(job["environment"], world_cord)
        validate_image_size(
            job["environment"], world_cord, getattr(args, "image_size", 1024)
        )
        validate_options(
            job["environment"],
            args.workflow,
            getattr(args, "action_mode", "single"),
            getattr(args, "max_actions_per_call", 3),
            getattr(args, "reasoning_history", -1),
            getattr(args, "image_history", -1),
            world_cord=world_cord,
        )
    identity = {
        "jobs": jobs,
        "prompt": args.prompt,
        "workflow": args.workflow,
        "world_cord": args.world_cord,
        "image_size": getattr(args, "image_size", 1024),
        "thinking": not getattr(args, "no_thinking", False),
        "run_options": {
            flag: getattr(args, flag[2:].replace("-", "_")) for flag in RUN_FLAGS
        },
        "action_mode": getattr(args, "action_mode", "single"),
        "max_actions_per_call": getattr(args, "max_actions_per_call", 3),
        "reasoning_history": getattr(args, "reasoning_history", -1),
        "image_history": getattr(args, "image_history", -1),
        "model": os.environ["RPENT_MODEL"],
        "source": source_identity(),
        "dependencies": dependency_identity(),
        "parameters": {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith("RPENT_")
            and k not in ("RPENT_BASE_URL", "RPENT_OUTPUT_DIR", "RPENT_SAM3_ENDPOINT")
        },
        "launcher_sha256": hashlib.sha256(
            (EVAL_DIR / "run.sh").read_bytes() + (EVAL_DIR / "planner.sh").read_bytes()
        ).hexdigest(),
    }
    # A directory cannot silently mix seeds, prompts or generation configurations.
    manifest = batch / "manifest.json"
    canonical = json.loads(json.dumps(identity))
    if manifest.exists() and json.loads(manifest.read_text()) != canonical:
        raise ValueError(
            "Resume configuration differs from manifest.json; use a new output directory"
        )
    atomic(manifest, canonical)
    endpoints = (
        args.endpoints.split(",") if args.endpoints else [os.environ["RPENT_BASE_URL"]]
    )
    slots = [gpu for gpu in args.gpus.split(",") for _ in range(args.per_gpu)]
    if not slots or args.per_gpu < 1:
        raise ValueError("At least one environment slot is required")
    gpu_names = list(dict.fromkeys(args.gpus.split(",")))
    pending = {gpu: queue.Queue() for gpu in gpu_names}
    initializing = {
        gpu: threading.Semaphore(args.max_initializing_per_gpu) for gpu in gpu_names
    }
    records = []
    outcomes = []
    exhausted = []
    current_attempts = {}
    failed_before_gate = threading.Event()
    gate = (
        batch / f"start-gate-{time.time_ns()}.json"
        if getattr(args, "start_barrier", False)
        else None
    )
    for index, job in enumerate(jobs):
        output = batch / job_name(job, args.prompt, args.workflow)
        row = completed(output)
        if row:
            records.append(row)
            outcomes.append({**row, "training_usable": True})
        elif row := evaluated(output):
            outcomes.append(row)
        else:
            gpu = str(job.get("gpu", gpu_names[index % len(gpu_names)]))
            if gpu not in pending:
                raise ValueError(f"task GPU {gpu} is outside --gpus")
            pending[gpu].put((job, output, index))
    atomic(batch / "valid-index.json", {"episodes": records})
    atomic(batch / "outcomes-index.json", {"episodes": outcomes})
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    waiting = sum(q.qsize() for q in pending.values())
    if gate and waiting > len(slots):
        raise ValueError(
            "start barrier requires one resident slot for every pending job"
        )
    started = time.time()

    def worker(index: int, gpu: str) -> None:
        if STOP.wait(index * args.start_interval):
            return
        while not STOP.is_set():
            try:
                job, output, job_index = pending[gpu].get_nowait()
            except queue.Empty:
                return
            output.mkdir(exist_ok=True)
            prior_failures = len(list(output.glob("attempt-*/INVALID.json")))
            for retry in range(prior_failures, args.infrastructure_retries + 1):
                if STOP.is_set():
                    return
                number = (
                    max(
                        [int(p.name.split("-")[-1]) for p in output.glob("attempt-*")]
                        + [0]
                    )
                    + 1
                )
                attempt = output / f"attempt-{number:03d}"
                attempt.mkdir()
                with LOCK:
                    current_attempts[str(output)] = attempt
                command = [
                    "bash",
                    str(EVAL_DIR / "run.sh"),
                    *job_arguments(job),
                    "--cuda-device",
                    gpu,
                    *episode_options(args),
                ]
                environment = {
                    **os.environ,
                    "RPENT_OUTPUT_DIR": str(attempt),
                    "RPENT_BASE_URL": endpoints[job_index % len(endpoints)],
                    **({"RPENT_START_GATE": str(gate)} if gate else {}),
                }
                while not initializing[gpu].acquire(timeout=0.5):
                    if STOP.is_set():
                        return
                startup_held = True
                begin = time.time()
                deadline = False
                with (attempt / "console.log").open("w") as log:
                    try:
                        process = subprocess.Popen(
                            command,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        while process.poll() is None:
                            if startup_held and (attempt / "states.json").exists():
                                initializing[gpu].release()
                                startup_held = False
                            if (
                                STOP.is_set()
                                or time.time() - begin >= args.episode_timeout
                            ):
                                deadline = not STOP.is_set()
                                stop_process(process)
                                break
                            time.sleep(0.5)
                        code = process.returncode
                    finally:
                        if startup_held:
                            initializing[gpu].release()
                status = {
                    "exit_code": code,
                    "elapsed_seconds": time.time() - begin,
                    "retry": retry,
                    "gpu": gpu,
                    "endpoint": environment["RPENT_BASE_URL"],
                }
                if STOP.is_set():
                    atomic(attempt / "INTERRUPTED.json", status)
                    return
                if row := terminal_failure(attempt, deadline=deadline, job=job):
                    names = [
                        "result.json",
                        "states.json",
                        "rollout.json",
                        *[path.name for path in attempt.glob("transcript_*.json")],
                        *[
                            str(path.relative_to(attempt))
                            for path in (attempt / "api_trace").rglob("*")
                            if path.is_file()
                        ],
                    ]
                    hashes = {
                        name: hashlib.sha256((attempt / name).read_bytes()).hexdigest()
                        for name in names
                        if (attempt / name).exists()
                    }
                    atomic(
                        attempt / "EVALUATED.json",
                        {"outcome": row, "status": status, "hashes": hashes},
                    )
                    with LOCK:
                        outcomes.append(row)
                        atomic(batch / "outcomes-index.json", {"episodes": outcomes})
                    print(
                        f"Evaluated {output.name}: {row['termination_reason']}; excluded from training",
                        flush=True,
                    )
                    break
                try:
                    if code:
                        raise ValueError(f"Worker exited {code}")
                    result, turns = audit_episode(attempt, verify_assets=True)
                    status.update(success=result["success"], turns=len(turns))
                    atomic(attempt / "COMPLETED.json", status)
                    row = completed(output)
                    if row is None:
                        raise ValueError("Sealed episode failed its second audit")
                    with LOCK:
                        records.append(row)
                        outcomes.append({**row, "training_usable": True})
                        atomic(batch / "valid-index.json", {"episodes": records})
                        atomic(batch / "outcomes-index.json", {"episodes": outcomes})
                    print(
                        f"Completed {output.name}: success={result['success']}",
                        flush=True,
                    )
                    break
                except (OSError, ValueError, KeyError, StopIteration) as error:
                    atomic(attempt / "INVALID.json", {**status, "error": str(error)})
                    print(f"Invalid {output.name}: {error}", flush=True)

            else:
                with LOCK:
                    exhausted.append(
                        {
                            "task": job,
                            "path": str(output),
                            "success": False,
                            "termination_reason": "infrastructure_error",
                        }
                    )
                    atomic(
                        batch / "infrastructure-errors.json", {"episodes": exhausted}
                    )
                if gate is not None and not gate.exists():
                    failed_before_gate.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(worker, index, gpu) for index, gpu in enumerate(slots)]
        try:
            while not all(future.done() for future in futures):
                for future in futures:
                    if future.done() and future.exception() is not None:
                        future.result()
                if gate is not None and not gate.exists():
                    if failed_before_gate.is_set():
                        raise RuntimeError(
                            "an environment failed before the shared start barrier"
                        )
                    with LOCK:
                        ready = [
                            path
                            for path in current_attempts.values()
                            if (path / "READY.json").exists()
                        ]
                    if len(ready) == waiting:
                        atomic(
                            gate,
                            {
                                "released_at": time.time(),
                                "ready": len(ready),
                                "attempts": [str(path) for path in ready],
                            },
                        )
                time.sleep(0.2)
            for future in futures:
                future.result()
        except BaseException:
            STOP.set()
            raise
    atomic(
        batch / "state.json",
        {
            "status": "paused" if STOP.is_set() else "finished",
            "expected": len(jobs),
            "valid": len(records),
            "evaluated": len(outcomes),
            "infrastructure_errors": len(exhausted),
            "successes": sum(x["success"] for x in outcomes),
            "elapsed_seconds": time.time() - started,
            "started_at": started,
            "finished_at": time.time(),
        },
    )
    if not STOP.is_set() and len(outcomes) + len(exhausted) != len(jobs):
        raise RuntimeError("Some episodes failed audit; all attempts are retained")


def summary(args: argparse.Namespace) -> None:
    """Print audited sealed episodes without changing a running collector."""
    batch = Path(args.batch_dir).resolve()
    rows = [
        row
        for output in sorted(batch.iterdir())
        if output.is_dir()
        if (row := completed(output)) is not None
    ]
    report = {
        "valid": len(rows),
        "evaluated": len(rows),
        "successes": sum(x["success"] for x in rows),
        "episodes": rows,
    }
    quarantined = [
        row
        for output in sorted(batch.iterdir())
        if output.is_dir()
        if completed(output) is None
        if (row := evaluated(output)) is not None
    ]
    report["evaluated"] += len(quarantined)
    report["successes"] += sum(row["success"] for row in quarantined)
    report["excluded_from_training"] = quarantined
    print(json.dumps(report, indent=2))


def build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Build public batch options without starting services."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collection = commands.add_parser("run")
    collection.add_argument("batch_dir")
    collection.add_argument("--jobs", required=True)
    collection.add_argument("--seeds", type=seed_values)
    collection.add_argument("--gpus", required=True)
    collection.add_argument("--per-gpu", type=int, default=1)
    collection.add_argument(
        "--max-initializing-per-gpu",
        type=int,
        default=2,
        help="Concurrent cold starts per GPU; does not limit resident episodes.",
    )
    collection.add_argument(
        "--sam3-timeout-s",
        type=float,
        default=None,
        help="SAM RPC deadline including queue time; default 120 seconds.",
    )
    collection.add_argument("--workflow", choices=WORKFLOWS, default=None)
    collection.add_argument("--world-cord", choices=("base", "general"), default=None)
    collection.add_argument("--action-mode", choices=("single", "batch"), default=None)
    collection.add_argument(
        "--perception-mode",
        choices=("single", "batch"),
        default=None,
        help="single tools or episode-only observe graph.",
    )
    collection.add_argument(
        "--max-observe-calls",
        type=int,
        default=None,
        help="Total observe subcall limit (default 8).",
    )
    collection.add_argument(
        "--observe-concurrency",
        type=int,
        default=None,
        help="Independent observe calls in flight (default 8).",
    )
    collection.add_argument("--max-actions-per-call", type=int, default=None)
    collection.add_argument("--reasoning-history", type=int, default=None)
    collection.add_argument("--image-history", type=int, default=None)
    collection.add_argument(
        "--no-thinking",
        "--no-reasoning",
        action="store_true",
        help="Disable Qwen thinking for every episode and pin it in the batch manifest",
    )
    collection.add_argument(
        "--image_size", type=int, choices=(1024, 512, 256), default=None
    )
    collection.add_argument(
        "--thinking", "--reasoning", dest="no_thinking", action="store_false"
    )
    collection.set_defaults(no_thinking=None)
    collection.add_argument("--max-turns", type=int, default=None)
    collection.add_argument("--max-tokens", type=int, default=None)
    collection.add_argument("--context-window", type=int, default=None)
    collection.add_argument("--compaction-fraction", type=float, default=None)
    collection.add_argument("--compaction-max-messages", type=int, default=None)
    collection.add_argument("--compaction-keep-messages", type=int, default=None)
    collection.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        default=None,
    )
    collection.add_argument("--temperature", type=float, default=None)
    collection.add_argument("--top-p", type=float, default=None)
    collection.add_argument("--top-k", type=int, default=None)
    collection.add_argument("--min-p", type=float, default=None)
    collection.add_argument("--presence-penalty", type=float, default=None)
    collection.add_argument("--repetition-penalty", type=float, default=None)
    collection.add_argument("--max-episode-steps", type=int, default=None)
    collection.add_argument("--planner-timeout-s", type=int, default=None)
    collection.add_argument("--observation-mode", choices=CONFIGS, default=None)
    collection.add_argument(
        "--start-barrier",
        action="store_true",
        help="Wait until every pending environment is resident before requesting the model",
    )
    collection.add_argument("--prompt", choices=PROMPT_VARIANTS, default=None)
    collection.add_argument(
        "--endpoints", help="Comma-separated local OpenAI base URLs"
    )
    collection.add_argument("--infrastructure-retries", type=int, default=1)
    collection.add_argument("--episode-timeout", type=int, default=14400)
    collection.add_argument(
        "--start-interval",
        type=float,
        default=1,
        help="Seconds between initial worker starts",
    )
    report = commands.add_parser("summary")
    report.add_argument("batch_dir")
    return parser, collection


def main() -> None:
    """Run the collection or summary command."""
    parser, _ = build_parser()
    args = parser.parse_args()
    {"run": run, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
