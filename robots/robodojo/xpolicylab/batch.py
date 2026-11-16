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

"""Schedule official task sequences through the single-task RPent launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path

from robots.robodojo.xpolicylab.__main__ import episode_count, verify_results
from rpent.evaluation import write_json_atomic
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon
from rpent.utils.logging import get_logger

logger = get_logger(__name__)


def task_plan(args: argparse.Namespace) -> list[dict]:
    """Read task membership and counts from the installed official runtime."""
    inventory = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(args.runtime_root / "scripts/internal/task_inventory.py"),
                "--format",
                "json",
                "--dimension",
                ",".join(args.dimensions),
            ],
            text=True,
        )
    )
    tasks = inventory["tasks"]
    if args.tasks:
        unknown = set(args.tasks) - {t["name"] for t in tasks}
        if unknown:
            raise ValueError(f"Tasks outside selected dimensions: {sorted(unknown)}")
        tasks = [t for t in tasks if t["name"] in args.tasks]
    if not tasks or any(not t["runnable"] for t in tasks):
        raise ValueError(
            "Official task inventory contains missing task implementations/configs"
        )
    return [
        {
            "task": t["name"],
            "dimension": t["dimension"],
            "episodes": episode_count(args.runtime_root, t["name"], args.episodes),
        }
        for t in tasks
    ]


def fingerprint(args: argparse.Namespace) -> str:
    """Reject changed code, configuration or memory when resuming a batch."""
    digest = hashlib.sha256()
    settings = {
        k: str(v) for k, v in vars(args).items() if k not in ("dry_run", "resume")
    }
    digest.update(json.dumps(settings, sort_keys=True).encode())
    for root in (get_repo_root(), args.runtime_root, args.runtime_root / "XPolicyLab"):
        for command in (["git", "rev-parse", "HEAD"], ["git", "diff", "HEAD"]):
            digest.update(subprocess.check_output(command, cwd=root))
    sources = [
        args.memory_dir,
        get_repo_root() / "robots/robodojo/prompts",
        get_repo_root() / "robots/robodojo/guides",
    ]
    if args.official_descriptions:
        sources.append(args.official_descriptions)
    for source in sources:
        files = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in files:
            if path.is_file() and "__pycache__" not in path.parts:
                digest.update(str(path.resolve()).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def task_command(
    args: argparse.Namespace, task: str, output: Path, worker: str
) -> list[str]:
    """Keep planner, memory and service options identical across workers."""
    env_gpu, policy_gpu = worker.split(":")
    command = [
        sys.executable,
        "-m",
        "robots.robodojo.xpolicylab",
        "--task",
        task,
        "--output-dir",
        str(output),
        "--env-cuda-device",
        env_gpu,
        "--policy-cuda-device",
        policy_gpu,
    ]
    for key in (
        "runtime_root",
        "seed",
        "episodes",
        "am_endpoint",
        "sam3_endpoint",
        "memory_dir",
        "official_descriptions",
        "planner",
        "model",
        "reasoning_effort",
        "timeout",
    ):
        value = getattr(args, key)
        if value is not None:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    return command


def run_batch(args: argparse.Namespace) -> None:
    """Run one task per GPU pair; retain interrupted tasks for explicit recovery."""
    root, output = args.runtime_root.resolve(), args.output_dir.resolve()
    args.runtime_root, args.output_dir = root, output
    args.memory_dir = args.memory_dir.resolve()
    if not args.memory_dir.is_dir():
        raise ValueError(f"Memory directory not found: {args.memory_dir}")
    workers = args.worker or [f"{args.env_cuda_device}:{args.policy_cuda_device}"]
    if any(len(w.split(":")) != 2 or not all(w.split(":")) for w in workers):
        raise ValueError("Each --worker must be ENV_GPU:POLICY_GPU")
    tasks = task_plan(args)
    frozen = fingerprint(args)
    plan = {
        "fingerprint": frozen,
        "seed": args.seed,
        "workers": workers,
        "tasks": tasks,
    }
    plan_path, status_path = output / "batch-plan.json", output / "batch-status.json"
    if args.resume:
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Batch inputs changed; use a new output directory")
        statuses = json.loads(status_path.read_text())
    else:
        output.mkdir(parents=True, exist_ok=False)
        statuses = {t["task"]: "pending" for t in tasks}
        write_json_atomic(plan_path, plan)
        write_json_atomic(status_path, statuses)
    for category in args.dimensions:
        selected = [t for t in tasks if t["dimension"] == category]
        logger.info(
            "%s: %d tasks, %d episodes",
            category,
            len(selected),
            sum(t["episodes"] for t in selected),
        )
    if args.dry_run:
        return
    # Freeze a plan even for dry runs; registration happens only when executing.
    package = Path(__file__).resolve().parent
    link = root / "XPolicyLab/policy/RPent"
    try:
        link.symlink_to(package, target_is_directory=True)
    except FileExistsError:
        if link.resolve() != package:
            raise FileExistsError(f"Another adapter exists at {link}") from None
    for t in tasks:
        task_output = output / t["task"]
        if statuses[t["task"]] == "complete":
            manifest = json.loads((task_output / "manifest.json").read_text())
            verify_results(
                root, Path(manifest["result_dir"]), task_output, t["episodes"]
            )
        elif task_output.exists():
            statuses[t["task"]] = "interrupted"
    pending = deque(t for t in tasks if statuses[t["task"]] == "pending")
    active = {}
    env = dict(os.environ, RPENT_XPL_BATCH="1", RPENT_REPO_ROOT=str(get_repo_root()))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(get_repo_root()), env.get("PYTHONPATH", "")]
    )
    try:
        while pending or active:
            for i, worker in enumerate(workers):
                if not pending or i in active:
                    continue
                if fingerprint(args) != frozen:
                    raise RuntimeError("Batch inputs changed during execution")
                task = pending.popleft()["task"]
                daemon = ProcessDaemon(
                    f"xpl-{task}",
                    task_command(args, task, output / task, worker),
                    cwd=str(get_repo_root()),
                    env_overrides=env,
                    log_path=str(output / f"{task}.log"),
                )
                daemon.subprocess_env["PYTHONPATH"] = env["PYTHONPATH"]
                active[i] = (task, daemon)
                statuses[task] = "running"
                write_json_atomic(status_path, statuses)
                daemon.start()
            for i, (task, daemon) in list(active.items()):
                code = daemon.poll()
                if code is not None:
                    statuses[task] = "complete" if code == 0 else "error"
                    daemon.stop()
                    del active[i]
                    write_json_atomic(status_path, statuses)
            time.sleep(1)
    finally:
        for task, daemon in active.values():
            daemon.stop(timeout=60)
            statuses[task] = "interrupted"
        write_json_atomic(status_path, statuses)
    # Give the official summarizer only this batch, not historical experiments.
    result_root = output / "official-results"
    result_root.mkdir(exist_ok=True)
    for task, status in statuses.items():
        if status != "complete":
            continue
        manifest = json.loads((output / task / "manifest.json").read_text())
        result = Path(manifest["result_dir"])
        destination = (
            result_root / task / "RPent/arx_x5" / f"{args.seed}_batch" / result.name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(result, target_is_directory=True)
    with (output / "summarize.log").open("w") as log:
        subprocess.run(
            ["bash", "scripts/robodojo.sh", "summarize"],
            cwd=root,
            env=dict(env, ROBODOJO_EVAL_ROOT=str(result_root)),
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    logger.info("Batch status: %s", dict(Counter(statuses.values())))
    if any(status != "complete" for status in statuses.values()):
        raise RuntimeError(
            f"Batch incomplete; inspect {status_path}. Interrupted tasks were not automatically rerun."
        )
