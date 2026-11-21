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

"""Qualify forty native LIBERO environments against a shared SAM3 pool."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from robots.rl.rl_episode import start_environment
from robots.rl.robot_spec import get_robot_spec
from rpent.evaluation import write_json_atomic as atomic
from rpent.utils.logging import init_output_dir
from tools.rl.compare_workflows import jobs
from tools.rl.run_batch import source_identity, stop_process

QUERIES = ["object", "bowl", "plate", "cabinet", "drawer", "bottle", "box", "gripper"]


def wait_for(path: Path, *, deadline: float) -> None:
    """Wait on an owned coordination file, bounded by a monotonic deadline."""
    while not path.exists():
        if path.name != "RELEASE.json" and (path.parent / "RELEASE.json").exists():
            raise RuntimeError("qualification canceled before action release")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"waiting for {path}")
        time.sleep(0.1)


def worker(args) -> None:
    """Own one real environment through construction, actions, hold and cleanup."""
    output = args.output / f"env-{args.worker:02d}"
    output.mkdir(parents=True)
    init_output_dir(output)
    task = jobs(seed=1)[args.worker]
    parser = argparse.ArgumentParser()
    get_robot_spec().add_cli_args(parser, use_dashboard=False)
    runtime = parser.parse_args(
        [
            "--sam3-endpoint",
            args.sam3_endpoint,
            "--workflow",
            "episode",
            "--cuda-device",
            args.gpus.split(",")[args.worker % len(args.gpus.split(","))],
            "--libero-type",
            "pro",
        ]
    )
    runtime.output_dir = output
    children = []

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    try:
        with contextlib.ExitStack() as release:
            toolkit, _ = start_environment(runtime, task, release)
            children = [
                int(x)
                for x in Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
                .read_text()
                .split()
            ]
            atomic(
                output / "READY.json",
                {
                    "pid": os.getpid(),
                    "children": children,
                    "task": task,
                    "time": time.time(),
                },
            )
            wait_for(args.output / "BURST.json", deadline=time.monotonic() + 1200)
            primitive = toolkit._primitives
            groups = []
            for camera in ("agentview", "wrist"):
                result = primitive._sam3.segment_queries(
                    primitive.images(0, [camera])[0], QUERIES
                )
                assert len(result) == len(QUERIES)
                groups.append([len(row) for row in result])
            position = primitive.measure().state()["pose"]["xyz"]
            moved = toolkit.execute_tool(
                "move_to", {"xyz": [position[0], position[1], position[2] + 0.02]}
            )
            gripped = toolkit.execute_tool("set_gripper", {"opening": 0.8})
            opened = toolkit.execute_tool("set_gripper", {"opening": 1.0})
            assert not any(result.is_error for result in (moved, gripped, opened))
            assert primitive.env_steps > 0
            atomic(
                output / "ACTIONS.json",
                {
                    "passed": True,
                    "time": time.time(),
                    "query_groups": groups,
                    "env_steps": primitive.env_steps,
                    "results": [result.data for result in (moved, gripped, opened)],
                },
            )
            wait_for(args.output / "RELEASE.json", deadline=time.monotonic() + 1200)
    finally:
        atomic(
            output / "CLOSED.json",
            {
                "children": children,
                "children_gone": all(
                    not Path(f"/proc/{pid}").exists() for pid in children
                ),
                "time": time.time(),
            },
        )


def sample_resources(output: Path) -> dict:
    """Record GPU queries independently of native environment correctness."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        row = {
            "time": time.time(),
            "gpu": result.stdout,
            "gpu_exit": result.returncode,
            "meminfo": Path("/proc/meminfo").read_text(),
        }
    except subprocess.TimeoutExpired:
        row = {"time": time.time(), "gpu_timeout": True}
    with (output / "resources.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    return row


def qualify(args) -> None:
    """Hold the complete native panel, then prove owned subprocess cleanup."""
    if args.hold_seconds < 300:
        raise ValueError("capacity hold must be at least 300 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    identity = source_identity()
    processes = {}
    started = time.monotonic()
    burst_at = held_at = None
    last_sample = 0
    gpu_ready = False
    completed = False
    held_seconds = 0.0
    try:
        while True:
            now = time.monotonic()
            if now - started > 1800:
                raise TimeoutError("capacity qualification exceeded 30 minutes")
            if (args.output.parent / "STOP_REQUESTED.json").exists():
                raise RuntimeError("stop requested")
            if now - last_sample >= 5:
                sample = sample_resources(args.output)
                gpu_ready |= sample.get("gpu_exit") == 0 and bool(sample.get("gpu"))
                last_sample = now
            for index, process in processes.items():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"capacity worker {index} exited {process.returncode}"
                    )
            # Bound cold starts on each GPU without lowering resident concurrency.
            gpus = args.gpus.split(",")
            for gpu_index in range(len(gpus)):
                initializing = sum(
                    index % len(gpus) == gpu_index
                    and not (args.output / f"env-{index:02d}/READY.json").exists()
                    for index in processes
                )
                candidates = [
                    index
                    for index in range(40)
                    if index % len(gpus) == gpu_index and index not in processes
                ]
                for index in candidates[: max(0, 2 - initializing)]:
                    command = [
                        sys.executable,
                        "-m",
                        "tools.rl.qualify_compare",
                        "--output",
                        str(args.output),
                        "--sam3-endpoint",
                        args.sam3_endpoint,
                        "--gpus",
                        args.gpus,
                        "--worker",
                        str(index),
                    ]
                    log = (args.output / f"worker-{index:02d}.log").open("w")
                    processes[index] = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    log.close()
            ready = len(list(args.output.glob("env-*/READY.json")))
            actions = len(list(args.output.glob("env-*/ACTIONS.json")))
            if ready == 40 and burst_at is None:
                burst_at = now
                atomic(
                    args.output / "BURST.json",
                    {"ready": 40, "released_at": time.time()},
                )
            if actions == 40 and held_at is None:
                held_at = now
            atomic(
                args.output / "progress.json",
                {
                    "ready": ready,
                    "actions": actions,
                    "hold_seconds": 0 if held_at is None else now - held_at,
                    "time": time.time(),
                },
            )
            if held_at is not None and now - held_at >= args.hold_seconds:
                if not gpu_ready:
                    raise RuntimeError(
                        "GPU resource observations unavailable; qualification incomplete"
                    )
                completed = True
                held_seconds = now - held_at
                break
            time.sleep(0.5)
    finally:
        atomic(args.output / "RELEASE.json", {"time": time.time()})
        cleanup_deadline = time.monotonic() + 120
        for process in processes.values():
            try:
                process.wait(timeout=max(0.1, cleanup_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                stop_process(process)
        cleanup = [
            json.loads(path.read_text())
            for path in args.output.glob("env-*/CLOSED.json")
        ]
        atomic(
            args.output / "cleanup.json",
            {
                "workers": len(cleanup),
                "children_gone": len(cleanup) == 40
                and all(row["children_gone"] for row in cleanup),
            },
        )
    assert completed and all(process.returncode == 0 for process in processes.values())
    assert len(cleanup) == 40 and all(row["children_gone"] for row in cleanup)
    assert source_identity() == identity, (
        "production source changed during qualification"
    )
    atomic(
        args.output / "verified.json",
        {
            "passed": True,
            "source": identity,
            "envs": 40,
            "hold_seconds": held_seconds,
            "actions": 40,
            "cleanup": True,
            "sam3_endpoint": args.sam3_endpoint,
            "gpus": args.gpus,
            "completed_at": time.time(),
        },
    )


def main() -> None:
    """Run a fresh capacity qualification or one of its owned workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sam3-endpoint", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--hold-seconds", type=int, default=300)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args)
    else:

        def stop(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        qualify(args)


if __name__ == "__main__":
    main()
