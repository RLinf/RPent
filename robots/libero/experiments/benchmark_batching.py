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

"""Run paired ordinary/batched LIBERO episodes with borrowed VLA endpoints."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from rpent.utils.logging import get_logger

logger = get_logger("batching_experiment")
MODULE = "robots.libero.experiments.benchmark_batching"


def write_json(path: Path, value: Any) -> None:
    """Atomically replace an experiment record."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def file_hashes(root: Path) -> dict[str, str]:
    """Fingerprint files in a frozen input directory."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class TimedRpc:
    """Record client RPC latency without modifying transport classes."""

    def __init__(self, client: Any, path: Path) -> None:
        self.client = client
        self.path = path
        self.events: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Delegate a request and retain failed requests as evidence."""
        start = time.perf_counter()
        error = None
        try:
            return self.client.call(method, *args, **kwargs)
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            event = {
                "method": method,
                "elapsed_s": time.perf_counter() - start,
                "error": error,
            }
            with self.lock, self.path.open("a") as stream:
                self.events.append(event)
                stream.write(json.dumps(event) + "\n")


def run_episode(config: dict[str, Any], cell: dict[str, Any], output: Path) -> dict:
    """Own one local simulator; borrow and close clients to shared services."""
    from robots.libero.env_client import LiberoEnvClient
    from robots.libero.experiments.batching import BatchToolkit, execution_policy
    from robots.libero.robot_spec import get_robot_spec
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.planner.codex import CodexPlanner
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient
    from rpent.robots.components.sam3_client import Sam3Client
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.logging import init_output_dir
    from rpent.utils.rpc import make_rpc_client, wait_for_ready

    init_output_dir(output)
    row = {**cell, "infrastructure_ok": False, "native_success": None}
    timers = []
    started = time.perf_counter()
    try:
        with ExitStack() as cleanup:

            def connect(endpoint: str, daemon: Any = None) -> TimedRpc:
                client = make_rpc_client(endpoint)
                cleanup.callback(client.close)
                wait_for_ready(client, daemon=daemon, timeout_s=300)
                timer = TimedRpc(client, output / "rpc.jsonl")
                timers.append(timer)
                return timer

            vla = connect(config["backends"][cell["backend"]])
            sam3 = connect(config["sam3_endpoint"])
            # Shared model startup/health checks are outside episode timing.
            started = time.perf_counter()
            port = pick_free_port()
            command = [
                sys.executable,
                "-m",
                "robots.libero.env_server",
                "--suite",
                config["suite"],
                "--task",
                str(cell["task"]),
                "--seed",
                str(cell["seed"]),
                "--max-episode-steps",
                str(config["max_episode_steps"]),
                "--transport",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--parent-watch",
            ]
            if config["cuda_device"] is not None:
                command += ["--cuda-device", str(config["cuda_device"])]
            daemon = ProcessDaemon(
                "batching_env",
                command,
                log_path=str(output / "env-server.log"),
                env_overrides={
                    "LIBERO_TYPE": "pro",
                    "MUJOCO_GL": "egl",
                    "PYTHONHASHSEED": "0",
                    "ROBOT_PLATFORM": "LIBERO",
                },
            )
            cleanup.callback(daemon.stop)
            write_json(output / "env-command.json", command)
            daemon.start()
            env_rpc = connect(f"http://127.0.0.1:{port}", daemon)
            env = LiberoEnvClient(
                env_rpc,
                expected_meta={
                    "suite": config["suite"],
                    "task": cell["task"],
                    "seed": cell["seed"],
                    "max_episode_steps": config["max_episode_steps"],
                },
            )
            memory_dir = output / "memory"
            shutil.copytree(config["memory_dir"], memory_dir)
            if file_hashes(memory_dir) != config["memory_sha256"]:
                raise ValueError("Copied memory differs from the frozen manifest")
            toolkit = BatchToolkit(
                batch_size=cell["batch_size"],
                event_path=output / "tools.jsonl",
                runtime_kwargs={
                    "env": env,
                    "model": Pi05VLAClient(vla, embodiment="libero"),
                    "sam3_client": Sam3Client(sam3),
                },
                dashboard_events=NullDashboardEventSink(),
                memory=MemoryManager(root=memory_dir, memory_access="read_only"),
                state_output_dir=output,
            )
            cleanup.callback(toolkit.close)
            image = output / "agentview.png" / "00.png"
            row["initial_image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
            tag = f"{config['suite'].removeprefix('libero_')}_t{cell['task']}_s{cell['seed']}"
            variables = {
                "suite": config["suite"],
                "task": cell["task"],
                "seed": cell["seed"],
                "recipe_tag": tag,
                "mode": "eval",
                "enable_direct_action": False,
                "memory_profile": "local",
                "memory_dir": str(memory_dir),
                "output_dir": str(output),
                "reference_tag": f"{config['suite'].removeprefix('libero_')}_t{cell['task']}_s0",
            }
            prompts = get_robot_spec().prompts
            system = prompts.render("system", variables=variables)
            system += "\n\nExperiment execution policy:\n" + execution_policy(
                cell["batch_size"]
            )
            (output / "system-prompt.txt").write_text(system)
            planner = CodexPlanner(
                output_dir=str(output),
                output_path=output / "planner.txt",
                dashboard_events=NullDashboardEventSink(),
                model=config["model"],
                timeout_s=config["planner_timeout_s"],
                reasoning_effort=config["reasoning_effort"],
            )
            result = planner.solve(
                system_prompt=system,
                user_message=prompts.render("user", variables=variables),
                toolkit=toolkit,
                max_turns=config["max_turns"],
            )
            write_json(
                output / "planner-result.json",
                {
                    "error": result.error,
                    "finish": result.finish_result,
                    "stats": result.stats,
                },
            )
            events = [event for timer in timers for event in timer.events]
            failed_rpc = [event for event in events if event["error"]]
            budget_error = (
                result.error
                == f"Codex SDK timed out after {config['planner_timeout_s']}s"
            )
            row.update(
                native_success=bool(env.terminated),
                infrastructure_ok=not failed_rpc and (not result.error or budget_error),
                error=result.error,
                planner_budget_exhausted=budget_error,
                llm_responses=result.stats.get("turns_used", 0),
                vla_calls=sum(e["method"] == "vla.predict" for e in events),
                vla_rpc_s=sum(
                    e["elapsed_s"] for e in events if e["method"] == "vla.predict"
                ),
                failed_rpcs=failed_rpc,
            )
            row["wall_s"] = time.perf_counter() - started
            tool_log = output / "tools.jsonl"
            dispatches = (
                [json.loads(line) for line in tool_log.read_text().splitlines()]
                if tool_log.exists()
                else []
            )
            sequences = [d for d in dispatches if d["name"] == "execute_sequence"]
            row["sequence_calls"] = len(sequences)
            row["sequence_actions"] = sum(d["actions_executed"] or 0 for d in sequences)
    except Exception as exc:
        row.update(infrastructure_ok=False, error=repr(exc))
    finally:
        row.setdefault("wall_s", time.perf_counter() - started)
        write_json(output / "result.json", row)
    return row


def summarize(rows: list[dict]) -> dict:
    """Separate task outcomes, infrastructure failures and successful pairs."""
    groups = defaultdict(list)
    for row in rows:
        if row["infrastructure_ok"]:
            groups[(row["backend"], row["batch_size"])].append(row)
    report = []
    for (backend, size), group in sorted(groups.items()):
        wall = sum(r["wall_s"] for r in group)
        vla = sum(r["vla_rpc_s"] for r in group)
        responses = sum(r["llm_responses"] for r in group)
        sequences = sum(r["sequence_calls"] for r in group)
        report.append(
            {
                "backend": backend,
                "batch_size": size,
                "valid_n": len(group),
                "successes": sum(r["native_success"] for r in group),
                "wall_mean_all_s": wall / len(group),
                "wall_mean_success_s": statistics.mean(
                    [r["wall_s"] for r in group if r["native_success"]]
                )
                if any(r["native_success"] for r in group)
                else None,
                "vla_rpc_mean_s": vla / len(group),
                "vla_time_share": vla / wall if wall else None,
                "vla_calls": sum(r["vla_calls"] for r in group),
                "llm_responses": responses,
                "vla_calls_per_llm_response": sum(r["vla_calls"] for r in group)
                / responses
                if responses
                else None,
                "mean_actions_per_sequence": sum(r["sequence_actions"] for r in group)
                / sequences
                if sequences
                else None,
            }
        )
    pairs = []
    for left, right in itertools.combinations(sorted(groups), 2):
        if left[0] != right[0] and left[1] != right[1]:
            continue
        a = {(r["task"], r["seed"]): r for r in groups[left]}
        b = {(r["task"], r["seed"]): r for r in groups[right]}
        keys = a.keys() & b.keys()
        successes = [
            k for k in keys if a[k]["native_success"] and b[k]["native_success"]
        ]
        pairs.append(
            {
                "from": left,
                "to": right,
                "complete_pairs": len(keys),
                "success_gain_pairs": sum(
                    not a[k]["native_success"] and b[k]["native_success"] for k in keys
                ),
                "success_loss_pairs": sum(
                    a[k]["native_success"] and not b[k]["native_success"] for k in keys
                ),
                "both_success_pairs": len(successes),
                "faster_pairs": sum(b[k]["wall_s"] < a[k]["wall_s"] for k in successes),
                "both_success_mean_saved_s": statistics.mean(
                    [a[k]["wall_s"] - b[k]["wall_s"] for k in successes]
                )
                if successes
                else None,
                "both_success_geometric_speedup": math.exp(
                    statistics.mean(
                        [math.log(a[k]["wall_s"] / b[k]["wall_s"]) for k in successes]
                    )
                )
                if successes
                else None,
            }
        )
    return {
        "completed": len(rows),
        "groups": report,
        "pairs": pairs,
        "infrastructure_excluded": [
            r["cell"] for r in rows if not r["infrastructure_ok"]
        ],
    }


def parser() -> argparse.ArgumentParser:
    """Build the experiment CLI without simulator or planner imports."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--memory-dir", type=Path, required=True, help="Frozen local evaluation memory"
    )
    ap.add_argument(
        "--provenance",
        type=Path,
        required=True,
        help="JSON with dataset, checkpoint and backend revisions",
    )
    ap.add_argument(
        "--backend", action="append", required=True, metavar="NAME=ENDPOINT"
    )
    ap.add_argument("--sam3-endpoint", required=True)
    ap.add_argument(
        "--suite", default="libero_object_task", choices=["libero_object_task"]
    )
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[0, 4],
        help="0: ordinary tools; 1: sequence tool control; N: up to N tools",
    )
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--planner-timeout-s", type=int, default=900)
    ap.add_argument("--max-episode-steps", type=int, default=10000)
    ap.add_argument("--cuda-device", type=int)
    ap.add_argument("--resume", action="store_true")
    return ap


def sweep(args: argparse.Namespace) -> None:
    """Run isolated cells; never overwrite or automatically retry an attempt."""
    import fcntl

    config = vars(args).copy()
    root = args.output_dir.resolve()
    config["output_dir"] = str(root)
    config["memory_dir"] = str(args.memory_dir.resolve())
    config["provenance"] = json.loads(args.provenance.read_text())
    config.pop("resume")
    backends = {}
    for item in config.pop("backend"):
        name, endpoint = item.split("=", 1)
        if (
            not name
            or not all(c.isalnum() or c in "-_" for c in name)
            or name in backends
            or not endpoint
        ):
            raise ValueError("Backends need unique simple names and nonempty endpoints")
        backends[name] = endpoint
    config["backends"] = backends
    for key in ("tasks", "seeds", "batch_sizes"):
        values = config[key]
        if len(values) != len(set(values)) or any(v < 0 for v in values):
            raise ValueError(f"{key} must contain unique nonnegative integers")
    if any(t >= 10 for t in config["tasks"]):
        raise ValueError("ObjectTask task IDs must be 0–9")
    if any(
        config[k] <= 0 for k in ("max_turns", "planner_timeout_s", "max_episode_steps")
    ):
        raise ValueError("Execution budgets must be positive")
    config["memory_sha256"] = file_hashes(args.memory_dir)
    if not config["memory_sha256"]:
        raise ValueError("Frozen memory must not be empty")
    if args.memory_dir.resolve() == root or args.memory_dir.resolve() in root.parents:
        raise ValueError("Output must not be inside the source memory")
    config["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    config["rpent_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / "sweep.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = root / "manifest.json"
        if args.resume:
            if json.loads(manifest.read_text()) != config:
                raise ValueError("Resume inputs differ from the saved manifest")
        else:
            write_json(manifest, config)
        rows = []
        arms = list(itertools.product(backends, config["batch_sizes"]))
        cells = list(itertools.product(config["tasks"], config["seeds"]))
        for index, (task, seed) in enumerate(cells):
            offset = index % len(arms)
            for backend, size in arms[offset:] + arms[:offset]:
                name = f"task{task}-seed{seed}-{backend}-batch{size}"
                directory = root / name
                if directory.exists():
                    # An interrupted cell with no result needs manual inspection.
                    row = json.loads((directory / "result.json").read_text())
                else:
                    if file_hashes(args.memory_dir) != config["memory_sha256"]:
                        raise ValueError("Source memory changed during sweep")
                    directory.mkdir()
                    cell = {
                        "cell": name,
                        "task": task,
                        "seed": seed,
                        "backend": backend,
                        "batch_size": size,
                    }
                    write_json(directory / "cell.json", cell)
                    command = [
                        sys.executable,
                        "-m",
                        MODULE,
                        "_episode",
                        str(root),
                        name,
                    ]
                    with (directory / "runner.log").open("w") as log:
                        child = subprocess.Popen(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env={
                                **os.environ,
                                "PYTHONHASHSEED": "0",
                                "ROBOT_PLATFORM": "LIBERO",
                            },
                            start_new_session=True,
                        )
                        worker_timeout = False
                        try:
                            child.wait(timeout=config["planner_timeout_s"] + 900)
                        except subprocess.TimeoutExpired:
                            worker_timeout = True
                        finally:
                            if child.poll() is None:
                                os.killpg(child.pid, signal.SIGINT)
                                try:
                                    child.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    os.killpg(child.pid, signal.SIGKILL)
                                    child.wait()
                    result_path = directory / "result.json"
                    if result_path.exists():
                        row = json.loads(result_path.read_text())
                    else:
                        row = {
                            **cell,
                            "infrastructure_ok": False,
                            "native_success": None,
                            "error": f"Worker exited {child.returncode} without a result",
                        }
                    if worker_timeout or child.returncode != 0:
                        row.update(
                            infrastructure_ok=False,
                            error="Worker timed out or exited abnormally",
                        )
                    previous = [
                        r
                        for r in rows
                        if r["task"] == task
                        and r["seed"] == seed
                        and r["infrastructure_ok"]
                    ]
                    if (
                        row["infrastructure_ok"]
                        and previous
                        and row["initial_image_sha256"]
                        != previous[0]["initial_image_sha256"]
                    ):
                        row.update(
                            infrastructure_ok=False, error="initial_image_mismatch"
                        )
                    write_json(result_path, row)
                rows.append(row)
                write_json(root / "summary.json", summarize(rows))
                logger.info(
                    "%s: native_success=%s infrastructure_ok=%s",
                    name,
                    row["native_success"],
                    row["infrastructure_ok"],
                )
                if not row["infrastructure_ok"]:
                    raise RuntimeError(
                        f"Infrastructure failure retained at {directory}; no automatic retry"
                    )
        write_json(root / "completed.json", {"episodes": len(rows)})


def main() -> None:
    """Run the sweep or an isolated episode worker."""
    if len(sys.argv) > 1 and sys.argv[1] == "_episode":
        root, name = Path(sys.argv[2]), sys.argv[3]
        run_episode(
            json.loads((root / "manifest.json").read_text()),
            json.loads((root / name / "cell.json").read_text()),
            root / name,
        )
    else:
        sweep(parser().parse_args())


if __name__ == "__main__":
    main()
