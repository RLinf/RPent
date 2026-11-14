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

"""Own bounded, independent simulator/toolkit workers on a remote GPU node."""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import uuid
from pathlib import Path

from robots.rl.remote import REMOTE_OPTION_DEFAULTS, remote_options
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade, make_rpc_client, wait_for_endpoint

logger = get_logger("rl_backend")


def validate_request(request: dict, gpus: set[int]) -> None:
    """Validate only the task and resources accepted by this backend."""
    if set(request) != {"task", "gpu", "max_episode_steps"}:
        raise ValueError("backend request requires task, gpu and max_episode_steps")
    if type(request["gpu"]) is not int or request["gpu"] not in gpus:
        raise ValueError("requested GPU is not assigned to this backend")
    if (
        type(request["max_episode_steps"]) is not int
        or request["max_episode_steps"] <= 0
    ):
        raise ValueError("max_episode_steps must be positive")
    task = request["task"]
    environment = task.get("environment")
    allowed = {
        "environment",
        "suite",
        "task",
        "task_name",
        "seed",
        "split",
        "task_config",
        "workflow",
        "observation_mode",
        "prompt",
        "libero_type",
        "init_state_id",
        *REMOTE_OPTION_DEFAULTS,
    }
    if set(task).difference(allowed) or environment not in (
        "libero",
        "robocasa",
        "robotwin",
    ):
        raise ValueError("unsupported remote simulator/task fields")
    remote_options(task)
    if type(task.get("seed")) is not int or task["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if "init_state_id" in task and (
        type(task["init_state_id"]) is not int
        or task["init_state_id"] < 0
        or task["init_state_id"] != task["seed"]
    ):
        raise ValueError("explicit init_state_id must match the native seed index")
    name = task.get("suite") if environment == "libero" else task.get("task_name")
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_]+", name) is None:
        raise ValueError("invalid native task identifier")
    if environment == "libero" and (
        type(task.get("task")) is not int or task["task"] < 0
    ):
        raise ValueError("LIBERO task index must be nonnegative")
    if task.get("libero_type") not in (None, "standard", "pro", "plus"):
        raise ValueError("invalid LIBERO variant")
    if task.get("split", "target") not in ("target", "pretrain", "all") or task.get(
        "task_config", "demo_randomized"
    ) not in ("demo_randomized", "demo_clean"):
        raise ValueError("invalid native task preset")


class BackendManager(RpcFacade):
    """Allocate GPU slots and retain every worker's evidence after release."""

    def __init__(
        self, output: Path, slots: dict[int, int], cold_starts: int, sam3_endpoint: str
    ):
        super().__init__()
        self.output, self.slots, self.sam3_endpoint = output, slots, sam3_endpoint
        self.output.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._workers: dict[str, dict] = {}
        self._cold = {gpu: threading.BoundedSemaphore(cold_starts) for gpu in slots}
        self._rpc.update(
            {
                "backend.open": self.open,
                "backend.call": self.call,
                "backend.renew": self.renew,
                "backend.release": self.release,
                "backend.status": self.status,
            }
        )
        # Registry mutations use _lock; independent worker RPCs must not share a write lock.
        self._readonly_methods.update(self._rpc)
        self._sweeper = threading.Thread(target=self._sweep, daemon=True)
        self._sweeper.start()

    def open(self, request: dict) -> dict:
        """Start exactly one owned task, with a per-GPU initialization limit."""
        validate_request(request, set(self.slots))
        gpu = request["gpu"]
        session = uuid.uuid4().hex
        output = self.output / session
        with self._lock:
            if self._shutdown_event.is_set():
                raise RuntimeError("backend is closing")
            if (
                sum(row["gpu"] == gpu for row in self._workers.values())
                >= self.slots[gpu]
            ):
                raise RuntimeError(f"GPU{gpu} episode slots are full")
            self._workers[session] = {
                "gpu": gpu,
                "last_seen": time.monotonic(),
                "daemon": None,
                "rpc": None,
                "output": output,
            }
        try:
            output.mkdir()
            config = {
                **request,
                "output_dir": str(output / "artifacts"),
                "sam3_endpoint": self.sam3_endpoint,
            }
            path = output / "configuration.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            with self._cold[gpu]:
                daemon = ProcessDaemon(
                    name=f"rl-{session}",
                    cmd=[
                        sys.executable,
                        "-m",
                        "robots.rl.remote",
                        "--configuration",
                        str(path),
                        "--parent-watch",
                    ],
                    log_path=str(output / "worker.log"),
                    cwd=str(get_repo_root()),
                )
                with self._lock:
                    self._workers[session]["daemon"] = daemon
                    daemon.start()
                rpc = make_rpc_client(wait_for_endpoint(daemon, timeout_s=1800))
                with self._lock:
                    self._workers[session]["rpc"] = rpc
                metadata = rpc.call("rl.describe", timeout_s=300)
                self._worker(session)
                (output / "identity.json").write_text(
                    json.dumps(
                        {
                            key: value
                            for key, value in metadata.items()
                            if key not in ("files", "images")
                        },
                        indent=2,
                    )
                    + "\n"
                )
            return {"session": session, **metadata}
        except BaseException:
            self.release(session)
            raise

    def _worker(self, session: str) -> dict:
        with self._lock:
            row = self._workers[session]
            row["last_seen"] = time.monotonic()
            return dict(row)

    def call(self, session: str, method: str, arguments: dict) -> dict:
        """Forward an allowed toolkit operation, without automatic replay."""
        if method not in ("rl.observe", "rl.execute", "rl.close", "rl.cancel"):
            raise ValueError("unsupported toolkit operation")
        row = self._worker(session)
        if row["rpc"] is None:
            raise RuntimeError("toolkit has not finished initializing")
        return row["rpc"].call(method, kwargs=arguments, timeout_s=600)

    def renew(self, session: str) -> dict:
        """Keep a worker leased while its planner waits for a model response."""
        self._worker(session)
        return {"renewed": True}

    def release(self, session: str) -> dict:
        """Close only the owned worker; its configuration, log and artifacts remain."""
        with self._lock:
            row = self._workers.pop(session, None)
        if row is not None and row["daemon"] is not None:
            row["daemon"].stop()
        return {"released": row is not None}

    def status(self) -> dict:
        """Count real ready workers separately from their reserved episode slots."""
        with self._lock:
            return {
                "reserved": len(self._workers),
                "ready": sum(row["rpc"] is not None for row in self._workers.values()),
                "per_gpu": {
                    gpu: sum(row["gpu"] == gpu for row in self._workers.values())
                    for gpu in self.slots
                },
            }

    def _sweep(self) -> None:
        while not self._shutdown_event.wait(30):
            with self._lock:
                expired = [
                    session
                    for session, row in self._workers.items()
                    if row["rpc"] is not None
                    and time.monotonic() - row["last_seen"] > 300
                ]
            for session in expired:
                logger.warning("Releasing expired remote episode %s", session)
                self.release(session)

    def close(self) -> None:
        self._shutdown_event.set()
        self._sweeper.join(timeout=31)
        with self._lock:
            sessions = list(self._workers)
        for session in sessions:
            self.release(session)


def main() -> None:
    """Start a backend on explicit GPUs; the default listener is loopback only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gpu-slots", required=True, help="Comma-separated GPU:slots, e.g. 0:7,1:14"
    )
    parser.add_argument("--max-initializing-per-gpu", type=int, default=2)
    parser.add_argument("--sam3-endpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28090)
    args = parser.parse_args()
    try:
        slots = dict(
            tuple(map(int, item.split(":"))) for item in args.gpu_slots.split(",")
        )
    except ValueError as exc:
        parser.error(str(exc))
    if (
        args.max_initializing_per_gpu < 1
        or not slots
        or any(gpu < 0 or count < 1 for gpu, count in slots.items())
    ):
        parser.error("GPU ordinals must be nonnegative and capacities positive")
    BackendManager(
        args.output_dir, slots, args.max_initializing_per_gpu, args.sam3_endpoint
    ).serve(transport="http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
