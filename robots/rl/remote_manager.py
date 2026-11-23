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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from robots.rl.contracts import CAMERAS, EPISODE_CONTRACT_VERSION
from robots.rl.remote import REMOTE_OPTION_DEFAULTS, remote_options
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade, make_rpc_client, wait_for_endpoint
from rpent.utils.rpc.http_rpc import EncodedRpcResponse, HttpRpcClient

logger = get_logger("rl_backend")


def validate_request(
    request: dict, gpus: set[int], simulator: str | None = None
) -> None:
    """Validate the episode protocol, task and resources before allocation."""
    if set(request) != {"task", "gpu", "max_episode_steps", "episode_contract_version"}:
        raise ValueError(
            "backend request requires task, gpu, max_episode_steps and episode_contract_version"
        )
    if request["episode_contract_version"] != EPISODE_CONTRACT_VERSION:
        raise ValueError("unsupported episode contract version")
    if type(request["gpu"]) is not int or request["gpu"] not in gpus:
        raise ValueError("requested GPU is not assigned to this backend")
    if (
        type(request["max_episode_steps"]) is not int
        or request["max_episode_steps"] <= 0
    ):
        raise ValueError("max_episode_steps must be positive")
    task = request["task"]
    environment = task.get("environment")
    if simulator is not None and environment != simulator:
        raise ValueError(f"backend serves {simulator}, requested {environment}")
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
    if environment == "robodojo":
        allowed.add("layout")
    if set(task).difference(allowed) or environment not in CAMERAS:
        raise ValueError("unsupported remote simulator/task fields")
    if environment == "robodojo" and (
        type(task.get("layout", 0)) is not int or task.get("layout", 0) < 0
    ):
        raise ValueError("RoboDojo layout must be a nonnegative integer")
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
        self,
        output: Path,
        slots: dict[int, int],
        cold_starts: int,
        sam3_endpoint: str | None = None,
        *,
        simulator: str | None = None,
    ):
        super().__init__()
        self.output, self.slots, self.sam3_endpoint = output, slots, sam3_endpoint
        self.simulator = simulator
        self.output.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._workers: dict[str, dict] = {}
        self._owners: dict[str, str] = {}
        self._cancelled_owners: set[str] = set()
        self._releases = ThreadPoolExecutor(
            max_workers=16, thread_name_prefix="rl-backend-release"
        )
        self._cold = {gpu: threading.BoundedSemaphore(cold_starts) for gpu in slots}
        self._rpc.update(
            {
                "backend.open": self.open,
                "backend.call": self.forward_call,
                "backend.renew": self.renew,
                "backend.release": self.release,
                "backend.status": self.status,
                "backend.cancel_owners": self.cancel_owners,
            }
        )
        # Registry mutations use _lock; independent worker RPCs must not share a write lock.
        self._readonly_methods.update(self._rpc)
        self._sweeper = threading.Thread(target=self._sweep, daemon=True)
        self._sweeper.start()

    def open(self, request: dict, owner_id: str | None = None) -> dict:
        """Start one worker; an optional attempt owner supports race-free cancellation."""
        validate_request(request, set(self.slots), self.simulator)
        if owner_id is not None:
            self._validate_owner(owner_id)
        if not self.sam3_endpoint:
            raise ValueError("perception workflows require a configured SAM3 endpoint")
        gpu = request["gpu"]
        session = uuid.uuid4().hex
        output = self.output / session
        with self._lock:
            if self._shutdown_event.is_set():
                raise RuntimeError("backend is closing")
            if owner_id in self._cancelled_owners:
                raise RuntimeError(f"remote attempt {owner_id} was cancelled")
            if owner_id is not None and owner_id in self._owners:
                raise ValueError(f"remote attempt {owner_id} is already open")
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
                "owner_id": owner_id,
                "released": threading.Event(),
                "closing": False,
            }
            if owner_id is not None:
                self._owners[owner_id] = session
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
                    start_new_session=True,
                )
                with self._lock:
                    row = self._workers[session]
                    if row["closing"] or owner_id in self._cancelled_owners:
                        raise RuntimeError(f"remote attempt {owner_id} was cancelled")
                    row["daemon"] = daemon
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
            return {"session": session, "owner_id": owner_id, **metadata}
        except BaseException:
            self.release(session)
            raise

    def _worker(self, session: str) -> dict:
        with self._lock:
            row = self._workers[session]
            if row["closing"] or row["owner_id"] in self._cancelled_owners:
                raise RuntimeError(f"remote session {session} is closing")
            row["last_seen"] = time.monotonic()
            return dict(row)

    def _toolkit_rpc(self, session: str, method: str) -> HttpRpcClient:
        if method not in ("rl.observe", "rl.execute", "rl.close", "rl.cancel"):
            raise ValueError("unsupported toolkit operation")
        row = self._worker(session)
        if row["rpc"] is None:
            raise RuntimeError("toolkit has not finished initializing")
        return row["rpc"]

    def call(self, session: str, method: str, arguments: dict) -> dict:
        """Call an allowed toolkit operation directly, without automatic replay."""
        return self._toolkit_rpc(session, method).call(
            method, kwargs=arguments, timeout_s=600
        )

    def forward_call(
        self, session: str, method: str, arguments: dict
    ) -> EncodedRpcResponse:
        """Relay an HTTP toolkit reply without decoding and re-encoding its artifacts."""
        return self._toolkit_rpc(session, method).forward(
            method, kwargs=arguments, timeout_s=600
        )

    def renew(self, session: str) -> dict:
        """Keep a worker leased while its planner waits for a model response."""
        self._worker(session)
        return {"renewed": True}

    def release(self, session: str) -> dict:
        """Close only the owned worker; its configuration, log and artifacts remain."""
        with self._lock:
            row = self._workers.get(session)
            if row is None:
                return {"released": False}
            # Keep the slot reserved until the owned process has really exited.
            already_closing = row.get("closing", False)
            if not already_closing:
                row["closing"] = True
                row.setdefault("released", threading.Event())
        if already_closing:
            if not row["released"].wait(120):
                raise RuntimeError(f"remote session {session} cleanup did not finish")
            if row.get("cleanup_error") is not None:
                raise RuntimeError(row["cleanup_error"])
            return {"released": True}
        try:
            if row["daemon"] is not None:
                row["daemon"].stop()
            with self._lock:
                self._workers.pop(session)
                owner_id = row.get("owner_id")
                if owner_id is not None:
                    self._owners.pop(owner_id)
        except BaseException as error:
            row["cleanup_error"] = repr(error)
            raise
        finally:
            row["released"].set()
        return {"released": True}

    @staticmethod
    def _validate_owner(owner_id: str) -> None:
        if (
            not isinstance(owner_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", owner_id) is None
        ):
            raise ValueError("remote owner_id must be one 32-character episode UUID")

    def cancel_owners(self, owner_ids: list[str]) -> dict:
        """Tombstone explicit attempts and release only their reserved/ready workers."""
        if not isinstance(owner_ids, list):
            raise ValueError("owner_ids must be a list of episode UUIDs")
        for owner_id in owner_ids:
            self._validate_owner(owner_id)
        owner_ids = list(dict.fromkeys(owner_ids))
        with self._lock:
            self._cancelled_owners.update(owner_ids)
            sessions = {owner_id: self._owners.get(owner_id) for owner_id in owner_ids}
            output_dirs = {
                owner_id: str(self._workers[session]["output"])
                for owner_id, session in sessions.items()
                if session is not None
            }
        errors = self._release_sessions(
            [session for session in sessions.values() if session is not None]
        )
        with self._lock:
            remaining = {
                owner_id: self._owners[owner_id]
                for owner_id in owner_ids
                if owner_id in self._owners
            }
        receipt = {
            "owner_ids": owner_ids,
            "sessions": sessions,
            "output_dirs": output_dirs,
            "remaining": remaining,
            "cleanup_complete": not errors and not remaining,
            "errors": errors,
        }
        cancellation_dir = self.output / "cancellations"
        cancellation_dir.mkdir(exist_ok=True)
        receipt_path = cancellation_dir / f"{uuid.uuid4().hex}.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        if not receipt["cleanup_complete"]:
            raise RuntimeError(
                f"remote attempt cleanup failed; see {receipt_path}: {errors or remaining}"
            )
        return receipt

    def _release_sessions(self, sessions: list[str]) -> dict[str, str]:
        futures = {
            session: self._releases.submit(self.release, session)
            for session in dict.fromkeys(sessions)
        }
        errors = {}
        for session, future in futures.items():
            try:
                future.result()
            except Exception as error:
                errors[session] = repr(error)
        return errors

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
        try:
            errors = self._release_sessions(sessions)
            if errors:
                raise RuntimeError(f"backend cleanup failed: {errors}")
        finally:
            self._releases.shutdown(wait=True)


def main() -> None:
    """Start a backend on explicit GPUs; the default listener is loopback only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulator", choices=tuple(CAMERAS), required=True)
    parser.add_argument(
        "--gpu-slots", required=True, help="Comma-separated GPU:slots, e.g. 0:7,1:14"
    )
    parser.add_argument("--max-initializing-per-gpu", type=int, default=2)
    parser.add_argument(
        "--sam3-endpoint",
        required=True,
        help="SAM3 endpoint used by episode perception.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28090)
    parser.add_argument("--parent-watch", action="store_true")
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
        args.output_dir,
        slots,
        args.max_initializing_per_gpu,
        args.sam3_endpoint,
        simulator=args.simulator,
    ).serve(
        transport="http", host=args.host, port=args.port, parent_watch=args.parent_watch
    )


if __name__ == "__main__":
    main()
