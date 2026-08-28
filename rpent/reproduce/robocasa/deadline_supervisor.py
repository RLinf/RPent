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

"""Bind planner timeout and driver commits to one monotonic deadline."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import signal
import stat
import subprocess as _subprocess
import sys
import threading
import time as _time
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rpent.reproduce.robocasa.secure_script import _read_source  # noqa: E402

DEADLINE_PROTOCOL = "monotonic_commit_gate_v1"
CONTRACT_NAME = "_deadline_contract.json"
GATE_NAME = "_deadline_commit.gate"
SEAL_NAME = "_deadline_seal.json"
FREEZE_NAME = "_deadline_freeze.json"
RECEIPT_PREFIX = "_deadline_commit_"
EXIT_DEADLINE = 20
EXIT_WRAPPER_ERROR = 70
DRIVER_STOP_FIELDS = frozenset(
    {"observed_state", "supervisor_stopped", "kill_sent", "kill_observed"}
)
DRIVER_STOP_STATES = frozenset({"exited", "T", "t", "Z", "X"})
DRIVER_ADAPTER_DEADLINE_EXIT = 75
RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "run_id",
        "nonce",
        "step",
        "deadline_monotonic_ns",
        "done_published_monotonic_ns",
        "contract_sha256",
    }
)


class DeadlineError(RuntimeError):
    """The deadline boundary could not be established or attested."""


def driver_stop_problem(value: Any) -> str | None:
    """Validate the frozen driver termination proof shared by all consumers."""
    if not isinstance(value, dict) or set(value) != DRIVER_STOP_FIELDS:
        return "driver stop proof has an invalid schema"
    state = value.get("observed_state")
    stopped = value.get("supervisor_stopped")
    kill_sent = value.get("kill_sent")
    kill_observed = value.get("kill_observed")
    if state not in DRIVER_STOP_STATES:
        return "driver stop proof has an invalid observed state"
    if any(type(item) is not bool for item in (stopped, kill_sent, kill_observed)):
        return "driver stop proof flags must be booleans"
    if stopped is not (state in {"T", "t"}):
        return "driver stop proof disagrees with the observed state"
    if kill_sent is not stopped:
        return "driver stop proof does not bind SIGKILL to supervisor stop"
    if kill_observed is not True:
        return "driver stop proof does not confirm driver termination"
    return None


def timeout_driver_exit_problem(
    driver_stop: Any,
    return_code: Any,
    *,
    success_latched: Any,
) -> str | None:
    """Bind a timeout's observed process exit to its frozen stop proof."""
    if problem := driver_stop_problem(driver_stop):
        return problem
    if type(return_code) is not int:
        return "driver return code must be an integer"
    if type(success_latched) is not bool:
        return "deadline success latch must be a boolean"
    assert isinstance(driver_stop, dict)
    passive_exit = driver_stop["observed_state"] in {"exited", "Z", "X"}
    if return_code == DRIVER_ADAPTER_DEADLINE_EXIT:
        return (
            None if passive_exit else "adapter deadline exit lacks a passive stop proof"
        )
    if return_code == -signal.SIGKILL:
        return (
            None
            if not passive_exit
            else "supervisor SIGKILL lacks a stopped-driver proof"
        )
    if return_code == 0 and success_latched:
        return (
            None
            if passive_exit
            else "successful driver exit lacks a passive stop proof"
        )
    return "driver return code is not a canonical timeout exit"


def normalize_deadline_outcome(
    raw_cause: Any,
    raw_rc: Any,
    *,
    supervisor_fired: bool,
    provider_violation: Any,
    forbidden_web_search_event_count: Any,
) -> tuple[Any, Any]:
    """Apply the shared outer-deadline precedence to a raw planner outcome."""
    if not supervisor_fired:
        return raw_cause, raw_rc
    provider_protocol_error = raw_cause == "planner_protocol_error" and (
        provider_violation is not None or forbidden_web_search_event_count != 0
    )
    if raw_cause in {"operator_interrupted", "wrapper_error"} or (
        provider_protocol_error
    ):
        return raw_cause, raw_rc
    return "planner_timeout", EXIT_DEADLINE


def deadline_receipt_problem(
    value: Any,
    *,
    step: int,
    contract: dict[str, Any],
    contract_sha256: str,
    require_before_deadline: bool,
) -> str | None:
    """Validate one post-publication receipt against its frozen contract."""
    if type(step) is not int or step <= 0:
        return "is not for a positive command step"
    if not isinstance(value, dict) or set(value) != RECEIPT_FIELDS:
        return "has an invalid schema"
    published_ns = value.get("done_published_monotonic_ns")
    started_ns = contract.get("started_monotonic_ns")
    deadline_ns = contract.get("deadline_monotonic_ns")
    if (
        value.get("schema_version") != 1
        or value.get("protocol") != DEADLINE_PROTOCOL
        or value.get("run_id") != contract.get("run_id")
        or value.get("nonce") != contract.get("nonce")
        or value.get("step") != step
        or value.get("deadline_monotonic_ns") != deadline_ns
        or value.get("contract_sha256") != contract_sha256
        or type(published_ns) is not int
        or type(started_ns) is not int
        or type(deadline_ns) is not int
    ):
        return "disagrees with the deadline contract"
    if published_ns < started_ns:
        return "claims publication before the deadline contract started"
    if require_before_deadline and published_ns >= deadline_ns:
        return "was published after the deadline"
    return None


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_parent(path: Path) -> int:
    parent = path.parent
    metadata = parent.lstat()
    if (
        parent.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise DeadlineError(f"unsafe deadline directory: {parent}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(parent, flags)


def _atomic_json(path: Path, value: Any) -> str:
    payload = _canonical_bytes(value)
    directory_fd = _safe_parent(path)
    temporary = f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd
        )
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)
    return _sha256_bytes(payload)


def _open_gate(path: Path, *, create: bool) -> int:
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    metadata = os.fstat(descriptor)
    path_metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
        or (metadata.st_dev, metadata.st_ino)
        != (path_metadata.st_dev, path_metadata.st_ino)
    ):
        os.close(descriptor)
        raise DeadlineError("deadline commit gate is unsafe")
    return descriptor


def _read_trusted_json(path: Path) -> tuple[dict[str, Any], str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_mode & 0o022
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > 2 * 1024 * 1024
        ):
            raise DeadlineError(f"unsafe deadline evidence: {path.name}")
        payload = b""
        while len(payload) <= before.st_size:
            chunk = os.read(descriptor, min(65536, before.st_size - len(payload)))
            if not chunk:
                break
            payload += chunk
        after = os.fstat(descriptor)
        if len(payload) != before.st_size or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise DeadlineError(f"deadline evidence changed while reading: {path.name}")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DeadlineError(f"invalid deadline evidence JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise DeadlineError(f"deadline evidence must be an object: {path.name}")
    return value, _sha256_bytes(payload)


def process_identity(pid: int) -> dict[str, int | str] | None:
    """Read the Linux PID identity fields needed before signaling a process group."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    close = raw.rfind(")")
    if close < 0:
        raise DeadlineError("driver /proc identity is malformed")
    fields = raw[close + 2 :].split()
    if len(fields) < 20:
        raise DeadlineError("driver /proc identity is truncated")
    return {
        "pid": pid,
        "state": fields[0],
        "pgid": int(fields[2]),
        "start_time_ticks": int(fields[19]),
    }


def _validate_process_identity(expected: dict[str, int]) -> dict[str, int | str] | None:
    observed = process_identity(expected["pid"])
    if observed is None:
        return None
    for key in ("pid", "pgid", "start_time_ticks"):
        if observed[key] != expected[key]:
            raise DeadlineError(f"driver process identity changed at {key}")
    return observed


def _journal_digest(path: Path, *, allow_empty: bool = False) -> str:
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
        or metadata.st_nlink != 1
        or (not allow_empty and metadata.st_size <= 0)
    ):
        raise DeadlineError(f"unsafe committed journal file: {path.name}")
    return _sha256_file(path)


def _read_journal_json(path: Path) -> dict[str, Any]:
    _journal_digest(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeadlineError(f"invalid committed journal JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise DeadlineError(f"committed journal must be an object: {path.name}")
    return value


def scan_committed_prefix(workdir: Path) -> dict[str, Any]:
    """Hash the exact contiguous prefix visible after the commit gate is sealed."""
    done_steps: set[int] = set()
    for path in workdir.glob("done_*.flag"):
        raw = path.name[5:-5]
        try:
            step = int(raw)
        except ValueError as exc:
            raise DeadlineError(f"invalid done marker name: {path.name}") from exc
        if path.name != f"done_{step:02d}.flag" or step in done_steps:
            raise DeadlineError(f"non-canonical done marker: {path.name}")
        done_steps.add(step)
    if not done_steps or 0 not in done_steps:
        raise DeadlineError("deadline freeze is missing done_00.flag")
    prefix_step = max(done_steps)
    if done_steps != set(range(prefix_step + 1)):
        raise DeadlineError("deadline freeze found non-contiguous done markers")

    files: dict[str, str] = {}
    commands: list[dict[str, Any]] = []
    final_state: dict[str, Any] | None = None
    for step in range(prefix_step + 1):
        suffix = f"{step:02d}"
        done = workdir / f"done_{suffix}.flag"
        state_path = workdir / f"state_{suffix}.json"
        files[done.name] = _journal_digest(done, allow_empty=True)
        state = _read_journal_json(state_path)
        files[state_path.name] = _journal_digest(state_path)
        if state.get("step") != step:
            raise DeadlineError(f"state_{suffix}.json has the wrong step")
        final_state = state
        if step:
            log_path = workdir / f"log_{suffix}.json"
            receipt_path = workdir / f"{RECEIPT_PREFIX}{suffix}.json"
            log = _read_journal_json(log_path)
            files[log_path.name] = _journal_digest(log_path)
            _read_journal_json(receipt_path)
            files[receipt_path.name] = _journal_digest(receipt_path)
            command = log.get("command")
            if not isinstance(command, dict) or not isinstance(
                command.get("action"), str
            ):
                raise DeadlineError(f"log_{suffix}.json has no command object")
            commands.append(command)
    assert final_state is not None
    trace = b"".join(
        (json.dumps(command, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for command in commands
    )
    raw_trace = workdir / "command_trace.jsonl"
    return {
        "prefix_step": prefix_step,
        "success": final_state.get("success") is True,
        "final_state_sha256": files[f"state_{prefix_step:02d}.json"],
        "command_trace_sha256": _sha256_bytes(trace),
        "raw_command_trace_sha256": _journal_digest(raw_trace, allow_empty=True),
        "files_sha256": files,
    }


def _remove_done_marker(path: Path) -> None:
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
        or metadata.st_nlink != 1
    ):
        raise DeadlineError(f"unsafe late commit marker: {path.name}")
    path.unlink()
    directory_fd = _safe_parent(path)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_deadline_receipt(
    path: Path,
    *,
    step: int,
    contract: dict[str, Any],
    contract_sha256: str,
) -> int:
    if step <= 0:
        raise DeadlineError("deadline receipts are only valid for command steps")
    if path.name != f"{RECEIPT_PREFIX}{step:02d}.json":
        raise DeadlineError(f"non-canonical deadline receipt: {path.name}")
    receipt, _digest = _read_trusted_json(path)
    published_ns = receipt.get("done_published_monotonic_ns")
    if deadline_receipt_problem(
        receipt,
        step=step,
        contract=contract,
        contract_sha256=contract_sha256,
        require_before_deadline=False,
    ):
        raise DeadlineError(f"invalid deadline receipt for step {step}")
    assert type(published_ns) is int
    return published_ns


def _seal_committed_prefix(
    workdir: Path, contract: dict[str, Any], contract_sha256: str
) -> list[int]:
    """Drop only a final marker whose post-publication receipt is late or absent."""
    done_steps = sorted(
        int(path.name[5:-5])
        for path in workdir.glob("done_*.flag")
        if path.name[5:-5].isdigit()
    )
    dropped: list[int] = []
    rejecting = False
    for step in done_steps:
        if step == 0:
            continue
        marker = workdir / f"done_{step:02d}.flag"
        receipt_path = workdir / f"{RECEIPT_PREFIX}{step:02d}.json"
        try:
            published_ns = _read_deadline_receipt(
                receipt_path,
                step=step,
                contract=contract,
                contract_sha256=contract_sha256,
            )
        except FileNotFoundError:
            rejecting = True
            published_ns = None
        if (
            published_ns is not None
            and published_ns >= contract["deadline_monotonic_ns"]
        ):
            rejecting = True
        if rejecting:
            _remove_done_marker(marker)
            dropped.append(step)

    remaining = {
        int(path.name[5:-5])
        for path in workdir.glob("done_*.flag")
        if path.name[5:-5].isdigit()
    }
    prefix_step = max(remaining, default=-1)
    for receipt_path in workdir.glob(f"{RECEIPT_PREFIX}*.json"):
        raw = receipt_path.name[len(RECEIPT_PREFIX) : -5]
        if not raw.isdigit():
            raise DeadlineError(f"invalid deadline receipt name: {receipt_path.name}")
        step = int(raw)
        if step <= 0:
            raise DeadlineError("deadline receipts are only valid for command steps")
        if receipt_path.name != f"{RECEIPT_PREFIX}{step:02d}.json":
            raise DeadlineError(f"non-canonical deadline receipt: {receipt_path.name}")
        if step <= prefix_step:
            continue
        published_ns = _read_deadline_receipt(
            receipt_path,
            step=step,
            contract=contract,
            contract_sha256=contract_sha256,
        )
        if published_ns < contract["deadline_monotonic_ns"]:
            raise DeadlineError("an on-time receipt is missing its done marker")
    return dropped


class DeadlineController:
    def __init__(
        self,
        *,
        workdir: Path,
        run_id: str,
        driver_identity: dict[str, int],
        timeout_seconds: float,
        kill_after_seconds: float,
        external_sha256: str,
    ) -> None:
        self.workdir = workdir
        self.run_id = run_id
        self.driver_identity = driver_identity
        self.timeout_ns = int(timeout_seconds * 1_000_000_000)
        self.kill_after_seconds = kill_after_seconds
        self.external_sha256 = external_sha256
        self.nonce = secrets.token_hex(16)
        self.cancelled = threading.Event()
        self.started = threading.Event()
        self.fired = False
        self.error: str | None = None
        self.cancelled_monotonic_ns: int | None = None
        self.contract: dict[str, Any] | None = None
        self.contract_sha256: str | None = None
        self.freeze_sha256: str | None = None
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self._state_lock = threading.Lock()

    def start(self, started_monotonic: float) -> None:
        with self._start_lock:
            if self.started.is_set():
                return
            gate = self.workdir / GATE_NAME
            descriptor = _open_gate(gate, create=True)
            os.close(descriptor)
            started_ns = int(started_monotonic * 1_000_000_000)
            deadline_ns = started_ns + self.timeout_ns
            self.contract = {
                "schema_version": 1,
                "protocol": DEADLINE_PROTOCOL,
                "run_id": self.run_id,
                "nonce": self.nonce,
                "started_monotonic_ns": started_ns,
                "deadline_monotonic_ns": deadline_ns,
                "timeout_ns": self.timeout_ns,
                "driver": self.driver_identity,
                "external_deadline_sha256": self.external_sha256,
            }
            self.contract_sha256 = _atomic_json(
                self.workdir / CONTRACT_NAME, self.contract
            )
            self.started.set()
            self._thread = threading.Thread(
                target=self._watch, name="robocasa-deadline", daemon=True
            )
            self._thread.start()

    def _watch(self) -> None:
        assert self.contract is not None
        deadline_ns = self.contract["deadline_monotonic_ns"]
        while True:
            remaining = max(0.0, (deadline_ns - _time.monotonic_ns()) / 1e9)
            self.cancelled.wait(remaining)
            observed_ns = _time.monotonic_ns()
            with self._state_lock:
                cancelled_ns = self.cancelled_monotonic_ns
                if cancelled_ns is not None and cancelled_ns < deadline_ns:
                    return
                if observed_ns >= deadline_ns:
                    self.fired = True
                    fire = True
                else:
                    fire = False
                    cancelled = self.cancelled.is_set()
            if fire:
                try:
                    self._freeze()
                except BaseException as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                return
            if cancelled:
                return

    def _acquire_gate(self, descriptor: int) -> None:
        lock_deadline = _time.monotonic() + self.kill_after_seconds
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if _time.monotonic() >= lock_deadline:
                    raise DeadlineError("timed out sealing the deadline commit gate")
                _time.sleep(0.01)

    @staticmethod
    def _driver_stop(state: str, *, supervisor_stopped: bool) -> dict[str, Any]:
        return {
            "observed_state": state,
            "supervisor_stopped": supervisor_stopped,
            "kill_sent": False,
            "kill_observed": state in {"exited", "Z", "X"},
        }

    def _stop_driver(self) -> dict[str, Any]:
        observed = _validate_process_identity(self.driver_identity)
        if observed is None:
            return self._driver_stop("exited", supervisor_stopped=False)
        state = str(observed["state"])
        if state in {"Z", "X"}:
            return self._driver_stop(state, supervisor_stopped=False)
        try:
            os.killpg(self.driver_identity["pgid"], signal.SIGSTOP)
        except ProcessLookupError:
            return self._driver_stop("exited", supervisor_stopped=False)
        stop_deadline = _time.monotonic() + self.kill_after_seconds
        while _time.monotonic() < stop_deadline:
            observed = _validate_process_identity(self.driver_identity)
            if observed is None:
                return self._driver_stop("exited", supervisor_stopped=False)
            state = str(observed["state"])
            if state in {"T", "t", "Z", "X"}:
                return self._driver_stop(state, supervisor_stopped=state in {"T", "t"})
            _time.sleep(0.01)
        raise DeadlineError("driver did not stop at the planner deadline")

    def _kill_driver(self, driver_stop: dict[str, Any]) -> None:
        if not driver_stop["supervisor_stopped"]:
            return
        observed = _validate_process_identity(self.driver_identity)
        if observed is None or observed["state"] not in {"T", "t"}:
            raise DeadlineError("stopped driver identity changed before SIGKILL")
        try:
            os.killpg(self.driver_identity["pgid"], signal.SIGKILL)
        except ProcessLookupError:
            raise DeadlineError("stopped driver exited before supervisor SIGKILL")
        driver_stop["kill_sent"] = True
        kill_deadline = _time.monotonic() + self.kill_after_seconds
        while _time.monotonic() < kill_deadline:
            observed = _validate_process_identity(self.driver_identity)
            if observed is None or observed["state"] in {"Z", "X"}:
                driver_stop["kill_observed"] = True
                return
            _time.sleep(0.01)
        raise DeadlineError("supervisor SIGKILL did not terminate the driver")

    def _freeze(self) -> None:
        assert self.contract is not None and self.contract_sha256 is not None
        gate_descriptor = _open_gate(self.workdir / GATE_NAME, create=False)
        try:
            self._acquire_gate(gate_descriptor)
            seal = {
                "schema_version": 1,
                "protocol": DEADLINE_PROTOCOL,
                "run_id": self.run_id,
                "nonce": self.nonce,
                "deadline_monotonic_ns": self.contract["deadline_monotonic_ns"],
                "sealed_monotonic_ns": _time.monotonic_ns(),
                "contract_sha256": self.contract_sha256,
            }
            seal_sha256 = _atomic_json(self.workdir / SEAL_NAME, seal)
            driver_stop = self._stop_driver()
            dropped_done_steps = _seal_committed_prefix(
                self.workdir, self.contract, self.contract_sha256
            )
            prefix = scan_committed_prefix(self.workdir)
            self._kill_driver(driver_stop)
            freeze = {
                "schema_version": 1,
                "protocol": DEADLINE_PROTOCOL,
                "run_id": self.run_id,
                "nonce": self.nonce,
                "deadline_monotonic_ns": self.contract["deadline_monotonic_ns"],
                "frozen_monotonic_ns": _time.monotonic_ns(),
                "contract_sha256": self.contract_sha256,
                "seal_sha256": seal_sha256,
                "driver": self.driver_identity,
                "driver_stop": driver_stop,
                "dropped_done_steps": dropped_done_steps,
                **prefix,
            }
            self.freeze_sha256 = _atomic_json(self.workdir / FREEZE_NAME, freeze)
        finally:
            try:
                fcntl.flock(gate_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(gate_descriptor)

    def finish(self) -> None:
        self.note_planner_terminal()
        if self._thread is not None:
            self._thread.join(timeout=self.kill_after_seconds * 2 + 5)
            if self._thread.is_alive():
                self.error = "deadline supervisor thread did not terminate"

    def status(self) -> dict[str, Any]:
        return {
            "protocol": DEADLINE_PROTOCOL,
            "run_id": self.run_id,
            "nonce": self.nonce,
            "fired": self.fired,
            "contract_sha256": self.contract_sha256,
            "freeze_sha256": self.freeze_sha256,
            "error": self.error,
        }

    def note_planner_terminal(self, observed_monotonic_ns: int | None = None) -> None:
        with self._state_lock:
            if self.fired:
                return
            observed = (
                _time.monotonic_ns()
                if observed_monotonic_ns is None
                else observed_monotonic_ns
            )
            if (
                self.cancelled_monotonic_ns is None
                or observed < self.cancelled_monotonic_ns
            ):
                self.cancelled_monotonic_ns = observed
            self.cancelled.set()


class _TimeProxy:
    def __init__(self, controller: DeadlineController) -> None:
        self._controller = controller

    def monotonic(self) -> float:
        value = _time.monotonic()
        self._controller.start(value)
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(_time, name)


class _ObservedProcess:
    def __init__(self, process: Any, controller: DeadlineController) -> None:
        self._process = process
        self._controller = controller

    def poll(self) -> int | None:
        value = self._process.poll()
        if value is not None:
            self._controller.note_planner_terminal()
        return value

    def wait(self, *args: Any, **kwargs: Any) -> int:
        value = self._process.wait(*args, **kwargs)
        self._controller.note_planner_terminal()
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._process, name)


class _SubprocessProxy:
    def __init__(self, controller: DeadlineController) -> None:
        self._controller = controller

    def Popen(self, *args: Any, **kwargs: Any) -> _ObservedProcess:  # noqa: N802
        return _ObservedProcess(
            _subprocess.Popen(*args, **kwargs),
            self._controller,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(_subprocess, name)


def _normalize_status(
    status_path: Path, controller: DeadlineController, external_rc: int
) -> int:
    try:
        status, _digest = _read_trusted_json(status_path)
    except (OSError, DeadlineError):
        status = {"state": "finished", "termination_cause": "wrapper_error"}
    supervisor = controller.status()
    status["external_deadline_fired"] = status.get("deadline_fired")
    status["deadline_supervisor"] = supervisor
    rc = external_rc
    if controller.error is not None or not controller.started.is_set():
        status["termination_cause"] = "wrapper_error"
        status["wrapper_rc"] = EXIT_WRAPPER_ERROR
        status["supervision_error"] = controller.error or "deadline did not start"
        rc = EXIT_WRAPPER_ERROR
    elif controller.fired:
        cause, wrapper_rc = normalize_deadline_outcome(
            status.get("termination_cause"),
            status.get("wrapper_rc"),
            supervisor_fired=True,
            provider_violation=status.get("provider_protocol_violation"),
            forbidden_web_search_event_count=status.get(
                "forbidden_web_search_event_count"
            ),
        )
        status["termination_cause"] = cause
        status["wrapper_rc"] = wrapper_rc
        if cause == "planner_timeout":
            status["deadline_fired"] = True
        rc = wrapper_rc
    _atomic_json(status_path, status)
    return rc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--driver-pid", type=int, required=True)
    parser.add_argument("--driver-pgid", type=int, required=True)
    parser.add_argument("--driver-start-time", type=int, required=True)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.script_args or args.script_args[0] != "--":
        parser.error("expected external deadline arguments after --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = _read_source(args.source, args.sha256)
    namespace: dict[str, Any] = {
        "__name__": "_rpent_external_deadline",
        "__file__": str(args.source),
        "__package__": None,
        "__builtins__": __builtins__,
    }
    exec(compile(source, str(args.source), "exec", dont_inherit=True), namespace)
    external_args = namespace["parse_args"](args.script_args[1:])
    workdir = args.workdir.resolve(strict=True)
    status_path = Path(external_args.status).resolve(strict=False)
    if status_path != workdir / "planner_status.json":
        raise DeadlineError(
            "external planner status must stay inside the rollout workdir"
        )
    driver_identity = {
        "pid": args.driver_pid,
        "pgid": args.driver_pgid,
        "start_time_ticks": args.driver_start_time,
    }
    observed = _validate_process_identity(driver_identity)
    if observed is None:
        raise DeadlineError("driver exited before the planner deadline was established")
    controller = DeadlineController(
        workdir=workdir,
        run_id=args.run_id,
        driver_identity=driver_identity,
        timeout_seconds=external_args.timeout_seconds,
        kill_after_seconds=external_args.kill_after_seconds,
        external_sha256=args.sha256,
    )
    namespace["time"] = _TimeProxy(controller)
    namespace["subprocess"] = _SubprocessProxy(controller)
    previous = sys.argv
    try:
        sys.argv = [str(args.source), *args.script_args[1:]]
        external_rc = int(namespace["main"](args.script_args[1:]))
    finally:
        sys.argv = previous
        controller.finish()
    return _normalize_status(status_path, controller, external_rc)


if __name__ == "__main__":
    raise SystemExit(main())
