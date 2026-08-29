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

"""Resume-aware multi-GPU scheduler for the frozen RoboCasa matrix."""

from __future__ import annotations

import fcntl
import json
import os
import queue
import stat
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .artifacts import Completion, artifact_directory
from .executor import (
    EXIT_ARTIFACT,
    EXIT_CANONICAL,
    EXIT_CONFIGURATION,
    EXIT_RETRYABLE_INFRA,
    Execution,
    ExecutorConfig,
    ensure_execution_run_manifest,
    execute_cell,
)
from .protocol import CELLS, Cell, Split
from .validator import validate_cell

SMOKE_CELLS = (
    Cell(Split.ATOMIC, "OpenDrawer", 1),
    Cell(Split.COMPOSITE_SEEN, "PrepareCoffee", 1),
    Cell(Split.COMPOSITE_UNSEEN, "ArrangeTea", 1),
    Cell(Split.COMPOSITE_UNSEEN, "HeatKebabSandwich", 1),
)

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_LOCK_FLAGS = (
    os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)


class PreflightFailed(RuntimeError):
    """The locked run preflight completed but rejected the configuration."""

    def __init__(self, report: dict):
        super().__init__("RoboCasa run preflight failed")
        self.report = report


def select_cells(selection: str) -> tuple[Cell, ...]:
    """Resolve a named selection while preserving frozen seed-major order."""
    if selection == "full":
        return CELLS
    if selection == "smoke-v1":
        return SMOKE_CELLS
    try:
        split = Split(selection)
    except ValueError as exc:
        raise ValueError(
            "selection must be full, smoke-v1, atomic, composite_seen, or "
            "composite_unseen"
        ) from exc
    return tuple(cell for cell in CELLS if cell.split is split)


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _open_lock_directory(path: Path, label: str) -> int:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        metadata = path.lstat()
        descriptor = os.open(path, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise RuntimeError(f"{label} must be a real private directory") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            path.is_symlink()
            or not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o700
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise RuntimeError(
                f"{label} must be owned by the current user with mode 0700"
            )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


@contextmanager
def _exclusive_lock(directory_descriptor: int, name: str, label: str) -> Iterator[None]:
    try:
        descriptor = os.open(name, _LOCK_FLAGS, 0o600, dir_fd=directory_descriptor)
    except OSError as exc:
        raise RuntimeError(f"cannot safely open {label} lock") from exc
    locked = False
    try:
        metadata = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise RuntimeError(
                f"{label} lock must be an owned, single-link 0600 regular file"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"{label} is already locked by another process") from exc
        locked = True
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _runner_locks(results_root: Path, gpus: tuple[int, ...]) -> Iterator[None]:
    results_descriptor = _open_lock_directory(results_root, "results root")
    gpu_root = Path("/tmp") / f"rpent-robocasa-locks-{os.geteuid()}"
    gpu_descriptor: int | None = None
    try:
        gpu_descriptor = _open_lock_directory(gpu_root, "GPU lock directory")
        with ExitStack() as locks:
            locks.enter_context(
                _exclusive_lock(
                    results_descriptor,
                    ".rpent-run.lock",
                    f"results root {results_root}",
                )
            )
            for gpu in sorted(gpus):
                locks.enter_context(
                    _exclusive_lock(
                        gpu_descriptor,
                        f"gpu-{gpu}.lock",
                        f"RoboCasa GPU {gpu}",
                    )
                )
            yield
    finally:
        if gpu_descriptor is not None:
            os.close(gpu_descriptor)
        os.close(results_descriptor)


def run_cells(
    config: ExecutorConfig,
    cells: Iterable[Cell],
    *,
    gpus: tuple[int, ...],
    max_attempts: int = 3,
    retry_backoff_seconds: float = 60.0,
    executor: Callable[[ExecutorConfig, Cell, int], Execution] | None = None,
    preflight: Callable[[], dict] | None = None,
) -> dict:
    """Run preflight and cells while owning the results root and requested GPUs."""
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("gpus must be a non-empty tuple of unique non-negative ids")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds cannot be negative")
    with _runner_locks(config.results_root, gpus):
        if preflight is not None:
            report = preflight()
            if report.get("ok") is not True:
                raise PreflightFailed(report)
        return _run_cells_locked(
            config,
            cells,
            gpus=gpus,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            executor=executor,
        )


def run_locked_preflight(results_root: Path, preflight: Callable[[], dict]) -> dict:
    """Run a standalone mutating preflight under the results-root lock."""
    with _runner_locks(results_root, ()):
        return preflight()


def _run_cells_locked(
    config: ExecutorConfig,
    cells: Iterable[Cell],
    *,
    gpus: tuple[int, ...],
    max_attempts: int = 3,
    retry_backoff_seconds: float = 60.0,
    executor: Callable[[ExecutorConfig, Cell, int], Execution] | None = None,
) -> dict:
    """Run cells with one serial worker per GPU and fail-closed scheduling."""
    started_wall = time.time()
    started_monotonic = time.monotonic()
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("gpus must be a non-empty tuple of unique non-negative ids")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds cannot be negative")

    fatal_path = config.results_root / "_FATAL_STOP.json"
    config.results_root.mkdir(parents=True, exist_ok=True)
    if fatal_path.exists() or fatal_path.is_symlink():
        raise RuntimeError(
            f"persistent fatal stop exists at {fatal_path}; inspect and remove it explicitly"
        )
    run_manifest = ensure_execution_run_manifest(config)
    run_config_sha256 = run_manifest["run_config_sha256"]
    requested = tuple(cells)
    existing = tuple(
        cell
        for cell in CELLS
        if (
            artifact_directory(config.results_root, cell) / f"{cell.tag}.completed.json"
        ).exists()
    )
    existing_validations = (
        validate_cell(
            config.results_root,
            cell,
            expected_run_config_sha256=run_config_sha256,
        )
        for cell in existing
    )
    if any(
        validation.provenance_problems
        and validation.result.completion is Completion.COMPLETED
        for validation in existing_validations
    ):
        raise ValueError(
            "existing completed cells do not belong to the active run configuration"
        )
    validations = {
        cell: validate_cell(
            config.results_root,
            cell,
            expected_run_config_sha256=run_config_sha256,
        )
        for cell in requested
    }
    pending = [cell for cell, validation in validations.items() if validation.resumable]
    skipped = len(requested) - len(pending)
    work: queue.Queue[Cell] = queue.Queue()
    for cell in pending:
        work.put(cell)

    stop = threading.Event()
    lock = threading.Lock()
    executions: list[Execution] = []
    fatal: dict | None = None

    def invoke(cell: Cell, gpu: int) -> Execution:
        if executor is None:
            return execute_cell(config, cell, gpu=gpu)
        return executor(config, cell, gpu)

    def worker(gpu: int) -> None:
        nonlocal fatal
        while not stop.is_set():
            try:
                cell = work.get_nowait()
            except queue.Empty:
                return
            try:
                for attempt in range(1, max_attempts + 1):
                    if stop.is_set():
                        return
                    try:
                        execution = invoke(cell, gpu)
                    except Exception as exc:
                        with lock:
                            if fatal is None:
                                fatal = {
                                    "schema_version": 1,
                                    "run_config_sha256": run_config_sha256,
                                    "cell": {
                                        "split": cell.split.value,
                                        "task": cell.task,
                                        "seed": cell.seed,
                                    },
                                    "attempt": attempt,
                                    "return_code": EXIT_CONFIGURATION,
                                    "termination_cause": "executor_exception",
                                    "reason": "executor raised an exception",
                                    "message": type(exc).__name__,
                                }
                                _atomic_json(fatal_path, fatal)
                        stop.set()
                        break
                    with lock:
                        executions.append(execution)
                    if execution.return_code == EXIT_CANONICAL:
                        break
                    if (
                        execution.return_code == EXIT_RETRYABLE_INFRA
                        and attempt < max_attempts
                    ):
                        if retry_backoff_seconds:
                            stop.wait(retry_backoff_seconds)
                        continue
                    reason = {
                        EXIT_RETRYABLE_INFRA: "infrastructure retries exhausted",
                        EXIT_ARTIFACT: "artifact or protocol validation failed",
                        EXIT_CONFIGURATION: "configuration failed",
                    }.get(execution.return_code, "unexpected executor return code")
                    with lock:
                        if fatal is None:
                            fatal = {
                                "schema_version": 1,
                                "run_config_sha256": run_config_sha256,
                                "cell": {
                                    "split": cell.split.value,
                                    "task": cell.task,
                                    "seed": cell.seed,
                                },
                                "attempt": attempt,
                                "return_code": execution.return_code,
                                "termination_cause": execution.termination_cause,
                                "reason": reason,
                                "message": execution.message,
                            }
                            _atomic_json(fatal_path, fatal)
                    stop.set()
                    break
            finally:
                work.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"robocasa-gpu-{gpu}")
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        stop.set()
        for thread in threads:
            thread.join()
        raise

    canonical = sum(
        validate_cell(
            config.results_root,
            cell,
            expected_run_config_sha256=run_config_sha256,
        ).result.canonical
        for cell in requested
    )
    return {
        "schema_version": 1,
        "run_config_sha256": run_config_sha256,
        "gpus": list(gpus),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_wall)),
        "elapsed_seconds": max(0.0, time.monotonic() - started_monotonic),
        "requested": len(requested),
        "skipped_canonical": skipped,
        "canonical": canonical,
        "complete": canonical == len(requested) and fatal is None,
        "fatal": fatal,
        "attempts": [
            {
                **asdict(item),
                "cell": {
                    "split": item.cell.split.value,
                    "task": item.cell.task,
                    "seed": item.cell.seed,
                },
                "workdir": str(item.workdir),
            }
            for item in executions
        ],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
