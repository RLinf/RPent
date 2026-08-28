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

"""Canonical completed-cell artifacts and atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator

from .protocol import Cell, command_problem


class Completion(str, Enum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class Integrity(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CellResult:
    completion: Completion
    outcome: Outcome
    integrity: Integrity
    reason: str | None = None

    def __post_init__(self) -> None:
        if (
            self.completion is not Completion.COMPLETED
            and self.outcome is not Outcome.UNKNOWN
        ):
            raise ValueError("incomplete cells cannot claim a task outcome")
        if (
            self.completion is not Completion.COMPLETED
            and self.integrity is Integrity.VALID
        ):
            raise ValueError("incomplete cells cannot be canonical")

    @property
    def canonical(self) -> bool:
        return (
            self.completion is Completion.COMPLETED
            and self.integrity is Integrity.VALID
        )


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _directory_problem(metadata: os.stat_result, label: str) -> str | None:
    if not stat.S_ISDIR(metadata.st_mode):
        return f"{label} must be a directory"
    if metadata.st_uid != os.geteuid():
        return f"{label} must be owned by the current user"
    if metadata.st_mode & 0o022:
        return f"{label} must not be writable by group or other"
    return None


def _component_name(value: str, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"artifact {label} is not one safe path component")
    return value


def _resolved_inside(path: Path, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"artifact {label} escapes or cannot resolve inside root"
        ) from exc
    return resolved


def _open_root(root: Path) -> tuple[int, Path]:
    try:
        descriptor = os.open(root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise ValueError(
            f"artifact root must be a real safe directory: {root}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if problem := _directory_problem(metadata, "artifact root"):
            raise ValueError(problem)
        resolved = root.resolve(strict=True)
        resolved_metadata = resolved.stat()
        if (metadata.st_dev, metadata.st_ino) != (
            resolved_metadata.st_dev,
            resolved_metadata.st_ino,
        ):
            raise ValueError("artifact root changed while it was being opened")
        return descriptor, resolved
    except Exception:
        os.close(descriptor)
        raise


def _open_child(
    parent_descriptor: int,
    parent_path: Path,
    name: str,
    root: Path,
    label: str,
    *,
    create: bool,
) -> int | None:
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
    except FileNotFoundError:
        if not create:
            return None
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            pass
        try:
            descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
        except OSError as exc:
            raise ValueError(f"artifact {label} could not be created safely") from exc
    except OSError as exc:
        raise ValueError(f"artifact {label} must be a real safe directory") from exc
    try:
        metadata = os.fstat(descriptor)
        if problem := _directory_problem(metadata, f"artifact {label}"):
            raise ValueError(problem)
        path = parent_path / name
        resolved = _resolved_inside(path, root, label)
        resolved_metadata = resolved.stat()
        if (metadata.st_dev, metadata.st_ino) != (
            resolved_metadata.st_dev,
            resolved_metadata.st_ino,
        ):
            raise ValueError(f"artifact {label} changed while it was being opened")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


@contextmanager
def _secure_artifact_directory(
    root: Path, cell: Cell, *, create: bool
) -> Iterator[tuple[Path, int | None]]:
    """Open the task directory through checked, non-symlink directory handles."""
    root_path = Path(root)
    split_name = _component_name(cell.split.value, "split")
    task_name = _component_name(cell.task, "task")
    directory = root_path / split_name / task_name
    root_descriptor, resolved_root = _open_root(root_path)
    split_descriptor: int | None = None
    task_descriptor: int | None = None
    try:
        try:
            directory.resolve(strict=False).relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            raise ValueError("artifact directory escapes root") from exc
        split_descriptor = _open_child(
            root_descriptor,
            root_path,
            split_name,
            resolved_root,
            "split",
            create=create,
        )
        if split_descriptor is None:
            yield directory, None
            return
        task_descriptor = _open_child(
            split_descriptor,
            root_path / split_name,
            task_name,
            resolved_root,
            "task",
            create=create,
        )
        yield directory, task_descriptor
    finally:
        if task_descriptor is not None:
            os.close(task_descriptor)
        if split_descriptor is not None:
            os.close(split_descriptor)
        os.close(root_descriptor)


def _atomic_write(directory_descriptor: int, name: str, data: bytes) -> None:
    name = _component_name(name, "file")
    temporary = f".{name}.{os.getpid()}.{secrets.token_hex(8)}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_descriptor)
    try:
        os.fchmod(descriptor, 0o600)
        remaining = memoryview(data)
        while remaining:
            written = os.write(descriptor, remaining)
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass


def artifact_directory(root: Path, cell: Cell) -> Path:
    """Return the cell path after validating every existing directory component."""
    with _secure_artifact_directory(root, cell, create=False) as (
        directory,
        _descriptor,
    ):
        return directory


@contextmanager
def secure_artifact_directory(
    root: Path, cell: Cell, *, create: bool = True
) -> Iterator[tuple[Path, int | None]]:
    """Yield a checked task-directory path and an open non-following handle."""
    with _secure_artifact_directory(root, cell, create=create) as opened:
        yield opened


@contextmanager
def secure_artifact_subdirectory(
    root: Path, cell: Cell, *components: str
) -> Iterator[tuple[Path, int]]:
    """Create and hold a checked descendant directory under one task."""
    with _secure_artifact_directory(root, cell, create=True) as (
        directory,
        task_descriptor,
    ):
        assert task_descriptor is not None
        root_resolved = Path(root).resolve(strict=True)
        current_path = directory
        current_descriptor = task_descriptor
        children: list[int] = []
        try:
            for index, raw_name in enumerate(components):
                name = _component_name(raw_name, f"subdirectory {index + 1}")
                descriptor = _open_child(
                    current_descriptor,
                    current_path,
                    name,
                    root_resolved,
                    f"subdirectory {index + 1}",
                    create=True,
                )
                assert descriptor is not None
                children.append(descriptor)
                current_descriptor = descriptor
                current_path = current_path / name
            yield current_path, current_descriptor
        finally:
            for descriptor in reversed(children):
                os.close(descriptor)


def atomic_write_artifact_file(
    directory_descriptor: int, name: str, data: bytes
) -> None:
    """Atomically publish one regular file through an open artifact dirfd."""
    _atomic_write(directory_descriptor, name, data)


def publish_completed_cell(
    root: Path,
    cell: Cell,
    result: CellResult,
    audit: dict[str, Any],
    commands: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Publish audit, normalized JSONL and commit manifest, commit marker last."""
    if not result.canonical:
        raise ValueError("only completed, valid cells may be published canonically")
    commands = list(commands)
    task_language = audit.get("task_language")
    for number, command in enumerate(commands, 1):
        if problem := command_problem(command, task_language=task_language):
            raise ValueError(f"command trace line {number} is invalid: {problem}")
    if (
        audit.get("task") != cell.task
        or type(audit.get("seed")) is not int
        or audit["seed"] != cell.seed
    ):
        raise ValueError("audit identity does not match cell")
    if audit.get("success") is not (result.outcome is Outcome.SUCCESS):
        raise ValueError("audit success disagrees with CellResult")
    run_config_sha256 = audit.get("run_config_sha256")
    if (
        not isinstance(run_config_sha256, str)
        or len(run_config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in run_config_sha256)
    ):
        raise ValueError("audit must contain a valid run_config_sha256")
    audit_name = f"{cell.tag}.json"
    trace_name = f"{cell.tag}.jsonl"
    commit_name = f"{cell.tag}.completed.json"
    audit_bytes = (
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    trace_bytes = b"".join(
        (json.dumps(command, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for command in commands
    )
    commit = {
        "schema_version": 1,
        "run_config_sha256": run_config_sha256,
        "cell": {"split": cell.split.value, "task": cell.task, "seed": cell.seed},
        "result": {
            key: value.value if isinstance(value, Enum) else value
            for key, value in asdict(result).items()
        },
        "files": {
            audit_name: hashlib.sha256(audit_bytes).hexdigest(),
            trace_name: hashlib.sha256(trace_bytes).hexdigest(),
        },
    }
    with _secure_artifact_directory(root, cell, create=True) as (
        _directory,
        directory_descriptor,
    ):
        assert directory_descriptor is not None
        _atomic_write(directory_descriptor, audit_name, audit_bytes)
        _atomic_write(directory_descriptor, trace_name, trace_bytes)
        _atomic_write(
            directory_descriptor,
            commit_name,
            (
                json.dumps(commit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode(),
        )
    return commit
