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

"""Build a canonical, task-isolated seed-0 memory pack."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .protocol import (
    EMPTY_MEMORY_TASKS,
    MEMORY_PAIR_TASKS,
    PROTOCOL_ID,
    command_problem,
)

SCHEMA_VERSION = 3
SOURCE_DIRS = {
    "atomic": "explore_atomic_recipe",
    "composite_seen": "exploration_seen_recipe",
    "composite_unseen": "exploration_unseen_recipe",
}
ALIASES = frozenset({"rldx_skill", "rldx_arm", "vla_act"})
LEGACY_NORMALIZATIONS = {
    "rldx_skill,rldx_arm,vla_act": "vla_act with task_language only",
    "set_gripper.width": "set_gripper.gripper",
    "scripted_grasp": "open, hover, descend, close, lift",
    "audit.command_sequence": "removed; paired JSONL is the sole command authority",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except Exception as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _normalized_trace(
    path: Path, task_language: str, *, normalize_vla: bool = True
) -> bytes:
    commands: list[dict[str, Any]] = []
    for number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="strict").splitlines(), 1
    ):
        if not raw.strip():
            raise ValueError(f"blank JSONL line {path}:{number}")
        try:
            command = json.loads(raw)
        except Exception as exc:
            raise ValueError(f"invalid JSONL {path}:{number}: {exc}") from exc
        if not isinstance(command, dict) or not isinstance(command.get("action"), str):
            raise ValueError(f"invalid command {path}:{number}")
        normalized = (
            _normalize_legacy_command(command, task_language, path, number)
            if normalize_vla
            else [command]
        )
        for item in normalized:
            if problem := command_problem(item, task_language=task_language):
                raise ValueError(f"invalid command {path}:{number}: {problem}")
            commands.append(item)
    if not commands:
        raise ValueError(f"empty command trace: {path}")
    return b"".join(
        (json.dumps(command, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for command in commands
    )


def _commands_from_trace_bytes(trace: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in trace.decode("utf-8").splitlines()]


def _audit_contains_command(value: Any) -> bool:
    if isinstance(value, dict):
        return "action" in value or any(
            _audit_contains_command(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_audit_contains_command(item) for item in value)
    return False


def _normalize_legacy_command(
    command: dict[str, Any], task_language: str, path: Path, number: int
) -> list[dict[str, Any]]:
    action = command["action"]
    if action in ALIASES:
        return [{"action": "vla_act", "prompt": task_language}]
    if action == "set_gripper" and "width" in command:
        if set(command) - {"action", "width", "steps"}:
            raise ValueError(f"invalid legacy set_gripper {path}:{number}")
        normalized = {"action": "set_gripper", "gripper": command["width"]}
        if "steps" in command:
            normalized["steps"] = command["steps"]
        return [normalized]
    if action != "scripted_grasp":
        return [command]
    if set(command) - {
        "action",
        "xyz",
        "approach_z",
        "grasp_z_offset",
        "step_clip",
    }:
        raise ValueError(f"invalid legacy scripted_grasp {path}:{number}")
    xyz = command.get("xyz")
    approach = command.get("approach_z", 0.10)
    offset = command.get("grasp_z_offset", 0.0)
    step_clip = command.get("step_clip", 0.02)
    if (
        not isinstance(xyz, list)
        or len(xyz) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in xyz
        )
        or isinstance(approach, bool)
        or not isinstance(approach, (int, float))
        or isinstance(offset, bool)
        or not isinstance(offset, (int, float))
    ):
        raise ValueError(f"invalid legacy scripted_grasp {path}:{number}")
    x, y, z = xyz
    return [
        {"action": "set_gripper", "gripper": -1.0, "steps": 4},
        {
            "action": "move_to",
            "xyz": [x, y, z + approach],
            "gripper": -1.0,
            "step_clip": step_clip,
            "tol": 0.012,
        },
        {
            "action": "move_to",
            "xyz": [x, y, z + offset],
            "gripper": -1.0,
            "step_clip": 0.012,
            "tol": 0.01,
        },
        {"action": "set_gripper", "gripper": 1.0, "steps": 14},
        {
            "action": "move_to",
            "xyz": [x, y, z + approach + 0.05],
            "gripper": "hold",
            "step_clip": 0.015,
            "tol": 0.012,
        },
    ]


def _atomic_write(path: Path, data: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def build_memory_pack(migration_root: Path, destination: Path) -> dict[str, Any]:
    """Pack exactly 43 successful audit/trace pairs plus seven empty entries."""
    migration_root = Path(migration_root)
    destination = Path(destination)
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError(f"memory destination must be a real directory: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"memory destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    task_to_source = {
        task: directory
        for directory, name in SOURCE_DIRS.items()
        for task in _tasks_for_directory(directory)
    }
    for task in sorted(MEMORY_PAIR_TASKS):
        source = migration_root / SOURCE_DIRS[task_to_source[task]]
        audit_source = source / f"{task}_s0.json"
        trace_source = source / f"recipe_{task}_s0.jsonl"
        for path in (audit_source, trace_source):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing or unsafe memory source: {path}")
        audit = _read_json(audit_source)
        if (
            audit.get("task") != task
            or type(audit.get("seed")) is not int
            or audit["seed"] != 0
        ):
            raise ValueError(f"memory identity mismatch: {audit_source}")
        if audit.get("success") is not True:
            raise ValueError(
                f"memory is not a successful seed-0 rollout: {audit_source}"
            )
        task_language = audit.get("task_language")
        if not isinstance(task_language, str) or not task_language.strip():
            raise ValueError(f"memory has no task language: {audit_source}")
        task_dir = destination / task
        task_dir.mkdir()
        audit_dest = task_dir / f"{task}_s0.json"
        trace_dest = task_dir / f"{task}_s0.jsonl"
        trace_bytes = _normalized_trace(trace_source, task_language)
        commands = _commands_from_trace_bytes(trace_bytes)
        audit.pop("command_sequence", None)
        if "n_commands" in audit:
            audit["n_commands"] = len(commands)
        if "manual_cmds" in audit:
            audit["manual_cmds"] = sum(
                command["action"] != "vla_act" for command in commands
            )
        if "vla_calls" in audit:
            audit["vla_calls"] = sum(
                command["action"] == "vla_act" for command in commands
            )
        audit_bytes = (
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _atomic_write(audit_dest, audit_bytes)
        _atomic_write(trace_dest, trace_bytes)
        entries.append(
            {
                "task": task,
                "memory": "seed0_success",
                "source": task_to_source[task],
                "files": {
                    audit_dest.name: hashlib.sha256(audit_bytes).hexdigest(),
                    trace_dest.name: hashlib.sha256(trace_bytes).hexdigest(),
                },
            }
        )
    for task in sorted(EMPTY_MEMORY_TASKS):
        (destination / task).mkdir()
        entries.append({"task": task, "memory": "empty", "source": None, "files": {}})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_ID,
        "source": "RoboCasa seed-0 migration recipes",
        "source_directories": SOURCE_DIRS,
        "legacy_normalizations": LEGACY_NORMALIZATIONS,
        "algorithm": "sha256",
        "entries": entries,
    }
    _atomic_write(
        destination / "manifest.json",
        (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )
    return manifest


def _tasks_for_directory(label: str) -> frozenset[str]:
    from .protocol import ATOMIC_TASKS, COMPOSITE_SEEN_TASKS, COMPOSITE_UNSEEN_TASKS

    return {
        "atomic": frozenset(ATOMIC_TASKS),
        "composite_seen": frozenset(COMPOSITE_SEEN_TASKS),
        "composite_unseen": frozenset(COMPOSITE_UNSEEN_TASKS),
    }[label]


def validate_memory_pack(root: Path) -> list[str]:
    """Validate identities, exact file names, empty whitelist and manifest hashes."""
    root = Path(root)
    problems: list[str] = []
    try:
        manifest = _read_json(root / "manifest.json")
    except ValueError as exc:
        return [str(exc)]
    entries = manifest.get("entries")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol") != PROTOCOL_ID
        or manifest.get("source") != "RoboCasa seed-0 migration recipes"
        or manifest.get("source_directories") != SOURCE_DIRS
        or manifest.get("legacy_normalizations") != LEGACY_NORMALIZATIONS
        or manifest.get("algorithm") != "sha256"
    ):
        problems.append("unsupported memory manifest")
    if not isinstance(entries, list) or len(entries) != 50:
        return problems + ["memory manifest must contain exactly 50 task entries"]
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("task"), str):
            problems.append("invalid memory manifest entry")
            continue
        task = entry["task"]
        if task in seen:
            problems.append(f"duplicate memory task: {task}")
        seen.add(task)
        directory = root / task
        if directory.is_symlink() or not directory.is_dir():
            problems.append(f"missing or unsafe memory directory for {task}")
            continue
        actual = {path.name for path in directory.iterdir()}
        expected_files = entry.get("files")
        if not isinstance(expected_files, dict) or actual != set(expected_files):
            problems.append(f"unexpected memory files for {task}: {sorted(actual)}")
            continue
        expected_kind = "empty" if task in EMPTY_MEMORY_TASKS else "seed0_success"
        if entry.get("memory") != expected_kind:
            problems.append(f"wrong memory kind for {task}")
        expected_source = next(
            (label for label in SOURCE_DIRS if task in _tasks_for_directory(label)),
            None,
        )
        if task in EMPTY_MEMORY_TASKS:
            expected_source = None
        if entry.get("source") != expected_source:
            problems.append(f"wrong memory provenance for {task}")
        for name, digest in expected_files.items():
            path = directory / name
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                problems.append(f"memory hash mismatch: {task}/{name}")
        if expected_kind == "seed0_success" and actual == {
            f"{task}_s0.json",
            f"{task}_s0.jsonl",
        }:
            try:
                audit = _read_json(directory / f"{task}_s0.json")
                if (
                    audit.get("task") != task
                    or type(audit.get("seed")) is not int
                    or audit.get("seed") != 0
                    or audit.get("success") is not True
                ):
                    problems.append(f"invalid seed-0 audit semantics for {task}")
                language = audit.get("task_language")
                if not isinstance(language, str) or not language.strip():
                    problems.append(f"missing task language for {task}")
                else:
                    trace_bytes = _normalized_trace(
                        directory / f"{task}_s0.jsonl", language, normalize_vla=False
                    )
                    commands = _commands_from_trace_bytes(trace_bytes)
                    if _audit_contains_command(audit):
                        problems.append(
                            f"seed-0 audit embeds commands instead of using JSONL for {task}"
                        )
                    expected_stats = {
                        "n_commands": len(commands),
                        "manual_cmds": sum(
                            command["action"] != "vla_act" for command in commands
                        ),
                        "vla_calls": sum(
                            command["action"] == "vla_act" for command in commands
                        ),
                    }
                    for key, expected_value in expected_stats.items():
                        if key in audit and audit[key] != expected_value:
                            problems.append(
                                f"seed-0 audit {key} differs from JSONL for {task}"
                            )
            except (OSError, UnicodeError, ValueError) as exc:
                problems.append(f"invalid memory semantics for {task}: {exc}")
    expected = set(MEMORY_PAIR_TASKS) | set(EMPTY_MEMORY_TASKS)
    if seen != expected:
        problems.append("memory manifest task set differs from protocol")
    allowed_root = seen | {"manifest.json"}
    extras = {path.name for path in root.iterdir()} - allowed_root
    if extras:
        problems.append(f"unexpected pack entries: {sorted(extras)}")
    return problems


# Public action-oriented name used by runners; retain the descriptive builder name.
pack_memory = build_memory_pack
