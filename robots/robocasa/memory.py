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

"""Verified RoboCasa task and global memory, shared by prompts and file tools."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpent.memory import MemoryManager
from rpent.tools.toolkit import readonly
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("robocasa_memory")
MEMORY_POLICIES = ("task-global", "task-only")
# Replaced with the immutable, verified HF data PR commit before publication.
DEFAULT_MEMORY_REVISION = "local-validation-pending-publication"
GLOBAL_FILE = "global/GLOBAL_MEMORY.md"
_TASK_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*\Z")
_TASK_FILE = re.compile(
    r"task_only/[A-Za-z][A-Za-z0-9]*(?:_s0\.json|_s0_recipe\.jsonl|\.md)\Z"
)


def corpus_digest(files: Mapping[str, str]) -> str:
    """Hash the canonical file-to-SHA256 mapping, independent of provenance text."""
    encoded = json.dumps(dict(files), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def task_files(task_name: str) -> tuple[str, str, str]:
    """Return canonical filenames for one task, without directory traversal."""
    if not _TASK_NAME.fullmatch(task_name):
        raise ValueError(f"invalid RoboCasa task name: {task_name!r}")
    return (
        f"task_only/{task_name}_s0.json",
        f"task_only/{task_name}_s0_recipe.jsonl",
        f"task_only/{task_name}.md",
    )


@dataclass(frozen=True)
class TaskMemory:
    """One validated corpus and the subset available to the current task."""

    root: Path
    policy: str
    hashes: dict[str, str]
    selected: tuple[str, ...]
    missing: tuple[str, ...]
    hf_revision: str | None

    @classmethod
    def load(
        cls,
        root: str | Path,
        task_name: str,
        *,
        policy: str = "task-global",
        hf_revision: str | None = None,
    ) -> TaskMemory:
        """Validate every declared file and select only this task's memory.

        Unlisted files are ignored, including stale files from an older HF
        snapshot. A listed but missing/modified file is a corrupt snapshot.
        """
        if policy not in MEMORY_POLICIES:
            raise ValueError(f"unsupported RoboCasa memory policy: {policy}")
        root = Path(root).expanduser().resolve()
        wanted = task_files(task_name)
        manifest_path = root / "CORPUS.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"cannot read RoboCasa corpus manifest: {manifest_path}"
            ) from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("unsupported RoboCasa CORPUS.json schema")
        hashes = manifest.get("files")
        if not isinstance(hashes, dict):
            raise ValueError("CORPUS.json files must map relative paths to SHA-256")
        for name, digest in hashes.items():
            if not isinstance(name, str) or not (
                name == GLOBAL_FILE or _TASK_FILE.fullmatch(name)
            ):
                raise ValueError(f"invalid RoboCasa corpus path: {name!r}")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"invalid SHA-256 for {name}")
            cls._verify_file(root, name, digest)
        if (wanted[0] in hashes) != (wanted[1] in hashes):
            raise ValueError(f"incomplete seed-0 JSON/JSONL pair for {task_name}")
        candidates = wanted + ((GLOBAL_FILE,) if policy == "task-global" else ())
        selected = tuple(name for name in candidates if name in hashes)
        missing = tuple(name for name in candidates if name not in hashes)
        for name in missing:
            logger.warning("memory layer not provided: %s", name)
        return cls(root, policy, hashes, selected, missing, hf_revision)

    @staticmethod
    def _verify_file(root: Path, name: str, digest: str) -> None:
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"memory file escapes corpus root: {name}")
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"declared memory file is missing: {name}") from exc
        if actual != digest:
            raise ValueError(f"memory SHA-256 mismatch: {name}")

    def metadata(self) -> dict[str, Any]:
        """Return public provenance for the run, without credentials."""
        return {
            "policy": self.policy,
            "corpus_sha256": corpus_digest(self.hashes),
            "hf_revision": self.hf_revision,
            "selected_files": list(self.selected),
            "missing_layers": list(self.missing),
        }


def memory_from_variables(variables: Mapping[str, object]) -> TaskMemory:
    """Resolve the same memory selection for prompt and toolkit construction."""
    return TaskMemory.load(
        str(variables["memory_dir"]),
        str(variables["task_name"]),
        policy=str(variables.get("memory_policy", "task-global")),
        hf_revision=variables.get("memory_revision") or None,
    )


class RoboCasaMemoryManager(MemoryManager):
    """Restrict RPent file tools to the current task and enabled global layer.

    This is a file-tool boundary, not an OS sandbox. Non-memory observations
    remain accessible through the shared tools' normal access rules.
    """

    def __init__(self, memory: TaskMemory, *, output_dir: Path | None = None):
        super().__init__(root=memory.root)
        self.selection = memory
        self._read_files: set[str] = set()
        self._output_dir = output_dir

    @property
    def unread_files(self) -> tuple[str, ...]:
        """Selected files not yet read completely through RPent's file tool."""
        return tuple(
            name for name in self.selection.selected if name not in self._read_files
        )

    def _relative_path(self, path: str) -> str | None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = get_repo_root() / candidate
        resolved = candidate.resolve()
        if candidate.is_relative_to(self.root) and not resolved.is_relative_to(
            self.root
        ):
            raise PermissionError(f"memory path escapes corpus root: {path}")
        if resolved.is_relative_to(self.root):
            return resolved.relative_to(self.root).as_posix()
        return None

    def get_common_tool_bindings(self):
        """Wrap shared read/list tools; keep existing read-only write policy."""
        bindings = super().get_common_tool_bindings()
        shared_read = bindings["read_text_file"][1]
        shared_list = bindings["list_dir"][1]

        @readonly
        def read_text_file(path: str, max_chars: int = 40000) -> dict:
            relative = self._relative_path(path)
            if relative is not None:
                if relative not in self.selection.selected:
                    raise PermissionError(
                        f"memory is outside current task/policy: {path}"
                    )
                self.selection._verify_file(
                    self.root, relative, self.selection.hashes[relative]
                )
            result = shared_read(path=path, max_chars=max_chars)
            if relative is not None and isinstance(result.get("content"), str):
                content = result["content"]
                complete = (
                    hashlib.sha256(content.encode()).hexdigest()
                    == self.selection.hashes[relative]
                )
                if complete:
                    self._read_files.add(relative)
                if self._output_dir is not None:
                    event = {
                        "path": relative,
                        "complete": complete,
                        "sha256": hashlib.sha256(content.encode()).hexdigest(),
                    }
                    with (self._output_dir / "memory_reads.jsonl").open("a") as handle:
                        handle.write(json.dumps(event) + "\n")
            return result

        @readonly
        def list_dir(path: str = "") -> dict:
            relative = self._relative_path(path) if path else None
            if relative is None:
                return shared_list(path=path)
            prefix = "" if relative == "." else relative.rstrip("/") + "/"
            entries = sorted(
                {
                    name.removeprefix(prefix).split("/")[0]
                    for name in self.selection.selected
                    if name.startswith(prefix)
                }
            )
            if not entries and relative != ".":
                raise PermissionError(
                    f"memory directory is outside current task/policy: {path}"
                )
            return {
                "path": str((self.root / relative).resolve()),
                "count": len(entries),
                "files": entries,
            }

        for name, handler in (
            ("read_text_file", read_text_file),
            ("list_dir", list_dir),
        ):
            spec = dict(bindings[name][0])
            spec["description"] += (
                " RoboCasa memory is limited to the current task and the selected global-memory policy."
            )
            bindings[name] = (spec, handler)
        return bindings
