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

"""RoboCasa task and global memory, shared by prompts and file tools."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpent.evaluation import write_json_atomic
from rpent.memory import MemoryManager
from rpent.memory.manager import _split_frontmatter, _validate
from rpent.tools.toolkit import readonly
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("robocasa_memory")
GLOBAL_FILE = "global/GLOBAL_MEMORY.md"
_TASK_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*\Z")
_SPLIT_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")


def task_files(task_name: str) -> tuple[str, str, str]:
    """Return canonical filenames for one task, without directory traversal."""
    if not _TASK_NAME.fullmatch(task_name):
        raise ValueError(f"invalid RoboCasa task name: {task_name!r}")
    return (
        f"task-specific/{task_name}_s0.json",
        f"task-specific/{task_name}_s0_recipe.jsonl",
        f"task-specific/{task_name}.md",
    )


def local_task_files(task_name: str, split: str) -> tuple[str, str, str]:
    """Return the native exploration filenames for the current task and split."""
    published = task_files(task_name)
    if not _SPLIT_NAME.fullmatch(split):
        raise ValueError(f"invalid RoboCasa split: {split!r}")
    return (
        f"task-specific/{task_name}_{split}_s0.json",
        f"task-specific/{task_name}_{split}_s0_recipe.jsonl",
        published[2],
    )


def _layer_file(name: object, directory: str) -> bool:
    if not isinstance(name, str):
        return False
    parts = name.split("/")
    return len(parts) == 2 and parts[0] == directory and parts[1].endswith(".md")


def selection_errors(
    memory: Mapping[str, Any], task_name: str, split: str
) -> list[str]:
    """Check recorded evaluation selection against task/profile boundaries."""
    profile = memory.get("profile", "hf")
    selected = memory.get("selected_files")
    missing = memory.get("missing_layers")
    families = memory.get("task_family", {})
    if (
        not isinstance(profile, str)
        or profile not in {"hf", "local"}
        or not isinstance(selected, list)
        or not all(isinstance(name, str) for name in selected)
        or len(selected) != len(set(selected))
        or not isinstance(missing, list)
        or not isinstance(families, dict)
    ):
        return ["invalid memory selection metadata"]
    published = task_files(task_name)
    native = local_task_files(task_name, split)
    wanted = published
    if profile == "local" and any(name in selected for name in native[:2]):
        wanted = native
    errors = []
    if any(
        (pair[0] in selected) != (pair[1] in selected) for pair in (published, native)
    ):
        errors.append("incomplete task audit/recipe pair")
    if published[0] in selected and native[0] in selected:
        errors.append("both published and native task pairs are selected")
    globals_ = [name for name in selected if _layer_file(name, "global")]
    if profile == "hf":
        if globals_ != [GLOBAL_FILE] or families:
            errors.append("HF evaluation requires its fixed global file")
    elif not globals_:
        errors.append("local evaluation requires global memory")
    identity = {"suite": "robocasa", "regime": split, "task_id": task_name}
    if any(
        not _layer_file(name, "task-family")
        or context != identity
        or name not in selected
        for name, context in families.items()
    ):
        errors.append("task-family identity does not match the current task/split")
    allowed = {*wanted, *globals_, *families}
    if any(name not in allowed for name in selected):
        errors.append("selected memory files violate the task/global boundary")
    if missing != [name for name in wanted if name not in selected]:
        errors.append("missing memory layers do not match the selected task files")
    return errors


@dataclass(frozen=True)
class TaskMemory:
    """The files available to one task for the duration of a run."""

    root: Path
    contents: dict[str, str]
    selected: tuple[str, ...]
    missing: tuple[str, ...]
    profile: str
    task_family: dict[str, dict[str, str]]

    @classmethod
    def load(
        cls,
        root: str | Path,
        task_name: str | None,
        *,
        profile: str = "hf",
        split: str = "target",
    ) -> TaskMemory:
        """Select conventional task/global paths without a dataset manifest.

        With no task yet, validate only the session's root and global
        layer before starting shared services. Task runs must supply their name.
        """
        if profile not in {"hf", "local"}:
            raise ValueError(f"unsupported RoboCasa memory profile: {profile}")
        root = Path(root).expanduser().resolve()
        wanted = task_files(task_name) if task_name is not None else ()
        if not root.is_dir():
            raise ValueError(f"RoboCasa memory directory not found: {root}")
        MemoryManager(root).check_layout()
        pairs = [wanted] if wanted else []
        if wanted and profile == "local":
            pairs.append(local_task_files(task_name, split))
        present_pairs = []
        for pair in pairs:
            present = [
                (root / name).exists() or (root / name).is_symlink()
                for name in pair[:2]
            ]
            if present[0] != present[1]:
                raise ValueError(f"incomplete seed-0 JSON/JSONL pair for {pair[0]}")
            if present[0]:
                present_pairs.append(pair)
        if len(present_pairs) > 1:
            raise ValueError(
                "both published and native task pairs exist; use separate --memory-dir directories"
            )
        if present_pairs:
            wanted = present_pairs[0]

        def read(name: str) -> str:
            path = root / name
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"memory file escapes memory root: {name}")
            try:
                return path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"cannot read memory file: {name}") from exc

        contents = {
            name: read(name)
            for name in wanted
            if (root / name).exists() or (root / name).is_symlink()
        }
        families = {}
        if profile == "local" and task_name is not None:
            identity = {"suite": "robocasa", "regime": split, "task_id": task_name}
            for path in sorted((root / "task-family").glob("*.md")):
                name = path.relative_to(root).as_posix()
                content = read(name)
                metadata, _ = _split_frontmatter(path)
                if all(metadata.get(key) == value for key, value in identity.items()):
                    _validate(metadata)
                    if metadata["scope"] != "task-family":
                        raise ValueError(f"invalid task-family scope: {name}")
                    contents[name] = content
                    families[name] = identity.copy()
        global_files = (
            [
                path.relative_to(root).as_posix()
                for path in sorted((root / "global").glob("*.md"))
            ]
            if profile == "local"
            else [GLOBAL_FILE]
            if (root / GLOBAL_FILE).exists()
            else []
        )
        if not global_files:
            required = "global/*.md" if profile == "local" else GLOBAL_FILE
            raise ValueError(f"task-global requires {root / required}")
        contents.update({name: read(name) for name in global_files})
        selected = tuple(contents)
        missing = tuple(name for name in wanted if name not in contents)
        for name in missing:
            logger.warning("memory layer not provided: %s", name)
        return cls(root, contents, selected, missing, profile, families)

    def metadata(self) -> dict[str, Any]:
        """Record the selected layers without pinning a data version."""
        return {
            "policy": "task-global",
            "selected_files": list(self.selected),
            "missing_layers": list(self.missing),
            "profile": self.profile,
            "task_family": self.task_family,
        }


def memory_from_variables(variables: Mapping[str, object]) -> TaskMemory:
    """Resolve the same memory selection for prompt and toolkit construction."""
    return TaskMemory.load(
        str(variables["memory_dir"]),
        str(variables["task_name"]),
        profile=str(variables.get("memory_profile") or "hf"),
        split=str(variables.get("split", "target")),
    )


class RoboCasaMemoryManager(MemoryManager):
    """Restrict RPent file tools to the current task and global layer.

    This is a file-tool boundary, not an OS sandbox. Non-memory observations
    remain accessible through the shared tools' normal access rules.
    """

    def __init__(self, memory: TaskMemory, *, output_dir: Path | None = None):
        super().__init__(root=memory.root)
        self.selection = memory
        self._read_files: set[str] = set()
        self._output_dir = output_dir
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            # A reused output directory starts a new audit for this run.
            (output_dir / "memory_reads.jsonl").write_text("", encoding="utf-8")
            write_json_atomic(output_dir / "memory.json", memory.metadata())

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
                        f"memory is outside current task/global selection: {path}"
                    )
                if (self.root / relative).read_text(
                    encoding="utf-8"
                ) != self.selection.contents[relative]:
                    raise ValueError(f"memory changed during this run: {relative}")
            result = shared_read(path=path, max_chars=max_chars)
            if relative is not None and isinstance(result.get("content"), str):
                content = result["content"]
                complete = content == self.selection.contents[relative]
                if complete:
                    self._read_files.add(relative)
                if self._output_dir is not None:
                    event = {
                        "path": relative,
                        "complete": complete,
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
                    f"memory directory is outside current task/global selection: {path}"
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
                " RoboCasa memory is limited to the current task and global memory."
            )
            bindings[name] = (spec, handler)
        return bindings
