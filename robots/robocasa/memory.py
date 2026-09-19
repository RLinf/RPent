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
from rpent.tools.toolkit import readonly
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("robocasa_memory")
MEMORY_POLICIES = ("task-global", "task-only")
GLOBAL_FILE = "global/GLOBAL_MEMORY.md"
_TASK_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*\Z")


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
    """The files available to one task for the duration of a run."""

    root: Path
    policy: str
    contents: dict[str, str]
    selected: tuple[str, ...]
    missing: tuple[str, ...]

    @classmethod
    def load(
        cls,
        root: str | Path,
        task_name: str | None,
        *,
        policy: str = "task-global",
    ) -> TaskMemory:
        """Select conventional task/global paths without a dataset manifest.

        With no task yet, validate only the session's root, policy and global
        layer before starting shared services. Task runs must supply their name.
        """
        if policy not in MEMORY_POLICIES:
            raise ValueError(f"unsupported RoboCasa memory policy: {policy}")
        root = Path(root).expanduser().resolve()
        wanted = task_files(task_name) if task_name is not None else ()
        if not root.is_dir():
            raise ValueError(f"RoboCasa memory directory not found: {root}")
        if wanted and (root / wanted[0]).exists() != (root / wanted[1]).exists():
            raise ValueError(f"incomplete seed-0 JSON/JSONL pair for {task_name}")
        candidates = wanted + ((GLOBAL_FILE,) if policy == "task-global" else ())
        contents = {}
        for name in candidates:
            path = root / name
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"memory file escapes memory root: {name}")
            if path.exists() or path.is_symlink():
                try:
                    contents[name] = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    raise ValueError(f"cannot read memory file: {name}") from exc
        if policy == "task-global" and GLOBAL_FILE not in contents:
            raise ValueError(f"task-global requires {root / GLOBAL_FILE}")
        selected = tuple(contents)
        missing = tuple(name for name in candidates if name not in contents)
        for name in missing:
            logger.warning("memory layer not provided: %s", name)
        return cls(root, policy, contents, selected, missing)

    def metadata(self) -> dict[str, Any]:
        """Record the selected layers without pinning a data version."""
        return {
            "policy": self.policy,
            "selected_files": list(self.selected),
            "missing_layers": list(self.missing),
        }


def memory_from_variables(variables: Mapping[str, object]) -> TaskMemory:
    """Resolve the same memory selection for prompt and toolkit construction."""
    return TaskMemory.load(
        str(variables["memory_dir"]),
        str(variables["task_name"]),
        policy=str(variables.get("memory_policy", "task-global")),
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
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            # A reused output directory starts a new audit, including when
            # switching between task-global and task-only.
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
                        f"memory is outside current task/policy: {path}"
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
