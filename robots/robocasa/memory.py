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

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpent.evaluation import write_json_atomic
from rpent.memory import MemoryManager
from rpent.memory.manager import _split_frontmatter, _validate
from rpent.planner.base import resolve_model
from rpent.robots.robot_spec import RunConfig
from rpent.utils.config import get_memory_dir, get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("robocasa_memory")
GLOBAL_FILE = "global/GLOBAL_MEMORY.md"
_TASK_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*\Z")
_SPLIT_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")


GPT5 = "GPT_5.5_xhigh"
ASTRA = "GPT_6_astra_high"
MEMORY_VERSIONS = ("auto", GPT5, ASTRA)
MEMORY_REVISION = "release/v0.1"


def validate_options(args: argparse.Namespace) -> None:
    """Reject HF corpus selectors for local memory and exploration."""
    if getattr(args, "memory_version", "auto") != "auto" and (
        getattr(args, "explore", False)
        or getattr(args, "memory_profile", None) == "local"
    ):
        raise ValueError(
            "--memory-version requires --memory-profile hf; "
            "local memory and exploration use --memory-dir"
        )


def select_version(
    version: str = "auto", *, model: str | None = None, planner: str = "api"
) -> str:
    """Select the corpus independently of the evaluation reasoning effort."""
    if version not in MEMORY_VERSIONS:
        raise ValueError(f"Unknown memory version: {version!r}")
    if version != "auto":
        return version
    resolved = resolve_model(planner, model)
    name = resolved.rsplit(":", 1)[-1].lower() if resolved else ""
    selected = {"gpt-5.5": GPT5, "gpt-6-astra": ASTRA}.get(name)
    if selected:
        return selected
    logger.warning(
        "No memory mapping for model %r (%s); using %s. "
        "Use --memory-version to select explicitly.",
        resolved,
        planner,
        GPT5,
    )
    return GPT5


def sync_version(
    *,
    version: str,
    cache_dir: Path,
    repo_id: str = "RLinf/RPent-memory",
) -> Path:
    """Download a selected corpus into the Hub's repository/revision cache."""
    from huggingface_hub import HfApi, snapshot_download

    if version not in (GPT5, ASTRA):
        raise ValueError("sync_version requires a resolved memory version")
    repo_id = os.environ.get("RPENT_MEMORY_HF_REPO", repo_id)
    commit = (
        HfApi().repo_info(repo_id, repo_type="dataset", revision=MEMORY_REVISION).sha
    )
    prefix = f"robocasa/{version}"
    repository_key = hashlib.sha256(repo_id.encode()).hexdigest()[:20]
    destination = cache_dir / repository_key / commit
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=commit,
        local_dir=str(destination),
        allow_patterns=[f"{prefix}/**"],
    )
    root = destination / prefix
    TaskMemory.load(root, None, profile="hf")
    return root


def prepare_memory(args: argparse.Namespace, config: RunConfig) -> None:
    """Bind the model-selected root before prompts, tools and task services."""
    validate_options(args)
    profile = getattr(args, "memory_profile", None) or (
        "local" if getattr(args, "explore", False) else "hf"
    )
    if profile == "local":
        return
    version = select_version(
        getattr(args, "memory_version", "auto"), model=args.model, planner=args.planner
    )
    root = sync_version(
        version=version,
        cache_dir=get_memory_dir("robocasa") / ".hub",
    )
    TaskMemory.load(root, args.task_name, profile="hf", split=args.split)
    config.prompt_vars.update(memory_dir=str(root), memory_version=version)
    # Runtime preflight receives args rather than RunConfig.
    args.memory_dir = str(root)


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

    file_tool_description = (
        " RoboCasa memory is limited to the current task and global memory."
    )

    def authorize_read(self, path: str | Path) -> Path:
        relative = self._relative_path(str(path))
        if relative is not None:
            if relative not in self.selection.selected:
                raise PermissionError(
                    f"memory is outside current task/global selection: {path}"
                )
            if (self.root / relative).read_text(
                encoding="utf-8"
            ) != self.selection.contents[relative]:
                raise ValueError(f"memory changed during this run: {relative}")
        return super().authorize_read(path)

    def record_read(self, path: Path, content: str) -> None:
        relative = self._relative_path(str(path))
        if relative is None:
            return
        complete = content == self.selection.contents[relative]
        if complete:
            self._read_files.add(relative)
        if self._output_dir is not None:
            event = {"path": relative, "complete": complete}
            with (self._output_dir / "memory_reads.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")

    def list_directory(self, path: str | Path) -> tuple[Path, list[str] | None]:
        relative = self._relative_path(str(path))
        if relative is None:
            return super().list_directory(path)
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
        return (self.root / relative).resolve(), entries
