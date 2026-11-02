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

"""Shared CLI/Dashboard memory preparation, before a task starts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path

from rpent.memory import MemoryManager
from rpent.memory.source import prepare_explicit_source
from rpent.planner.base import resolve_model
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.utils.config import get_memory_dir


def prepare_run_memory(args: Namespace, spec: RobotSpec, config: RunConfig) -> None:
    """Set the single root used by prompts, tools and replay for this task."""
    # Pin the same resolved Codex environment model for selection and planner.
    args.model = resolve_model(args.planner, args.model)
    if spec.prepare_memory is not None:
        spec.prepare_memory(args, config)
        _prepare_exploration_copy(args, spec, config)
        return
    profile = getattr(args, "memory_profile", None) or (
        "local" if getattr(args, "explore", False) else "hf"
    )
    if profile == "local":
        _prepare_exploration_copy(args, spec, config)
        return
    if args.memory_repo is not None:
        root = prepare_explicit_source(args.memory_repo, spec.name)
        config.prompt_vars["memory_dir"] = str(root)
        args.memory_dir = str(root)
        return
    MemoryManager(get_memory_dir(spec.name)).sync(remote_repo=spec.memory_repo_id)


def _prepare_exploration_copy(
    args: Namespace, spec: RobotSpec, config: RunConfig
) -> None:
    """Give each harness exploration run its own mutable corpus and provenance."""
    if args.planner != "deepseek_harness" or not getattr(args, "explore", False):
        return
    source = (
        Path(
            config.prompt_vars.get("memory_dir")
            or args.memory_dir
            or get_memory_dir(spec.name)
        )
        .expanduser()
        .resolve()
    )
    destination = Path(config.output_dir).resolve() / "memory"
    if destination.exists():
        raise ValueError(
            f"Exploration memory destination already exists: {destination}"
        )
    if source == destination or source in destination.parents:
        raise ValueError("Exploration output must be outside its source memory corpus")
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded = {".git", ".versions"}

    def manifest(root: Path) -> dict[str, str]:
        files = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(part in excluded for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Memory source contains a symbolic link: {relative}")
            if path.is_file():
                files[relative.as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        return files

    files = manifest(source) if source.exists() else {}
    receipt_path = source.parent / f"{source.name}.receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    with tempfile.TemporaryDirectory(
        prefix=".exploration-memory-", dir=destination.parent
    ) as temp:
        copied = Path(temp) / "corpus"
        if source.exists():
            shutil.copytree(source, copied, ignore=shutil.ignore_patterns(*excluded))
        else:
            copied.mkdir()
        if manifest(copied) != files:
            raise ValueError(
                "Memory source changed while preparing the exploration copy"
            )
        copied.rename(destination)
    provenance = {
        "source_root": str(source),
        "source_kind": "published" if receipt is not None else "local",
        "source_receipt": receipt,
        "source_files_sha256": files,
        "writable_root": str(destination),
        "planner": args.planner,
        "model": args.model,
    }
    provenance_path = destination.parent / "memory-source.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    config.prompt_vars["memory_dir"] = str(destination)
    config.prompt_vars["memory_inbox"] = str(
        destination / "_internal" / "inbox" / config.recipe_tag
    )
    config.prompt_vars["memory_source"] = str(provenance_path)
    args.memory_dir = str(destination)
