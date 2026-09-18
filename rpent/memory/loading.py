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

from argparse import Namespace
from pathlib import Path

from rpent.memory import MemoryManager
from rpent.memory.versions import (
    ASTRA,
    replay_directory,
    resolve_model,
    select_version,
    sync_version,
)
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.utils.config import get_memory_dir


def validate_memory_options(args: Namespace) -> None:
    """Reject remote version overrides for local/exploration corpora."""
    version = getattr(args, "memory_version", "auto")
    if version != "auto":
        if (
            getattr(args, "explore", False)
            or getattr(args, "memory_profile", None) == "local"
        ):
            raise ValueError(
                "--memory-version requires --memory-profile hf; local memory and exploration use --memory-dir"
            )
        if getattr(args, "robot_name", "libero") != "libero":
            raise ValueError("--memory-version is currently supported only for LIBERO")


def prepare_run_memory(args: Namespace, spec: RobotSpec, config: RunConfig) -> None:
    """Set the single root used by prompts, tools and replay for this task."""
    validate_memory_options(args)
    # Pin the same resolved Codex environment model for selection and planner.
    args.model = resolve_model(args.planner, args.model)
    profile = getattr(args, "memory_profile", None) or (
        "local" if getattr(args, "explore", False) else "hf"
    )
    if profile == "local":
        return
    if spec.name == "libero":
        version = select_version(
            getattr(args, "memory_version", "auto"),
            model=args.model,
            planner=args.planner,
        )
        if args.planner == "flash" and version == ASTRA:
            raise ValueError(
                f"{ASTRA} has no Flash/Task Card replay assets; choose GPT_5.5_xhigh"
            )
        root = sync_version(
            version=version,
            repo_id=spec.memory_repo_id,
            cache_dir=get_memory_dir(spec.name) / ".versions",
        )
        config.prompt_vars["memory_dir"] = root
        config.prompt_vars["memory_version"] = version
    else:
        root = MemoryManager(get_memory_dir(spec.name)).sync(
            remote_repo=spec.memory_repo_id
        )
    if args.planner == "flash":
        replay_directory(Path(root))
