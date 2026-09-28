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

from rpent.memory import MemoryManager
from rpent.planner.base import resolve_model
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.utils.config import get_memory_dir


def prepare_run_memory(args: Namespace, spec: RobotSpec, config: RunConfig) -> None:
    """Set the single root used by prompts, tools and replay for this task."""
    # Pin the same resolved Codex environment model for selection and planner.
    args.model = resolve_model(args.planner, args.model)
    if spec.prepare_memory is not None:
        spec.prepare_memory(args, config)
        return
    profile = getattr(args, "memory_profile", None) or (
        "local" if getattr(args, "explore", False) else "hf"
    )
    if profile == "local":
        return
    MemoryManager(get_memory_dir(spec.name)).sync(remote_repo=spec.memory_repo_id)
