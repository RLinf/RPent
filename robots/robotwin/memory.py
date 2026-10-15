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

"""Explicit RoboTwin HF corpus selection with the unversioned default."""

import os
from argparse import Namespace

from rpent.memory import MemoryManager
from rpent.memory.model import validate_options
from rpent.memory.source import prepare_explicit_source
from rpent.robots.robot_spec import RunConfig
from rpent.utils.config import get_memory_dir


def prepare_memory(args: Namespace, config: RunConfig) -> None:
    """Bind one corpus for CLI and Dashboard without changing local memory."""
    validate_options(args)
    profile = getattr(args, "memory_profile", None) or (
        "local" if getattr(args, "explore", False) else "hf"
    )
    if profile == "local":
        return
    version = getattr(args, "memory_version", "auto")
    source = getattr(args, "memory_repo", None)
    if version == "auto" and source is None:
        root = MemoryManager(get_memory_dir("robotwin")).sync(
            remote_repo="RLinf/RPent-memory"
        )
    else:
        source = source or os.environ.get("RPENT_MEMORY_HF_REPO", "RLinf/RPent-memory")
        root = prepare_explicit_source(
            source, "robotwin", version=None if version == "auto" else version
        )
    config.prompt_vars.update(memory_dir=str(root), memory_version=version)
    args.memory_dir = str(root)
