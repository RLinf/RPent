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

"""Shared construction of per-toolkit memory managers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rpent.memory import MemoryManager


@dataclass(frozen=True)
class ToolkitMemoryConfig:
    """Inputs for one toolkit session's memory manager."""

    root: str | Path
    mode: str
    cell_tag: str


def create_toolkit_memory(config: ToolkitMemoryConfig) -> MemoryManager:
    """Create a fresh memory manager with access scoped to ``config.mode``."""
    exploration = config.mode == "exploration"
    return MemoryManager(
        root=config.root,
        memory_access="inbox_write" if exploration else "read_only",
        inbox_cell_tag=config.cell_tag if exploration else None,
    )
