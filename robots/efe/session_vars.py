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

"""Run identity and prompt variables shared by both EFE entry points.

``rpent --robot efe`` and the Harbor runner render the same prompt template.
Derive its variables in one place so the two paths cannot drift. The memory
root stays a caller decision: Harbor gives each trial an empty directory,
while the CLI reads one shared corpus.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Harbor starts one trial per task, so its memory root needs no per-task cell.
DEFAULT_INBOX_TAG = "efe"


def recipe_tag(task_name: str, instruction: str) -> str:
    """Return the stable per-task tag for run directories and memory cells."""
    digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:8]
    return f"{task_name or 'efe'}_{digest}"


def inbox_path(memory_dir: Path, cell_tag: str) -> Path:
    """Return the inbox that ``MemoryManager`` exposes for this tag."""
    return memory_dir / "_internal" / "inbox" / cell_tag


def prompt_vars(
    *,
    workdir: Path,
    output_dir: Path,
    memory_dir: Path,
    inbox_tag: str,
    session_max: int,
    skills: bool,
) -> dict[str, object]:
    """Return the variables ``robots.efe.prompt_bundle`` reads."""
    return {
        "workdir": str(workdir),
        "output_dir": str(output_dir),
        "memory_dir": str(memory_dir),
        "memory_inbox": str(inbox_path(memory_dir, inbox_tag)),
        "session_max": session_max,
        "skills": skills,
    }
