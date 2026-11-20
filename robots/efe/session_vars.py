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

"""Run identity and prompt variables shared by RPent and Harbor.

``rpent --robot efe`` and Harbor render the same prompt template. Derive its
variables in one place so the two paths cannot drift. The memory root stays a
caller decision: Harbor gives each trial an empty directory, while a local CLI
run can use a shared corpus.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def recipe_tag(task_name: str, instruction: str) -> str:
    """Return the stable per-task tag for run directories and memory cells."""
    digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:8]
    return f"{task_name}_{digest}"


def prompt_vars(
    *,
    workdir: Path,
    output_dir: Path,
    memory_dir: Path,
    skills: bool,
) -> dict[str, object]:
    """Return the variables ``robots.efe.prompt_bundle`` reads."""
    return {
        "workdir": str(workdir),
        "output_dir": str(output_dir),
        "memory_dir": str(memory_dir),
        "skills": skills,
    }
