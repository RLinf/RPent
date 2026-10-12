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

"""Shared contracts for robot-specific run-result artifacts."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunFinalizationContext:
    """Structured runner state supplied to a robot result finalizer."""

    output_dir: Path
    robot_name: str
    task_desc: Mapping[str, Any]
    environment_success: bool | None
    agent_error: str | None
    elapsed_s: float
    planner: str
    model: str | None
    reasoning_effort: str
    max_turns: int
    planner_timeout_s: int | None
    finish_result: Mapping[str, Any] | None
    stats: Mapping[str, Any]


RunFinalizer = Callable[[RunFinalizationContext], Path | str | None]


def write_json_atomic(
    destination: Path | str,
    record: Mapping[str, Any],
) -> Path:
    """Write one JSON object atomically and return its destination path."""
    path = Path(destination)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
