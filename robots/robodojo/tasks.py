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

"""RoboDojo task inventory (read-only view of the workspace task registry)."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_WORKSPACE = "/home/admin/robodojo_pro6000_ws"


def robodojo_source_root() -> Path:
    env = os.environ.get("ROBODOJO_SOURCE_ROOT")
    if env:
        return Path(env)
    return Path(DEFAULT_WORKSPACE) / "src" / "RoboDojo"


def task_config_dir() -> Path:
    return robodojo_source_root() / "task" / "RoboDojo" / "config"


def is_available() -> bool:
    return task_config_dir().is_dir()


def list_tasks() -> list[str]:
    """All RoboDojo task names (config ymls, excluding ``_task.yml``)."""
    cfg_dir = task_config_dir()
    if not cfg_dir.is_dir():
        return []
    return sorted(p.stem for p in cfg_dir.glob("*.yml") if p.stem != "_task")


def task_config_path(task_name: str) -> Path:
    return task_config_dir() / f"{task_name}.yml"


def task_summary(task_name: str) -> dict:
    """Lightweight static summary of a task config (objects per category)."""
    import yaml

    path = task_config_path(task_name)
    if not path.exists():
        return {"task": task_name, "error": f"config not found: {path}"}
    data = yaml.safe_load(path.read_text(errors="replace")) or {}
    summary: dict = {"task": task_name}
    for section in ("Rigid", "Articulation", "Geometry", "Cloth"):
        items = data.get(section) or []
        labels: list[str] = []
        for group in items:
            for cat in group.get("category", []):
                labels.extend(cat.get("label", []) or [cat.get("name", "?")])
        if labels:
            summary[section.lower()] = sorted(set(labels))
    return summary


def validate_task(task_name: str) -> str | None:
    """Return an error string if the task is unknown, else None."""
    if not is_available():
        return None  # workspace not configured here; defer validation
    if task_name not in list_tasks():
        return f"unknown RoboDojo task {task_name!r}; known tasks: " + ", ".join(
            list_tasks()
        )
    return None
