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

from pathlib import Path


def task_config_dir(source_root: str | Path) -> Path:
    return Path(source_root).expanduser() / "task" / "RoboDojo" / "config"


def is_available(source_root: str | Path | None = None) -> bool:
    return source_root is not None and task_config_dir(source_root).is_dir()


def list_tasks(source_root: str | Path | None = None) -> list[str]:
    """All RoboDojo task names (config ymls, excluding ``_task.yml``)."""
    if source_root is None:
        return []
    cfg_dir = task_config_dir(source_root)
    if not cfg_dir.is_dir():
        return []
    return sorted(p.stem for p in cfg_dir.glob("*.yml") if p.stem != "_task")


def task_config_path(task_name: str, source_root: str | Path) -> Path:
    return task_config_dir(source_root) / f"{task_name}.yml"


def task_summary(task_name: str, source_root: str | Path | None = None) -> dict:
    """Lightweight static summary of a task config (objects per category)."""
    import yaml

    if source_root is None:
        return {"task": task_name}
    path = task_config_path(task_name, source_root)
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


def validate_task(task_name: str, source_root: str | Path | None = None) -> str | None:
    """Return an error string if the task is unknown, else None."""
    if not is_available(source_root):
        return None  # workspace not configured here; defer validation
    if task_name not in list_tasks(source_root):
        return f"unknown RoboDojo task {task_name!r}; known tasks: " + ", ".join(
            list_tasks(source_root)
        )
    return None
