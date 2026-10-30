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

"""Read schema-v5 trajectory records, loading attachments on demand."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np


def iter_trajectories(
    input_root: Path | str, pattern: str = "*"
) -> Iterator[dict[str, Any]]:
    """Read task, action, and prediction records without loading attachments."""
    root = Path(input_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Trajectory input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(str(root))

    if (root / "task.json").is_file():
        task_paths = [root / "task.json"]
    else:
        task_paths = sorted(
            {
                path.resolve()
                for job in root.glob(pattern)
                if job.is_dir()
                for path in job.rglob("task.json")
                if path.is_file()
            }
        )
    if not task_paths:
        raise ValueError(f"No trajectories found under {root} (pattern={pattern!r})")

    for task_path in task_paths:
        try:
            document = json.loads(task_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{task_path}:{exc.lineno}: {exc.msg}") from exc
        if not isinstance(document, dict) or document.get("schema_version") != 5:
            raise ValueError(f"{task_path}: expected trajectory schema_version 5")
        if not isinstance(document.get("task"), dict):
            raise ValueError(f"{task_path}: expected a task object")

        trajectory_root = task_path.parent
        yield {
            "root": trajectory_root,
            "schema_version": document["schema_version"],
            "task": document["task"],
            "actions": _read_jsonl(trajectory_root / "actions.jsonl"),
            "predictions": _read_jsonl(trajectory_root / "predictions.jsonl"),
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a record object")
            records.append(record)
    return records


def validate_image(
    value: dict[str, Any] | None,
    name: str,
    *,
    expected_shape: tuple[int, int, int] | None = None,
) -> None:
    """Check uint8 HWC image metadata without loading image files."""
    value = value or {}
    shape = value.get("shape", [])
    if (
        value.get("$type") not in ("ndarray", "artifact_ref")
        or len(shape) != 3
        or shape[-1] != 3
        or (expected_shape is not None and tuple(shape) != expected_shape)
        or value.get("dtype") != "uint8"
    ):
        raise ValueError(f"Invalid {name} metadata")
    if value["$type"] == "artifact_ref" and (
        value.get("kind") != "ndarray" or not value.get("path")
    ):
        raise ValueError(f"Invalid {name} artifact reference")


def read_array(trajectory_root: Path | str, value: dict[str, Any]) -> np.ndarray:
    """Load an inline or artifact ndarray, checking its dtype, shape, and path."""
    if value.get("$type") == "ndarray":
        array = np.asarray(_inline_values(value["value"]), dtype=value["dtype"])
        # Empty nested lists cannot retain dimensions after their first zero.
        if array.size == 0:
            array = array.reshape(value["shape"])
    elif value.get("$type") == "artifact_ref" and value.get("kind") == "ndarray":
        root = Path(trajectory_root).expanduser().resolve()
        relative = Path(value["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Array artifact must stay inside {root}: {relative}")
        with path.open("rb") as file:
            # EnvState stores ndarray artifacts as NPY, without pickle data.
            array = np.lib.format.read_array(file, allow_pickle=False)
    else:
        raise ValueError("Expected an encoded ndarray or ndarray artifact_ref")

    if array.dtype.hasobject:
        raise ValueError("Object arrays are not supported")
    if value.get("dtype") is not None and array.dtype != np.dtype(value["dtype"]):
        raise ValueError(
            f"Array dtype mismatch: expected {value['dtype']}, got {array.dtype}"
        )
    if value.get("shape") is not None and array.shape != tuple(value["shape"]):
        raise ValueError(
            f"Array shape mismatch: expected {value['shape']}, got {array.shape}"
        )
    return array


def _inline_values(value: Any) -> Any:
    if isinstance(value, list):
        return [_inline_values(item) for item in value]
    if isinstance(value, dict) and value.get("$type") == "float":
        return float(value["value"])
    return value
