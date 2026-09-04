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

"""Export successful LIBERO episodes to the LeRobot format used by RLinf."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rpent.flywheel.episode import validate_episode

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _features() -> dict[str, dict[str, Any]]:
    return {
        "image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }


def _successful_episodes(
    data_root: Path, suite: str, task_id: int
) -> list[tuple[Path, dict[str, Any]]]:
    task_root = data_root / "raw" / "libero" / suite / f"task_{task_id:02d}"
    episodes = []
    for path in sorted(task_root.glob("seed_*/episode_*")):
        if not path.is_dir() or path.name.endswith(".partial"):
            continue
        metadata = validate_episode(path)
        if metadata["suite"] != suite or metadata["task_id"] != task_id:
            raise ValueError(f"episode metadata does not match its directory: {path}")
        if metadata["is_success"]:
            episodes.append((path, metadata))
    if not episodes:
        raise ValueError(f"no successful episodes found under {task_root}")
    return episodes


def export_lerobot(
    data_root: Path | str,
    *,
    suite: str,
    task_id: int,
    dataset_id: str | None = None,
    output_root: Path | str | None = None,
) -> dict[str, Any]:
    """Export all finalized successful episodes for one LIBERO task."""
    if not _NAME.fullmatch(suite):
        raise ValueError(f"invalid LIBERO suite: {suite!r}")
    dataset_id = dataset_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not _NAME.fullmatch(dataset_id):
        raise ValueError(f"invalid dataset ID: {dataset_id!r}")

    data_root = Path(data_root).expanduser().resolve()
    episodes = _successful_episodes(data_root, suite, task_id)
    languages = {metadata["task_language"] for _, metadata in episodes}
    if len(languages) != 1:
        raise ValueError("successful episodes have different task descriptions")
    language = languages.pop()

    parent = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else data_root / "datasets" / "lerobot" / suite / f"task_{task_id:02d}"
    )
    destination = parent / dataset_id
    partial = parent / f"{dataset_id}.partial"
    if destination.exists() or partial.exists():
        raise FileExistsError(f"dataset already exists: {destination}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("install RPent with the 'flywheel' extra") from exc
    parent.mkdir(parents=True, exist_ok=True)

    repo_id = f"rpent/{suite}-task-{task_id:02d}-{dataset_id}"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=partial,
        robot_type="panda",
        fps=20,
        features=_features(),
        use_videos=False,
        image_writer_threads=2,
    )
    frame_count = 0
    source_ids = []
    for path, metadata in episodes:
        count = metadata["training_step_count"]
        with np.load(path / "transitions.npz", allow_pickle=False) as data:
            for index in range(count):
                dataset.add_frame(
                    {
                        "image": data["main_images"][index],
                        "wrist_image": data["wrist_images"][index],
                        "state": data["states"][index],
                        "actions": data["actions"][index],
                    },
                    task=language,
                )
        dataset.save_episode()
        frame_count += count
        source_ids.append(metadata["episode_id"])

    reopened = LeRobotDataset(repo_id, root=partial)
    if len(reopened) != frame_count:
        raise RuntimeError("LeRobot frame count changed after reopening")
    manifest = {
        "schema_version": 1,
        "repo_id": repo_id,
        "suite": suite,
        "task_id": task_id,
        "task_language": language,
        "source_episode_ids": source_ids,
        "episode_count": len(source_ids),
        "frame_count": frame_count,
    }
    (partial / "meta" / "rpent_flywheel.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, destination)
    return {"dataset_path": str(destination), **manifest}
