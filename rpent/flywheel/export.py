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

"""Export VLA training frames and VLM planner turns, and load VLA datasets."""

from __future__ import annotations

import json
import mimetypes
import shutil
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np

from rpent.flywheel.flywheel import (
    read_frame,
    resolve_action_config,
    select_vla_segments,
)
from rpent.flywheel.reader import (
    has_confirmed_environment_success,
    iter_jsonl,
    iter_tasks,
    iter_trajectories,
)
from rpent.robots import get_robot_spec

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


def export_dataset(
    input_root: Path | str,
    output_dir: Path | str,
    *,
    pattern: str = "*",
    horizon: int = 10,
    repo_id: str = "rpent/vla-sft",
    image_writer_threads: int = 0,
) -> dict[str, Any]:
    """Export selected actions to train; horizon only counts complete, unpadded windows.

    Images are written synchronously when image_writer_threads is zero.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if (
        isinstance(image_writer_threads, bool)
        or not isinstance(image_writer_threads, int)
        or image_writer_threads < 0
    ):
        raise ValueError("image_writer_threads must be a non-negative integer")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    report = {
        "schema_version": 5,
        "repo_id": repo_id,
        "dataset_paths": {"train": None},
        "horizon": horizon,
        "robot": None,
        "fps": None,
        "episode_counts": {"train": 0},
        "frame_counts": {"train": 0},
        "window_counts": {"train": 0},
        "trajectories": [],
    }
    dataset = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as tmp:
        root = Path(tmp)
        try:
            for trajectory in iter_trajectories(input_root, pattern):
                selected = select_vla_segments(trajectory)
                source = {
                    "source": str(trajectory["root"]),
                    "task_id": trajectory["task"]["task_id"],
                    "action_count": len(trajectory["actions"]),
                    "selected_action_count": sum(
                        segment["stop_index"] - segment["start_index"]
                        for segment in selected["segments"]
                    ),
                    "exported_action_count": 0,
                    "frame_counts": {"train": 0},
                    "window_counts": {"train": 0},
                    "episodes": [],
                    "rejected": selected["rejected"],
                }
                report["trajectories"].append(source)
                if selected["segments"]:
                    task = trajectory["task"]
                    robot = task["config"]["data"]["arguments"]["robot_name"]
                    if report["robot"] is not None and robot != report["robot"]:
                        raise ValueError("Export each robot into a separate dataset")
                    robot_config = get_robot_spec(robot).flywheel
                for segment in selected["segments"]:
                    records = trajectory["actions"][
                        segment["start_index"] : segment["stop_index"]
                    ]
                    config = resolve_action_config(
                        robot_config, records[0]["action"]["format"]
                    )
                    image_fields = set(config["images"])
                    use_videos = config.get("use_videos", False)
                    for record in records:
                        frame = read_frame(
                            trajectory["root"],
                            record,
                            segment["instruction"],
                            config,
                        )
                        if dataset is None:
                            features = {
                                name: {
                                    "dtype": ("video" if use_videos else "image")
                                    if name in image_fields
                                    else str(value.dtype),
                                    "shape": value.shape,
                                    "names": ["height", "width", "channel"]
                                    if name in image_fields
                                    else None,
                                }
                                for name, value in frame.items()
                                if name != "prompt"
                            }
                            features["prompt"] = {
                                "dtype": "string",
                                "shape": (1,),
                                "names": None,
                            }
                            dataset = LeRobotDataset.create(
                                repo_id=repo_id,
                                root=root / "train",
                                robot_type=robot,
                                fps=config["fps"],
                                features=features,
                                use_videos=use_videos,
                                video_backend="pyav",
                                image_writer_threads=image_writer_threads,
                            )
                            report["dataset_paths"]["train"] = "train"
                            report.update(robot=robot, fps=config["fps"])
                        dataset.add_frame(frame, task=segment["instruction"])
                    dataset.save_episode()
                    frame_count = len(records)
                    window_count = max(0, frame_count - horizon + 1)
                    report["frame_counts"]["train"] += frame_count
                    report["window_counts"]["train"] += window_count
                    source["frame_counts"]["train"] += frame_count
                    source["window_counts"]["train"] += window_count
                    source["episodes"].append(
                        {
                            "split": "train",
                            "episode_index": report["episode_counts"]["train"],
                            "start_index": segment["start_index"],
                            "stop_index": segment["stop_index"],
                        }
                    )
                    report["episode_counts"]["train"] += 1
                    source["exported_action_count"] += frame_count
                source["excluded_action_count"] = (
                    source["action_count"] - source["exported_action_count"]
                )
                non_vla_count = sum(
                    action.get("source") != "vla" for action in trajectory["actions"]
                )
                source["excluded_action_counts"] = {
                    "non_vla": non_vla_count,
                    "unselected_vla": source["action_count"]
                    - non_vla_count
                    - source["selected_action_count"],
                }
            report["trajectory_count"] = len(report["trajectories"])
            report["exported_trajectory_count"] = sum(
                source["exported_action_count"] > 0 for source in report["trajectories"]
            )
            if dataset is not None:
                if config.get("metadata") == "rldx":
                    _write_rldx_metadata(dataset, config)
        finally:
            if dataset is not None:
                dataset.stop_image_writer()
        if dataset is not None:
            # Images are embedded in Parquet or encoded as videos.
            shutil.rmtree(root / "train" / "images")
        (root / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        root.rename(destination)
    return {"output_dir": str(destination), **report}


def _write_rldx_metadata(dataset: LeRobotDataset, config: dict[str, Any]) -> None:
    """Save RLDX field groups and statistics, including quantiles."""
    from lerobot.datasets.utils import serialize_dict, write_json

    modality = {
        "video": {
            field["path"][-1]: {"original_key": name}
            for name, field in config["images"].items()
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    groups = {
        "state": {
            field["name"]: field["shape"][0] for field in config["state"]["fields"]
        },
        "action": config["action"]["groups"],
    }
    for kind, fields in groups.items():
        modality[kind] = {}
        start = 0
        for name, size in fields.items():
            modality[kind][name] = {"start": start, "end": start + size}
            start += size
    statistics = {}
    stored = dataset.hf_dataset.with_format("numpy")
    for name in ("observation.state", "action"):
        statistics[name] = {
            key: dataset.meta.stats[name][key] for key in ("min", "max", "mean", "std")
        }
        q01, q99 = np.quantile(stored[name], (0.01, 0.99), axis=0)
        statistics[name].update(q01=q01, q99=q99)
    write_json(modality, dataset.root / "meta" / "modality.json")
    write_json(serialize_dict(statistics), dataset.root / "meta" / "stats.json")


def load_dataset(
    output_dir: Path | str,
) -> LeRobotDataset | list[dict[str, Any]]:
    """Load exported training frames without building action windows."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(output_dir).expanduser().resolve()
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report["frame_counts"]["train"] == 0:
        return []
    return LeRobotDataset(
        report["repo_id"],
        root=root / report["dataset_paths"]["train"],
        video_backend="pyav",
    )


def _copy_images(
    row: dict,
    root: Path,
    output: Path,
    source_paths: dict[str, str],
    images_by_digest: dict[str, str],
) -> list[str]:
    """Store each distinct image once and replace source paths with dataset paths."""
    artifacts = [
        block["artifact"]
        for message in row["messages"]
        for block in message["content"]
        if block["type"] == "image"
    ]
    paths = {artifact["path"] for artifact in artifacts}
    missing = sorted(
        path
        for path in paths
        if path not in source_paths and not (root / path).is_file()
    )
    if missing:
        return missing
    for artifact in artifacts:
        path = artifact["path"]
        if path not in source_paths:
            source = root / path
            if not source.resolve().is_relative_to(root / "artifacts"):
                raise ValueError(f"Image artifact outside trajectory: {source}")
            data = source.read_bytes()
            digest = sha256(data).hexdigest()
            if digest not in images_by_digest:
                suffix = mimetypes.guess_extension(artifact["media_type"]) or ".bin"
                target = f"images/{digest}{suffix}"
                (output / target).write_bytes(data)
                images_by_digest[digest] = target
            source_paths[path] = images_by_digest[digest]
        artifact["path"] = source_paths[path]
    return []


def export_vlm_samples(
    input_root: Path | str,
    output_dir: Path | str,
    *,
    pattern: str = "*",
    include_unsuccessful: bool = False,
) -> dict[str, Any]:
    """Export API turns with their recorded request history and portable images."""
    from rpent.flywheel.vlm_api import TurnConversionError, api_turn_to_sample

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "include_unsuccessful": include_unsuccessful,
        "episode_count": 0,
        "selected_episode_count": 0,
        "exported_episode_count": 0,
        "turn_count": 0,
        "sample_count": 0,
        "skipped_backend_counts": Counter(),
        "excluded": [],
    }
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as tmp:
        output = Path(tmp)
        (output / "images").mkdir()
        images_by_digest = {}
        with (output / "samples.jsonl").open("w", encoding="utf-8") as rows:
            for root, task in iter_tasks(input_root, pattern):
                selected = include_unsuccessful or has_confirmed_environment_success(
                    task
                )
                report["episode_count"] += 1
                if selected:
                    report["selected_episode_count"] += 1
                samples_before = report["sample_count"]
                source_paths = {}
                for planner in iter_jsonl(root / "planners.jsonl"):
                    report["turn_count"] += 1
                    source = {
                        "episode_id": task["task_id"],
                        "turn_id": planner["planner_id"],
                    }
                    backend = planner["backend"]
                    if backend != "api":
                        report["skipped_backend_counts"][backend] += 1
                    if not selected:
                        report["excluded"].append(
                            {**source, "reason": "no_confirmed_environment_success"}
                        )
                        continue
                    if backend != "api":
                        report["excluded"].append(
                            {
                                **source,
                                "reason": "unsupported_backend",
                                "backend": backend,
                            }
                        )
                        continue
                    try:
                        sample = api_turn_to_sample(task, planner)
                    except (ValueError, KeyError) as exc:
                        report["excluded"].append(
                            {
                                **source,
                                "reason": exc.reason
                                if isinstance(exc, TurnConversionError)
                                else "invalid_turn",
                                "detail": str(exc),
                            }
                        )
                        continue
                    row = asdict(sample)
                    missing = _copy_images(
                        row, root, output, source_paths, images_by_digest
                    )
                    if missing:
                        report["excluded"].append(
                            {
                                **source,
                                "reason": "missing_images",
                                "paths": missing,
                            }
                        )
                        continue
                    rows.write(
                        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    report["sample_count"] += 1
                if report["sample_count"] > samples_before:
                    report["exported_episode_count"] += 1
        report["image_count"] = len(images_by_digest)
        report["excluded_episode_counts"] = {
            "no_confirmed_environment_success": report["episode_count"]
            - report["selected_episode_count"]
        }
        counts = Counter(item["reason"] for item in report["excluded"])
        report["excluded_turn_counts"] = {
            reason: counts[reason]
            for reason in (
                "no_confirmed_environment_success",
                "unsupported_backend",
                "incomplete_response",
                "invalid_tool_arguments",
                "missing_images",
                "invalid_turn",
            )
        }
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output.rename(destination)
    return report
