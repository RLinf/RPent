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

"""Offline tests for rendering a grounded map as a report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from rpent.tools.map_view import (
    HTML_FILE,
    ISOMETRIC_FILE,
    OBJECTS_CSV,
    TOPDOWN_FILE,
    build_report,
    main,
)

FRAMES = 2
# A floor point satisfies dot(point, normal) == offset.
FLOOR_OFFSET = -0.5


def _write_map(
    map_dir: Path,
    *,
    gravity: bool = True,
    floor: bool = True,
    semantic_points: bool = True,
    overlay: bool = True,
) -> None:
    map_dir.mkdir(parents=True)
    camera_to_world = np.tile(np.eye(4), (FRAMES, 1, 1))
    camera_to_world[1, 0, 3] = 1.0
    np.save(map_dir / "cameras_c2w.npy", camera_to_world)
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "map_name": map_dir.name,
        "frame": "model",
        "frames": FRAMES,
        "has_depth": True,
        "has_world_points": True,
        "cameras": {"frame": "model"},
        "camera_to_world": {
            "path": "cameras_c2w.npy",
            "convention": "camera_to_world",
            "shape": [FRAMES, 4, 4],
        },
        "gravity": (
            {"status": "estimated", "up": [0.0, 0.0, 1.0]} if gravity else None
        ),
        "floor": (
            {
                "status": "estimated",
                "normal": [0.0, 0.0, 1.0],
                "offset": FLOOR_OFFSET,
            }
            if floor
            else None
        ),
        "scale": {
            "status": "arbitrary",
            "metric": False,
            "meters_per_unit": None,
            "source": None,
        },
        "alignment": {"status": "unavailable", "reason": "test map"},
    }
    (map_dir / "map.json").write_text(json.dumps(metadata))

    if semantic_points:
        np.savez_compressed(
            map_dir / "semantic_points.npz",
            points=np.asarray(
                [
                    [-1.0, -1.0, FLOOR_OFFSET],
                    [0.0, 0.0, FLOOR_OFFSET],
                    [1.0, 1.0, FLOOR_OFFSET],
                    [0.0, 0.0, 0.4],
                    [0.1, 0.1, 0.6],
                ],
                dtype=np.float32,
            ),
            labels=np.asarray([0, 0, 0, 0, 1], dtype=np.int32),
            scores=np.asarray([0.9, 0.9, 0.9, 0.8, 0.7], dtype=np.float32),
            object_ids=np.asarray([0, 0, 0, 1, 2], dtype=np.int32),
            frame_indices=np.asarray([0, 1, 0, 0, 1], dtype=np.int32),
            label_names=np.asarray(["chair", "desk"]),
        )

    document: dict[str, Any] = {
        "schema_version": 1,
        "map_frame": "model",
        "labels": ["chair", "desk"],
        "keyframes": [0, 1],
        "detections": [
            {
                "label": "chair",
                "keyframe": 0,
                "timestamp_s": 0.0,
                "pushed_base_pose": None,
                "score": 0.9,
                "box": [1.0, 2.0, 3.0, 4.0],
                "mask_pixels": 40,
                "valid_depth_pixels": 40,
                "mask": "annotations/mask_chair_000000.png",
                "lifted": True,
                "lift_reason": None,
                "object_id": "chair_0",
            },
            {
                "label": "desk",
                "keyframe": 1,
                "timestamp_s": 1.0,
                "pushed_base_pose": None,
                "score": 0.6,
                "box": [2.0, 3.0, 4.0, 5.0],
                "mask_pixels": 36,
                "valid_depth_pixels": 0,
                "mask": "annotations/mask_desk_000001.png",
                "lifted": False,
                "lift_reason": "the mask falls outside the model crop",
                "object_id": None,
            },
        ],
        "objects": [
            {
                "id": "chair_0",
                "label": "chair",
                "frame": "model",
                "observation_count": 2,
                "point_count": 3,
                "centroid": [0.0, 0.0, 0.4],
                "aabb_min": [-0.1, -0.1, 0.3],
                "aabb_max": [0.1, 0.1, 0.6],
                "extent": [0.2, 0.2, 0.3],
                "radius": 0.1,
                "robot_frame": {
                    "frame": "robot_base",
                    "point_count": 3,
                    "centroid": [1.0, 2.0, 3.0],
                    "extent": [0.2, 0.2, 0.3],
                    "radius": 0.1,
                    "meters_per_unit": 0.5,
                },
            },
            {
                "id": "desk_0",
                "label": "desk",
                "frame": "model",
                "observation_count": 1,
                "point_count": 2,
                "centroid": [0.1, 0.1, 0.6],
                "aabb_min": [0.0, 0.0, 0.5],
                "aabb_max": [0.2, 0.2, 0.7],
                "extent": [0.2, 0.2, 0.2],
                "radius": 0.05,
            },
        ],
        "evaluation": {
            "labels": {
                "chair": {
                    "detections": 1,
                    "lifted_detections": 1,
                    "objects": 1,
                    "best_score": 0.9,
                    "median_nearest_centroid_distance": None,
                },
                "desk": {
                    "detections": 1,
                    "lifted_detections": 0,
                    "objects": 1,
                    "best_score": 0.6,
                    "median_nearest_centroid_distance": None,
                },
            }
        },
    }
    (map_dir / "objects.json").write_text(json.dumps(document))
    if overlay:
        annotations = map_dir / "annotations"
        annotations.mkdir()
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            annotations / "overlay_000000.png"
        )


def _assert_png(path: Path) -> None:
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.size[0] > 0 and image.size[1] > 0


def test_build_report_writes_plots_tables_and_gallery(tmp_path: Path) -> None:
    map_dir = tmp_path / "workspace"
    _write_map(map_dir)

    written = build_report(map_dir)

    report_dir = map_dir / "report"
    assert written == {
        "topdown": str(report_dir / TOPDOWN_FILE),
        "isometric": str(report_dir / ISOMETRIC_FILE),
        "html": str(report_dir / HTML_FILE),
        "csv": str(report_dir / OBJECTS_CSV),
    }
    _assert_png(report_dir / TOPDOWN_FILE)
    _assert_png(report_dir / ISOMETRIC_FILE)
    assert (report_dir / "annotations" / "overlay_000000.png").is_file()

    page = (report_dir / HTML_FILE).read_text()
    assert "floor at z=0" in page
    assert "chair_0" in page
    assert "desk (1)" in page
    assert "annotations/overlay_000000.png" in page
    # Only frame 0 has an overlay, so frame 1 adds no gallery figure.
    assert page.count("<figure><img") == 1

    with (report_dir / OBJECTS_CSV).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == ["chair_0", "desk_0"]
    chair, desk = rows
    assert chair["label"] == "chair"
    assert float(chair["centroid_z"]) == pytest.approx(0.4)
    assert float(chair["robot_centroid_x"]) == pytest.approx(1.0)
    assert desk["robot_centroid_x"] == ""


def test_report_without_depth_keeps_tables(tmp_path: Path) -> None:
    map_dir = tmp_path / "archive"
    _write_map(map_dir, gravity=False, floor=False, semantic_points=False)
    (map_dir / "cameras_c2w.npy").unlink()

    written = build_report(map_dir)

    assert set(written) == {"html", "csv"}
    assert not (map_dir / "report" / TOPDOWN_FILE).exists()
    page = (map_dir / "report" / HTML_FILE).read_text()
    assert "no lifted semantic points" in page
    assert "map frame as recorded" in page


def test_report_falls_back_to_the_camera_path(tmp_path: Path) -> None:
    map_dir = tmp_path / "sweep"
    _write_map(map_dir, gravity=False, floor=False)

    build_report(map_dir)

    page = (map_dir / "report" / HTML_FILE).read_text()
    assert "camera-aligned" in page


def test_report_requires_grounding(tmp_path: Path) -> None:
    map_dir = tmp_path / "raw"
    _write_map(map_dir)
    (map_dir / "objects.json").unlink()

    with pytest.raises(FileNotFoundError, match="ground_map"):
        build_report(map_dir)


def test_main_writes_to_named_output_dir(tmp_path: Path) -> None:
    map_dir = tmp_path / "workspace"
    _write_map(map_dir)
    output_dir = tmp_path / "elsewhere"

    main(
        [
            "--map-path",
            str(map_dir),
            "--output-dir",
            str(output_dir),
            "--max-points",
            "2",
        ]
    )

    assert (output_dir / HTML_FILE).is_file()
    assert (output_dir / OBJECTS_CSV).is_file()
    _assert_png(output_dir / TOPDOWN_FILE)
