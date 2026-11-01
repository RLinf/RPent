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

"""Offline tests for lifting open-vocabulary detections onto a built map."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from toolkits.grounding.client import (
    Detection,
    DetectSegmentResult,
)
from toolkits.grounding.ground_map import (
    LocalMapSource,
    PreprocessSpec,
    ground_map,
    lift_mask,
    save_grounded_map,
    select_keyframes,
)

SOURCE_SIZE = (12, 16)
MODEL_SIZE = (8, 16)
FRAMES = 2


def _write_frame(path: Path, value: int) -> None:
    image = np.full((*SOURCE_SIZE, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path)


def _write_map(
    map_dir: Path,
    *,
    has_depth: bool = True,
    write_c2w: bool = True,
    camera_offset: float = 1.0,
) -> dict[str, Any]:
    map_dir.mkdir(parents=True)
    frames_dir = map_dir / "frames"
    frames_dir.mkdir()
    for index in range(FRAMES):
        _write_frame(frames_dir / f"{index:06d}.png", 10 + 10 * index)
    intrinsic = np.tile(
        np.asarray([[10.0, 0.0, 8.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]),
        (FRAMES, 1, 1),
    )
    extrinsic = np.tile(np.eye(3, 4), (FRAMES, 1, 1))
    camera_to_world = np.tile(np.eye(4), (FRAMES, 1, 1))
    camera_to_world[1, 0, 3] = camera_offset
    if has_depth:
        depth = np.full((1, FRAMES, MODEL_SIZE[0], MODEL_SIZE[1], 1), 2.0)
        np.save(map_dir / "depth.npy", depth)
        np.save(
            map_dir / "world_points_conf.npy",
            np.ones((FRAMES, MODEL_SIZE[0], MODEL_SIZE[1]), dtype=np.float32),
        )
    if write_c2w:
        np.save(map_dir / "cameras_c2w.npy", camera_to_world)
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "backend": "gct_stream",
        "map_name": map_dir.name,
        "frame": "model",
        "frames": FRAMES,
        "has_depth": has_depth,
        "has_world_points": has_depth,
        "preprocess": {
            "resize_width": 16,
            "patch_size": 2,
            "crop": "center",
            "input_size": list(MODEL_SIZE),
        },
        "cameras": {
            "frame": "model",
            "extrinsic_convention": "world_to_camera",
            "intrinsic_convention": "pixel",
            "shape": [FRAMES, 3, 4],
            "extrinsic": extrinsic.tolist(),
            "intrinsic_shape": [FRAMES, 3, 3],
            "intrinsic": intrinsic.tolist(),
        },
        "camera_to_world": {
            "path": "cameras_c2w.npy",
            "convention": "camera_to_world",
            "shape": [FRAMES, 4, 4],
        },
        "gravity": None,
        "floor": None,
        "scale": {
            "status": "arbitrary",
            "metric": False,
            "meters_per_unit": None,
            "source": None,
        },
        "alignment": {"status": "unavailable", "reason": "test map"},
    }
    entries = [
        {
            "index": index,
            "timestamp_s": float(index),
            "image": f"frames/{index:06d}.png",
            "source_size": list(SOURCE_SIZE),
            "pushed_base_pose": [float(index), 0.0, 0.0, 0.0, 0.0, 0.0],
            "model_keyframe": True,
        }
        for index in range(FRAMES)
    ]
    (map_dir / "keyframe_index.json").write_text(json.dumps(entries))
    (map_dir / "map.json").write_text(json.dumps(metadata))
    return metadata


class _FakePerception:
    """Detect a chair on the 3x3 source patch that keeps the object fixed."""

    def __init__(self) -> None:
        self.frames: list[int] = []

    def detect_segment(
        self, image: Any, *, labels: list[str], min_score: float = 0.3
    ) -> DetectSegmentResult:
        frame = np.asarray(Image.open(io.BytesIO(bytes(image))).convert("RGB"))
        value = int(frame[0, 0, 0])
        frame_index = 0 if value < 20 else 1
        self.frames.append(frame_index)
        # The camera moves +1 in x between frames. At depth 2 and fx 10 a
        # six-pixel shift keeps the patch on the same world point.
        columns = (6, 7, 8) if frame_index == 0 else (1, 2, 3)
        mask = np.zeros(SOURCE_SIZE, dtype=bool)
        mask[4:7, list(columns)] = True
        detections = tuple(
            Detection(
                label=label,
                score=0.9 if frame_index == 0 else 0.7,
                box=[float(columns[0]), 4.0, float(columns[-1]), 7.0],
                mask=mask,
            )
            for label in labels
        )
        return DetectSegmentResult(
            found=True,
            detections=detections,
            image_shape=SOURCE_SIZE,
        )


def test_select_keyframes_spans_the_archive() -> None:
    assert select_keyframes(5, max_keyframes=5) == [0, 1, 2, 3, 4]
    assert select_keyframes(10, max_keyframes=4) == [0, 3, 6, 9]
    assert select_keyframes(10, keyframes=[8, 2, 8]) == [2, 8]
    with pytest.raises(ValueError):
        select_keyframes(10, keyframes=[10])
    with pytest.raises(ValueError):
        select_keyframes(0)


def test_source_mask_maps_through_center_crop() -> None:
    spec = PreprocessSpec(resize_width=16, patch_size=2, input_size=MODEL_SIZE)
    mask = np.zeros(SOURCE_SIZE, dtype=bool)
    mask[4:7, 4:9] = True
    warped = spec.source_mask_to_model(mask, SOURCE_SIZE)
    # The model keeps resized rows 2..9, so source rows 4..6 become 2..4.
    assert warped.shape == MODEL_SIZE
    assert warped[2:5, 4:9].all()
    assert warped[:2].sum() == 0
    assert warped[5:].sum() == 0
    assert warped.sum() == mask.sum()


def test_lift_mask_uses_the_camera_pose() -> None:
    depth = np.full(MODEL_SIZE, 2.0, dtype=np.float32)
    mask = np.zeros(MODEL_SIZE, dtype=bool)
    mask[4, 8] = True
    intrinsic = np.asarray([[10.0, 0.0, 8.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    points = lift_mask(mask, depth, None, intrinsic, pose)
    # u == cx and v == cy place the point on the optical axis at z = 2.
    np.testing.assert_allclose(points, [[1.0, 2.0, 5.0]])

    confidence = np.zeros(MODEL_SIZE, dtype=np.float32)
    assert lift_mask(mask, depth, confidence, intrinsic, pose).shape == (0, 3)


def test_local_source_derives_poses_without_c2w_file(tmp_path: Path) -> None:
    map_dir = tmp_path / "map"
    _write_map(map_dir, write_c2w=False, camera_offset=0.0)
    source = LocalMapSource(map_dir)
    assert source.has_depth
    frame = source.load(0)
    assert frame.source_size == SOURCE_SIZE
    assert frame.timestamp_s == 0.0
    np.testing.assert_allclose(frame.camera_to_world, np.eye(4))


def test_ground_map_fuses_cross_frame_observations(tmp_path: Path) -> None:
    map_dir = tmp_path / "map"
    _write_map(map_dir)
    source = LocalMapSource(map_dir)
    perception = _FakePerception()

    result = ground_map(
        source, perception, ["chair"], max_keyframes=2, min_mask_pixels=1
    )

    assert result.keyframes == [0, 1]
    assert len(result.observations) == 2
    assert all(observation.lifted for observation in result.observations)
    assert len(result.objects) == 1
    instance = result.objects[0]
    assert instance.label == "chair"
    assert len(instance.members) == 2
    centroid = instance.summary["centroid"]
    assert centroid[2] == pytest.approx(2.0, abs=0.05)
    assert centroid[0] == pytest.approx(-0.2, abs=0.05)
    assert result.semantic_points.shape == (18, 3)
    assert result.semantic_labels.tolist() == [0] * 18
    assert result.summary["labels"]["chair"]["objects"] == 1


def test_ground_map_2d_fallback_for_archive_maps(tmp_path: Path) -> None:
    map_dir = tmp_path / "map"
    _write_map(map_dir, has_depth=False)
    source = LocalMapSource(map_dir)
    assert not source.has_depth
    with pytest.raises(RuntimeError):
        ground_map(
            source,
            _FakePerception(),
            ["chair"],
            max_keyframes=2,
            min_mask_pixels=1,
        )

    result = ground_map(
        source,
        _FakePerception(),
        ["chair"],
        max_keyframes=2,
        min_mask_pixels=1,
        allow_2d=True,
    )
    assert len(result.observations) == 2
    assert not any(observation.lifted for observation in result.observations)
    assert result.objects == []
    assert result.semantic_points.shape == (0, 3)
    assert result.observations[0].lift_reason == "the map has no depth"


def test_save_grounded_map_writes_the_contract(tmp_path: Path) -> None:
    map_dir = tmp_path / "map"
    _write_map(map_dir)
    result = ground_map(
        LocalMapSource(map_dir),
        _FakePerception(),
        ["chair"],
        max_keyframes=2,
        min_mask_pixels=1,
    )

    written = save_grounded_map(result)

    objects = json.loads((map_dir / "objects.json").read_text())
    assert objects["schema_version"] == 1
    assert objects["map_path"] == str(map_dir)
    assert objects["preprocess"]["input_size"] == list(MODEL_SIZE)
    assert objects["objects"][0]["id"] == "chair-0"
    assert objects["objects"][0]["observation_count"] == 2
    assert objects["detections"][0]["mask"] == "annotations/000000_chair.png"
    assert objects["evaluation"]["labels"]["chair"]["lifted_detections"] == 2
    assert (map_dir / "annotations" / "000000_chair.png").is_file()
    assert (map_dir / "annotations" / "overlay_000000.png").is_file()
    assert (map_dir / "semantic_points.npz").is_file()
    assert (map_dir / "semantic_points.ply").is_file()
    assert (map_dir / "evaluation.json").is_file()
    assert written["objects"] == str(map_dir / "objects.json")

    with np.load(map_dir / "semantic_points.npz") as arrays:
        assert arrays["points"].shape == (18, 3)
        assert arrays["label_names"].tolist() == ["chair"]
        np.testing.assert_array_equal(arrays["object_ids"], np.zeros(18, dtype=int))

    header = (map_dir / "semantic_points.ply").read_bytes().split(b"end_header\n")[0]
    assert b"element vertex 18" in header
    assert b"property uchar label" in header
