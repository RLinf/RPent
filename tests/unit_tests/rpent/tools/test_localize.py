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

"""Offline tests for map localization and replay evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from rpent.tools.ground_map import PreprocessSpec
from rpent.tools.localize import (
    FeatureMatches,
    FeatureSet,
    LocalizationResult,
    MapLocalizer,
    evaluate,
    pose_error,
    preprocess_image,
)

SIZE = 6
POSE_X = (0.0, 0.5, 1.0)


class _FakeBackend:
    """Return a dense grid of identical features for every image."""

    def extract(self, image: np.ndarray) -> FeatureSet:
        array = np.asarray(image)
        height, width = array.shape[:2]
        ys, xs = np.mgrid[0:height, 0:width]
        keypoints = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
        descriptors = np.ones((keypoints.shape[0], 4), dtype=np.float32)
        return FeatureSet(keypoints, descriptors, (width, height), (width, height))

    def match(self, query: FeatureSet, keyframe: FeatureSet) -> FeatureMatches:
        count = min(query.count, keyframe.count)
        indices = np.arange(count, dtype=np.int64)
        return FeatureMatches(indices, indices, np.ones(count, np.float32))


def _write_map(map_dir: Path, *, poses: np.ndarray | None = None) -> np.ndarray:
    map_dir.mkdir(parents=True)
    frames = len(POSE_X) if poses is None else len(poses)
    if poses is None:
        poses = np.tile(np.eye(4), (frames, 1, 1))
        for index, x in enumerate(POSE_X):
            poses[index, 0, 3] = x
    depth = 4.0 + np.random.default_rng(0).random((frames, SIZE, SIZE))
    depth = depth.astype(np.float32)
    intrinsic = np.asarray(
        [[6.0, 0.0, (SIZE - 1) / 2.0], [0.0, 6.0, (SIZE - 1) / 2.0], [0.0, 0.0, 1.0]]
    )
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "map_name": map_dir.name,
        "frame": "model",
        "frames": frames,
        "has_depth": True,
        "has_world_points": True,
        "preprocess": {
            "resize_width": SIZE,
            "patch_size": 1,
            "input_size": [SIZE, SIZE],
            "crop": "center",
        },
        "cameras": {
            "frame": "model",
            "intrinsic": [intrinsic.tolist() for _ in range(frames)],
        },
        "camera_to_world": {
            "path": "cameras_c2w.npy",
            "convention": "camera_to_world",
            "shape": [frames, 4, 4],
        },
        "scale": {
            "status": "arbitrary",
            "metric": False,
            "meters_per_unit": None,
            "source": None,
        },
        "alignment": {"status": "unavailable", "reason": "test map"},
    }
    (map_dir / "map.json").write_text(json.dumps(metadata))
    np.save(map_dir / "cameras_c2w.npy", np.asarray(poses, dtype=np.float64))
    np.save(map_dir / "depth.npy", depth)
    np.save(
        map_dir / "world_points_conf.npy",
        np.ones((frames, SIZE, SIZE), dtype=np.float32),
    )
    (map_dir / "keyframe_index.json").write_text(
        json.dumps(
            [{"index": index, "model_keyframe": True} for index in range(frames)]
        )
    )
    frames_dir = map_dir / "frames"
    frames_dir.mkdir()
    for index in range(frames):
        Image.fromarray(np.full((SIZE, SIZE, 3), 40 + index, dtype=np.uint8)).save(
            frames_dir / f"{index:06d}.png"
        )
    return depth


def test_preprocess_matches_the_server_geometry() -> None:
    spec = PreprocessSpec()
    portrait = np.zeros((960, 540, 3), dtype=np.uint8)
    landscape = np.zeros((540, 960, 3), dtype=np.uint8)

    assert preprocess_image(portrait, spec).shape[:2] == (518, 518)
    assert preprocess_image(landscape, spec).shape[:2] == (294, 518)
    assert spec.resized_size((960, 540)) == (924, 518)
    assert spec.crop_top((960, 540)) == 203
    assert spec.crop_top((540, 960)) == 0


def test_preprocess_rejects_non_rgb_input() -> None:
    with pytest.raises(ValueError, match="RGB"):
        preprocess_image(np.zeros((10, 10), dtype=np.uint8), PreprocessSpec())


def test_pose_error_measures_translation_and_rotation() -> None:
    ground_truth = np.eye(4)
    estimated = np.eye(4)
    estimated[:3, 3] = [2.0, 0.0, 0.0]
    estimated[:3, :3] = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    translation, rotation = pose_error(estimated, ground_truth)

    assert translation == pytest.approx(2.0)
    assert rotation == pytest.approx(90.0)


def test_localize_recovers_the_matching_keyframe_pose(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    map_dir = tmp_path / "workspace"
    _write_map(map_dir)
    localizer = MapLocalizer(map_dir, backend=_FakeBackend())

    for frame, expected_x in enumerate(POSE_X):
        result = localizer.localize(localizer.load_image(frame), candidates=[frame])

        assert result.status == "ok", result.reason
        assert result.keyframe == frame
        assert result.selection == "explicit"
        assert result.inliers >= 30
        assert result.reprojection_rms_px < 1e-3
        assert result.pose is not None
        assert result.pose[0, 3] == pytest.approx(expected_x, abs=1e-4)
        assert np.allclose(result.pose[:3, :3], np.eye(3), atol=1e-6)


def test_localize_flags_two_equally_supported_places(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    map_dir = tmp_path / "ambiguous"
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[1, 0, 3] = 3.0
    _write_map(map_dir, poses=poses)
    localizer = MapLocalizer(map_dir, backend=_FakeBackend())

    result = localizer.localize(localizer.load_image(0), candidates=[0, 1])

    assert result.inliers >= 30
    assert result.ambiguous
    assert result.status == "ambiguous"
    assert result.second_keyframe in {0, 1}
    assert result.second_keyframe != result.keyframe
    assert result.second_inliers >= 30
    assert "distinct pose" in (result.reason or "")


def test_localize_rejects_a_fully_hidden_candidate_set(tmp_path: Path) -> None:
    map_dir = tmp_path / "workspace"
    _write_map(map_dir)
    localizer = MapLocalizer(map_dir, backend=_FakeBackend())

    with pytest.raises(ValueError, match="no model keyframes remain"):
        localizer.localize(localizer.load_image(0), exhaustive=True, exclude=[0, 1, 2])


def test_evaluate_replays_held_out_frames(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    map_dir = tmp_path / "workspace"
    _write_map(map_dir, poses=np.tile(np.eye(4), (3, 1, 1)))
    localizer = MapLocalizer(map_dir, backend=_FakeBackend())
    output = tmp_path / "eval.json"

    payload = evaluate(
        map_dir,
        queries=[0, 1, 2],
        exclusion=0,
        exclusion_radius=0.0,
        exclusion_rotation=0.0,
        localizer=localizer,
        output=output,
    )

    summary = payload["summary"]
    assert summary["attempted"] == 3
    assert summary["localized"] == 3
    assert summary["accurate"] == 3
    assert summary["success_rate"] == pytest.approx(1.0)
    assert summary["translation_error"]["median"] == pytest.approx(0.0, abs=1e-6)
    assert summary["rotation_error_deg"]["median"] == pytest.approx(0.0, abs=1e-6)
    assert summary["translation_unit"] == "map units"
    assert [record["frame"] for record in payload["results"]] == [0, 1, 2]
    assert all(record["accurate"] for record in payload["results"])
    assert output.is_file()
    assert json.loads(output.read_text())["summary"]["localized"] == 3


class _StubSource:
    frame_count = 3

    def __init__(self, poses: np.ndarray | None = None) -> None:
        if poses is None:
            poses = np.tile(np.eye(4), (self.frame_count, 1, 1))
        self.camera_to_world = np.asarray(poses, dtype=np.float64)

    def load(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(camera_to_world=self.camera_to_world[index])


class _StubLocalizer:
    def __init__(
        self,
        map_path: Path,
        results: list[LocalizationResult],
        poses: np.ndarray | None = None,
    ) -> None:
        self.map_path = str(map_path)
        self.source = _StubSource(poses)
        self.model_indices = [0, 1, 2]
        self.all_indices = [0, 1, 2]
        self.scale = {"metric": False, "meters_per_unit": None}
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def load_image(self, index: int) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def localize(self, image: np.ndarray, **kwargs: Any) -> LocalizationResult:
        self.calls.append(kwargs)
        return self._results.pop(0)


def _result(
    *,
    pose: np.ndarray | None,
    latency_ms: float,
    query_keypoints: int = 300,
    status: str | None = None,
    ambiguous: bool = False,
) -> LocalizationResult:
    if status is None:
        status = "ok" if pose is not None else "lost"
    poses = pose is not None
    reason = None
    if status == "ambiguous":
        reason = "test ambiguity"
    elif status != "ok":
        reason = "test loss"
    return LocalizationResult(
        status=status,
        frame="model",
        pose=pose,
        keyframe=0 if poses else None,
        matches=100 if poses else 0,
        correspondences=100 if poses else 0,
        inliers=90 if poses else 0,
        inlier_ratio=0.9 if poses else 0.0,
        reprojection_rms_px=1.0 if poses else None,
        query_keypoints=query_keypoints,
        candidates_checked=2,
        selection="explicit",
        ambiguous=ambiguous,
        latency_ms=latency_ms,
        scale={"metric": False, "meters_per_unit": None},
        reason=reason,
    )


def test_evaluate_aggregates_errors_and_hides_neighbors(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "stub_map"
    offset = np.eye(4)
    offset[0, 3] = 0.2
    stub = _StubLocalizer(
        map_path,
        [
            _result(pose=offset, latency_ms=10.0),
            _result(pose=None, latency_ms=20.0),
        ],
    )
    output = tmp_path / "stub_eval.json"

    payload = evaluate(
        map_path,
        queries=[1, 2],
        exclusion=1,
        exclusion_radius=0.0,
        exclusion_rotation=0.0,
        localizer=stub,
        output=output,
    )

    summary = payload["summary"]
    assert summary["attempted"] == 2
    assert summary["localized"] == 1
    assert summary["accurate"] == 1
    assert summary["ambiguous"] == 0
    assert summary["lost"] == 1
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["translation_error"]["median"] == pytest.approx(0.2)
    assert summary["queries_with_200_keypoints"] == 2
    assert summary["success_rate_with_200_keypoints"] == pytest.approx(0.5)
    assert summary["latency_ms"]["median"] == pytest.approx(15.0)
    # The query and its neighbors are hidden from candidate selection.
    assert stub.calls[0]["exclude"] == [0, 1, 2]
    assert stub.calls[1]["exclude"] == [1, 2]
    assert stub.calls[0]["exhaustive"] is True
    assert output.is_file()


def test_evaluate_hides_only_near_identical_poses(tmp_path: Path) -> None:
    map_path = tmp_path / "stub_map"
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[0, 0, 3] = 0.05
    poses[2, 0, 3] = 0.5
    stub = _StubLocalizer(
        map_path,
        [_result(pose=np.eye(4), latency_ms=1.0)],
        poses=poses,
    )

    payload = evaluate(
        map_path,
        queries=[1],
        exclusion=0,
        exclusion_radius=0.1,
        exclusion_rotation=10.0,
        localizer=stub,
        output=tmp_path / "spatial_eval.json",
    )

    # Frame 0 is a near-identical pose; frame 2 is farther along the path.
    assert stub.calls[0]["exclude"] == [0, 1]
    assert payload["exclusion"] == {
        "index": 0,
        "radius": 0.1,
        "rotation_deg": 10.0,
    }
    assert payload["summary"]["localized"] == 1


def test_evaluate_does_not_count_ambiguous_as_success(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "stub_map"
    stub = _StubLocalizer(
        map_path,
        [
            _result(
                pose=np.eye(4),
                latency_ms=1.0,
                status="ambiguous",
                ambiguous=True,
            )
        ],
    )

    payload = evaluate(
        map_path,
        queries=[1],
        exclusion=0,
        exclusion_radius=0.0,
        exclusion_rotation=0.0,
        localizer=stub,
        output=tmp_path / "ambiguous_eval.json",
    )

    summary = payload["summary"]
    assert summary["attempted"] == 1
    assert summary["localized"] == 0
    assert summary["accurate"] == 0
    assert summary["ambiguous"] == 1
    assert summary["lost"] == 0
    assert summary["success_rate"] == pytest.approx(0.0)
