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

"""Offline tests for map-based corridor clearance."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from toolkits.localization.clearance import (
    ClearanceGeometry,
    ClearanceRequest,
    ClearanceResult,
    MapClearance,
    check_corridor,
)

#: A pose looking along model ``+x`` with camera ``+y`` pointing up.
LOOK_ALONG_X = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _floor(
    x_max: float = 2.0, half_width: float = 1.0, step: float = 0.05
) -> np.ndarray:
    """Return a dense floor plane at ``z = 0``."""
    xs = np.arange(0.0, x_max + step, step)
    ys = np.arange(-half_width, half_width + step, step)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1), np.zeros(grid_x.size)], axis=1
    )


def _write_map(
    root: Path,
    serialized: np.ndarray,
    *,
    camera_height: float = 0.2,
    floor_offset: float = 0.0,
    up: list[float] | None = None,
    meters_per_unit: float | None = None,
    confidence: np.ndarray | None = None,
) -> Path:
    """Write the map artifacts a clearance check reads."""
    count = (
        int(serialized.shape[1]) if serialized.ndim == 5 else int(serialized.shape[0])
    )
    metadata = {
        "schema_version": 2,
        "frames": count,
        "gravity": {
            "status": "estimated",
            "up": [0.0, 0.0, 1.0] if up is None else up,
        },
        "floor": {
            "status": "estimated",
            "offset": floor_offset,
            "camera_height": camera_height,
        },
        "scale": {
            "metric": meters_per_unit is not None,
            "meters_per_unit": meters_per_unit,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "map.json").write_text(json.dumps(metadata))
    np.save(root / "world_points.npy", serialized.astype(np.float32))
    if confidence is not None:
        np.save(root / "world_points_conf.npy", confidence.astype(np.float32))
    return root


def _serialized(points: np.ndarray, frames: int | None = None) -> np.ndarray:
    """Wrap one ``[N,3]`` point set as ``[1, F, N, 1, 3]`` frames."""
    count = 1 if frames is None else int(frames)
    stacked = np.repeat(points[None, None, :, None, :], count, axis=1)
    return stacked.astype(np.float32)


def _wall(x: float = 1.0, z: float = 0.1) -> np.ndarray:
    """Three distinct points inside a corridor running along ``+x``."""
    return np.asarray([[x, -0.1, z], [x, 0.0, z], [x, 0.1, z]], dtype=np.float64)


class TestClearanceGeometry:
    def test_reads_gravity_floor_and_metric_scale(self) -> None:
        geometry = ClearanceGeometry.from_metadata(
            {
                "gravity": {"status": "estimated", "up": [0.0, 1.0, 0.0]},
                "floor": {"status": "estimated", "offset": 0.5, "camera_height": 0.5},
                "scale": {"meters_per_unit": 2.0},
            }
        )

        assert geometry.usable
        assert geometry.camera_height == pytest.approx(0.5)
        assert geometry.meters_per_unit == pytest.approx(2.0)
        assert geometry.floor_offset == pytest.approx(0.5)
        assert np.allclose(geometry.rotation @ np.asarray([0.0, 1.0, 0.0]), [0, 0, 1])

    def test_camera_height_anchors_scale_only_when_map_has_none(self) -> None:
        metadata = {
            "gravity": {"status": "estimated", "up": [0.0, 0.0, 1.0]},
            "floor": {"status": "estimated", "offset": 0.0, "camera_height": 0.4},
            "scale": {"meters_per_unit": None},
        }

        anchored = ClearanceGeometry.from_metadata(metadata, camera_height_m=0.6)
        assert anchored.meters_per_unit == pytest.approx(1.5)

        with pytest.raises(ValueError, match="positive"):
            ClearanceGeometry.from_metadata(metadata, camera_height_m=0.0)

    @pytest.mark.parametrize(
        ("metadata", "reason"),
        [
            ({"floor": {"status": "estimated"}}, "gravity"),
            (
                {
                    "gravity": {"status": "estimated", "up": [0.0, 0.0, 1.0]},
                    "floor": {"status": "missing"},
                },
                "floor",
            ),
            (
                {
                    "gravity": {"status": "estimated", "up": [0.0, 0.0, 1.0]},
                    "floor": {
                        "status": "estimated",
                        "offset": 0.0,
                        "camera_height": -1.0,
                    },
                },
                "not positive",
            ),
        ],
    )
    def test_unusable_geometry_explains_why(self, metadata: dict, reason: str) -> None:
        geometry = ClearanceGeometry.from_metadata(metadata)

        assert not geometry.usable
        assert geometry.status == "unusable"
        assert reason in (geometry.reason or "")


class TestCorridorScan:
    def test_reports_blocking_points_and_nearest_obstacle(self) -> None:
        floor = _floor()
        wall = np.asarray(
            [[1.0, -0.2, 0.1], [1.0, 0.0, 0.1], [1.0, 0.2, 0.1]], dtype=np.float64
        )

        scan = check_corridor(
            np.concatenate([floor, wall]),
            origin=np.zeros(3),
            forward=np.asarray([1.0, 0.0]),
            length=2.0,
            half_width=0.5,
            min_height=0.05,
            max_height=1.0,
        )

        assert scan.status == "blocked"
        assert scan.blocked_points == 3
        assert scan.coverage == pytest.approx(1.0)
        assert scan.nearest_index is not None
        assert scan.nearest is not None
        assert scan.nearest["distance"] == pytest.approx(1.0)
        assert scan.nearest["height"] == pytest.approx(0.1)
        assert abs(scan.nearest["lateral"]) <= 0.2

    def test_clear_needs_every_bin_observed(self) -> None:
        scan = check_corridor(
            _floor(),
            origin=np.zeros(3),
            forward=np.asarray([1.0, 0.0]),
            length=2.0,
            half_width=0.5,
            min_height=0.05,
            max_height=1.0,
        )
        assert scan.status == "clear"
        assert scan.coverage == pytest.approx(1.0)
        assert scan.blocked_points == 0
        assert scan.observed_distance is None

    def test_missing_geometry_is_unmapped_not_clear(self) -> None:
        scan = check_corridor(
            _floor(x_max=1.0),
            origin=np.zeros(3),
            forward=np.asarray([1.0, 0.0]),
            length=2.0,
            half_width=0.5,
            min_height=0.05,
            max_height=1.0,
        )

        assert scan.status == "unmapped"
        assert 1.0 <= (scan.observed_distance or 0.0) <= 1.2
        assert 0.0 < scan.coverage < 1.0
        assert not bool(scan.observed_bins.all())

    def test_points_outside_the_corridor_or_height_band_are_ignored(self) -> None:
        outside = np.asarray(
            [
                [1.0, 1.0, 0.1],  # too far to the side
                [1.0, 0.0, 2.0],  # above the height band
                [-0.5, 0.0, 0.1],  # behind the camera
                [1.0, 0.0, 0.01],  # below the obstacle height
            ]
        )

        scan = check_corridor(
            np.concatenate([_floor(), outside]),
            origin=np.zeros(3),
            forward=np.asarray([1.0, 0.0]),
            length=2.0,
            half_width=0.5,
            min_height=0.05,
            max_height=1.0,
        )

        assert scan.status == "clear"
        assert scan.blocked_points == 0

    def test_single_stray_point_does_not_block(self) -> None:
        points = np.concatenate(
            [_floor(), np.asarray([[1.0, 0.0, 0.1], [1.1, 0.0, 0.1]])]
        )

        scan = check_corridor(
            points,
            origin=np.zeros(3),
            forward=np.asarray([1.0, 0.0]),
            length=2.0,
            half_width=0.5,
            min_height=0.05,
            max_height=1.0,
            blocking_points=3,
        )

        assert scan.blocked_points == 2
        assert scan.status == "clear"

    def test_heading_rotates_the_corridor(self) -> None:
        wall = np.asarray(
            [[1.0, -0.2, 0.1], [1.0, 0.0, 0.1], [1.0, 0.2, 0.1]], dtype=np.float64
        )
        points = np.concatenate([_floor(), wall])

        straight = check_corridor(
            points,
            origin=np.zeros(3),
            forward=np.asarray([1.0, 0.0]),
            length=2.0,
            half_width=0.5,
            min_height=0.05,
            max_height=1.0,
        )
        turned = check_corridor(
            points,
            origin=np.zeros(3),
            forward=np.asarray([0.0, 1.0]),
            length=2.0,
            half_width=0.5,
            min_height=0.05,
            max_height=1.0,
        )

        assert straight.status == "blocked"
        assert turned.status != "blocked"


class TestMapClearance:
    def test_check_reports_distance_in_camera_heights(self, tmp_path: Path) -> None:
        floor = np.concatenate([_floor(), _wall()])
        root = _write_map(
            tmp_path / "map", _serialized(floor, frames=1), camera_height=0.2
        )
        clearance = MapClearance(root, pixel_stride=1)

        result = clearance.check(
            LOOK_ALONG_X,
            ClearanceRequest(
                length=12.0, half_width=2.0, min_height=0.05, max_height=1.0
            ),
        )

        assert result.status == "blocked"
        assert result.unit == "camera_heights"
        assert result.unit_scale == pytest.approx(5.0)
        assert result.nearest is not None
        assert result.nearest["distance"] == pytest.approx(5.0)
        assert result.nearest["height"] == pytest.approx(0.5)
        assert result.frame == "model"

    def test_metric_request_needs_a_metric_anchor(self, tmp_path: Path) -> None:
        root = _write_map(
            tmp_path / "map", _serialized(_floor(), frames=1), camera_height=0.4
        )

        plain = MapClearance(root, pixel_stride=1).check(
            LOOK_ALONG_X, ClearanceRequest(unit="m")
        )
        anchored = MapClearance(root, pixel_stride=1, camera_height_m=0.8).check(
            LOOK_ALONG_X, ClearanceRequest(unit="m", length=1.5)
        )

        assert plain.status == "unusable"
        assert "metric scale" in (plain.reason or "")
        assert anchored.status == "clear"
        assert anchored.unit_scale == pytest.approx(2.0)
        assert anchored.meters_per_unit == pytest.approx(2.0)

    def test_low_confidence_points_are_ignored(self, tmp_path: Path) -> None:
        points = np.concatenate([_floor(), _wall()])
        confidence = np.ones((1, 1, points.shape[0], 1, 1), dtype=np.float32)
        confidence[0, 0, -3:] = 0.0
        root = _write_map(
            tmp_path / "map",
            _serialized(points, frames=1),
            confidence=confidence,
        )

        result = MapClearance(root, pixel_stride=1).check(
            LOOK_ALONG_X,
            ClearanceRequest(
                unit="map_units", length=2.0, half_width=0.5, min_height=0.05
            ),
        )

        assert result.status == "clear"
        assert result.blocked_points == 0
        assert result.point_count == points.shape[0] - 3

    def test_duplicate_frames_are_voxel_deduped(self, tmp_path: Path) -> None:
        floor = _floor()
        root = _write_map(tmp_path / "map", _serialized(floor, frames=2))

        clearance = MapClearance(root, pixel_stride=1)

        assert clearance.points.shape[0] == floor.shape[0]

    def test_nearest_obstacle_gets_semantic_label(self, tmp_path: Path) -> None:
        wall = _wall()
        points = np.concatenate([_floor(), wall])
        root = _write_map(
            tmp_path / "map",
            _serialized(points, frames=1),
            camera_height=0.2,
            floor_offset=0.0,
        )
        np.savez(
            root / "semantic_points.npz",
            points=np.asarray([[1.0, -0.1, 0.1]], dtype=np.float32),
            labels=np.asarray([0], dtype=np.int32),
            scores=np.asarray([0.9], dtype=np.float32),
            object_ids=np.asarray([0], dtype=np.int32),
            frame_indices=np.asarray([0], dtype=np.int32),
            label_names=np.asarray(["chair"]),
        )
        (root / "objects.json").write_text(
            json.dumps({"objects": [{"id": "obj_0001"}]})
        )

        result = MapClearance(root, pixel_stride=1).check(
            LOOK_ALONG_X,
            ClearanceRequest(
                unit="map_units", length=2.0, half_width=0.5, min_height=0.05
            ),
        )

        assert result.status == "blocked"
        assert result.nearest is not None
        assert result.nearest["label"] == "chair"
        assert result.nearest["object_id"] == "obj_0001"

    def test_missing_pose_is_unlocalized(self, tmp_path: Path) -> None:
        root = _write_map(tmp_path / "map", _serialized(_floor(), frames=1))

        result = MapClearance(root, pixel_stride=1).check(
            None,
            ClearanceRequest(),
            localization={"status": "lost", "reason": "few matches"},
        )

        assert result.status == "unlocalized"
        assert result.pose is None
        assert result.localization == {"status": "lost", "reason": "few matches"}

    def test_near_vertical_axis_is_unusable(self, tmp_path: Path) -> None:
        root = _write_map(tmp_path / "map", _serialized(_floor(), frames=1))
        vertical = np.eye(4)
        vertical[2, 3] = 1.0

        result = MapClearance(root, pixel_stride=1).check(vertical, ClearanceRequest())

        assert result.status == "unusable"
        assert "vertical" in (result.reason or "")

    def test_non_finite_pose_is_rejected(self, tmp_path: Path) -> None:
        root = _write_map(tmp_path / "map", _serialized(_floor(), frames=1))
        pose = np.eye(4)
        pose[0, 3] = math.nan

        with pytest.raises(ValueError, match="finite"):
            MapClearance(root, pixel_stride=1).check(pose, ClearanceRequest())


class TestClearanceResult:
    def test_round_trip_preserves_the_record(self, tmp_path: Path) -> None:
        root = _write_map(
            tmp_path / "map",
            _serialized(
                np.concatenate([_floor(), _wall()]),
                frames=1,
            ),
            camera_height=0.2,
        )
        result = MapClearance(root, pixel_stride=1).check(
            LOOK_ALONG_X,
            ClearanceRequest(length=2.0, half_width=0.5, min_height=0.05),
        )

        restored = ClearanceResult.from_dict(json.loads(json.dumps(result.to_dict())))

        assert restored.status == result.status
        assert restored.unit == result.unit
        assert restored.nearest == result.nearest
        assert restored.coverage_bins == result.coverage_bins
        assert restored.pose is not None
        assert np.allclose(restored.pose, result.pose)

    @pytest.mark.parametrize(
        "changes",
        [
            {"status": ""},
            {"unit": "metres"},
            {"length": "long"},
            {"pose": [[1.0, 0.0], [0.0, 1.0]]},
            {"blocked_points": -1},
        ],
    )
    def test_malformed_records_are_rejected(self, changes: dict) -> None:
        payload = {
            "status": "clear",
            "unit": "camera_heights",
            "length": 2.0,
            "half_width": 0.75,
            "heading_deg": 0.0,
            "min_height": 0.05,
            "max_height": 4.0,
            "frame": "model",
            "point_count": 10,
            "blocked_points": 0,
            "coverage_bins": [True],
            "latency_ms": 1.0,
        }
        payload.update(changes)

        with pytest.raises(ValueError):
            ClearanceResult.from_dict(payload)


def test_cli_checks_an_archived_pose(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from toolkits.localization.clearance import main

    root = _write_map(
        tmp_path / "map",
        _serialized(_floor(), frames=1),
        camera_height=0.2,
    )
    np.save(root / "cameras_c2w.npy", LOOK_ALONG_X[None])

    main(
        [
            "--map-path",
            str(root),
            "--frame-index",
            "0",
            "--length",
            "10",
            "--half-width",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "clear"
    assert payload["keyframe"] == 0
