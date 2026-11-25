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

"""Offline contracts for visible changes between recorded RGB-D frames."""

from types import SimpleNamespace

import numpy as np
import pytest

from toolkits.image_diff import image_diff
from toolkits.observation import GeometryObservation


def _frame(step, *, rgb=None, **geometry):
    if rgb is None:
        rgb = np.zeros((4, 5, 3), np.uint8)
    calibration = {
        "depth_m": np.ones(rgb.shape[:2]),
        "intrinsics": np.eye(3),
        "camera_to_world": np.eye(4),
        "world_frame_id": "table",
    }
    # Build the metric map from the finite samples so invalid depth is exercised
    # without introducing floating-point warnings during observation creation.
    depth = geometry.get("depth_m", calibration["depth_m"])
    if depth is not None and not np.isfinite(depth).all():
        calibration["world_xyz"] = np.ones(rgb.shape, dtype=float)
    return GeometryObservation(
        camera="overhead", step=step, rgb=rgb, **(calibration | geometry)
    )


def _diff(before, after, **arguments):
    frames = {before.step: before, after.step: after, -1: after}
    return image_diff(
        camera="overhead",
        before_step=before.step,
        observations=SimpleNamespace(
            get_observation=lambda *, camera, step: frames[step]
        ),
        **arguments,
    )


def test_reports_separate_change_boxes_and_denominators():
    before_rgb = np.zeros((4, 5, 3), np.uint8)
    before_rgb[1, 1] = [250, 0, 0]
    before_rgb[3, 4] = [200, 0, 0]
    after_rgb = before_rgb.copy()
    after_rgb[0, 1] = [0, 20, 0]
    after_rgb[1, 1] = [240, 0, 0]
    after_rgb[3, 4] = [0, 0, 0]

    before_depth = np.ones((4, 5))
    before_depth[0, 0] = 0
    after_depth = np.ones((4, 5))
    after_depth[0, 0] = 4
    after_depth[0, 1] = np.nan
    after_depth[1, 0] = np.inf
    after_depth[2, 0] = 4
    after_depth[0, 2] += 0.0625
    after_depth[1, 3] -= 0.125
    after_depth[2, 4] += 0.25
    mask = np.ones((4, 5), dtype=bool)
    mask[2, 0] = False

    before = _frame(7, rgb=before_rgb, depth_m=before_depth)
    after = _frame(8, rgb=after_rgb, depth_m=after_depth, valid_mask=mask)
    result = _diff(before, after, depth_threshold_m=0.125)
    assert result.data == {
        "status": "compared",
        "camera": "overhead",
        "before_step": 7,
        "after_step": 8,
        "image_shape": [4, 5],
        "rgb": {
            "status": "compared",
            "compared_pixels": 20,
            "changed_pixels": 2,
            "changed_ratio": 2 / 20,
            "bbox_rc": [0, 1, 4, 5],
        },
        "depth": {
            "status": "compared",
            "compared_pixels": 16,
            "changed_pixels": 2,
            "changed_ratio": 2 / 16,
            "bbox_rc": [1, 3, 3, 5],
            "valid_ratio": 16 / 20,
        },
    }
    assert result.images == []


def test_identical_views_return_zero_and_no_box():
    pose = np.eye(4)
    pose[0, 3] = 1e-6
    result = _diff(_frame(0), _frame(1, camera_to_world=pose), after_step=1)
    for modality in ("rgb", "depth"):
        assert result.data[modality]["changed_ratio"] == 0
        assert result.data[modality]["bbox_rc"] is None


def test_depth_from_world_points_is_measured_along_the_camera_axis():
    pose = np.array([[0, 0, 1, 1], [0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 0, 1]])
    before_xyz = np.broadcast_to([2.0, 2.0, 3.0], (4, 5, 3)).copy()
    after_xyz = before_xyz.copy()
    after_xyz[2, 1, 0] += 0.25
    after_xyz[2, 3, 1] += 0.5
    calibration = {"depth_m": None, "intrinsics": None, "camera_to_world": pose}
    result = _diff(
        _frame(0, world_xyz=before_xyz, **calibration),
        _frame(1, world_xyz=after_xyz, **calibration),
        mode="depth",
    )
    assert result.data["depth"]["bbox_rc"] == [2, 1, 3, 2]
    assert result.data["depth"]["changed_ratio"] == 1 / 20


@pytest.mark.parametrize(
    ("mode", "status"),
    [("rgb", "compared"), ("depth", "need_observation"), ("both", "partial")],
)
def test_rgb_remains_available_when_depth_coverage_is_too_low(mode, status):
    valid = np.zeros((4, 5), dtype=bool)
    valid[2, 3] = True
    result = _diff(_frame(0), _frame(1, valid_mask=valid), mode=mode).data

    assert result["status"] == status
    assert ("rgb" in result) is (mode != "depth")
    if mode == "rgb":
        assert "depth" not in result and "reason" not in result
    else:
        assert result["reason"] == "insufficient_common_depth"
        assert result["depth"] == {
            "status": "need_observation",
            "reason": "insufficient_common_depth",
            "compared_pixels": 1,
            "valid_ratio": 1 / 20,
        }


def test_depth_coverage_threshold_is_configurable_but_requires_a_sample():
    valid = np.zeros((4, 5), dtype=bool)
    empty = _diff(_frame(0), _frame(1, valid_mask=valid), min_depth_coverage=0)
    assert empty.data["depth"]["compared_pixels"] == 0
    assert empty.data["status"] == "partial"
    valid[2, 3] = True
    covered = _diff(
        _frame(0), _frame(1, valid_mask=valid), mode="depth", min_depth_coverage=0.05
    )
    assert covered.data["status"] == "compared"
    assert covered.data["depth"]["changed_ratio"] == 0
    assert "rgb" not in covered.data


@pytest.mark.parametrize(
    ("geometry", "reason"),
    [
        ({"rgb": np.zeros((2, 5, 3), np.uint8)}, "different_image_shape"),
        ({"world_frame_id": "other_frame"}, "different_world_frame"),
        (
            {"camera_to_world": None, "world_xyz": np.ones((4, 5, 3))},
            "missing_camera_pose",
        ),
        (
            {
                "camera_to_world": np.array(
                    [
                        [1, 0, 0, 0.1],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )
            },
            "camera_pose_changed",
        ),
        (
            {"intrinsics": None, "world_xyz": np.ones((4, 5, 3))},
            "incomplete_camera_intrinsics",
        ),
        ({"intrinsics": np.diag([2, 1, 1])}, "camera_intrinsics_changed"),
    ],
)
def test_mismatched_view_returns_no_change_metrics(geometry, reason):
    result = _diff(_frame(0), _frame(1, **geometry))
    assert result.data["status"] == "need_observation"
    assert result.data["reason"] == reason
    assert "rgb" not in result.data and "depth" not in result.data


def test_wrong_provider_identity_and_no_later_record_are_reported():
    wrong_step = image_diff(
        camera="overhead",
        before_step=0,
        after_step=1,
        observations=SimpleNamespace(get_observation=lambda **_: _frame(0)),
    )
    assert wrong_step.is_error
    assert "different camera or step" in wrong_step.data["error"]
    no_later = _diff(_frame(0), _frame(0))
    assert no_later.is_error
    assert "later after_step" in no_later.data["error"]


def test_one_centimetre_threshold_is_inclusive_at_sensor_scale():
    before_depth = np.full((4, 5), 0.55)
    after_depth = before_depth.copy()
    after_depth[1, 4] = 0.56
    after_depth[2, 2] = 0.55999
    # The second exact centimetre change rounds down under direct subtraction.
    before_depth[0, 1] = 0.56
    after_depth[0, 1] = 0.57
    before_depth[3, 3] = 0.56
    after_depth[3, 3] = 0.56999
    before, after = _frame(0, depth_m=before_depth), _frame(1, depth_m=after_depth)
    first = _diff(before, after, mode="depth").data
    assert first == _diff(before, after, mode="depth").data
    assert first["depth"]["changed_pixels"] == 2
    assert first["depth"]["changed_ratio"] == 0.1
    assert first["depth"]["bbox_rc"] == [0, 1, 2, 5]
    np.testing.assert_array_equal(before.depth_m, before_depth)
    np.testing.assert_array_equal(after.depth_m, after_depth)
