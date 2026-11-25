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

"""Offline placement clearance contracts on synthetic RGB-D observations."""

from types import SimpleNamespace

import numpy as np
import pytest

from toolkits.check_free_space import check_free_space
from toolkits.observation import GeometryObservation


@pytest.fixture
def table():
    x, y = np.meshgrid(np.linspace(-0.2, 0.2, 81), np.linspace(0.2, -0.2, 81))
    return np.stack((x, y, np.full_like(x, 0.5)), axis=-1)


def _view(points, *, up=(0, 0, 1), valid=None):
    return GeometryObservation(
        camera="overhead",
        step=3,
        world_frame_id="table",
        rgb=np.zeros(points.shape, dtype=np.uint8),
        world_xyz=points,
        intrinsics=np.array([[100, 0, 40], [0, 100, 40], [0, 0, 1]]),
        camera_to_world=np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ]
        )
        @ np.diag([1, -1, -1, 1]),
        up_direction=up,
        valid_mask=valid,
    )


def _check(view, **overrides):
    arguments = {
        "camera": "overhead",
        "center_xyz": [0, 0, 0.5],
        "world_frame_id": "table",
        "footprint_radius_m": 0.1,
        "clearance_m": 0.08,
    }
    return check_free_space(
        **(arguments | overrides),
        observations=SimpleNamespace(get_observation=lambda **_: view),
    )


def test_clear_from_pinhole_depth_and_rotated_world_up():
    size = 81
    view = GeometryObservation(
        camera="overhead",
        step=3,
        world_frame_id="table",
        rgb=np.zeros((size, size, 3), dtype=np.uint8),
        depth_m=np.full((size, size), 0.5),
        intrinsics=np.array([[100, 0, 40], [0, 100, 40], [0, 0, 1]]),
        camera_to_world=np.array(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 1],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        ),
        up_direction=np.array([0, 1, 0]),
    )
    result = _check(view, center_xyz=[0, 0.5, 0], max_slope_deg=0)
    assert result.data["status"] == "free"
    assert result.data["camera"] == "overhead" and result.data["step"] == 3
    assert result.data["plane"]["normal_xyz"] == pytest.approx([0, 1, 0])
    assert result.data["slope_deg"] == pytest.approx(0)
    assert result.data["support_coverage"] == 1
    assert result.data["obstacle_points"] == 0 and result.images == []


def test_measured_obstacle_blocks_even_with_incomplete_support_coverage(table):
    patch = (np.abs(table[..., 0] - 0.05) < 0.025) & (np.abs(table[..., 1]) < 0.026)
    table[patch, 2] += 0.04
    visible = ~((table[..., 0] < -0.04) & (table[..., 1] < -0.03))
    data = _check(_view(table, valid=visible)).data

    assert data["status"] == "blocked" and data["reason"] == "obstacle_in_clearance"
    assert data["support_coverage"] < 1
    assert data["obstacle_points"] == int(patch.sum())
    assert data["max_obstacle_height_m"] == pytest.approx(0.04)


def test_clutter_outside_the_requested_footprint_is_excluded(table):
    patch = (np.abs(table[..., 0] - 0.125) < 0.012) & (np.abs(table[..., 1]) < 0.012)
    table[patch, 2] += 0.04
    data = _check(_view(table)).data
    assert data["status"] == "free" and data["obstacle_points"] == 0


def test_visible_slope_is_compared_with_the_requested_limit(table):
    slope = np.tan(np.radians(15))
    # Each point lies on both the original camera ray and z = 0.5 + slope*x.
    depth_scale = 1 / (1 + 2 * slope * table[..., 0])
    table[..., :2] *= depth_scale[..., None]
    table[..., 2] = 1 - 0.5 * depth_scale
    view = _view(table)
    blocked = _check(view).data
    assert blocked["status"] == "blocked" and blocked["reason"] == "support_too_steep"
    assert blocked["slope_deg"] == pytest.approx(15)
    accepted = _check(view, max_slope_deg=16).data
    assert accepted["status"] == "free" and accepted["support_coverage"] == 1


@pytest.mark.parametrize("occlusion", ["central_hole", "frame_edge"])
def test_missing_support_returns_unknown(table, occlusion):
    if occlusion == "central_hole":
        visible = (np.abs(table[..., 0]) > 0.04) | (np.abs(table[..., 1]) > 0.04)
    else:
        visible = table[..., 0] <= 0
    data = _check(_view(table, valid=visible)).data
    assert (
        data["status"] == "unknown"
        and data["reason"] == "insufficient_support_coverage"
    )
    assert 0 < data["support_coverage"] < 1


@pytest.mark.parametrize(
    ("offset", "reason"),
    [(0.12, "surface_above_clearance"), (-0.035, "support_discontinuity")],
)
def test_unknown_on_overhang_or_discontinuous_surface(table, offset, reason):
    patch = (np.abs(table[..., 0]) < 0.025) & (np.abs(table[..., 1]) < 0.025)
    table[patch, 2] += offset
    data = _check(_view(table)).data
    assert data["status"] == "unknown" and data["reason"] == reason


def test_plane_at_the_wrong_height_cannot_replace_the_expected_support(table):
    table[..., 2] += 0.006
    data = _check(_view(table)).data
    assert data["status"] == "unknown" and data["reason"] == "support_height_mismatch"


def test_up_and_support_points_are_required(table):
    no_up = _check(_view(table, up=None)).data
    assert no_up["status"] == "unknown" and no_up["reason"] == "missing_up_direction"
    empty = _check(_view(table, valid=np.zeros(table.shape[:2], dtype=bool))).data
    assert (
        empty["status"] == "unknown"
        and empty["reason"] == "insufficient_support_points"
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"step": 2}, "different camera or step"),
        ({"world_frame_id": "other"}, "Expected world frame"),
        ({"clearance_m": 0.002}, "smaller than clearance_m"),
    ],
)
def test_rejects_mismatched_source_or_ambiguous_clearance(table, arguments, message):
    result = _check(_view(table), **arguments)
    assert result.is_error and message in result.data["error"]


@pytest.mark.parametrize("missing", [0, np.nan, np.inf])
def test_missing_depth_inside_a_covered_cell_is_still_unknown(missing):
    depth = np.full((81, 81), 0.5)
    depth[42:44, 44:46] = missing
    with np.errstate(invalid="ignore"):
        view = GeometryObservation(
            camera="overhead",
            step=3,
            world_frame_id="table",
            rgb=np.zeros((81, 81, 3), np.uint8),
            depth_m=depth,
            intrinsics=np.array([[100, 0, 40], [0, 100, 40], [0, 0, 1]]),
            camera_to_world=np.array(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 1],
                    [0, 0, 0, 1],
                ]
            ),
            up_direction=[0, 0, 1],
        )
    data = _check(view).data
    assert data["support_coverage"] == 1
    assert data["status"] == "unknown"
    assert data["reason"] == "unobserved_support_pixels"
    np.testing.assert_array_equal(view.depth_m, depth)


def test_dense_world_points_without_calibration_cannot_confirm_visibility(table):
    view = GeometryObservation(
        camera="overhead",
        step=3,
        world_frame_id="table",
        rgb=np.zeros(table.shape, dtype=np.uint8),
        world_xyz=table,
        up_direction=[0, 0, 1],
    )
    data = _check(view).data
    assert data["support_coverage"] == 1
    assert data["status"] == "unknown"
    assert data["reason"] == "missing_camera_calibration"


def test_footprint_crossing_frame_edge_stays_unknown_with_all_grid_cells_filled():
    depth = np.full((81, 60), 0.5)
    view = GeometryObservation(
        "overhead",
        3,
        "table",
        np.zeros((*depth.shape, 3), np.uint8),
        depth_m=depth,
        intrinsics=np.array([[100, 0, 40], [0, 100, 40], [0, 0, 1]]),
        camera_to_world=np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 1],
                [0, 0, 0, 1],
            ]
        ),
        up_direction=[0, 0, 1],
    )
    data = _check(view).data
    # The right image boundary projects to x=19.5*0.5/100=0.0975 m.
    assert data["support_coverage"] == 1
    assert data["status"] == "unknown"
    assert data["reason"] == "footprint_outside_camera_view"


def test_small_depth_noise_remains_supported(table):
    rows, cols = np.indices(table.shape[:2])
    table[..., 2] += np.where((rows + cols) % 2 == 0, 0.0005, -0.0005)
    view = _view(table)
    data = _check(view).data
    assert data["status"] == "free" and data["obstacle_points"] == 0
    assert data == _check(view).data
    np.testing.assert_array_equal(view.world_xyz, table)


def test_foreground_outside_footprint_can_occlude_the_clearance():
    depth = np.full((81, 81), 0.5)
    depth[40, 16] = 0.14
    view = GeometryObservation(
        "overhead",
        3,
        "table",
        np.zeros((*depth.shape, 3), np.uint8),
        depth_m=depth,
        intrinsics=np.array([[100, 0, 100], [0, 100, 40], [0, 0, 1]]),
        camera_to_world=np.array(
            [
                [1, 0, 0, 0.3],
                [0, -1, 0, 0],
                [0, 0, -1, 1],
                [0, 0, 0, 1],
            ]
        ),
        up_direction=[0, 0, 1],
    )
    # At this pixel the table hit is x=-0.12, the observed foreground is
    # x=0.1824 and the ray crosses the top of the clearance at x=-0.0528.
    data = _check(view).data
    assert data["support_coverage"] == 1 and data["obstacle_points"] == 0
    assert data["status"] == "unknown"
    assert data["reason"] == "unobserved_clearance_pixels"


def test_camera_inside_clearance_cannot_confirm_free_space():
    depth = np.full((81, 81), 0.05)
    view = GeometryObservation(
        "overhead",
        3,
        "table",
        np.zeros((*depth.shape, 3), np.uint8),
        depth_m=depth,
        intrinsics=np.array([[10, 0, 40], [0, 10, 40], [0, 0, 1]]),
        camera_to_world=np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0.55],
                [0, 0, 0, 1],
            ]
        ),
        up_direction=[0, 0, 1],
    )
    data = _check(view).data
    assert data["status"] == "unknown" and data["reason"] == "camera_below_clearance"
