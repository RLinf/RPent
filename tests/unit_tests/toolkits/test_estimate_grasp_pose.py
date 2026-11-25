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

"""Offline geometry contracts for parallel-jaw grasp candidates."""

from types import SimpleNamespace

import numpy as np
import pytest

from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from toolkits.estimate_grasp_pose import estimate_grasp_pose
from toolkits.observation import GeometryObservation
from toolkits.segment import SegmentTool


def _rectangle(length=0.12, width=0.04, yaw_deg=37):
    x, y = np.meshgrid(
        np.linspace(-length / 2, length / 2, 61), np.linspace(-width / 2, width / 2, 41)
    )
    angle = np.radians(yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.stack(
        (0.2 + c * x - s * y, -0.1 + s * x + c * y, np.full_like(x, 0.45)), axis=-1
    )


def _record(
    tmp_path, points=None, *, up=(0, 0, 1), valid=None, mask=None, found=True, **depth
):
    state = EnvState(tmp_path)
    with state.record_step(state={}) as step:
        pass
    shape = points.shape[:2] if points is not None else depth["depth_m"].shape
    view = GeometryObservation(
        camera="overhead",
        step=step,
        world_frame_id="table",
        rgb=np.zeros((*shape, 3), dtype=np.uint8),
        world_xyz=points,
        valid_mask=valid,
        up_direction=up,
        **depth,
    )
    observations = SimpleNamespace(get_observation=lambda **_: view)
    if mask is None and found:
        mask = np.ones(shape, dtype=bool)
    segmenter = SimpleNamespace(
        segment=lambda *a, **kw: Sam3Result(found=found, mask=mask, score=0.8)
    )
    tool = SegmentTool(observations=observations, state=state, segmenter=segmenter)
    result = tool.segment(camera="overhead", prompt="rectangle").data
    return SimpleNamespace(
        state=state, observations=observations, view=view, segment=result
    )


def _estimate(source, **overrides):
    args = {
        "camera": "overhead",
        "segment_artifact": source.segment["segment_artifact"],
    }
    return estimate_grasp_pose(
        **(args | overrides), observations=source.observations, state=source.state
    )


def test_rotated_rectangle_returns_ranked_right_handed_grasps(tmp_path):
    scene = _record(tmp_path, _rectangle())
    artifacts = set(scene.state.get(scene.view.step).artifacts)
    result = _estimate(scene, top_k=4)
    data = result.data

    assert data["status"] == "estimated" and data["world_frame_id"] == "table"
    assert data["step"] == scene.view.step and data["units"] == "m"
    assert data["pca_yaw_ambiguous"] is False
    assert len(data["candidates"]) == 4
    best = data["candidates"][0]
    expected_minor = [-np.sin(np.radians(37)), np.cos(np.radians(37)), 0]
    assert abs(
        np.asarray(best["rotation_matrix"])[:, 0] @ expected_minor
    ) == pytest.approx(1)
    assert best["estimated_width_m"] == pytest.approx(0.04, abs=0.002)
    assert best["suggested_opening_m"] == pytest.approx(
        best["estimated_width_m"] + 0.004
    )
    assert best["score"] == pytest.approx(1)

    scores = []
    for candidate in data["candidates"]:
        assert candidate["center_xyz"] == pytest.approx([0.2, -0.1, 0.45])
        rotation = np.asarray(candidate["rotation_matrix"])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1)
        assert rotation[:, 2] == pytest.approx([0, 0, -1])
        scores.append(candidate["score"])
    assert scores == sorted(scores, reverse=True)
    assert scene.state.latest_step == scene.view.step and result.images == []
    assert scene.state.get(scene.view.step).artifacts == artifacts


def test_opening_limit_applies_to_span_plus_both_margins(tmp_path):
    scene = _record(tmp_path, _rectangle())
    rejected = _estimate(scene, max_opening_m=0.042).data
    assert rejected["status"] == "no_candidate"
    assert rejected["reason"] == "opening_limit" and rejected["candidates"] == []
    available = _estimate(scene, max_opening_m=0.042, jaw_margin_m=0).data
    assert available["status"] == "estimated"
    assert 0 < len(available["candidates"]) < 3
    assert all(item["suggested_opening_m"] <= 0.042 for item in available["candidates"])


def test_bounds_sampling_on_pinhole_depth_with_non_z_up(tmp_path):
    depth = np.full((81, 121), 0.45)
    scene = _record(
        tmp_path,
        up=(0, 1, 0),
        depth_m=depth,
        intrinsics=np.array([[450, 0, 60], [0, 450, 40], [0, 0, 1]]),
        camera_to_world=np.array(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 1],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        ),
    )
    result = _estimate(scene, top_k=1).data
    assert result["status"] == "estimated"
    assert result["n_valid"] == depth.size and 20 <= result["n_used"] <= 4096
    (candidate,) = result["candidates"]
    assert candidate["center_xyz"] == pytest.approx([0, 0.55, 0], abs=0.002)
    assert np.asarray(candidate["rotation_matrix"])[:, 2] == pytest.approx([0, -1, 0])


def test_square_uses_deterministic_samples_when_pca_yaw_is_ambiguous(tmp_path):
    scene = _record(tmp_path, _rectangle(length=0.06, width=0.06, yaw_deg=22))
    first = _estimate(scene, top_k=6).data
    assert first["status"] == "estimated" and first["pca_yaw_ambiguous"] is True
    assert first == _estimate(scene, top_k=6).data
    axes = [np.asarray(c["rotation_matrix"])[:, 0] for c in first["candidates"]]
    assert all(
        abs(axis @ other) < 0.999 for i, axis in enumerate(axes) for other in axes[:i]
    )


def test_ignores_pixels_outside_mask_invalid_depth_and_isolated_outliers(tmp_path):
    points = _rectangle()
    mask = np.ones(points.shape[:2], dtype=bool)
    mask[0, 0] = False
    valid = np.ones(points.shape[:2], dtype=bool)
    valid[0, 1] = False
    points[0, :3] = [40, 50, 60]
    scene = _record(tmp_path, points, mask=mask, valid=valid)
    data = _estimate(scene, top_k=1).data
    assert data["status"] == "estimated"
    assert data["n_valid"] == int((mask & valid).sum())
    assert data["n_used"] == data["n_valid"] - 1
    (candidate,) = data["candidates"]
    center = np.asarray(candidate["center_xyz"])
    assert center == pytest.approx([0.2, -0.1, 0.45], abs=0.002)
    closing = np.asarray(candidate["rotation_matrix"])[:, 0]
    radius = np.max(np.abs((points[mask & valid] - center) @ closing))
    assert radius + 0.002 <= candidate["suggested_opening_m"] / 2 + 1e-12
    assert _estimate(scene, max_opening_m=0.15).data["status"] == "no_candidate"


def test_explicit_approach_is_normalized_in_the_world_frame(tmp_path):
    scene = _record(tmp_path, _rectangle()[..., [0, 2, 1]], up=None)
    estimate = _estimate(scene, approach_xyz=[0, -10, 0], top_k=1).data
    rotation = np.asarray(estimate["candidates"][0]["rotation_matrix"])
    assert rotation[:, 2] == pytest.approx([0, -1, 0])
    assert _estimate(scene).data["reason"] == "missing_approach_direction"
    bad_axis = _estimate(scene, approach_xyz=[0, 0, 0])
    assert bad_axis.is_error and "nonzero approach_xyz" in bad_axis.data["error"]


@pytest.mark.parametrize(
    "condition, reason",
    [
        ("sparse", "insufficient_valid_points"),
        ("line", "degenerate_projected_cloud"),
        ("missing", "missing_segment_mask"),
    ],
)
def test_unusable_geometry_produces_no_candidates(tmp_path, condition, reason):
    points = _rectangle(yaw_deg=0)
    valid = None
    if condition == "sparse":
        valid = np.zeros(points.shape[:2], dtype=bool)
        valid.flat[:10] = True
    elif condition == "line":
        points[..., 1] = 0
    scene = _record(tmp_path, points, valid=valid, found=condition != "missing")
    data = _estimate(scene).data
    assert data["status"] == "need_observation"
    assert data["reason"] == reason and data["candidates"] == []


@pytest.mark.parametrize(
    "bad_record",
    [
        {"camera": "wrist"},
        {"source_step": 2},
        {"world_frame_id": "foreign"},
        {"image_shape": [1, 1]},
        {"mask_artifact": "foreign.npz"},
    ],
)
def test_refuses_to_use_mismatched_segment_records(tmp_path, bad_record):
    scene = _record(tmp_path, _rectangle())
    artifact = scene.segment["segment_artifact"]
    step = scene.view.step
    scene.state.save(
        artifact, scene.state.load(artifact, step=step) | bad_record, step=step
    )
    result = _estimate(scene)
    assert result.is_error and "candidates" not in result.data


def test_small_visible_tab_cannot_disappear_from_the_opening_limit(tmp_path):
    points = _rectangle(yaw_deg=0)
    # Two extra support extrema extend the observed 12 cm by 4 cm rectangle.
    # The hull contains a 12 cm by 11 cm rectangle, so every yaw needs >= 11 cm.
    points[-1, 1] = [0.14, -0.01, 0.45]
    points[-1, -2] = [0.26, -0.01, 0.45]
    # Keep the entire bottom edge of the rectangle observable.
    scene = _record(tmp_path, points)
    rejected = _estimate(scene, max_opening_m=0.06, jaw_margin_m=0).data
    assert (
        rejected["status"] == "no_candidate" and rejected["reason"] == "opening_limit"
    )
    data = _estimate(scene, top_k=1, jaw_margin_m=0).data
    (candidate,) = data["candidates"]
    assert candidate["estimated_width_m"] == pytest.approx(0.04, abs=0.002)
    assert candidate["suggested_opening_m"] == pytest.approx(0.11, abs=0.002)
    np.testing.assert_array_equal(scene.view.world_xyz, points)


def test_rigidly_rotated_cuboid_has_known_center_opening_and_approach(tmp_path):
    x, y, z = np.meshgrid(
        np.linspace(-0.06, 0.06, 31),
        np.linspace(-0.02, 0.02, 21),
        np.linspace(-0.015, 0.015, 5),
    )
    yaw, roll = np.radians([37, 30])
    c, s = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    u, v = c * x - s * y, s * x + c * y
    points = np.stack([u + 0.2, cr * v - sr * z - 0.1, sr * v + cr * z + 0.45], axis=-1)
    up = [0, -sr, cr]
    scene = _record(tmp_path, points.reshape(21, -1, 3), up=up)
    result = _estimate(scene, top_k=1).data
    assert result["status"] == "estimated"
    (best,) = result["candidates"]
    assert best["center_xyz"] == pytest.approx([0.2, -0.1, 0.45], abs=1e-12)
    assert best["suggested_opening_m"] == pytest.approx(0.044, abs=1e-12)
    rotation = np.asarray(best["rotation_matrix"])
    assert abs(rotation[:, 0] @ [-s, cr * c, sr * c]) == pytest.approx(1)
    assert rotation[:, 2] == pytest.approx([0, sr, -cr])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)


def test_uneven_sampling_centers_symmetric_jaws_on_observed_extrema(tmp_path):
    x = np.concatenate([np.linspace(-0.02, -0.012, 20), np.linspace(-0.012, 0.02, 6)])
    x, y = np.meshgrid(x, np.linspace(-0.08, 0.08, 21))
    points = np.stack((x, y, np.full_like(x, 0.4)), axis=-1)
    scene = _record(tmp_path, points)

    result = _estimate(scene, top_k=1, max_opening_m=0.05).data
    assert result["status"] == "estimated"
    (best,) = result["candidates"]
    closing = np.asarray(best["rotation_matrix"])[:, 0]
    np.testing.assert_allclose(np.abs(closing), [1, 0, 0], atol=1e-12)
    # The visible x limits are -20 mm and +20 mm regardless of pixel density.
    assert best["center_xyz"] == pytest.approx([0, 0, 0.4], abs=1e-12)
    assert best["suggested_opening_m"] == pytest.approx(0.044, abs=1e-12)
    assert _estimate(scene, max_opening_m=0.043).data["status"] == "no_candidate"


def test_all_yaws_enclose_rotated_partial_cloud_including_one_sided_tip(tmp_path):
    x, y = np.meshgrid(np.linspace(-0.02, 0.02, 21), np.linspace(-0.07, 0.07, 41))
    visible = (x < 0.005) | (y < 0.01)
    x[20, -1] = 0.042
    visible[20, -1] = True
    yaw = np.radians(31)
    c, s = np.cos(yaw), np.sin(yaw)
    points = np.stack(
        (0.35 + c * x - s * y, -0.2 + s * x + c * y, np.full_like(x, 0.6)), axis=-1
    )
    scene = _record(tmp_path, points, mask=visible)
    margin = 0.003
    limit = 0.16
    result = _estimate(scene, top_k=6, max_opening_m=limit, jaw_margin_m=margin).data

    assert result["status"] == "estimated"
    assert len(result["candidates"]) == 6
    axes = []
    for candidate in result["candidates"]:
        rotation = np.asarray(candidate["rotation_matrix"])
        closing = rotation[:, 0]
        center = np.asarray(candidate["center_xyz"])
        opening = candidate["suggested_opening_m"]
        projected = (points[visible] - center) @ closing
        assert np.max(np.abs(projected)) + margin <= opening / 2 + 1e-12
        assert np.min(projected) == pytest.approx(-np.max(projected), abs=1e-12)
        assert opening <= limit
        assert center[2] == pytest.approx(0.6, abs=1e-12)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1)
        assert all(abs(closing @ other) < 0.999 for other in axes)
        axes.append(closing)
    np.testing.assert_array_equal(scene.view.world_xyz, points)
