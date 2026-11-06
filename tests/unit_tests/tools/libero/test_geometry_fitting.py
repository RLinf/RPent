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

"""Analytic geometric references are independent of the fitting implementation."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.libero.geometry import (
    FitOptions,
    ShapeFit,
    clean_part_cloud,
    fit_axisymmetric,
    fit_circle,
    fit_part,
    fit_plane,
    geometry_fields,
)

ROTATION = Rotation.from_euler("xyz", [22, -31, 17], degrees=True).as_matrix()
TRANSLATION = np.array([0.4, -0.2, 0.9])


def ring(degrees=300, count=300):
    theta = np.linspace(0, np.radians(degrees), count)
    xyz = np.column_stack((0.05 * np.cos(theta), 0.05 * np.sin(theta), np.zeros(count)))
    return xyz @ ROTATION.T + TRANSLATION


def body(slope=0, degrees=220):
    theta, axial = np.meshgrid(
        np.linspace(0, np.radians(degrees), 32), np.linspace(-0.06, 0.06, 13)
    )
    radius = 0.045 + slope * axial
    xyz = np.column_stack(
        (
            (radius * np.cos(theta)).ravel(),
            (radius * np.sin(theta)).ravel(),
            axial.ravel(),
        )
    )
    return xyz @ ROTATION.T + TRANSLATION


def test_cleanup_removes_table_and_disconnected_object_without_inventing_points():
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    rim = np.column_stack(
        (0.05 * np.cos(theta), 0.05 * np.sin(theta), np.full(360, 0.12))
    )
    xx, yy = np.meshgrid(np.linspace(-0.1, 0.1, 20), np.linspace(-0.1, 0.1, 20))
    table = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    other = rim + [0.3, 0.3, 0]
    points = np.concatenate((rim, table, other, [[np.nan, 0, 0]]))
    cleaned = clean_part_cloud(points, part="rim", anchor=rim[0], context_points=table)
    assert cleaned.quality["background_plane"]["n_removed"] >= len(table)
    assert 200 < len(cleaned.points) <= len(rim)
    np.testing.assert_array_equal(cleaned.points, points[cleaned.indices])
    np.testing.assert_allclose(cleaned.points[:, 2], 0.12)
    assert np.max(cleaned.points[:, 0]) < 0.06


def test_plane_target_keeps_its_own_plane_and_world_origin():
    xx, yy = np.meshgrid(np.arange(10) * 0.004, np.arange(10) * 0.004)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    cleaned = clean_part_cloud(
        points, part="plane", anchor=points[0], context_points=points
    )
    assert any(np.all(cleaned.points == [0, 0, 0], axis=1))
    assert "background_plane" not in cleaned.quality
    assert fit_plane(cleaned.points).accepted


def test_no_cluster_at_visual_anchor_is_rejected():
    cleaned = clean_part_cloud(ring(), part="rim", anchor=np.array([10, 10, 10]))
    assert len(cleaned.points) == 0
    assert cleaned.quality["rejection"] == "no_sufficient_target_connected_component"


def test_inner_contour_can_omit_anchor_without_disabling_background_cleanup():
    theta = np.linspace(np.radians(-110), np.radians(110), 360)
    rim = np.column_stack(
        (0.05 * np.cos(theta), 0.05 * np.sin(theta), np.full(360, 0.12))
    )
    anchor = np.array([-0.05, 0, 0.12])
    xx, yy = np.meshgrid(np.linspace(-0.1, 0.1, 20), np.linspace(-0.1, 0.1, 20))
    table = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    other = rim[::4] + [0.3, 0.3, 0]
    points = np.concatenate((rim, table, other))
    original = clean_part_cloud(points, part="rim", anchor=anchor, context_points=table)
    assert len(original.points) == 0
    cleaned = clean_part_cloud(
        points,
        part="rim",
        anchor=anchor,
        context_points=table,
        require_anchor_connection=False,
    )
    assert cleaned.quality["background_plane"]["n_removed"] >= len(table)
    assert cleaned.quality["component_selection"] == "largest_component"
    np.testing.assert_array_equal(cleaned.points, points[cleaned.indices])
    assert np.max(cleaned.points[:, 0]) < 0.06
    np.testing.assert_allclose(cleaned.points[:, 2], 0.12)
    fit = fit_circle(cleaned.points, anchor=anchor)
    assert fit.accepted, fit.summary()
    np.testing.assert_allclose(fit.parameters["center"], [0, 0, 0.12], atol=1e-6)


def test_nearly_collinear_context_does_not_establish_a_background_plane():
    theta = np.linspace(0, 2 * np.pi, 200)
    points = np.column_stack(
        (0.05 * np.cos(theta), 0.05 * np.sin(theta), np.full(200, 0.12))
    )
    rng = np.random.default_rng(11)
    context = np.column_stack((np.linspace(-0.1, 0.1, 100), np.zeros((100, 2))))
    context += rng.normal(0, 0.00001, context.shape)
    cleaned = clean_part_cloud(
        points, part="rim", anchor=points[0], context_points=context
    )
    assert "background_plane" not in cleaned.quality
    assert len(cleaned.points) >= 150


def test_circle_recovers_tilted_boundary_with_noise_and_outliers():
    rng = np.random.default_rng(2)
    points = ring() + rng.normal(0, 0.0003, (300, 3))
    outliers = TRANSLATION + rng.uniform(-0.12, 0.12, (70, 3))
    fit = fit_circle(np.concatenate((points, outliers)), anchor=ring()[0])
    assert fit.accepted, fit.summary()
    np.testing.assert_allclose(fit.parameters["center"], TRANSLATION, atol=0.001)
    assert abs(fit.parameters["radius"] - 0.05) < 0.001
    assert abs(np.dot(fit.parameters["normal"], ROTATION[:, 2])) > 0.999
    assert fit.quality["angular_coverage_deg"] > 290
    assert fit.quality["calibrated_error_m"] is None


def test_short_arc_is_rejected_despite_zero_residual():
    fit = fit_circle(ring(degrees=55), anchor=ring(degrees=55)[0])
    assert not fit.accepted and fit.reason == "insufficient_circle_angular_coverage"
    assert fit.parameters == {}
    assert fit.quality["residual_p95_m"] < 1e-8


def test_filled_disk_is_not_a_circular_boundary():
    rng = np.random.default_rng(3)
    xy = rng.uniform(-0.05, 0.05, (800, 2))
    xy = xy[np.linalg.norm(xy, axis=1) < 0.05]
    fit = fit_circle(np.column_stack((xy, np.zeros(len(xy)))))
    assert not fit.accepted


def test_small_square_boundary_is_not_accepted_under_a_coarse_absolute_threshold():
    u = np.linspace(-0.01, 0.01, 70)
    points = np.concatenate(
        [
            np.column_stack((u, np.full(70, 0.01), np.zeros(70))),
            np.column_stack((u, np.full(70, -0.01), np.zeros(70))),
            np.column_stack((np.full(70, 0.01), u, np.zeros(70))),
            np.column_stack((np.full(70, -0.01), u, np.zeros(70))),
        ]
    )
    fit = fit_circle(points)
    assert not fit.accepted and fit.reason == "noncircular_boundary_residual"
    assert fit.quality["residual_p95_m"] < 0.003


def test_small_true_circle_still_passes_the_relative_shape_gate():
    points = (ring() - TRANSLATION) * 0.2 + TRANSLATION
    fit = fit_circle(points)
    assert fit.accepted, fit.summary()
    assert fit.parameters["radius"] == pytest.approx(0.01, abs=1e-6)


def test_ransac_plane_recovers_visible_patch_with_background():
    rng = np.random.default_rng(4)
    xx, yy = np.meshgrid(np.linspace(-0.1, 0.1, 20), np.linspace(-0.05, 0.05, 16))
    original = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    points = original @ ROTATION.T + TRANSLATION
    points += rng.normal(0, 0.0003, points.shape)
    points = np.concatenate((points, TRANSLATION + rng.uniform(-0.2, 0.2, (70, 3))))
    fit = fit_plane(points, camera_origin=TRANSLATION + ROTATION[:, 2])
    assert fit.accepted, fit.summary()
    assert np.dot(fit.parameters["normal"], ROTATION[:, 2]) > 0.999
    assert (
        abs(np.dot(np.array(fit.parameters["point"]) - TRANSLATION, ROTATION[:, 2]))
        < 0.001
    )
    assert "width" not in fit.parameters


def test_collinear_samples_do_not_determine_a_plane_or_circle():
    points = np.column_stack((np.linspace(0, 0.1, 100), np.zeros((100, 2))))
    assert not fit_plane(points).accepted
    assert not fit_circle(points).accepted


def test_cylinder_axis_and_radius_with_handle_outliers():
    rng = np.random.default_rng(5)
    points = body() + rng.normal(0, 0.0002, (416, 3))
    handle = np.column_stack(
        (
            np.full(35, 0.085),
            np.linspace(-0.015, 0.015, 35),
            np.linspace(-0.035, 0.035, 35),
        )
    )
    points = np.concatenate((points, handle @ ROTATION.T + TRANSLATION))
    fit = fit_axisymmetric(points)
    assert fit.accepted and fit.model == "cylinder", fit.summary()
    assert abs(np.dot(fit.parameters["axis_direction"], ROTATION[:, 2])) > 0.995
    assert abs(fit.parameters["radius_at_visible_midpoint"] - 0.045) < 0.002
    axis_point = np.array(fit.parameters["axis_point"])
    assert np.linalg.norm(np.cross(axis_point - TRANSLATION, ROTATION[:, 2])) < 0.002
    assert fit.parameters["height"] is None and fit.parameters["body_center"] is None
    assert fit.quality["n_inliers"] < len(points)


def test_frustum_is_selected_for_tapered_body():
    rng = np.random.default_rng(6)
    points = body(slope=0.2) + rng.normal(0, 0.0002, (416, 3))
    fit = fit_axisymmetric(points)
    assert fit.accepted and fit.model == "frustum", fit.summary()
    assert abs(np.dot(fit.parameters["axis_direction"], ROTATION[:, 2])) > 0.995
    assert abs(abs(fit.parameters["radius_slope"]) - 0.2) < 0.02
    assert abs(fit.parameters["radius_at_visible_midpoint"] - 0.045) < 0.002


def test_small_visible_body_arc_cannot_establish_axis_and_radius():
    fit = fit_axisymmetric(body(degrees=35))
    assert not fit.accepted, fit.summary()


def test_sphere_does_not_pass_as_a_well_supported_cup_body():
    rng = np.random.default_rng(10)
    vectors = rng.normal(size=(600, 3))
    points = (
        0.05 * vectors / np.linalg.norm(vectors, axis=1, keepdims=True) + TRANSLATION
    )
    fit = fit_axisymmetric(points)
    assert not fit.accepted, fit.summary()


@pytest.mark.parametrize("fitter", [fit_plane, fit_circle, fit_axisymmetric])
def test_invalid_or_empty_cloud_has_no_fabricated_geometry(fitter):
    for points in [np.empty((0, 3)), np.full((50, 3), np.nan)]:
        fit = fitter(points)
        assert not fit.accepted and fit.parameters == {}


def test_invalid_point_shape_is_an_input_error():
    with pytest.raises(ValueError, match="N,3"):
        fit_circle(np.ones((20, 2)))
    with pytest.raises(ValueError):
        FitOptions(distance_threshold_m=0)


def test_full_height_requires_two_consistent_observed_end_boundaries():
    rim = fit_circle(ring() + 0.06 * ROTATION[:, 2])
    bottom = fit_part(ring() - 0.06 * ROTATION[:, 2], "bottom")
    fields, _ = geometry_fields({"rim": rim}, ["rim_center", "height", "body_center"])
    assert fields["rim_center"]["value"] is not None
    assert (
        fields["height"]["value"] is None
        and fields["body_center"]["evidence_status"] == "unknown"
    )
    fields, checks = geometry_fields(
        {"rim": rim, "bottom": bottom}, ["height", "body_center", "axis", "radius"]
    )
    assert fields["height"]["value"] == pytest.approx(0.12, abs=0.001)
    np.testing.assert_allclose(fields["body_center"]["value"], TRANSLATION, atol=0.001)
    np.testing.assert_allclose(
        fields["axis"]["value"]["direction"], ROTATION[:, 2], atol=0.001
    )
    assert fields["radius"]["value"] == pytest.approx(0.05, abs=0.001)
    assert fields["height"]["evidence_status"] == "inferred"
    assert checks[0]["consistent"] is True


def test_different_object_part_cannot_be_composed_into_a_cup():
    rim = fit_circle(ring())
    body_fit = ShapeFit(
        "cylinder",
        accepted=True,
        parameters={"axis_point": [3, 3, 3], "axis_direction": [0, 0, 1]},
    )
    fields, checks = geometry_fields(
        {"rim": rim, "body": body_fit}, ["rim_center", "axis"]
    )
    assert fields["rim_center"]["value"] is None and fields["axis"]["value"] is None
    assert checks[0]["consistent"] is False


def test_same_circle_cannot_be_both_rim_and_bottom():
    circle = fit_circle(ring())
    fields, checks = geometry_fields(
        {"rim": circle, "bottom": circle}, ["rim_center", "bottom_center", "height"]
    )
    assert all(item["value"] is None for item in fields.values())
    assert checks[0]["reason"] == "end_boundaries_not_distinct_at_fit_threshold"


def test_end_boundaries_cannot_cut_through_the_observed_body():
    rim = fit_circle(ring() + 0.02 * ROTATION[:, 2])
    bottom = fit_circle(ring() - 0.02 * ROTATION[:, 2])
    body_fit = ShapeFit(
        "cylinder",
        accepted=True,
        parameters={
            "axis_point": TRANSLATION.tolist(),
            "axis_direction": ROTATION[:, 2].tolist(),
            "visible_axial_range": [-0.06, 0.06],
        },
    )
    fields, checks = geometry_fields(
        {"rim": rim, "bottom": bottom, "body": body_fit}, ["height", "axis"]
    )
    assert all(item["value"] is None for item in fields.values())
    assert any(
        item.get("test") == "end_boundaries_enclose_visible_body"
        and not item["consistent"]
        for item in checks
    )


def test_invalid_float_thresholds_are_rejected():
    for value in [float("nan"), float("inf"), -1]:
        with pytest.raises(ValueError):
            FitOptions(distance_threshold_m=value)
