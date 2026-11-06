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

"""Candidate centers retain uncertainty and never replace missing depth with zeros."""

import numpy as np

from tools.libero.candidate_geometry import estimate_reference_center
from tools.libero.geometry import ShapeFit, fit_circle
from tools.libero.reference_geometry import ReferenceFit


def test_short_arc_candidate_does_not_need_anchor_on_the_same_contour():
    theta = np.linspace(-0.9, 0.9, 160)
    points = np.column_stack(
        (0.1 + 0.04 * np.cos(theta), 0.2 + 0.04 * np.sin(theta), np.full(160, 0.8))
    )
    strict = fit_circle(points)
    assert strict.reason == "insufficient_circle_angular_coverage"
    result = estimate_reference_center(
        points,
        "inner_rim",
        points.reshape(10, 16, 3),
        np.ones((10, 16), dtype=bool),
        np.array([0.2, 0.2, 0.85]),
        None,
        ReferenceFit(strict, points, points, np.ones((10, 16), dtype=bool)),
    )
    np.testing.assert_allclose(result.center, [0.1, 0.2, 0.8], atol=1e-6)
    assert result.method == "relaxed_inner_edge_circle"
    assert result.summary()["strict_accepted"] is False
    assert result.quality["strict_rejection"] == strict.reason
    assert not strict.accepted


def test_partial_rectangle_estimate_is_labelled_without_certifying_hidden_edges():
    x, y = np.meshgrid(np.linspace(0, 0.06, 40), np.linspace(-0.02, 0.02, 30))
    world = np.stack((x, y, np.full(x.shape, 0.8)), axis=-1)
    points = world.reshape(-1, 3)
    mask = np.ones(x.shape, dtype=bool)
    result = estimate_reference_center(
        points,
        "rectangular_top",
        world,
        mask,
        points[0],
        None,
        ReferenceFit(
            ShapeFit("rectangle", reason="four_complete_physical_edges_not_observed"),
            points[:0],
            points[:0],
            ~mask,
        ),
    )
    np.testing.assert_allclose(result.center, [0.03, 0, 0.8], atol=0.001)
    assert result.quality["four_physical_edges_certified"] is False
    assert result.method == "visible_plane_minimum_area_rectangle"


def test_missing_depth_has_no_candidate_and_no_fabricated_origin():
    empty = np.empty((0, 3))
    mask = np.ones((4, 4), dtype=bool)
    result = estimate_reference_center(
        empty,
        "inner_rim",
        np.full((4, 4, 3), np.nan),
        mask,
        None,
        None,
        ReferenceFit(ShapeFit("circle"), empty, empty, mask),
    )
    assert result is None
