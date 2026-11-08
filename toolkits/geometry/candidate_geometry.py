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

"""Observation-only candidate centers for permissive error evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from toolkits.geometry.geometry import (
    FitOptions,
    _basis,
    fit_axisymmetric,
    fit_circle,
    fit_plane,
)
from toolkits.geometry.reference_geometry import (
    ReferenceFit,
    _outer_boundary,
    _plane_mask,
    fit_barrel_body,
)


@dataclass
class CenterEstimate:
    """A finite hypothesis, with no claim that strict geometry gates passed."""

    center: list[float]
    method: str
    points: np.ndarray
    quality: dict[str, Any]
    assumptions: list[str]

    def summary(self) -> dict[str, Any]:
        """Return JSON metadata while storing source points in a separate artifact."""
        return {
            "value": self.center,
            "method": self.method,
            "evaluation_only": True,
            "strict_accepted": False,
            "quality": self.quality,
            "assumptions": self.assumptions,
        }


def permissive_options(options: FitOptions | None = None) -> FitOptions:
    """Relax quality gates, retaining the metric residual threshold and sample cap."""
    return replace(
        options or FitOptions(),
        min_points=6,
        min_inlier_ratio=0.1,
        min_circle_coverage_deg=1.0,
        min_body_coverage_deg=1.0,
        max_condition=1e12,
        max_relative_residual=0.99,
    )


def _sample(points: np.ndarray, limit: int) -> np.ndarray:
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > limit:
        points = points[np.linspace(0, len(points) - 1, limit).astype(int)]
    return points


def _rectangle_center(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Minimum-area box center of visible planar support, without edge certification."""
    from scipy.spatial import ConvexHull, QhullError

    origin = points.mean(axis=0)
    basis = np.column_stack(_basis(normal))
    xy = (points - origin) @ basis
    try:
        hull = xy[ConvexHull(xy).vertices]
        vectors = np.roll(hull, -1, axis=0) - hull
        angles = np.unique(np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi / 2))
    except QhullError:
        angles = np.array([0.0])
    if len(angles) > 128:
        angles = angles[np.linspace(0, len(angles) - 1, 128).astype(int)]
    best = None
    for angle in angles:
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        projected = xy @ rotation
        low, high = np.quantile(projected, [0.002, 0.998], axis=0)
        area = float(np.prod(high - low))
        center = origin + basis @ rotation @ ((low + high) / 2)
        if best is None or area < best[0]:
            best = (area, center)
    return best[1]


def estimate_reference_center(
    points: np.ndarray,
    part: str,
    world: np.ndarray,
    mask: np.ndarray,
    anchor: np.ndarray | None,
    camera_origin: np.ndarray | None,
    reference: ReferenceFit,
    options: FitOptions | None = None,
) -> CenterEstimate | None:
    """Estimate a rejected reference from this frame alone, without requesting a view.

    Try a relaxed shape fit first. If boundary evidence or fitting is unavailable,
    use the visible face/edge extent and explicitly report that approximation.
    Finite masked depth is required; missing observations never become zeros.
    No simulator model, object pose, reference annotation or truth is an input.
    """
    options = permissive_options(options)
    cloud = _sample(np.asarray(points, dtype=float), options.max_points)
    source = "cleaned_part_points"
    if not len(cloud):
        cloud = _sample(world[mask].astype(float), options.max_points)
        source = "finite_segmented_points_after_empty_cleanup"
    if not len(cloud):
        return None
    quality = {
        "strict_rejection": reference.fit.reason,
        "strict_fit_quality": reference.fit.quality,
        "options": asdict(options),
        "input_source": source,
        "n_input_points": len(cloud),
        "calibrated_error_m": None,
    }

    def estimate(center, method, support, **extra):
        return CenterEstimate(
            np.asarray(center, dtype=float).tolist(),
            method,
            support,
            {**quality, **extra},
            [
                "candidate for error evaluation; strict reliability gates did not pass",
                "visible support may be incomplete or belong to a different contour",
            ],
        )

    if part == "inner_rim":
        # The material anchor may lie on a different contour. It still selected
        # the object, but does not veto a candidate circle in evaluation mode.
        circle = fit_circle(cloud, options, camera_origin=camera_origin)
        quality["relaxed_fit"] = circle.summary()
        if circle.accepted:
            return estimate(
                circle.parameters["center"], "relaxed_inner_edge_circle", cloud
            )
        return estimate(
            (cloud.min(axis=0) + cloud.max(axis=0)) / 2,
            "visible_inner_edge_extent_midpoint",
            cloud,
        )

    if part == "bottle_segment":
        body = fit_barrel_body(cloud, anchor, options)
        if not body.accepted:
            body = fit_axisymmetric(cloud, options)
        quality["relaxed_body_fit"] = body.summary()
        if body.accepted:
            # This point is the midpoint of the visible fitted lateral span.
            # It does not certify complete main-barrel endpoints.
            return estimate(
                body.parameters["axis_point"],
                "relaxed_axis_visible_span_midpoint",
                cloud[body.inlier_indices],
                endpoint_evidence="visible_samples_only; neck/base limits uncertified",
            )
        return estimate(
            (cloud.min(axis=0) + cloud.max(axis=0)) / 2,
            "visible_bottle_extent_midpoint",
            cloud,
        )

    plane = fit_plane(cloud, options, camera_origin=camera_origin)
    quality["relaxed_plane_fit"] = plane.summary()
    if plane.accepted:
        origin = np.asarray(plane.parameters["point"])
        normal = np.asarray(plane.parameters["normal"])
        support = cloud[plane.inlier_indices]
        face = _plane_mask(world, mask, cloud, plane, options)
    else:
        origin = cloud.mean(axis=0)
        if len(cloud) < 3:
            return estimate(origin, "finite_visible_samples_mean", cloud)
        _, _, axes = np.linalg.svd(cloud - origin, full_matrices=False)
        normal = axes[-1]
        support = cloud
        face = mask & np.isfinite(world).all(axis=2)
        quality["plane_fallback"] = (
            "PCA of visible samples without support certification"
        )
    if part == "rectangular_top":
        return estimate(
            _rectangle_center(support, normal),
            "visible_plane_minimum_area_rectangle",
            support,
            four_physical_edges_certified=False,
        )

    observed = _sample(reference.observed_points, options.max_points)
    boundary_source = "strict_physical_boundary"
    if len(observed) < options.min_points:
        observed = _sample(world[_outer_boundary(face)], options.max_points)
        boundary_source = "visible_plane_mask_contour_without_physical_edge_check"
    projected = observed - ((observed - origin) @ normal)[:, None] * normal
    circle = fit_circle(
        projected, options, camera_origin=camera_origin, anchor_on_boundary=False
    )
    quality.update(boundary_source=boundary_source, relaxed_circle_fit=circle.summary())
    if circle.accepted:
        return estimate(
            circle.parameters["center"], "relaxed_planar_boundary_circle", projected
        )
    projected_support = support - ((support - origin) @ normal)[:, None] * normal
    return estimate(
        _rectangle_center(projected_support, normal),
        "visible_planar_extent_midpoint",
        projected_support,
    )
