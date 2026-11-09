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

"""Deterministic, metre-based cleanup and bounded shape fitting for visible parts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

Part = Literal["rim", "body", "bottom", "plane"]


@dataclass(frozen=True)
class FitOptions:
    """Algorithm gates, not a calibrated sensor-error model."""

    distance_threshold_m: float = 0.003
    voxel_size_m: float = 0.001
    min_points: int = 30
    max_points: int = 2000
    ransac_iterations: int = 384
    min_inlier_ratio: float = 0.6
    min_circle_coverage_deg: float = 180.0
    min_body_coverage_deg: float = 120.0
    max_condition: float = 1e5
    max_relative_residual: float = 0.05

    def __post_init__(self) -> None:
        if not np.isfinite(
            [self.distance_threshold_m, self.voxel_size_m, self.max_condition]
        ).all() or not (0 < self.distance_threshold_m and 0 < self.voxel_size_m):
            raise ValueError("geometry thresholds must be positive metres")
        if self.min_points < 6 or self.max_points < self.min_points:
            raise ValueError("invalid geometry point limits")
        if not 0 < self.min_inlier_ratio <= 1 or self.ransac_iterations < 1:
            raise ValueError("invalid geometry consensus limits")
        if not (
            0 < self.min_circle_coverage_deg <= 360
            and 0 < self.min_body_coverage_deg <= 360
            and self.max_condition > 1
        ):
            raise ValueError("invalid geometry observability gates")
        if (
            not np.isfinite(self.max_relative_residual)
            or not 0 < self.max_relative_residual < 1
        ):
            raise ValueError("invalid relative shape-residual gate")


@dataclass
class CleanedCloud:
    """Selected observed samples and their indices in the input point array."""

    points: np.ndarray
    indices: np.ndarray
    quality: dict[str, Any]


@dataclass
class ShapeFit:
    """Fitted parameters with observable support and explicit rejected results."""

    model: str
    accepted: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    inlier_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    reason: str | None = None
    assumptions: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Serialize small fit metadata; keep inlier arrays in artifacts."""
        return {
            "model": self.model,
            "accepted": self.accepted,
            "parameters": self.parameters if self.accepted else {},
            "quality": self.quality,
            "reason": self.reason,
            "assumptions": self.assumptions,
        }


def _points(value: np.ndarray) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point cloud must have shape [N,3] in metres")
    return points


def _direction(vector: np.ndarray, towards: np.ndarray | None = None) -> np.ndarray:
    vector = vector / np.linalg.norm(vector)
    index = int(np.argmax(np.abs(vector)))
    if (towards is not None and vector @ towards < 0) or (
        towards is None and vector[index] < 0
    ):
        vector = -vector
    return vector


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    first = np.cross(normal, reference)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def _coverage(xy: np.ndarray) -> float:
    angles = np.sort(np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    return float(np.degrees(2 * np.pi - gaps.max()))


def _support_quality(
    residual: np.ndarray, inliers: np.ndarray, options: FitOptions
) -> dict[str, Any]:
    absolute = np.abs(residual[inliers])
    return {
        "n_points": len(residual),
        "n_inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()) if len(inliers) else 0.0,
        "residual_median_m": float(np.median(absolute)) if len(absolute) else None,
        "residual_p95_m": float(np.quantile(absolute, 0.95)) if len(absolute) else None,
        "distance_threshold_m": options.distance_threshold_m,
        "calibrated_error_m": None,
    }


def _plane_consensus(
    points: np.ndarray, options: FitOptions, *, normal_hint: np.ndarray | None = None
) -> np.ndarray:
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(options.ransac_iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        if normal_hint is not None and abs(normal @ normal_hint) < np.cos(
            np.radians(15)
        ):
            continue
        inliers = np.abs((points - a) @ normal) <= options.distance_threshold_m
        if inliers.sum() > best.sum():
            best = inliers
    return best


def fit_plane(
    points: np.ndarray,
    options: FitOptions | None = None,
    *,
    camera_origin: np.ndarray | None = None,
) -> ShapeFit:
    """Fit a visible planar patch using RANSAC and SVD; no full-face dimensions."""
    options = options or FitOptions()
    points = _points(points)
    result = ShapeFit("plane", assumptions=["one dominant planar visible patch"])
    if len(points) < options.min_points or not np.isfinite(points).all():
        result.reason = "insufficient_finite_points"
        return result
    inliers = _plane_consensus(points, options)
    if inliers.sum() < options.min_points:
        result.reason = "insufficient_plane_consensus"
        return result
    for _ in range(2):
        origin = points[inliers].mean(axis=0)
        _, singular, axes = np.linalg.svd(points[inliers] - origin, full_matrices=False)
        normal = axes[-1]
        residual = (points - origin) @ normal
        inliers = np.abs(residual) <= options.distance_threshold_m
        if inliers.sum() < options.min_points:
            result.reason = "insufficient_plane_consensus"
            return result
    result.quality = _support_quality(residual, inliers, options)
    # Collinear samples cannot determine a plane, even with zero residual.
    spread = singular / np.sqrt(inliers.sum())
    result.quality["in_plane_spread_m"] = spread[:2].tolist()
    if inliers.mean() < options.min_inlier_ratio:
        result.reason = "low_plane_inlier_ratio"
    elif (
        spread[1] < 2 * options.distance_threshold_m
        or spread[1] / max(spread[0], 1e-12) < 0.03
    ):
        result.reason = "degenerate_plane_support"
    else:
        normal = _direction(
            normal, None if camera_origin is None else camera_origin - origin
        )
        result.accepted = True
        result.parameters = {
            "point": origin.tolist(),
            "normal": normal.tolist(),
            "offset": -float(origin @ normal),
            "normal_sign": "toward_camera"
            if camera_origin is not None
            else "canonical_undirected",
        }
        result.inlier_indices = np.flatnonzero(inliers)
    return result


def clean_part_cloud(
    points: np.ndarray,
    *,
    part: Part,
    anchor: np.ndarray | None = None,
    context_points: np.ndarray | None = None,
    options: FitOptions | None = None,
    require_anchor_connection: bool = True,
    up_direction: np.ndarray | None = None,
) -> CleanedCloud:
    """Keep target-connected observed samples and reject conservative outliers.

    Context planes perpendicular to an explicitly supplied up direction may be
    removed for body/rim only, when separated from the visual anchor. The target's own plane or bottom
    is never removed by this step. All returned points come from the input.
    Resolved contours may omit the visual material point; disabling its
    connection requirement selects the largest component while retaining the
    anchor for background-plane separation.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    options = options or FitOptions()
    points = _points(points)
    if up_direction is not None:
        up_direction = np.asarray(up_direction, dtype=float)
        if (
            up_direction.shape != (3,)
            or not np.isfinite(up_direction).all()
            or np.linalg.norm(up_direction) == 0
        ):
            raise ValueError("up_direction must be a finite nonzero 3-vector")
        up_direction = up_direction / np.linalg.norm(up_direction)
    indices = np.flatnonzero(np.isfinite(points).all(axis=1))
    quality: dict[str, Any] = {
        "n_input": len(points),
        "n_finite": len(indices),
        "options": asdict(options),
    }
    if not len(indices):
        return CleanedCloud(points[indices], indices, quality)
    _, unique = np.unique(
        np.floor(points[indices] / options.voxel_size_m).astype(np.int64),
        axis=0,
        return_index=True,
    )
    indices = indices[np.sort(unique)]
    if len(indices) > options.max_points:
        indices = indices[
            np.random.default_rng(0).choice(
                len(indices), options.max_points, replace=False
            )
        ]
        indices.sort()
    quality["n_voxel_sampled"] = len(indices)
    if (
        part in ("rim", "body")
        and context_points is not None
        and anchor is not None
        and up_direction is not None
    ):
        context = _points(context_points)
        context = context[np.isfinite(context).all(axis=1)]
        if len(context) > options.max_points:
            context = context[
                np.random.default_rng(0).choice(
                    len(context), options.max_points, replace=False
                )
            ]
        if len(context) >= options.min_points:
            support = _plane_consensus(context, options, normal_hint=up_direction)
            if support.sum() >= options.min_points and support.mean() >= 0.5:
                origin = context[support].mean(axis=0)
                _, singular, axes = np.linalg.svd(
                    context[support] - origin, full_matrices=False
                )
                normal = axes[-1]
                distance = abs(float((np.asarray(anchor) - origin) @ normal))
                spread = singular / np.sqrt(support.sum())
                observable = (
                    spread[1] > 2 * options.distance_threshold_m
                    and spread[1] / max(spread[0], 1e-12) > 0.03
                )
                if (
                    observable
                    and abs(normal @ up_direction) >= np.cos(np.radians(15))
                    and distance > 3 * options.distance_threshold_m
                ):
                    keep = (
                        np.abs((points[indices] - origin) @ normal)
                        > 1.5 * options.distance_threshold_m
                    )
                    quality["background_plane"] = {
                        "point": origin.tolist(),
                        "normal": normal.tolist(),
                        "n_removed": int((~keep).sum()),
                        "context_inlier_ratio": float(support.mean()),
                        "in_plane_spread_m": spread[:2].tolist(),
                    }
                    indices = indices[keep]
    quality["n_after_background"] = len(indices)
    if len(indices) >= options.min_points:
        samples = points[indices]
        tree = cKDTree(samples)
        distances, _ = tree.query(samples, k=min(8, len(samples)))
        spacing = float(np.median(distances[:, 1]))
        connection_radius = max(4 * options.distance_threshold_m, 3 * spacing)
        pairs = tree.query_pairs(connection_radius, output_type="ndarray")
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
            shape=(len(samples), len(samples)),
        )
        _, labels = connected_components(graph, directed=False)
        counts = np.bincount(labels)
        if anchor is not None and require_anchor_connection:
            distance, seed = tree.query(np.asarray(anchor, dtype=float))
            label = labels[seed] if distance <= connection_radius else -1
        else:
            label = int(counts.argmax())
        quality["component_selection"] = (
            "visual_anchor"
            if anchor is not None and require_anchor_connection
            else "largest_component"
        )
        if label < 0 or counts[label] < options.min_points:
            quality["rejection"] = "no_sufficient_target_connected_component"
            indices = indices[:0]
        else:
            indices = indices[labels == label]
        quality["connection_radius_m"] = connection_radius
    quality["n_target_component"] = len(indices)
    if len(indices) >= options.min_points:
        distance, _ = cKDTree(points[indices]).query(
            points[indices], k=min(12, len(indices))
        )
        mean_distance = distance[:, 1:].mean(axis=1)
        median = np.median(mean_distance)
        mad = np.median(np.abs(mean_distance - median))
        limit = max(3 * median, median + 4.5 * 1.4826 * mad)
        indices = indices[mean_distance <= limit]
        quality["outlier_neighbor_limit_m"] = float(limit)
    quality["n_cleaned"] = len(indices)
    return CleanedCloud(points[indices], indices, quality)


def _circle_from_three(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    a, b, c = points
    u, v = b - a, c - a
    cross = np.cross(u, v)
    denominator = 2 * (cross @ cross)
    if denominator < 1e-18:
        return None
    center = (
        a + ((u @ u) * np.cross(v, cross) + (v @ v) * np.cross(cross, u)) / denominator
    )
    return center, _direction(cross), float(np.linalg.norm(a - center))


def _circle_residual(
    points: np.ndarray, center: np.ndarray, normal: np.ndarray, radius: float
) -> np.ndarray:
    delta = points - center
    height = delta @ normal
    radial = np.sqrt(np.maximum(0, np.einsum("ij,ij->i", delta, delta) - height**2))
    return np.hypot(height, radial - radius)


def _anchor_error(
    anchor: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    boundary: bool,
) -> float:
    delta = np.asarray(anchor) - center
    height = float(delta @ normal)
    radial = np.linalg.norm(delta - height * normal)
    return float(
        np.hypot(height, radial - radius if boundary else max(0, radial - radius))
    )


def fit_circle(
    points: np.ndarray,
    options: FitOptions | None = None,
    *,
    anchor: np.ndarray | None = None,
    camera_origin: np.ndarray | None = None,
    anchor_on_boundary: bool = True,
) -> ShapeFit:
    """Fit an observed 3-D circular boundary; reject short arcs and nonplanarity."""
    from scipy.optimize import least_squares

    options = options or FitOptions()
    points = _points(points)
    result = ShapeFit("circle", assumptions=["selected part is one circular boundary"])
    if len(points) < options.min_points or not np.isfinite(points).all():
        result.reason = "insufficient_finite_points"
        return result
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), dtype=bool)
    best_error = np.inf
    span = np.linalg.norm(np.ptp(points, axis=0))
    for _ in range(options.ransac_iterations):
        candidate = _circle_from_three(
            points[rng.choice(len(points), 3, replace=False)]
        )
        if candidate is None:
            continue
        center, normal, radius = candidate
        if not options.distance_threshold_m < radius < 2 * span:
            continue
        if (
            anchor is not None
            and _anchor_error(anchor, center, normal, radius, anchor_on_boundary)
            > 3 * options.distance_threshold_m
        ):
            continue
        residual = _circle_residual(points, center, normal, radius)
        inliers = residual <= options.distance_threshold_m
        error = float(np.median(residual[inliers])) if inliers.any() else np.inf
        if (inliers.sum(), -error) > (best.sum(), -best_error):
            best, best_error = inliers, error
    if best.sum() < options.min_points:
        result.reason = "insufficient_circle_consensus"
        return result
    for _ in range(2):
        origin = points[best].mean(axis=0)
        _, _, axes = np.linalg.svd(points[best] - origin, full_matrices=False)
        normal = axes[-1]
        basis = axes[:2].T
        xy = (points[best] - origin) @ basis
        linear = np.column_stack((2 * xy, np.ones(len(xy))))
        coefficients = np.linalg.lstsq(linear, (xy**2).sum(axis=1), rcond=None)[0]
        initial_radius = np.sqrt(
            max(1e-12, coefficients[2] + coefficients[:2] @ coefficients[:2])
        )
        fit = least_squares(
            lambda p: np.linalg.norm(xy - p[:2], axis=1) - p[2],
            np.r_[coefficients[:2], initial_radius],
            loss="soft_l1",
            f_scale=options.distance_threshold_m,
            max_nfev=150,
        )
        center = origin + basis @ fit.x[:2]
        radius = float(fit.x[2])
        residual = _circle_residual(points, center, normal, radius)
        best = residual <= options.distance_threshold_m
        if best.sum() < options.min_points:
            result.reason = "insufficient_circle_consensus"
            return result
    xy = (points[best] - center) @ basis
    coverage = _coverage(xy)
    result.quality = _support_quality(residual, best, options)
    result.quality["angular_coverage_deg"] = coverage
    relative_residual = result.quality["residual_p95_m"] / max(
        radius, options.distance_threshold_m
    )
    result.quality["relative_surface_residual_p95"] = relative_residual
    if best.mean() < options.min_inlier_ratio:
        result.reason = "low_circle_inlier_ratio"
    elif coverage < options.min_circle_coverage_deg:
        result.reason = "insufficient_circle_angular_coverage"
    elif relative_residual > options.max_relative_residual:
        result.reason = "noncircular_boundary_residual"
    elif (
        not fit.success or radius <= options.distance_threshold_m or radius >= 2 * span
    ):
        result.reason = "unstable_circle_fit"
    elif (
        anchor is not None
        and _anchor_error(anchor, center, normal, radius, anchor_on_boundary)
        > 3 * options.distance_threshold_m
    ):
        result.reason = "circle_inconsistent_with_part_anchor"
    else:
        normal = _direction(
            normal, None if camera_origin is None else camera_origin - center
        )
        result.accepted = True
        result.parameters = {
            "center": center.tolist(),
            "normal": normal.tolist(),
            "radius": radius,
            "normal_sign": "toward_camera"
            if camera_origin is not None
            else "canonical_undirected",
        }
        result.inlier_indices = np.flatnonzero(best)
    return result


def _body_axes(points: np.ndarray) -> list[np.ndarray]:
    from scipy.spatial import cKDTree

    _, _, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    _, neighbors = cKDTree(points).query(points, k=min(16, len(points)))
    local = points[neighbors]
    local -= local.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", local, local)
    _, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    _, normal_axes = np.linalg.eigh(normals.T @ normals)
    return [normal_axes[:, 0], *axes]


def fit_axisymmetric(points: np.ndarray, options: FitOptions | None = None) -> ShapeFit:
    """Compare robust circular-cylinder and circular-frustum lateral fits.

    Axial limits describe visible inlier samples, never complete object height.
    Axis sign is a convention. Complete endpoints require separate observations.
    """
    from scipy.optimize import least_squares

    options = options or FitOptions()
    points = _points(points)
    failed = ShapeFit("axisymmetric", reason="insufficient_finite_points")
    if len(points) < options.min_points or not np.isfinite(points).all():
        return failed
    origin = np.median(points, axis=0)
    span = float(np.linalg.norm(np.ptp(points, axis=0)))
    if span < 4 * options.distance_threshold_m:
        failed.reason = "insufficient_spatial_extent"
        return failed
    candidates = []
    for initial_axis in _body_axes(points):
        initial_axis = _direction(initial_axis)
        u, v = _basis(initial_axis)
        xy = (points - origin) @ np.column_stack((u, v))
        coefficients = np.linalg.lstsq(
            np.column_stack((2 * xy, np.ones(len(xy)))), (xy**2).sum(axis=1), rcond=None
        )[0]
        offsets = np.clip(coefficients[:2], -0.8 * span, 0.8 * span)
        radius = np.clip(
            np.median(np.linalg.norm(xy - offsets, axis=1)),
            2 * options.distance_threshold_m,
            span,
        )
        for conical in (False, True):

            def decode(p):
                center = origin + p[0] * u + p[1] * v
                axis = initial_axis + p[2] * u + p[3] * v
                axis /= np.linalg.norm(axis)
                return center, axis, p[4], p[5] if conical else 0.0

            def residual_function(p):
                center, axis, radius, slope = decode(p)
                delta = points - center
                axial = delta @ axis
                radial = np.linalg.norm(delta - axial[:, None] * axis, axis=1)
                return (radial - radius - slope * axial) / np.sqrt(1 + slope * slope)

            initial = [*offsets, 0, 0, radius] + ([0.0] if conical else [])
            lower = [-span, -span, -2, -2, options.distance_threshold_m] + (
                [-0.75] if conical else []
            )
            upper = [span, span, 2, 2, 2 * span] + ([0.75] if conical else [])
            fit = least_squares(
                residual_function,
                initial,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=options.distance_threshold_m,
                max_nfev=200,
                x_scale="jac",
            )
            residual = residual_function(fit.x)
            inliers = np.abs(residual) <= options.distance_threshold_m
            if inliers.sum() < options.min_points:
                continue
            center, axis, radius, slope = decode(fit.x)
            if axis[np.argmax(np.abs(axis))] < 0:
                axis, slope = -axis, -slope
            axial = (points[inliers] - center) @ axis
            low, high = np.quantile(axial, [0.01, 0.99])
            midpoint = (low + high) / 2
            axis_point = center + midpoint * axis
            reference_radius = radius + slope * midpoint
            bu, bv = _basis(axis)
            coverage = _coverage(
                (points[inliers] - axis_point) @ np.column_stack((bu, bv))
            )
            jac = fit.jac[inliers]
            column_norm = np.linalg.norm(jac, axis=0)
            singular = np.linalg.svd(
                jac / np.maximum(column_norm, 1e-12), compute_uv=False
            )
            condition = float(singular[0] / max(singular[-1], 1e-12))
            quality = _support_quality(residual, inliers, options)
            relative_residual = quality["residual_p95_m"] / max(
                reference_radius, options.distance_threshold_m
            )
            quality.update(
                angular_coverage_deg=coverage,
                normalized_jacobian_condition=condition,
                relative_surface_residual_p95=relative_residual,
            )
            model = ShapeFit(
                "frustum" if conical else "cylinder",
                quality=quality,
                assumptions=[
                    "one circular axisymmetric lateral surface",
                    "handle/interior/background are outliers",
                    "axis sign is conventional; axial endpoints are visible sample limits",
                ],
            )
            if not fit.success:
                model.reason = "axisymmetric_solver_did_not_converge"
            elif inliers.mean() < options.min_inlier_ratio:
                model.reason = "low_axisymmetric_inlier_ratio"
            elif coverage < options.min_body_coverage_deg:
                model.reason = "insufficient_body_angular_coverage"
            elif high - low < 4 * options.distance_threshold_m:
                model.reason = "insufficient_visible_axial_span"
            elif condition > options.max_condition:
                model.reason = "ill_conditioned_axisymmetric_fit"
            elif relative_residual > options.max_relative_residual:
                model.reason = (
                    "axisymmetric_surface_residual_too_large_relative_to_radius"
                )
            elif (
                min(radius + slope * low, radius + slope * high)
                <= options.distance_threshold_m
            ):
                model.reason = "nonpositive_frustum_radius"
            else:
                model.accepted = True
                model.inlier_indices = np.flatnonzero(inliers)
                model.parameters = {
                    "axis_point": axis_point.tolist(),
                    "axis_direction": axis.tolist(),
                    "radius_at_visible_midpoint": float(reference_radius),
                    "radius_slope": float(slope),
                    "visible_axial_range": [
                        float(low - midpoint),
                        float(high - midpoint),
                    ],
                    "visible_axial_extent": float(high - low),
                    "visible_end_radii": [
                        float(radius + slope * low),
                        float(radius + slope * high),
                    ],
                    "height": None,
                    "body_center": None,
                }
            # Compare both models on the same complete sample set with bounded
            # outlier cost. The extra frustum slope must earn its complexity.
            score = len(points) * np.log(
                max(
                    float(
                        np.mean(
                            np.minimum(
                                (residual / options.distance_threshold_m) ** 2, 25
                            )
                        )
                    ),
                    1e-12,
                )
            ) + len(initial) * np.log(len(points))
            quality["model_selection_score"] = float(score)
            candidates.append(model)
    accepted = [model for model in candidates if model.accepted]
    if not accepted:
        if candidates:
            failed = max(candidates, key=lambda item: item.quality["inlier_ratio"])
        return failed
    best_cylinder = min(
        (m for m in accepted if m.model == "cylinder"),
        key=lambda m: m.quality["model_selection_score"],
        default=None,
    )
    best_frustum = min(
        (m for m in accepted if m.model == "frustum"),
        key=lambda m: m.quality["model_selection_score"],
        default=None,
    )
    if best_frustum is not None and (
        best_cylinder is None
        or (
            best_frustum.quality["model_selection_score"] + 10
            < best_cylinder.quality["model_selection_score"]
            and abs(best_frustum.parameters["radius_slope"])
            * best_frustum.parameters["visible_axial_extent"]
            > 2 * options.distance_threshold_m
        )
    ):
        chosen = best_frustum
    else:
        chosen = best_cylinder or best_frustum
    chosen.quality["candidate_models"] = [
        {
            "model": m.model,
            "accepted": m.accepted,
            "reason": m.reason,
            "score": m.quality["model_selection_score"],
        }
        for m in candidates
    ]
    return chosen


def fit_part(
    points: np.ndarray,
    part: Part,
    *,
    options: FitOptions | None = None,
    anchor: np.ndarray | None = None,
    camera_origin: np.ndarray | None = None,
) -> ShapeFit:
    """Choose the deterministic model for a semantically selected visible part."""
    if part in ("rim", "bottom"):
        return fit_circle(
            points,
            options,
            anchor=anchor,
            camera_origin=camera_origin,
            anchor_on_boundary=part == "rim",
        )
    if part == "body":
        return fit_axisymmetric(points, options)
    if part == "plane":
        return fit_plane(points, options, camera_origin=camera_origin)
    raise ValueError(f"unsupported geometric part: {part}")


def part_candidate_mask(
    mask: np.ndarray,
    world: np.ndarray,
    image: np.ndarray,
    part: Part,
    *,
    camera_transform: np.ndarray | None = None,
    options: FitOptions | None = None,
) -> np.ndarray:
    """Use observed mask/RGB/depth edges for circular boundaries, not cup interiors."""
    from scipy.ndimage import binary_erosion

    options = options or FitOptions()
    if part not in ("rim", "bottom"):
        return mask.copy()
    boundary = mask & ~binary_erosion(mask)
    if part == "bottom":
        return boundary
    # A whole-cup SAM mask may include its inner floor. RGB edges and the
    # foreground side of depth jumps supply candidates; 3-D circle consensus
    # must still establish the requested boundary near the visual rim anchor.
    gray = np.asarray(image[..., :3], dtype=float).mean(axis=2) / 255
    gradients = np.gradient(gray)
    strength = np.hypot(*gradients)
    cutoff = max(0.04, float(np.quantile(strength[mask], 0.8)))
    edges = strength >= cutoff
    if camera_transform is not None:
        z = (world - camera_transform[:3, 3]) @ camera_transform[:3, 2]
        for axis in (0, 1):
            for shift in (-1, 1):
                neighbor = np.roll(z, shift, axis=axis)
                edges |= np.isfinite(neighbor) & (
                    neighbor - z > 2 * options.distance_threshold_m
                )
    return mask & (boundary | edges)


def geometry_fields(
    fits: dict[str, ShapeFit], requested: list[str], options: FitOptions | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Derive explicitly defined quantities, requiring observed ends for full height."""
    options = options or FitOptions()
    checks: list[dict[str, Any]] = []
    bad_parts: set[str] = set()
    body = fits.get("body")
    if body is not None and body.accepted:
        axis = np.asarray(body.parameters["axis_direction"])
        axis_point = np.asarray(body.parameters["axis_point"])
        for part in ("rim", "bottom"):
            circle = fits.get(part)
            if circle is None or not circle.accepted:
                continue
            center = np.asarray(circle.parameters["center"])
            offset = float(np.linalg.norm(np.cross(center - axis_point, axis)))
            alignment = abs(float(np.dot(circle.parameters["normal"], axis)))
            consistent = offset <= max(
                3 * options.distance_threshold_m, 0.15 * circle.parameters["radius"]
            ) and alignment >= np.cos(np.radians(15))
            checks.append(
                {
                    "parts": ["body", part],
                    "consistent": bool(consistent),
                    "axis_offset_m": offset,
                    "normal_alignment": alignment,
                }
            )
            if not consistent:
                bad_parts.update(("body", part))
    ends = None
    rim, bottom = fits.get("rim"), fits.get("bottom")
    if rim is not None and bottom is not None and rim.accepted and bottom.accepted:
        top = np.asarray(rim.parameters["center"])
        base = np.asarray(bottom.parameters["center"])
        height = float(np.linalg.norm(top - base))
        if height > 4 * options.distance_threshold_m:
            direction = (top - base) / height
            consistent = all(
                abs(np.dot(circle.parameters["normal"], direction))
                >= np.cos(np.radians(15))
                for circle in (rim, bottom)
            )
            if body is not None and body.accepted:
                end_positions = sorted(
                    (
                        float((top - axis_point) @ axis),
                        float((base - axis_point) @ axis),
                    )
                )
                visible_low, visible_high = body.parameters["visible_axial_range"]
                encloses_body = (
                    end_positions[0] <= visible_low + 3 * options.distance_threshold_m
                    and end_positions[1]
                    >= visible_high - 3 * options.distance_threshold_m
                )
                checks.append(
                    {
                        "parts": ["rim", "body", "bottom"],
                        "consistent": bool(encloses_body),
                        "test": "end_boundaries_enclose_visible_body",
                    }
                )
                if not encloses_body:
                    bad_parts.update(("rim", "body", "bottom"))
            checks.append({"parts": ["rim", "bottom"], "consistent": bool(consistent)})
            if consistent and not bad_parts.intersection(("rim", "body", "bottom")):
                ends = {
                    "body_center": ((top + base) / 2).tolist(),
                    "height": height,
                    "axis": {
                        "point": ((top + base) / 2).tolist(),
                        "direction": direction.tolist(),
                        "sign": "bottom_to_rim",
                    },
                }
            elif not consistent:
                bad_parts.update(("rim", "bottom"))
        else:
            checks.append(
                {
                    "parts": ["rim", "bottom"],
                    "consistent": False,
                    "reason": "end_boundaries_not_distinct_at_fit_threshold",
                    "separation_m": height,
                }
            )
            bad_parts.update(("rim", "bottom"))
    definitions = {
        "rim_center": ("rim", "center_of_fitted_visible_rim_circle", "m"),
        "rim_radius": ("rim", "radius_of_fitted_visible_rim_circle", "m"),
        "radius": ("rim", "radius_of_fitted_visible_rim_circle", "m"),
        "bottom_center": (
            "bottom",
            "center_of_fitted_visible_bottom_boundary_circle",
            "m",
        ),
        "axis": ("body", "axis_of_fitted_axisymmetric_body", "m_and_unit_direction"),
        "body_radii": ("body", "radii_at_visible_axial_sample_limits", "m"),
        "body_center": (
            "ends",
            "midpoint_of_observed_rim_and_bottom_circle_centers",
            "m",
        ),
        "height": ("ends", "distance_between_observed_coaxial_end_centers", "m"),
        "plane": (
            "plane",
            "fitted_visible_plane_not_complete_object_face",
            "m_and_unit_normal",
        ),
        "plane_normal": ("plane", "normal_of_fitted_visible_plane", "unit_direction"),
    }
    fields = {}
    for name in requested:
        part, definition, unit = definitions[name]
        value = None
        source_parts = ["rim", "bottom"] if part == "ends" else [part]
        assumptions = []
        fit = fits.get(part)
        reason = fit.reason if fit is not None else "required_part_not_observed"
        if part == "ends" and ends is not None:
            value = ends[name]
            assumptions = [
                "rim and bottom are circular coaxial end boundaries",
                "body center is axial midpoint, not mass or volume centroid",
            ]
        elif name == "axis" and ends is not None:
            value, source_parts = ends["axis"], ["rim", "bottom"]
            definition = "line_from_observed_bottom_center_to_rim_center"
            assumptions = ["rim and bottom are coaxial end boundaries"]
        elif fit is not None and fit.accepted and part not in bad_parts:
            parameters = fit.parameters
            if name in ("rim_center", "bottom_center"):
                value = parameters["center"]
            elif name in ("rim_radius", "radius"):
                value = parameters["radius"]
            elif name == "axis":
                value = {
                    "point": parameters["axis_point"],
                    "direction": parameters["axis_direction"],
                    "sign": "canonical_undirected",
                }
            elif name == "body_radii":
                value = {
                    "radii": parameters["visible_end_radii"],
                    "axial_range": parameters["visible_axial_range"],
                    "axis_point": parameters["axis_point"],
                    "axis_direction": parameters["axis_direction"],
                }
            elif name == "plane":
                value = parameters
            elif name == "plane_normal":
                value = parameters["normal"]
            assumptions = fit.assumptions
        if bad_parts.intersection(source_parts):
            value, reason = None, "inconsistent_part_fits"
        if part == "ends" and value is None:
            reason = "both_consistent_end_boundaries_required"
        fields[name] = {
            "value": value,
            "definition": definition,
            "unit": unit,
            "evidence_status": "inferred" if value is not None else "unknown",
            "source_parts": source_parts,
            "assumptions": assumptions,
            "reason": None if value is not None else reason,
            "quality": {p: fits[p].quality for p in source_parts if p in fits},
        }
    return fields, checks
