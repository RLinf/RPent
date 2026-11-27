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

"""Circle fitting and shared geometry helpers for calibrated RGB-D measurements."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FitOptions:
    """Algorithm gates, not a calibrated sensor-error model."""

    distance_threshold_m: float = 0.003
    _: KW_ONLY
    min_points: int = 30
    max_points: int = 2000
    ransac_iterations: int = 384
    min_inlier_ratio: float = 0.6
    min_circle_coverage_deg: float = 180.0
    max_relative_residual: float = 0.05

    def __post_init__(self) -> None:
        if not np.isfinite(self.distance_threshold_m) or self.distance_threshold_m <= 0:
            raise ValueError("geometry distance threshold must be positive metres")
        if self.min_points < 6 or self.max_points < self.min_points:
            raise ValueError("invalid geometry point limits")
        if not 0 < self.min_inlier_ratio <= 1 or self.ransac_iterations < 1:
            raise ValueError("invalid geometry consensus limits")
        if not 0 < self.min_circle_coverage_deg <= 360:
            raise ValueError("invalid circle coverage gate")
        if (
            not np.isfinite(self.max_relative_residual)
            or not 0 < self.max_relative_residual < 1
        ):
            raise ValueError("invalid relative shape-residual gate")


@dataclass(kw_only=True)
class ShapeFit:
    """Keep computed parameters separate from their quality acceptance."""

    accepted: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def warn(self, reason: str) -> None:
        """Retain an estimate while marking an unmet quality requirement."""
        self.accepted = False
        self.reason = self.reason or reason
        if reason not in self.warnings:
            self.warnings.append(reason)


def _points(value: np.ndarray) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point cloud must have shape [N,3] in metres")
    return points


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    first = np.cross(normal, reference)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def _coverage(xy: np.ndarray, min_points: int) -> float:
    angles = np.sort(np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    # Allow two sample intervals for uneven spacing along a measured arc.
    # Larger gaps between separate arcs contribute no observed perimeter.
    max_gap = 2 * (2 * np.pi / min_points)
    if np.all(gaps <= max_gap):
        return 360.0
    return float(np.degrees(gaps[gaps <= max_gap].sum()))


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
    return center, cross / np.linalg.norm(cross), float(np.linalg.norm(a - center))


def _circle_residual(
    points: np.ndarray, center: np.ndarray, normal: np.ndarray, radius: float
) -> np.ndarray:
    delta = points - center
    height = delta @ normal
    radial = np.sqrt(np.maximum(0, np.einsum("ij,ij->i", delta, delta) - height**2))
    return np.hypot(height, radial - radius)


def fit_circle(
    points: np.ndarray,
    options: FitOptions | None = None,
    *,
    inward_half_steps: np.ndarray | None = None,
) -> ShapeFit:
    """Fit a circular boundary, optionally bracketed by inward pixel transitions."""
    from scipy.optimize import least_squares

    options = options or FitOptions()
    points = _points(points)
    if inward_half_steps is not None:
        inward_half_steps = _points(inward_half_steps)
        if inward_half_steps.shape != points.shape:
            raise ValueError("each circle sample needs one inward half-step")
    result = ShapeFit()
    if (
        len(points) < 3
        or not np.isfinite(points).all()
        or (inward_half_steps is not None and not np.isfinite(inward_half_steps).all())
    ):
        result.reason = "insufficient_finite_points"
        return result
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), dtype=bool)
    best_error = np.inf
    span = np.linalg.norm(np.ptp(points, axis=0))
    radius_limit = 2 * span
    for _ in range(options.ransac_iterations):
        candidate = _circle_from_three(
            points[rng.choice(len(points), 3, replace=False)]
        )
        if candidate is None:
            continue
        center, normal, radius = candidate
        if not options.distance_threshold_m < radius < radius_limit:
            continue
        residual = _circle_residual(points, center, normal, radius)
        inliers = residual <= options.distance_threshold_m
        error = float(np.median(residual[inliers])) if inliers.any() else np.inf
        rank = (inliers.sum(), -error)
        if rank > (best.sum(), -best_error):
            best, best_error = inliers, error
            if best.all():
                # Refinement depends only on the consensus set.
                break
    if best.sum() < 3:
        result.reason = "insufficient_circle_consensus"
        return result
    for _ in range(2):
        origin = points[best].mean(axis=0)
        _, _, axes = np.linalg.svd(points[best] - origin, full_matrices=False)
        normal = axes[-1]
        basis = axes[:2].T
        xy = (points[best] - origin) @ basis
        linear = np.column_stack((2 * xy, np.ones(len(xy))))
        if np.linalg.matrix_rank(linear) < 3:
            result.reason = "degenerate_circle_support"
            return result
        coefficients = np.linalg.lstsq(linear, (xy**2).sum(axis=1), rcond=None)[0]
        initial_radius = np.sqrt(
            max(1e-12, coefficients[2] + coefficients[:2] @ coefficients[:2])
        )
        projected = (points - origin) @ basis
        lower = np.r_[projected.min(axis=0) - radius_limit, np.finfo(float).eps]
        upper = np.r_[projected.max(axis=0) + radius_limit, radius_limit]
        # Short arcs admit arbitrarily large circles. Refinement must preserve
        # the observed-extent limit already used for RANSAC hypotheses.
        initial = np.clip(
            np.r_[coefficients[:2], initial_radius],
            np.nextafter(lower, upper),
            np.nextafter(upper, lower),
        )
        fit = least_squares(
            lambda p: np.linalg.norm(xy - p[:2], axis=1) - p[2],
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=options.distance_threshold_m,
            max_nfev=150,
        )
        center = origin + basis @ fit.x[:2]
        radius = float(fit.x[2])
        residual = _circle_residual(points, center, normal, radius)
        best = residual <= options.distance_threshold_m
        if best.sum() < 3:
            result.reason = "insufficient_circle_consensus"
            return result
    xy = (points[best] - center) @ basis
    coverage = _coverage(xy, options.min_points)
    result.quality = _support_quality(residual, best, options)
    result.quality["angular_coverage_deg"] = coverage
    relative_residual = result.quality["residual_p95_m"] / max(
        radius, options.distance_threshold_m
    )
    result.quality["relative_surface_residual_p95"] = relative_residual
    if inward_half_steps is not None:
        # The boundary lies between the last exterior ray and the first
        # interior ray. A midpoint can miss a small circle by half a pixel.
        step = inward_half_steps[best] @ basis
        inside = np.linalg.norm(xy + step, axis=1)
        outside = np.linalg.norm(xy - step, axis=1)
        miss = np.maximum(0.0, np.maximum(inside - radius, radius - outside))
        residual = np.hypot(miss, (points[best] - center) @ normal)
        relative_residual = float(np.quantile(residual, 0.95)) / radius
        result.quality["relative_pixel_interval_residual_p95"] = relative_residual
    at_extent_limit = bool(
        np.any(fit.active_mask) or radius >= radius_limit * (1 - 1e-5)
    )
    result.quality.update(
        radius_limit_m=float(radius_limit), refinement_at_extent_limit=at_extent_limit
    )
    if not fit.success or not np.isfinite(fit.x).all() or radius <= 0:
        result.reason = "unstable_circle_fit"
        return result
    result.accepted = True
    result.parameters = {
        "center": center.tolist(),
        "normal": (normal if normal[np.argmax(abs(normal))] >= 0 else -normal).tolist(),
        "radius": radius,
    }
    if best.sum() < options.min_points:
        result.warn("insufficient_circle_consensus")
    if best.mean() < options.min_inlier_ratio:
        result.warn("low_circle_inlier_ratio")
    if coverage < options.min_circle_coverage_deg:
        result.warn("insufficient_circle_angular_coverage")
    if relative_residual > options.max_relative_residual:
        result.warn("noncircular_boundary_residual")
    if radius <= options.distance_threshold_m or at_extent_limit:
        result.warn("unstable_circle_fit")
    return result


def ray_plane_offsets(
    points: np.ndarray, camera: np.ndarray, center: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    """Return offsets from the center where the viewing rays meet the plane."""
    rays = points - camera
    denominator = rays @ normal
    scale = np.divide(
        (center - camera) @ normal,
        denominator,
        out=np.zeros_like(denominator),
        where=abs(denominator) > 1e-12,
    )
    offset = (camera - center) + rays * scale[..., None]
    return np.where(scale[..., None] > 0, offset, np.nan)


def ray_plane_radius(
    points: np.ndarray, camera: np.ndarray, center: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    """Measure radius at each viewing ray's forward intersection with a plane."""
    radius = np.linalg.norm(ray_plane_offsets(points, camera, center, normal), axis=-1)
    return np.where(np.isfinite(radius), radius, np.inf)


def opening_covers_center(
    below: np.ndarray, valid: np.ndarray, offsets_xy: np.ndarray
) -> bool:
    """Check whether a region bounded by measured deeper rays encloses the center."""
    from scipy.ndimage import binary_propagation

    if len(offsets_xy) < 4 or min(below.shape) < 2:
        return False
    opening = below[:-1, :-1] & below[1:, :-1] & below[1:, 1:] & below[:-1, 1:]
    if not opening.any():
        return False
    if not valid.all():
        # Keep a dropout inside an observed opening only when deeper rays
        # separate it from every measured surface and from the image border.
        surface = valid & ~below
        exposed = (
            surface[:-1, :-1] | surface[1:, :-1] | surface[1:, 1:] | surface[:-1, 1:]
        )
        exposed[[0, -1]] |= ~opening[[0, -1]]
        exposed[:, [0, -1]] |= ~opening[:, [0, -1]]
        opening = ~binary_propagation(exposed, mask=~opening)

    # Cross the projected perimeter with a half-line from the requested center.
    # Its edges join measured rays even when the enclosed depths have dropouts.
    xy = np.full((*below.shape, 2), np.nan)
    xy[below] = offsets_xy
    horizontal = np.diff(opening, axis=0, prepend=False, append=False)
    vertical = np.diff(opening, axis=1, prepend=False, append=False)
    a = np.concatenate((xy[:, :-1][horizontal], xy[:-1][vertical]))
    b = np.concatenate((xy[:, 1:][horizontal], xy[1:][vertical]))
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    if ((cross == 0) & ((a * b).sum(axis=1) <= 0)).any():
        return True
    straddles = (a[:, 1] > 0) != (b[:, 1] > 0)
    crosses_right = np.where(b[:, 1] > a[:, 1], cross > 0, cross < 0)
    return bool(np.count_nonzero(straddles & crosses_right) % 2)


def observed_silhouette(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Select material pixels adjacent to an in-frame, measured non-object pixel."""
    from scipy.ndimage import binary_dilation

    return mask & valid & binary_dilation(~mask & valid)


def outer_boundary(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Select material along the measured exterior, excluding interior holes."""
    from scipy.ndimage import binary_fill_holes

    return mask & observed_silhouette(binary_fill_holes(mask), valid)
