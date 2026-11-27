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


"""Locate an upright round body's axis on its measured support plane."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, find_objects

from toolkits.geometry.geometry import (
    FitOptions,
    ShapeFit,
    _basis,
    fit_circle,
    outer_boundary,
    ray_plane_offsets,
    ray_plane_radius,
)
from toolkits.geometry.rgbd_parts import _circle, _plane
from toolkits.observation import GeometryObservation

_UPRIGHT_COSINE = float(np.cos(np.deg2rad(15)))


def _support_plane(obs, mask, cloud, tol):
    up = obs.up_direction
    low, high = np.quantile(cloud @ up, [0.01, 0.99])
    box = find_objects(mask, max_label=1)[0]
    pad = max(5, min(32, min(s.stop - s.start for s in box) // 3))
    roi = tuple(slice(max(0, s.start - pad), s.stop + pad) for s in box)
    region, valid, world = mask[roi], obs.valid_mask[roi], obs.world_xyz[roi]
    # Search around the low part of the object in world coordinates. This also
    # works for a rolled camera and excludes surfaces behind the upper silhouette.
    near_foot = (
        region & valid & (abs(world @ up - low) <= max(2 * tol, 0.06 * (high - low)))
    )
    if not near_foot.any():
        return None
    distance, nearest = distance_transform_edt(~near_foot, return_indices=True)
    outside = valid & ~binary_dilation(region) & (distance <= pad)
    neighbors = world[outside]
    paired = world[tuple(nearest[:, outside])]
    close = np.linalg.norm(neighbors - paired, axis=1) <= max(
        4 * tol, 0.10 * (high - low)
    )
    plane = _measured_plane(neighbors[close], up, tol)
    if plane is None:
        return None
    origin, normal, support = plane
    below = neighbors[(neighbors - origin) @ normal < -2 * tol]
    return origin, normal, support, below


def _measured_plane(neighbors, up, tol):
    if len(neighbors) < 15:
        return None
    sample = neighbors[:: max(1, len(neighbors) // 1500)]
    plane = _plane(sample, up, tol / 2)
    if plane is None:
        return None
    origin, normal = plane
    if normal @ up < _UPRIGHT_COSINE:
        return None
    support = neighbors[abs((neighbors - origin) @ normal) <= tol / 2]
    if len(support) < max(15, 0.65 * len(neighbors)):
        return None
    # A line at a depth discontinuity cannot establish a support plane.
    spread = np.linalg.svd(support - support.mean(axis=0), compute_uv=False)
    if spread[1] < tol * np.sqrt(len(support)):
        return None
    return origin, normal, support


def _wall_section(points, options):
    if len(points) < max(12, options.min_points // 2):
        return None
    limit = min(750, options.max_points)
    if len(points) > limit:
        points = points[np.linspace(0, len(points) - 1, limit, dtype=int)]
    fit = _circle(
        points,
        np.array([0.0, 0.0, 1.0]),
        tolerance=options.distance_threshold_m * 2 / 3,
    )
    if fit is None:
        return None
    center, radius = fit
    if radius <= 2 * options.distance_threshold_m:
        return None
    delta = points[:, :2] - center[:2]
    residual = abs(np.linalg.norm(delta, axis=1) - radius)
    inliers = residual <= min(options.distance_threshold_m, 0.06 * radius)
    if inliers.sum() < max(12, options.min_inlier_ratio * len(points)):
        return None
    # A single view usually sees only the front of the body. Count populated
    # sectors so that a few handle points cannot turn a short arc into a broad one.
    angles = np.arctan2(delta[inliers, 1], delta[inliers, 0])
    bins = np.floor(angles * 12 / np.pi).astype(int) % 24
    occupied = np.bincount(bins, minlength=24) >= max(2, 0.015 * inliers.sum())
    coverage = int(occupied.sum()) * 15
    if coverage < 90:
        return None
    return points[inliers, 2].mean(), center[:2], radius, coverage


def _cylindrical_base(obs, mask, cloud, options):
    """Intersect a near-upright sidewall axis with its observed support plane."""
    tol = options.distance_threshold_m
    plane = _support_plane(obs, mask, cloud, tol)
    if plane is None:
        return ShapeFit(reason="base_support_plane_not_observed")
    origin, normal, support, below = plane
    basis = np.column_stack((*_basis(normal), normal))
    local = (cloud - origin) @ basis
    low, height = np.quantile(local[:, 2], [0.01, 0.99])
    if low < -tol:
        return ShapeFit(reason="base_support_plane_above_object")
    if height <= 6 * tol:
        return ShapeFit(reason="cylindrical_sidewall_not_observed")

    # Start above the rounded heel, and stop below the shoulder or mouth.
    levels = np.linspace(0.08, 0.48, 6) * height
    sections = []
    for lo, hi in zip(levels, levels[1:]):
        band = local[(local[:, 2] >= lo) & (local[:, 2] < hi)]
        sections.append(_wall_section(band, options))
    for start in range(len(sections) - 2):
        window = sections[start : start + 3]
        if any(section is None for section in window):
            continue
        heights, centers, radii, coverage = map(np.asarray, zip(*window))
        radius = np.median(radii)
        if np.ptp(radii) > min(0.10 * radius, tol + 0.05 * np.ptp(heights)):
            continue
        dh = heights - heights.mean()
        slope = (dh @ centers) / (dh @ dh)
        axis = basis @ np.r_[slope, 1.0]
        axis /= np.linalg.norm(axis)
        if min(axis @ obs.up_direction, axis @ normal) < _UPRIGHT_COSINE:
            continue
        base = centers.mean(axis=0) - slope * heights.mean()
        drift = np.linalg.norm(centers - base - heights[:, None] * slope, axis=1)
        if drift.max() > min(tol, 0.08 * radius):
            continue

        # Require measured material near the foot as well as adjacent support
        # near the inferred footprint. A hanging cylinder has only the latter.
        bottom = local[(local[:, 2] >= -tol / 2) & (local[:, 2] <= 2 * tol)]
        radial = np.linalg.norm(bottom[:, :2] - base, axis=1)
        near_wall = abs(radial - radius) <= max(2 * tol, 0.2 * radius)
        contact = (support - origin) @ basis[:, :2] - base
        support_radius = np.linalg.norm(contact, axis=1)
        near_support = support_radius <= radius + 3 * tol
        if near_wall.sum() < 5 or near_support.sum() < 10:
            return ShapeFit(reason="base_contact_not_observed")
        base_world = origin + basis[:, :2] @ base
        contact_occluded = (support_radius <= radius + tol).sum() < 5
        if contact_occluded:
            # Check where the deeper rays passed the foot, not where they hit the floor.
            crossing = ray_plane_radius(
                below, obs.camera_to_world[:3, 3], base_world, normal
            )
            if (
                np.count_nonzero(
                    (crossing >= radius - tol) & (crossing <= radius + tol)
                )
                >= 5
            ):
                return ShapeFit(reason="base_support_gap_observed")
        return ShapeFit(
            accepted=True,
            parameters={"center": base_world.tolist()},
            warnings=["base_contact_occluded"] if contact_occluded else [],
            quality={
                "center_source": "sidewall_axis_support_plane",
                "axis_world": axis.tolist(),
                "support_normal_world": normal.tolist(),
                "sidewall_radius_m": float(radius),
                "section_heights_m": heights.tolist(),
                "min_occupied_arc_deg": int(min(coverage)),
                "axis_deviation_m": float(drift.max()),
                "contact_samples": int(near_wall.sum()),
                "support_samples": int(near_support.sum()),
            },
        )
    return ShapeFit(reason="cylindrical_sidewall_not_observed")


def _azimuth_sectors(xy):
    return np.floor(np.arctan2(xy[:, 1], xy[:, 0]) * 6 / np.pi).astype(int) % 12


def _rim_base_on_support(obs, mask, cloud, center, axis, axes, radius, tol):
    """Intersect the centerline with a local support observed around the footprint."""
    box = find_objects(mask, max_label=1)[0]
    pad = max(6, min(40, min(s.stop - s.start for s in box) // 3))
    roi = tuple(slice(max(0, s.start - pad), s.stop + pad) for s in box)
    outside = obs.valid_mask[roi] & ~binary_dilation(mask[roi], iterations=2)
    neighbors = obs.world_xyz[roi][outside]
    xy = (neighbors - center) @ axes
    distance = np.linalg.norm(xy, axis=1) - radius
    # A bowl can hide its foot. Look around its actual footprint for support,
    # within a bounded vertical distance of the lowest measured material.
    low = np.quantile(cloud @ obs.up_direction, 0.02)
    height = neighbors @ obs.up_direction - low
    band = max(4 * tol, 0.25 * radius)
    near = (
        (abs(distance) <= band)
        & (height <= tol)
        & (height >= -max(4 * tol, 0.3 * radius))
    )
    plane = _measured_plane(neighbors[near], obs.up_direction, tol)
    if plane is None:
        return ShapeFit(reason="base_support_plane_not_observed")
    origin, normal, support = plane
    if min(axis @ normal, axis @ obs.up_direction) < _UPRIGHT_COSINE:
        return ShapeFit(reason="body_axis_not_upright")
    around = (support - center) @ axes
    if (np.bincount(_azimuth_sectors(around), minlength=12) >= 3).sum() < 6:
        return ShapeFit(reason="base_support_plane_not_observed")
    gap = float(np.quantile((cloud - origin) @ normal, 0.01))
    distance_to_plane = (center - origin) @ normal
    clearance = max(2 * tol, min(3 * tol, 0.15 * radius))
    if not -tol <= gap <= clearance or distance_to_plane <= tol:
        return ShapeFit(reason="base_contact_not_observed")
    base = center - axis * (distance_to_plane / (axis @ normal))
    # A ring of deeper readings can show an opening even when a countertop
    # surrounds the bowl. Locate the rays at the support height, not the floor.
    world = obs.world_xyz[roi]
    below = obs.valid_mask[roi] & ~mask[roi] & ((world - origin) @ normal < -2 * tol)
    plane_axes = np.column_stack(_basis(normal))
    crossings = (
        ray_plane_offsets(world[below], obs.camera_to_world[:3, 3], base, normal)
        @ plane_axes
    )
    contact = (support - base) @ plane_axes
    nearest_support = np.full(12, np.inf)
    np.minimum.at(
        nearest_support, _azimuth_sectors(contact), np.linalg.norm(contact, axis=1)
    )
    radial = np.linalg.norm(crossings, axis=1)
    near = radial <= radius + band
    sectors = _azimuth_sectors(crossings[near])
    gap_sectors = sectors[radial[near] < nearest_support[sectors]]
    if (np.bincount(gap_sectors, minlength=12) >= 3).sum() >= 6:
        return ShapeFit(reason="base_support_gap_observed")
    return ShapeFit(
        accepted=True,
        parameters={"center": base.tolist()},
        quality={
            "axis_world": axis.tolist(),
            "support_normal_world": normal.tolist(),
            "support_samples": len(support),
            "observed_clearance_m": gap,
        },
        warnings=["base_contact_occluded"] if gap > tol else [],
    )


def _concentric_surface(cloud, center, axes, axis, radius, tol):
    """Check that the measured interior rises towards the rim around the same axis."""
    delta = cloud - center
    xy = delta @ axes
    radial = np.linalg.norm(xy, axis=1) / radius
    z = delta @ axis
    if (
        np.quantile(z, 0.99) > 2 * tol
        or np.quantile(radial, 0.98) > 1 + 2 * tol / radius
    ):
        return False
    sectors = np.floor(np.arctan2(xy[:, 1], xy[:, 0]) * 4 / np.pi).astype(int) % 8
    # Follow the surface outwards along the same directions. The near rim of a
    # deep bowl often hides half its interior, even with a well-measured rim.
    rings = [abs(radial - r) < 0.05 for r in (0.35, 0.65, 0.90)]
    profiles = []
    for k in range(8):
        samples = [z[ring & (sectors == k)] for ring in rings]
        if min(map(len, samples)) >= 3:
            profiles.append([np.median(band) for band in samples])
    if len(profiles) < 4:
        return False
    low, high = np.quantile(profiles, [0.1, 0.9], axis=0)
    levels = np.median(profiles, axis=0)
    return (
        np.all(high - low <= 2 * tol)
        and min(np.diff(levels)) >= -tol
        and levels[-1] - levels[0] > tol
    )


def fit_supported_base(
    obs: GeometryObservation,
    mask: np.ndarray,
    cloud: np.ndarray,
    options: FitOptions,
) -> ShapeFit:
    """Fit the base of an upright cylinder or a concentric bowl or plate."""
    cylinder = _cylindrical_base(obs, mask, cloud, options)
    if cylinder.accepted or cylinder.reason == "base_support_gap_observed":
        return cylinder
    edge = obs.world_xyz[outer_boundary(mask, obs.valid_mask)]
    if len(edge) > options.max_points:
        edge = edge[np.linspace(0, len(edge) - 1, options.max_points, dtype=int)]
    rim = fit_circle(edge, options)
    if not rim.accepted or rim.quality["angular_coverage_deg"] < 235:
        if cylinder.reason == "cylindrical_sidewall_not_observed":
            return ShapeFit(reason="round_body_axis_not_observed")
        return cylinder
    center = np.asarray(rim.parameters["center"])
    axis = np.asarray(rim.parameters["normal"])
    if axis @ obs.up_direction < 0:
        axis = -axis
    axes = np.column_stack(_basis(axis))
    radius = rim.parameters["radius"]
    tol = options.distance_threshold_m
    if not _concentric_surface(cloud, center, axes, axis, radius, tol):
        return ShapeFit(reason="round_body_profile_not_observed")
    fit = _rim_base_on_support(obs, mask, cloud, center, axis, axes, radius, tol)
    if fit.accepted:
        fit.quality.update(
            center_source="circular_rim_axis_support_plane",
            rim_radius_m=radius,
            rim_coverage_deg=rim.quality["angular_coverage_deg"],
        )
    return fit
