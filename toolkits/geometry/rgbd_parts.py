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

"""Fit physical features using a single segmented object's calibrated RGB-D samples."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.ndimage import binary_erosion, label
from scipy.optimize import least_squares

from toolkits.geometry.depth_foreground import local_depth_range
from toolkits.geometry.geometry import _basis


def _circle(points, normal, *, tolerance=0.0025):
    """RANSAC with equal azimuth weights; refit against the measured ridge."""
    if len(points) < 7:
        return None
    origin = np.median(points, axis=0)
    axes = np.stack(_basis(normal), axis=1)
    xy = (points - origin) @ axes
    rng = np.random.default_rng(19)
    samples = rng.integers(len(xy), size=(320, 3))
    a, b, c = xy[samples].transpose(1, 0, 2)
    ab, ac = b - a, c - a
    det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    ok = abs(det) > 1e-8
    if not ok.any():
        return None
    ab, ac, a, det = ab[ok], ac[ok], a[ok], det[ok]
    vb, vc = (ab * ab).sum(axis=1) / 2, (ac * ac).sum(axis=1) / 2
    centers = (
        a
        + np.stack(
            (vb * ac[:, 1] - vc * ab[:, 1], vc * ab[:, 0] - vb * ac[:, 0]), axis=1
        )
        / det[:, None]
    )
    radii = np.linalg.norm(a - centers, axis=1)
    span = max(np.ptp(xy, axis=0))
    residual = abs(
        np.linalg.norm(xy[None, :, :] - centers[:, None, :], axis=2) - radii[:, None]
    )
    support = np.maximum(0, 1 - (residual / tolerance) ** 2).sum(axis=1)
    support[(radii > span * 1.3) | (radii < span * 0.12)] = 0
    k = np.argmax(support)
    if support[k] < 5:
        return None
    active = residual[k] < tolerance * 1.75
    out = least_squares(
        lambda p: np.linalg.norm(xy[active] - p[:2], axis=1) - p[2],
        [*centers[k], radii[k]],
        loss="soft_l1",
        f_scale=0.0009,
    )
    if not out.success or not np.isfinite(out.x).all() or out.x[2] <= 0:
        return None
    return origin + axes @ out.x[:2], out.x[2]


def _plane(points, up, tol=0.002, *, min_up=0.52):
    if len(points) < 6:
        return None
    rng = np.random.default_rng(37)
    s = points[rng.integers(len(points), size=(180, 3))]
    normals = np.cross(s[:, 1] - s[:, 0], s[:, 2] - s[:, 0])
    length = np.linalg.norm(normals, axis=1)
    ok = length > 1e-8
    if not ok.any():
        return None
    s, normals = s[ok], normals[ok] / length[ok, None]
    offsets = (normals * s[:, 0]).sum(axis=1)
    sampled = points[:: max(1, len(points) // 1300)]
    residual = abs(sampled @ normals.T - offsets)
    counts = (residual < tol).sum(axis=0)
    counts[abs(normals @ up) < min_up] = 0
    k = np.argmax(counts)
    if counts[k] < 5:
        return None
    good = abs(points @ normals[k] - offsets[k]) < tol
    return _refit_plane(points[good], up)


def _refit_plane(support, up):
    origin = support.mean(axis=0)
    normal = np.linalg.svd(support - origin, full_matrices=False)[2][-1]
    if normal @ up < 0:
        normal = -normal
    return origin, normal


def rim(points, up, *, outer=False):
    """Estimate the upper rim plane and radius from the highest points in each sector."""
    normal = up.copy()
    center = np.median(points, axis=0)
    for _ in range(4):
        axes = np.stack(_basis(normal), axis=1)
        flat = (points - center) @ axes
        angles = np.arctan2(flat[:, 1], flat[:, 0])
        groups = np.minimum(71, ((angles + np.pi) * (72 / (2 * np.pi))).astype(int))
        height = (points - center) @ normal
        radial = np.linalg.norm(flat, axis=1)
        ridge = []
        for group in range(72):
            idx = np.flatnonzero(groups == group)
            if len(idx) < (1 if outer else 3):
                continue
            # Sample the selected radial edge within each sector's extreme band.
            top = np.quantile(height[idx], 0.98)
            band = idx[(height[idx] >= top - 0.0012) & (height[idx] <= top + 0.0018)]
            if not len(band):
                continue
            order = band[np.argsort(radial[band])]
            edge = order[-1] if outer else order[int(0.28 * (len(order) - 1))]
            ridge.append(points[edge])
        ridge = np.asarray(ridge)
        plane = _plane(ridge, up)
        if plane is None:
            return None
        p, normal = plane
        ridge = ridge[abs((ridge - p) @ normal) < 0.0035]
        circ = _circle(ridge, normal)
        if circ is None:
            return None
        center, radius = circ
        center -= normal * ((center - p) @ normal)
    return center, normal, radius


def face_candidates(
    points: np.ndarray, up: np.ndarray, cancel: Callable[[], None] = lambda: None
):
    """Return distinct upward-facing planes from highest to lowest."""
    faces = []
    remaining = points
    while len(remaining) >= 25:
        heights = remaining @ up
        best = None
        strength = 0.0
        # Height bands also expose faces on objects dominated by vertical walls.
        for level in (heights, -heights):
            for cutoff in np.unique(np.quantile(level, (0.0, 0.55, 0.79, 0.90))):
                cancel()
                plane = _plane(remaining[level >= cutoff], up, 0.0018, min_up=0.91)
                if plane is None:
                    continue
                origin, normal = plane
                residual = abs((remaining - origin) @ normal)
                inliers = residual < 0.0019
                count = int(inliers.sum())
                score = count * (normal @ up) ** 3
                if count >= 25 and normal @ up >= 0.91 and score > strength:
                    strength = score
                    best = inliers
        if best is None:
            break
        support = remaining[best]
        origin, normal = _refit_plane(support, up)
        faces.append((origin, normal, support))
        # A smaller face must compete only with points still unexplained.
        remaining = remaining[~best]
    return sorted(faces, key=lambda face: face[2].mean(axis=0) @ up, reverse=True)


def object_points(world, valid, mask, camera_forward, *, keep_small_faces=False):
    """Remove depth discontinuities and isolated specks from the object cloud."""
    yy, xx = np.nonzero(mask & valid)
    if not len(yy):
        return np.empty((0, 3))
    region = np.s_[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    xyz = world[region]
    kept = mask[region] & valid[region]
    depth = xyz @ camera_forward
    spread = local_depth_range(depth, valid[region])
    kept &= spread < 0.014
    kept = binary_erosion(kept, border_value=0)
    components, count = label(kept)
    if count:
        areas = np.bincount(components.ravel())[1:]
        # A raised face can be much smaller than the surface below it. Keep
        # the bulk-cloud filter for body and rim fits that use the whole cloud.
        minimum = 15 if keep_small_faces else max(15, areas.max() * 0.06)
        keep = np.r_[False, areas >= minimum]
        kept = keep[components]
    return xyz[kept]
