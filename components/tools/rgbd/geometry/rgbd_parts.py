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

import numpy as np
from scipy.ndimage import binary_erosion, label
from scipy.optimize import least_squares

from components.tools.rgbd.geometry.depth_foreground import local_depth_range
from components.tools.rgbd.geometry.geometry import FitOptions, _basis, fit_circle


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
    center = origin + axes @ out.x[:2]
    return center, out.x[2], active


def _plane(points, up, tol=0.002):
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
    counts[abs(normals @ up) < 0.52] = 0
    k = np.argmax(counts)
    if counts[k] < 5:
        return None
    good = abs(points @ normals[k] - offsets[k]) < tol
    origin = np.mean(points[good], axis=0)
    normal = np.linalg.svd(points[good] - origin, full_matrices=False)[2][-1]
    if normal @ up < 0:
        normal = -normal
    return origin, normal


def rim(points, up):
    """Find the upper lip ridge independently in radial sectors, including on tilted vessels."""
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
            if len(idx) < 3:
                continue
            # The inner side of the measured lip within the local top band.
            top = np.quantile(height[idx], 0.98)
            band = idx[(height[idx] >= top - 0.0012) & (height[idx] <= top + 0.0018)]
            if not len(band):
                continue
            order = band[np.argsort(radial[band])]
            ridge.append(points[order[int(0.28 * (len(order) - 1))]])
        ridge = np.asarray(ridge)
        plane = _plane(ridge, up)
        if plane is None:
            return None
        p, normal = plane
        ridge = ridge[abs((ridge - p) @ normal) < 0.0035]
        circ = _circle(ridge, normal)
        if circ is None:
            return None
        center, radius, inliers = circ
        center -= normal * ((center - p) @ normal)
    q = ridge[inliers] - center
    angle = np.arctan2(q @ axes[:, 1], q @ axes[:, 0])
    angles = np.sort(np.mod(angle, 2 * np.pi))
    coverage = 360 - np.rad2deg(np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])))
    error = np.hypot(
        np.linalg.norm(q - np.outer(q @ normal, normal), axis=1) - radius, q @ normal
    )
    return {
        "center": center,
        "normal": normal,
        "radius": float(radius),
        "residual": float(np.median(error)),
        "coverage": float(coverage),
        "n_support": len(q),
    }


def _face(points, up):
    """Largest observed upward plane. Exclude other planes before measuring its footprint."""
    top = points @ up
    choices = []
    for threshold in (0.15, 0.55, 0.79):
        sample = points[top >= np.quantile(top, threshold)]
        plane = _plane(sample, up, 0.0018)
        if plane is None:
            continue
        p, n = plane
        active = abs((points - p) @ n) < 0.0019
        support = points[active]
        if len(support) < 10:
            continue
        choices.append((len(support) * max(0, n @ up) ** 3, p, n, support))
    return max(choices, key=lambda c: c[0])[1:] if choices else None


def measure(
    points, up, quantity, camera_right=None, camera_toward=None, *, boundary_points=None
):
    """Return metric scalar/vector geometry and diagnostics without object class information."""
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 25 or up is None:
        return None, {"reason": "insufficient_valid_object_samples_or_up_direction"}
    up = np.array(up, dtype=float)
    up /= np.linalg.norm(up)
    if quantity in ("inner_rim_center", "inner_rim_radius", "rim_extrema"):
        fit = rim(points, up)
        if fit is None:
            return None, {"reason": "upper_circular_ridge_not_resolved"}
        center, n, r = fit.pop("center"), fit.pop("normal"), fit.pop("radius")
        if quantity == "rim_extrema":
            if camera_right is None or camera_toward is None:
                return None, {"reason": "camera_basis_required"}
            right = camera_right - (camera_right @ n) * n
            if np.linalg.norm(right) < 1e-5:
                return None, {"reason": "degenerate_camera_basis"}
            right /= np.linalg.norm(right)
            near = np.cross(n, right)
            if near @ camera_toward < 0:
                near = -near
            value = [
                center - r * right,
                center + r * right,
                center + r * near,
                center - r * near,
            ]
        else:
            value = center if quantity == "inner_rim_center" else r
        return np.asarray(value).tolist(), fit
    if quantity == "body_segment_center":
        return barrel(points, up)
    if quantity == "object_height":
        bounds = np.quantile(points @ up, [0.002, 0.998])
        return float(bounds[1] - bounds[0]), {
            "reason": "visible_projected_height",
            "n_support": len(points),
        }
    face = _face(points, up)
    if face is None:
        return None, {"reason": "upward_face_not_resolved"}
    p, n, support = face
    if quantity == "plate_center":
        if boundary_points is None or not len(boundary_points):
            return None, {"reason": "plate_rim_not_observed"}
        # Fit the observed silhouette in the plane of the plate's supporting face.
        # The lip may sit above the surface on which an object would be placed.
        options = FitOptions()
        if len(boundary_points) > options.max_points:
            idx = np.linspace(
                0, len(boundary_points) - 1, options.max_points, dtype=int
            )
            boundary_points = boundary_points[idx]
        flat_rim = boundary_points - np.outer((boundary_points - p) @ n, n)
        fit = fit_circle(flat_rim, options)
        if not fit.parameters:
            return None, {"reason": fit.reason}
        center = np.asarray(fit.parameters["center"])
        radius = fit.parameters["radius"]
        quality = {
            **fit.quality,
            "n_support": len(fit.inlier_indices),
            "radius": radius,
            "warnings": fit.warnings,
        }
        delta = points - center
        projected = delta - np.outer(delta @ n, n)
        middle = points[np.linalg.norm(projected, axis=1) < 0.48 * radius]
        if len(middle) >= 15:
            center = center + np.median((middle - center) @ n) * n
            quality["central_face_support"] = len(middle)
        return center.tolist(), quality
    resolved_cap = False
    if quantity == "lid_center" and n @ up < 0.85:
        # An obliquely viewed can may expose more of its wall than of its lid.
        levels = points @ up
        cap = points[levels >= np.quantile(levels, 0.79)]
        other = _face(cap, up) if len(cap) >= 25 else None
        if other is not None:
            top, axis, samples = other
            if axis @ up >= 0.90 and top @ up > p @ up + 0.002:
                p, n, support = top, axis, samples
                resolved_cap = True
    from scipy.spatial import ConvexHull, QhullError

    axes = np.stack(_basis(n), axis=1)
    xy = (support - p) @ axes
    try:
        hull = ConvexHull(xy)
    except QhullError:
        return None, {"reason": "degenerate_face_extent"}
    if quantity in ("rectangle_center", "top_corners", "top_size"):
        edges = np.diff(xy[hull.vertices], axis=0, append=xy[hull.vertices[:1]])
        orientations = np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2)
        directions = np.stack((np.cos(orientations), np.sin(orientations)), axis=1)
        second = np.stack((-directions[:, 1], directions[:, 0]), axis=1)
        # Trim isolated edge pixels but retain the physical four extremities.
        candidate = support[:: max(1, len(support) // 2500)]
        proj = (candidate - p) @ axes
        lo, hi = np.quantile(proj @ directions.T, [0.002, 0.998], axis=0)
        low, high = np.quantile(proj @ second.T, [0.002, 0.998], axis=0)
        k = np.argmin((hi - lo) * (high - low))
        d, e = axes @ directions[k], axes @ second[k]
        center = p + (hi[k] + lo[k]) / 2 * d + (high[k] + low[k]) / 2 * e
        extents = np.array([hi[k] - lo[k], high[k] - low[k]])
        corners = np.array(
            [
                center + i * d * extents[0] / 2 + j * e * extents[1] / 2
                for i, j in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            ]
        )
        if quantity == "top_corners":
            if camera_right is None or camera_toward is None:
                return None, {"reason": "camera_basis_required"}
            order = np.argsort((corners - center) @ camera_toward)
            far, near = corners[order[:2]], corners[order[2:]]
            value = np.r_[
                far[np.argsort(far @ camera_right)],
                near[np.argsort(near @ camera_right)],
            ]
        else:
            value = center if quantity == "rectangle_center" else np.sort(extents)[::-1]
        return value.tolist(), {
            "n_support": len(support),
            "residual": float(np.median(abs((support - p) @ n))),
        }
    angles = np.linspace(0, 2 * np.pi, 91)[:-1]
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    edges = support[np.argmax(xy @ directions.T, axis=0)]
    circ = _circle(edges, n)
    if circ is None:
        return None, {"reason": "circular_face_not_resolved"}
    center, r, active = circ
    quality = {"n_support": int(active.sum()), "radius": r}
    if resolved_cap:
        quality["warnings"] = ["upper_lid_plane_selected_above_side_wall"]
    return center.tolist(), quality


def object_points(world, valid, mask, camera_forward=None):
    """Remove mixed pixels at silhouettes and detached foreground/background patches."""
    yy, xx = np.nonzero(mask & valid)
    if not len(yy):
        return np.empty((0, 3))
    region = np.s_[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    xyz = world[region]
    kept = mask[region] & valid[region]
    if camera_forward is not None:
        depth = xyz @ camera_forward
        spread = local_depth_range(depth, valid[region])
        kept &= spread < 0.014
    kept = binary_erosion(kept, border_value=0)
    components, count = label(kept)
    if count:
        areas = np.bincount(components.ravel())[1:]
        # Both sides of an aperture can be disconnected in depth.
        keep = np.r_[False, areas >= max(15, areas.max() * 0.06)]
        kept = keep[components]
    return xyz[kept]


def barrel(points, up):
    """Fit multiple observed sections, then locate the widest straight cylindrical interval."""
    points = points[:: max(1, len(points) // 8000)]
    mean = np.median(points, axis=0)
    pcs = np.linalg.svd(points - mean, full_matrices=False)[2]
    alternatives = []
    for axis in [up, pcs[0]]:
        axis = axis.copy()
        for iteration in range(2):
            axial = points @ axis
            lo, hi = np.quantile(axial, [0.001, 0.999])
            width = (hi - lo) / 24
            if width <= 0:
                break
            sections = []
            for i in range(24):
                mid = lo + (i + 0.5) * width
                block = points[abs(axial - mid) < width * 0.50]
                if len(block) < 12:
                    continue
                block = block[:: max(1, len(block) // 100)]
                circle = _circle(block, axis, tolerance=0.0014)
                if circle is None:
                    continue
                center, radius, good = circle
                center += (mid - center @ axis) * axis
                sections.append((i, center, radius, int(good.sum())))
            if len(sections) < 5:
                break
            radii = np.array([v[2] for v in sections])
            radius = np.quantile(radii, 0.77)
            groups = []
            g = []
            for value in sections:
                if abs(value[2] - radius) < max(0.0014, radius * 0.06):
                    if g and value[0] != g[-1][0] + 1:
                        groups.append(g)
                        g = []
                    g.append(value)
                elif g:
                    groups.append(g)
                    g = []
            if g:
                groups.append(g)
            if not groups:
                break
            group = max(groups, key=len)
            if len(group) < 3:
                break
            centers = np.array([v[1] for v in group])
            center = np.median(centers, axis=0)
            center += ((centers[0] + centers[-1]) / 2 - center) @ axis * axis
            spread = np.median(
                np.linalg.norm(
                    (centers - center) - np.outer((centers - center) @ axis, axis),
                    axis=1,
                )
            )
            alternatives.append(
                (len(group) / (1 + spread / 0.0008), center, spread, len(group))
            )
            improved = np.linalg.svd(centers - centers.mean(0), full_matrices=False)[2][
                0
            ]
            if improved @ axis < 0:
                improved = -improved
            if improved @ axis < 0.92:
                break
            axis = improved
    if not alternatives:
        return None, {"reason": "stable_cylindrical_sections_not_resolved"}
    _, center, residual, n = max(alternatives, key=lambda v: v[0])
    return center.tolist(), {
        "residual": float(residual),
        "n_sections": n,
        "warnings": ["barrel_endpoints_inferred_from_radius_profile"],
    }


def cuboid_vertices(top, up, support_level, measured_height):
    """Project four measured top corners to an observed contacting plane of an upright cuboid."""
    if any(v is None for v in (top, up, support_level, measured_height)):
        return None
    top = np.asarray(top, dtype=float)
    if top.shape != (4, 3):
        return None
    heights = top @ up - support_level
    if not np.isfinite(heights).all():
        return None
    if np.min(heights) <= 0 or np.max(abs(heights - measured_height)) > 0.005:
        return None
    normal = np.cross(top[1] - top[0], top[2] - top[0])
    if (
        abs(normal @ up) < 0.985 * np.linalg.norm(normal)
        or np.linalg.norm(normal) < 1e-8
    ):
        return None
    return np.vstack((top, top - heights[:, None] * up)).tolist()
