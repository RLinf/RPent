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

"""Rectangular faces from measured plane boundaries and their connected side walls."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    find_objects,
    label,
)
from scipy.spatial import ConvexHull, QhullError

from components.tools.rgbd.geometry.depth_foreground import local_depth_range
from components.tools.rgbd.geometry.geometry import _basis
from components.tools.rgbd.geometry.rgbd_parts import _face


def _span(values):
    return float(np.ptp(values)) if len(values) else 0.0


def _directions(xy):
    hull = ConvexHull(xy)
    vertices = xy[hull.vertices]
    edge = np.roll(vertices, -1, axis=0) - vertices
    axes = edge / np.maximum(np.linalg.norm(edge, axis=1)[:, None], 1e-12)
    cross = np.stack((-axes[:, 1], axes[:, 0]), axis=1)
    areas = np.ptp(vertices @ axes.T, axis=0) * np.ptp(vertices @ cross.T, axis=0)
    idx = int(np.argmin(areas))
    return np.stack((axes[idx], cross[idx]), axis=1), hull.volume / max(
        areas[idx], 1e-12
    )


def _wall_limits(world, valid, uv, height, bounds, boundary, pitch):
    """A vertical material edge can continue beyond the visible roof in an oblique view."""
    spans = bounds[1] - bounds[0]
    dr = world[2:, 1:-1] - world[:-2, 1:-1]
    dc = world[1:-1, 2:] - world[1:-1, :-2]
    normal = np.cross(dr, dc)
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-12)
    normal_valid = (
        valid[2:, 1:-1] & valid[:-2, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
    )
    valid = valid[1:-1, 1:-1]
    uv = uv[1:-1, 1:-1]
    h = height[1:-1, 1:-1]
    # The gate scales with measured span and sampling resolution.
    tol = max(0.0015, 2 * pitch)
    inside = (h < -2 * pitch) & (h > -1.3 * max(spans))
    updates = bounds.copy()
    for axis in range(2):
        tangent = 1 - axis
        for edge in bounds[:, axis]:
            plane = (
                valid
                & normal_valid
                & inside
                & (abs(uv[:, :, axis] - edge) < tol)
                & (abs(normal @ boundary[:, axis]) > 0.94)
            )
            plane &= (
                uv[:, :, tangent] >= bounds[0, tangent] - 0.28 * spans[tangent]
            ) & (uv[:, :, tangent] <= bounds[1, tangent] + 0.28 * spans[tangent])
            # A shallow measured seam can split a wall in normal space. Permit a
            # connection only through depth samples close to this same wall.
            wall_distance = abs(uv[:, :, axis] - edge)
            along = (
                uv[:, :, tangent] >= bounds[0, tangent] - 0.28 * spans[tangent]
            ) & (uv[:, :, tangent] <= bounds[1, tangent] + 0.28 * spans[tangent])
            connected, count = label(
                valid & inside & along & (wall_distance < max(0.007, 5 * pitch))
            )
            if not count:
                continue
            for i, box in enumerate(find_objects(connected), 1):
                if box is None:
                    continue
                region = (connected[box] == i) & plane[box]
                if region.sum() < 24:
                    continue
                v = uv[box][:, :, tangent][region]
                z = h[box][region]
                if z.max() < -max(0.018, 0.17 * max(spans)) or _span(z) < max(
                    0.005, 4 * pitch
                ):
                    continue
                if (
                    np.mean(
                        (v >= bounds[0, tangent] - tol)
                        & (v <= bounds[1, tangent] + tol)
                    )
                    < 0.14
                ):
                    continue
                for side in range(2):
                    limit = float(np.quantile(v, 0.0005 if side == 0 else 0.9995))
                    sign = -1 if side == 0 else 1
                    extension = sign * (limit - bounds[side, tangent])
                    if not 2 * pitch < extension < 0.25 * spans[tangent]:
                        continue
                    end = abs(v - limit) < max(2.5 * pitch, 0.012 * spans[tangent])
                    if end.sum() < 5 or _span(z[end]) < max(
                        4 * pitch, 0.025 * np.ptp(z)
                    ):
                        continue
                    if sign * (limit - updates[side, tangent]) > 0:
                        updates[side, tangent] = limit
    return updates


def observed_face(observation, region, cloud):
    """Return observed rectangular side positions and per-edge support in world coordinates."""
    up = observation.up_direction
    if up is None or observation.camera_to_world is None:
        return None
    face = _face(cloud, up)
    if face is None:
        return None
    origin, n, _ = face
    if n @ up < 0.91:
        return None
    boxes = find_objects(region, max_label=1)
    if not boxes or boxes[0] is None:
        return None
    y, x = boxes[0]
    H, W = region.shape
    pad = max(12, round(0.65 * max(y.stop - y.start, x.stop - x.start)))
    sl = np.s_[
        max(0, y.start - pad) : min(H, y.stop + pad),
        max(0, x.start - pad) : min(W, x.stop + pad),
    ]
    world = observation.world_xyz[sl]
    valid = observation.valid_mask[sl]
    h = (world - origin) @ n
    optical = (
        world - observation.camera_to_world[:3, 3]
    ) @ observation.camera_to_world[:3, 2]
    aligned = valid & (abs(h) < 0.0013)
    jump = local_depth_range(optical, valid)
    interior = aligned & (jump < 0.008)
    cc, count = label(interior)
    if not count:
        return None
    weights = np.bincount(cc[region[sl]], minlength=count + 1)
    weights[0] = 0
    if weights.max() < 20:
        return None
    core = cc == weights.argmax()
    roof = binary_dilation(core, iterations=2, mask=aligned)
    if roof.sum() > max(70, region.sum() * 4.0):
        return None
    basis = np.stack(_basis(n), axis=1)
    pts = world[roof]
    xy = (pts - origin) @ basis
    try:
        rot, fill = _directions(xy)
    except QhullError:
        return None
    if fill < 0.66:
        return None
    axes = basis @ rot
    uv = (world - origin) @ axes
    bounds = np.quantile(uv[roof], [0.0005, 0.9995], axis=0)
    size = bounds[1] - bounds[0]
    pitches = np.r_[
        np.linalg.norm(np.diff(world, axis=0)[roof[1:] & roof[:-1]], axis=1),
        np.linalg.norm(np.diff(world, axis=1)[roof[:, 1:] & roof[:, :-1]], axis=1),
    ]
    if not len(pitches):
        return None
    pitch = float(np.median(pitches))
    if min(size) < max(0.002, 4 * pitch):
        return None
    rows, cols = np.indices(roof.shape)
    in_frame = (
        (rows + sl[0].start > 1)
        & (rows + sl[0].start < H - 2)
        & (cols + sl[1].start > 1)
        & (cols + sl[1].start < W - 2)
    )
    near = distance_transform_edt(~roof, return_distances=False, return_indices=True)
    outside = binary_dilation(roof, iterations=3) & ~roof & valid
    foreground = outside & (optical[tuple(near)] - optical > max(0.009, 5 * pitch))
    # A side needs an observed transition off the face. A depth dropout leaves it unknown.
    physical = roof & in_frame & binary_dilation(outside & ~aligned)
    physical &= ~binary_dilation(foreground, iterations=3)
    edge = uv[physical]
    supported = np.zeros((2, 2), dtype=bool)
    edge_tolerance = np.maximum(1.8 * pitch, 0.012 * size)
    for d in range(2):
        tangent = 1 - d
        # Corners alone can belong to either of their adjoining sides.
        side = (edge[:, tangent] > bounds[0, tangent] + edge_tolerance[tangent]) & (
            edge[:, tangent] < bounds[1, tangent] - edge_tolerance[tangent]
        )
        for j in range(2):
            sel = side & (abs(edge[:, d] - bounds[j, d]) < edge_tolerance[d])
            supported[j, d] = (
                sel.sum() >= 3 and _span(edge[sel, tangent]) >= 0.17 * size[tangent]
            )
    clipped = bool(np.any(roof & ~in_frame))
    measured = bounds
    if clipped:
        measured = _wall_limits(world, valid, uv, h, bounds, axes, pitch)
    center = origin + np.mean(measured, axis=0) @ axes.T
    dims = measured[1] - measured[0]
    return {
        "origin": origin,
        "axes": axes,
        "bounds": measured,
        "supported": supported,
        "center": center,
        "sizes": dims,
        "clipped": clipped,
        "wall_extended": bool(np.any(abs(measured - bounds) > 2 * pitch)),
        "pitch": pitch,
        "n_support": int(roof.sum()),
        "fill": float(fill),
    }


def rectangle_value(face, name, camera):
    center, axes, extents = face["center"], face["axes"], face["sizes"]
    if name == "rectangle_center":
        return center.tolist()
    if name == "top_size":
        return np.sort(extents)[::-1].tolist()
    if camera is None:
        return None
    corners = np.array(
        [
            center + i * axes[:, 0] * extents[0] / 2 + j * axes[:, 1] * extents[1] / 2
            for i, j in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        ]
    )
    depth = (corners - center) @ (-camera[:3, 2])
    right = (corners - center) @ camera[:3, 0]
    order = np.argsort(depth, kind="stable")
    # At a diagonal the two middle corners can exchange depth order within
    # the spatial resolution. Keep center and side lengths available.
    towards_span = np.linalg.norm(axes.T @ camera[:3, 2])
    right_span = np.linalg.norm(axes.T @ camera[:3, 0])
    margin = 2 * face["pitch"]
    if (
        min(towards_span, right_span) < 1e-5
        or depth[order[2]] - depth[order[1]] <= margin * towards_span
        or min(np.ptp(right[order[:2]]), np.ptp(right[order[2:]]))
        <= margin * right_span
    ):
        return None
    return corners[
        np.r_[
            order[:2][np.argsort(right[order[:2]])],
            order[2:][np.argsort(right[order[2:]])],
        ]
    ].tolist()
