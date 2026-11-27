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

"""Fit rectangular faces from measured planar boundaries in the primary image."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    find_objects,
    label,
)
from scipy.spatial import ConvexHull, QhullError

from toolkits.geometry.depth_foreground import local_depth_range
from toolkits.geometry.geometry import _basis, opening_covers_center, ray_plane_offsets

PLANE_TOLERANCE_M = 0.0013


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


def _measure_sides(
    object_mask, contour, valid, world, plane, camera, axes, bounds, pitch, *, clipped
):
    """Locate the four sides using outward, in-frame depth transitions."""
    origin, normal = plane
    # Leave room for all four neighbors, even at the boundary of the working crop.
    rows, cols = np.nonzero(contour[1:-1, 1:-1])
    rows, cols = rows + 1, cols + 1
    uv = (world[rows, cols] - origin) @ axes
    neighbors = (rows[:, None] + [-1, 1, 0, 0], cols[:, None] + [0, 0, -1, 1])
    # Compare pixel positions on the fitted plane. A depth jump alone can also
    # come from an occluder or a neighbor on the inward side of the contour.
    xyz = world[neighbors]
    step = ray_plane_offsets(xyz, camera, origin, normal) @ axes - uv[:, None, :]
    height = (xyz - origin) @ normal
    # A row or column projects to a line on the plane. When the opposite pixel
    # lies on the face, it directly measures the outward direction and pitch.
    opposite = [1, 0, 3, 2]
    coplanar = valid[neighbors] & (abs(height) <= PLANE_TOLERANCE_M)
    step = np.where(coplanar[:, opposite, None], -step[:, opposite], step)
    transition = valid[neighbors] & (height < -PLANE_TOLERANCE_M)
    foreground = valid[neighbors] & (height > PLANE_TOLERANCE_M)
    # An internal step can bound a recessed face. At the object silhouette
    # only a ray reaching a farther surface establishes an edge.
    transition |= foreground & object_mask[neighbors]
    foreground &= ~object_mask[neighbors]

    refined = bounds.copy()
    supported = np.zeros((2, 2), dtype=bool)
    used = np.zeros(len(rows), dtype=bool)
    size = bounds[1] - bounds[0]
    if clipped and min(size) < 10 * pitch:
        return refined, supported, 0
    tolerance = np.maximum((4 if clipped else 1.8) * pitch, 0.012 * size)
    for axis in range(2):
        tangent = 1 - axis
        along = uv[:, tangent] - bounds[0, tangent]
        margin = 3 * pitch if clipped else tolerance[tangent]
        interior = (along > margin) & (along < size[tangent] - margin)
        for end, sign in enumerate((-1, 1)):
            outward = sign * step[:, :, axis] > 0.15 * pitch
            near = abs(uv[:, axis] - bounds[end, axis]) < tolerance[axis]
            side = (
                interior
                & near
                & (outward & transition).any(axis=1)
                & ~(outward & foreground).any(axis=1)
            )
            if not side.any():
                continue
            if not clipped:
                if side.sum() >= 3 and _span(along[side]) >= 0.17 * size[tangent]:
                    supported[end, axis] = True
                    used |= side
                continue
            offset = np.median(uv[side, axis])
            side &= abs(uv[:, axis] - offset) <= max(1.5 * pitch, 0.003 * size[axis])
            # Distributed points on one line establish a side. Two isolated
            # clusters near opposite corners can have the same overall span.
            counts, _ = np.histogram(along[side], bins=16, range=(0, size[tangent]))
            runs = np.lib.stride_tricks.sliding_window_view(counts >= 2, 4)
            if runs.all(axis=1).any():
                supported[end, axis] = True
                refined[end, axis] = np.median(uv[side, axis])
                used |= side
    return refined, supported, int((used & ~object_mask[rows, cols]).sum())


def observed_face(observation, region, *, object_mask, plane):
    """Measure rectangular sides on the selected object plane."""
    up = observation.up_direction
    if up is None or observation.camera_to_world is None:
        return None
    origin, n = plane
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
    aligned = valid & (abs(h) < PLANE_TOLERANCE_M)
    jump = local_depth_range(optical, valid)
    interior = aligned & (jump < 0.008) & object_mask[sl]
    cc, count = label(interior)
    if not count:
        return None
    weights = np.bincount(cc[region[sl]], minlength=count + 1)
    weights[0] = 0
    if weights.max() < 20:
        return None
    core = cc == weights.argmax()
    roof = binary_dilation(
        core,
        iterations=2,
        mask=aligned & region[sl],
    )
    if roof.sum() > max(70, region.sum() * 4.0):
        return None
    # A mask can end one pixel inside any of the measured sides.
    roof = binary_dilation(roof, mask=aligned)
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
    foreground = (
        outside
        & ~object_mask[sl]
        & (optical[tuple(near)] - optical > max(0.009, 5 * pitch))
    )
    # A side needs an observed transition off the face. A depth dropout leaves it unknown.
    transition = (h < -PLANE_TOLERANCE_M) | (~aligned & object_mask[sl])
    physical = roof & in_frame & binary_dilation(outside & transition)
    physical &= ~binary_dilation(foreground, iterations=3)
    edge = uv[physical]
    edge_tolerance = np.maximum(1.8 * pitch, 0.012 * size)
    # An ellipse fills pi/4 of its enclosing box and touches all four sides.
    # If the footprint fits that area as well as a rectangle's, require a
    # physical boundary sample near two adjoining sides.
    if fill <= (1 + np.pi / 4) / 2:
        side_distance = np.minimum(abs(edge - bounds[0]), abs(edge - bounds[1]))
        near_corner = side_distance <= np.minimum(edge_tolerance, 0.1 * size)
        if not near_corner.all(axis=1).any():
            return None
    image_borders = [
        name
        for name, reached in (
            ("left", sl[1].start == 0 and roof[:, :2].any()),
            ("right", sl[1].stop == W and roof[:, -2:].any()),
            ("top", sl[0].start == 0 and roof[:2].any()),
            ("bottom", sl[0].stop == H and roof[-2:].any()),
        )
        if reached
    ]
    camera = observation.camera_to_world[:3, 3]
    contour = roof & ~binary_erosion(roof) & in_frame if image_borders else physical
    bounds, supported, adjacent = _measure_sides(
        object_mask[sl],
        contour,
        valid,
        world,
        plane,
        camera,
        axes,
        bounds,
        pitch,
        clipped=bool(image_borders),
    )
    center = origin + np.mean(bounds, axis=0) @ axes.T
    if supported.all():
        # A sightline reaching a lower surface inside the four sides disproves
        # material there. Missing depth and foreground occluders cannot do so.
        extent = find_objects(roof, max_label=1)[0]
        below = valid[extent] & (h[extent] < -PLANE_TOLERANCE_M)
        crossings = ray_plane_offsets(world[extent][below], camera, center, n) @ axes
        inside = (abs(crossings) < (bounds[1] - bounds[0]) / 2 - edge_tolerance).all(
            axis=1
        )
        large_cutout = np.count_nonzero(inside) > max(5, 0.15 * roof.sum())
        if large_cutout or opening_covers_center(below, valid[extent], crossings):
            return None
    return {
        "supported": supported,
        "center": center,
        "sizes": bounds[1] - bounds[0],
        "clipped": bool(image_borders),
        "image_borders": image_borders,
        "pitch": pitch,
        "n_support": int(roof.sum()),
        "adjacent_edge_samples": adjacent,
    }
