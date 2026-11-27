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


"""Resolve a nearby uniformly colored object using metric depth boundaries."""

import re

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    find_objects,
    label,
)

from components.tools.rgbd.geometry.depth_foreground import local_depth_range

# Hue is cyclic. These are ordinary sRGB hue angles, independent of the camera and object class.
HUES = {
    "red": 0,
    "orange": 30,
    "yellow": 57,
    "green": 120,
    "cyan": 180,
    "blue": 230,
    "purple": 280,
    "pink": 330,
}


def _rectangular_top(world, valid, mask, up):
    from scipy.spatial import ConvexHull, QhullError

    from components.tools.rgbd.geometry.geometry import _basis
    from components.tools.rgbd.geometry.rgbd_parts import _face

    cloud = world[valid & mask]
    if len(cloud) < 30:
        return False
    face = _face(cloud, up)
    if face is None:
        return False
    center, normal, samples = face
    if normal @ up < 0.90:
        return False
    xy = (samples - center) @ np.stack(_basis(normal), axis=1)
    try:
        hull = ConvexHull(xy)
    except QhullError:
        return False
    vertices = xy[hull.vertices]
    edges = np.diff(vertices, axis=0, append=vertices[:1])
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    transverse = np.stack((-directions[:, 1], directions[:, 0]), axis=1)
    box_area = np.min(
        np.ptp(vertices @ directions.T, axis=0)
        * np.ptp(vertices @ transverse.T, axis=0)
    )
    return box_area > 0 and hull.volume >= 0.9 * box_area


def _isolated_contour(cc, areas, jump, optical, valid, world, up):
    """A distant compact contour must be unique and show foreground or contact evidence."""
    candidates = []
    height, width = valid.shape
    for component, box in enumerate(find_objects(cc), start=1):
        if (
            box is None
            or not max(35, 0.00015 * height * width)
            <= areas[component]
            <= 0.18 * height * width
        ):
            continue
        ys, xs = box
        if ys.start < 3 or xs.start < 3 or ys.stop > height - 3 or xs.stop > width - 3:
            continue
        if areas[component] < 0.50 * (ys.stop - ys.start) * (xs.stop - xs.start):
            continue
        roi = np.s_[ys.start - 3 : ys.stop + 3, xs.start - 3 : xs.stop + 3]
        mask = cc[roi] == component
        near = binary_dilation(mask, iterations=2)
        shell = near & ~mask
        outside = binary_dilation(near) & ~near & valid[roi]
        boundary = float(np.mean(jump[roi][shell] > 0.012))
        if boundary < 0.50 or outside.sum() < 10:
            continue
        isolated = (
            boundary >= 0.9
            and np.quantile(optical[roi][outside], 0.25)
            > np.quantile(optical[roi][mask], 0.75) + 0.012
        )
        if world is not None and up is not None:
            if not _rectangular_top(world[roi], valid[roi], mask, up):
                continue
            if not isolated:
                from components.tools.rgbd.geometry.depth_foreground import (
                    nearby_support_height,
                )

                region = cc == component
                low, top = np.quantile(world[region] @ up, [0.002, 0.998])
                support = nearby_support_height(world, valid, region, up, top)
                if support is None or abs(low - support) > 0.007 or top - low < 0.006:
                    continue
        elif not isolated:
            continue
        candidates.append(
            (
                component,
                boundary,
                "unique_occluding_contour"
                if isolated
                else "unique_rectangular_top_on_support",
            )
        )
        if len(candidates) > 1:
            return None
    return candidates[0] if candidates else None


def described_hue(query):
    phrase = re.split(r"\bwith\b|,|\bexcluding\b", query.lower(), maxsplit=1)[0]
    words = set(
        re.findall(
            r"\b(?:red|orange|yellow|green|cyan|blue|purple|pink|white|black|gray|grey|brown)\b",
            phrase,
        )
    )
    return words.pop() if len(words) == 1 and words.issubset(HUES) else None


def _color_pixels(rgb, valid, color):
    hue, sat, val = np.moveaxis(
        np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=float) / 255, -1, 0
    )
    delta = np.abs((360 * hue - HUES[color] + 180) % 360 - 180)
    return valid & (delta < 24) & (sat > 0.43) & (val > 0.18)


def _components(rgb, optical, valid, color, *, match=None):
    if match is None:
        match = _color_pixels(rgb, valid, color)
    jump = local_depth_range(optical, valid)
    cc, n = label(match & (jump < 0.015))
    return cc, n, jump


def _top_footprint(world, valid, mask, up):
    """Rectangularity and side ratio of the resolved top, including its eroded edge."""
    from scipy.spatial import ConvexHull, QhullError

    from components.tools.rgbd.geometry.geometry import _basis

    levels = world @ up
    top = np.quantile(levels[mask], 0.95)
    band = valid & (abs(levels - top) < 0.0008)
    top_pixels = mask & band
    if np.count_nonzero(top_pixels) < max(10, 0.08 * mask.sum()):
        return 0.0, 0.0, False
    # Extend the measured top at its own boundary. Dilation from the whole
    # silhouette can reach a separate coplanar object beside the lower wall.
    face = binary_dilation(top_pixels, iterations=3, mask=band)
    xy = world[face] @ np.stack(_basis(up), axis=1)
    try:
        hull = ConvexHull(xy)
    except QhullError:
        return 0.0, 0.0, False
    vertices = xy[hull.vertices]
    edge = np.diff(vertices, axis=0, append=vertices[:1])
    angle = np.arctan2(edge[:, 1], edge[:, 0])
    dirs = np.stack((np.cos(angle), np.sin(angle)), axis=1)
    width = np.ptp(vertices @ dirs.T, axis=0)
    height = np.ptp(vertices @ np.stack((-dirs[:, 1], dirs[:, 0])), axis=0)
    best = np.argmin(width * height)
    a, b = sorted((width[best], height[best]))
    pitches = np.r_[
        np.linalg.norm(np.diff(world, axis=0)[face[:-1] & face[1:]], axis=1),
        np.linalg.norm(np.diff(world, axis=1)[face[:, :-1] & face[:, 1:]], axis=1),
    ]
    if not len(pitches) or a < 4 * np.median(pitches):
        return 0.0, 0.0, False
    axis = dirs[best]
    uv = vertices @ np.stack((axis, [-axis[1], axis[0]]), axis=1)

    def straight_edge(axis):
        along, across = uv[:, axis], uv[:, 1 - axis]
        tolerance = 0.04 * np.ptp(along) + 0.0005
        return any(
            np.ptp(across[abs(along - edge) <= tolerance]) > 0.58 * np.ptp(across)
            for edge in (along.min(), along.max())
        )

    return hull.volume / (a * b), a / b, int(straight_edge(0)) + int(straight_edge(1))


def _has_flat_side(world, mask, up):
    """Distinguish a partly covered cuboid wall from a cut through a round object."""
    from components.tools.rgbd.geometry.geometry import _basis

    axes = np.stack(_basis(up), axis=1)
    dr = world[2:, 1:-1] - world[:-2, 1:-1]
    dc = world[1:-1, 2:] - world[1:-1, :-2]
    cross = np.cross(dr, dc)
    lengths = np.linalg.norm(cross, axis=2)
    normals = cross / np.maximum(lengths[..., None], 1e-12)
    vertical = (
        binary_erosion(mask)[1:-1, 1:-1]
        & (lengths > 1e-10)
        & (abs(normals @ up) < 0.25)
    )
    if vertical.sum() < 12:
        return False
    points = world[mask]
    span = max(
        np.max(np.ptp(np.quantile(points @ axes, [0.03, 0.97], axis=0), axis=0)),
        np.ptp(np.quantile(points @ up, [0.03, 0.97])),
    )
    points = world[1:-1, 1:-1][vertical]
    horizontal = normals[vertical] @ axes
    angle = np.arctan2(horizontal[:, 1], horizontal[:, 0])
    for direction in np.linspace(0, np.pi, 60, endpoint=False):
        aligned = abs((angle - direction + np.pi / 2) % np.pi - np.pi / 2) < np.deg2rad(
            6
        )
        if aligned.sum() < max(12, 0.25 * len(points)):
            continue
        face = points[aligned]
        tangent = axes @ [-np.sin(direction), np.cos(direction)]
        width = np.ptp(np.quantile(face @ tangent, [0.05, 0.95]))
        height = np.ptp(np.quantile(face @ up, [0.05, 0.95]))
        if width >= 0.24 * span and height >= 0.15 * span:
            return True
    return False


def _unresolved_fragment(match, components, chosen, optical, valid):
    """Account for separate colored slivers erased by depth-edge filtering."""
    raw, _ = label(match, structure=np.ones((3, 3), dtype=bool))
    areas = np.bincount(raw.ravel())
    for i, box in enumerate(find_objects(raw), 1):
        if box is None or areas[i] < 12:
            continue
        y, x = box
        roi = np.s_[max(0, y.start - 4) : y.stop + 4, max(0, x.start - 4) : x.stop + 4]
        material = raw[roi] == i
        labels = components[roi][material]
        if (labels == chosen).any() or np.count_nonzero(labels) >= max(
            20, 0.50 * len(labels)
        ):
            continue
        outer = binary_dilation(material, iterations=3) & ~material & valid[roi]
        near = distance_transform_edt(
            ~material, return_distances=False, return_indices=True
        )
        in_front = optical[roi][tuple(near)] - optical[roi] > 0.018
        if outer.any() and np.mean(in_front[outer]) >= 0.12:
            return True
    return False


def unique_colored_box(observation, color, *, cube):
    """Find distinct, resolved faces of the described color across the whole view."""
    world, valid = observation.world_xyz, observation.valid_mask
    camera, up = observation.camera_to_world, observation.up_direction
    if camera is None or up is None:
        return None, "box_orientation_or_camera_unavailable"
    optical = (world - camera[:3, 3]) @ camera[:3, 2]
    match = _color_pixels(observation.rgb, valid, color)
    cc, count, jump = _components(observation.rgb, optical, valid, color, match=match)
    areas = np.bincount(cc.ravel(), minlength=count + 1)
    matches = []
    obscured = []
    for i, box in enumerate(find_objects(cc), start=1):
        if box is None or not (20 if cube else 55) <= areas[i] <= valid.size * 0.18:
            continue
        y, x = box
        short_side = min(y.stop - y.start, x.stop - x.start)
        if short_side < (4 if cube else 8):
            continue
        roi = np.s_[max(0, y.start - 5) : y.stop + 5, max(0, x.start - 5) : x.stop + 5]
        local = cc[roi] == i
        fill, ratio, edges = _top_footprint(world[roi], valid[roi], local, up)
        shell = binary_dilation(local, iterations=2) & ~local
        if np.mean(jump[roi][shell] > 0.012) < 0.35:
            continue
        outer = (
            binary_dilation(local, iterations=4)
            & ~binary_dilation(local, iterations=2)
            & valid[roi]
        )
        nearest = distance_transform_edt(
            ~local, return_distances=False, return_indices=True
        )
        in_front = optical[roi][tuple(nearest)] - optical[roi] > 0.035
        occlusion = float(np.mean(in_front[outer])) if outer.any() else 0.0
        if cube and edges >= 1 and fill >= 0.50 and ratio >= 0.25 and occlusion >= 0.10:
            obscured.append(i)
        if areas[i] < 55 or short_side < 8:
            continue
        complete = fill >= 0.90 and (not cube or ratio >= 0.80)
        if complete and not cube:
            # A colored stripe on a fixture may intersect a horizontal plane.
            # A freestanding block either occludes its surround or rises from support.
            from components.tools.rgbd.geometry.depth_foreground import (
                nearby_support_height,
            )

            isolated = outer.any() and np.quantile(optical[roi][outer], 0.25) > (
                np.quantile(optical[roi][local], 0.75) + 0.012
            )
            if not isolated:
                low, high = np.quantile(world[roi][local] @ up, [0.002, 0.998])
                support = nearby_support_height(world, valid, cc == i, up, high)
                if support is None or abs(support - low) > 0.007 or high - low < 0.006:
                    continue
        if complete and (occlusion < 0.22 or (cube and ratio >= 0.92)):
            kind = "unique_measured_colored_box"
        else:
            if (
                areas[i] < 180
                or edges < 2
                or fill < 0.55
                or ratio < 0.4
                or occlusion < 0.12
                or not _has_flat_side(world[roi], local, up)
            ):
                continue
            kind = "occluded_box_only_visible_material"
        matches.append((i, kind))
        if len(matches) > 1:
            return None, "multiple_matching_colored_boxes"
    if not matches:
        return None, "no_resolved_face_of_described_color_and_shape"
    component, kind = matches[0]
    if any(i != component for i in obscured) or (
        cube and _unresolved_fragment(match, cc, component, optical, valid)
    ):
        return None, "another_occluded_colored_instance"
    return cc == component, kind


def colored_region(rgb, optical, valid, rc, color, *, world=None, up=None):
    h, w = valid.shape
    cc, n, jump = _components(rgb, optical, valid, color)
    if not n:
        return None
    r, c = rc
    search = max(8, round(0.065 * min(h, w)))
    sy = slice(max(0, r - search), min(h, r + search + 1))
    sx = slice(max(0, c - search), min(w, c + search + 1))
    yy, xx = np.mgrid[sy, sx]
    costs = np.full(n + 1, np.inf)
    np.minimum.at(costs, cc[sy, sx].ravel(), np.hypot(yy - r, xx - c).ravel())
    areas = np.bincount(cc.ravel(), minlength=n + 1)
    costs[(areas < max(35, 0.00015 * w * h)) | (areas > 0.18 * w * h)] = np.inf
    costs[0] = np.inf
    # Printed patches and highlights on larger objects rarely have a depth boundary around their contour.
    ranked = []
    for component in np.flatnonzero(costs <= search):
        mask = cc == component
        shell = binary_dilation(mask, iterations=2) & ~mask
        boundary = float(np.mean(jump[shell] > 0.012)) if shell.any() else 0.0
        if boundary >= 0.50:
            ranked.append((boundary - costs[component] / search, component, boundary))
    ranked.sort(reverse=True)
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.06:
        return None
    if ranked:
        _, chosen, boundary = ranked[0]
        scope = "near_native_point"
    else:
        unique = _isolated_contour(cc, areas, jump, optical, valid, world, up)
        if unique is None:
            return None
        chosen, boundary, scope = unique
    mask = cc == chosen
    distance = distance_transform_edt(mask)
    if distance.max() < 3:
        return None
    rows, cols = np.nonzero(distance >= max(2.0, 0.60 * distance.max()))
    i = np.argmin((rows - r) ** 2 + (cols - c) ** 2)
    ys, xs = np.nonzero(mask)
    return (
        mask,
        [int(rows[i]), int(cols[i])],
        {
            "color": color,
            "anchor_distance_px": float(np.min(np.hypot(ys - r, xs - c))),
            "depth_boundary_fraction": boundary,
            "search_scope": scope,
        },
    )
