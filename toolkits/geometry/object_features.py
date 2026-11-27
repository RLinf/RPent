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

"""Fit geometric centers on a specified feature of a segmented object."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Literal, get_args

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    find_objects,
    generate_binary_structure,
    label,
    minimum_filter,
)

from toolkits.geometry.geometry import (
    FitOptions,
    ShapeFit,
    _basis,
    fit_circle,
    opening_covers_center,
    outer_boundary,
    ray_plane_offsets,
    ray_plane_radius,
)
from toolkits.geometry.measured_rectangle import PLANE_TOLERANCE_M, observed_face
from toolkits.geometry.rgbd_parts import (
    _plane,
    _refit_plane,
    face_candidates,
    object_points,
    rim,
)
from toolkits.geometry.supported_base import fit_supported_base
from toolkits.observation import GeometryObservation

RingFeature = Literal["upper_inner_ring", "upper_outer_ring"]
GeometryFeature = Literal[RingFeature, "upper_face", "supported_round_base"]
GeometryQuantity = Literal[
    "ring_center", "disk_center", "rectangle_center", "base_center"
]
QUANTITIES = get_args(GeometryQuantity)
RING_FEATURES = get_args(RingFeature)
FEATURES_BY_QUANTITY = {
    "ring_center": RING_FEATURES,
    "disk_center": ("upper_face",),
    "rectangle_center": ("upper_face",),
    "base_center": ("supported_round_base",),
}

CENTER_DEFINITIONS = {
    "ring_center": "center_of_selected_observed_circular_boundary",
    "disk_center": "center_of_selected_observed_circular_planar_surface",
    "rectangle_center": "center_of_selected_observed_rectangular_planar_surface",
    "base_center": "intersection_of_rotational_symmetry_axis_and_observed_support_plane",
}


def _circle_on_plane(points, origin, normal, options, *, inward_half_steps=None):
    """Fit observed boundary samples in the selected plane, with its upward normal."""
    if len(points) > options.max_points:
        indices = np.linspace(0, len(points) - 1, options.max_points, dtype=int)
        points = points[indices]
        if inward_half_steps is not None:
            inward_half_steps = inward_half_steps[indices]
    offsets = (points - origin) @ normal
    fit = fit_circle(
        points - np.outer(offsets, normal), options, inward_half_steps=inward_half_steps
    )
    if fit.parameters:
        fit.parameters["normal"] = normal.tolist()
        fit.quality["radius"] = fit.parameters["radius"]
    return fit


def _boundary_steps(face, adjacent):
    """Yield the raster step from each face pixel to a selected neighbor."""
    row, col = np.nonzero(face & binary_dilation(adjacent))
    padded = np.pad(adjacent, 1)
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        keep = padded[row + dr + 1, col + dc + 1]
        yield row[keep], col[keep], dr, dc


def _outer_edge_steps(on_plane, valid, mask, height, camera_height, band, *, min_drop):
    """Yield face pixels and the raster direction of each measured outer boundary."""
    # The segmentation can stop one pixel inside a measured planar face.
    on_plane = on_plane | (
        binary_dilation(on_plane) & ~mask & valid & (abs(height) <= band)
    )
    exterior = valid & ~binary_fill_holes(on_plane)
    for row, col, dr, dc in _boundary_steps(on_plane, exterior):
        neighbor = (row + dr, col + dc)
        # Inside the object either height transition defines an edge. Across
        # the silhouette only a farther background sample establishes an edge.
        physical = np.where(
            mask[neighbor],
            abs(height[neighbor]) > band,
            np.sign(camera_height) * (height[row, col] - height[neighbor]) > min_drop,
        )
        yield row[physical], col[physical], dr, dc


def _visible_outer_edge(on_plane, valid, mask, height, camera_height, band):
    edge = np.zeros_like(mask)
    for row, col, _, _ in _outer_edge_steps(
        on_plane, valid, mask, height, camera_height, band, min_drop=band
    ):
        edge[row, col] = True
    return edge


def _plane_noise_band(height, measured, max_band):
    """Estimate height scatter from second differences along measured pixel triples."""
    scatter = []
    for h, valid in ((height, measured), (height.T, measured.T)):
        triples = valid[:-2] & valid[1:-1] & valid[2:]
        scatter.append(abs(np.diff(h, n=2, axis=0))[triples])
    samples = np.concatenate(scatter)
    if len(samples):
        return np.clip(2 * np.median(samples), 1e-6, max_band)
    return None


def _tapered_cap_edges(world, material, raised, normal, options, cancel):
    """Locate planar caps whose flanks descend by less than a depth step per pixel."""
    edges = np.zeros_like(material)
    regions, _ = label(raised)
    for index, box in enumerate(find_objects(regions), start=1):
        cancel()
        patch = regions[box] == index
        xyz = world[box]
        points = xyz[patch]
        if (
            len(points) < options.min_points
            or np.ptp(points @ normal) <= options.distance_threshold_m
        ):
            continue
        upper = rim(points, normal, outer=True)
        if upper is None:
            continue
        elevation = (points - upper[0]) @ upper[1]
        top = points[elevation >= np.quantile(elevation, 0.95)]
        if len(top) < 3:
            continue
        origin, axis = _refit_plane(top, normal)
        height = (xyz - origin) @ axis
        tolerance = _plane_noise_band(height, patch, options.distance_threshold_m)
        if tolerance is None:
            continue
        level = np.quantile(height[patch], 0.98)
        near = patch & (abs(height - level) <= tolerance)
        if near.sum() < options.min_points:
            continue
        level = np.median(height[near])
        distance = abs(height - level)
        cap = patch & (distance <= tolerance)
        # A planar patch persists when the band narrows. An arbitrary contour
        # across a smoothly rounded surface loses most of its samples.
        if cap.sum() < options.min_inlier_ratio * np.count_nonzero(
            patch & (distance <= 2 * tolerance)
        ):
            continue
        exterior = ~binary_fill_holes(cap)
        lower = exterior & material[box] & (height < level - tolerance)
        edges[box] |= cap & binary_dilation(lower)
    return edges


def _raised_outer_rim(world, valid, mask, up, camera, seed, options, cancel):
    """Find an upper circular edge where the measured surface drops inside the mask."""
    band = options.distance_threshold_m
    material = mask & valid
    normal = seed[1] if seed is not None else up
    elevation = world @ normal
    raised = (
        material if seed is None else material & (elevation > seed[0] @ normal + band)
    )
    if not raised.any():
        return None
    neighbor = minimum_filter(
        np.where(material, elevation, np.inf),
        footprint=generate_binary_structure(2, 1),
        mode="constant",
        cval=np.inf,
    )
    caps = _tapered_cap_edges(world, material, raised, normal, options, cancel)
    steps = raised & ((elevation > neighbor + band) | caps)
    components, _ = label(steps, structure=np.ones((3, 3)))
    boxes = find_objects(components)
    counts = np.bincount(components.ravel())[1:]
    best = None
    best_height = -np.inf
    for index in np.flatnonzero(counts >= options.min_points):
        cancel()
        roi = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in boxes[index])
        step = components[roi] == index + 1
        xyz = world[roi]
        upper = rim(xyz[step], up, outer=True)
        if upper is None:
            continue
        origin, axis, _ = upper
        view = (camera - origin) @ axis
        if view <= 0:
            continue
        height = (xyz - origin) @ axis
        on_plane = material[roi] & (abs(height) <= band)
        # A lip can also drop inwards into an opening. Only its exterior
        # supplies samples for the outer ring.
        ledge = _visible_outer_edge(on_plane, valid[roi], mask[roi], height, view, band)
        edge = step & (ledge | (on_plane & caps[roi]))
        fit = _circle_on_plane(xyz[edge], origin, axis, options)
        if (
            fit.accepted
            and (level := np.dot(fit.parameters["center"], up)) > best_height
        ):
            if fit.quality["angular_coverage_deg"] < 235:
                fit.warn("partial_circular_arc")
            best, best_height = fit, level
    if best is None and seed is not None:
        # A measured circular flank above the silhouette rules out the shoulder.
        # Its cross-section alone does not locate an upper edge.
        crossing = raised & binary_dilation(material & ~binary_fill_holes(raised))
        crossing &= elevation <= seed[0] @ normal + 2 * band
        if (
            crossing.sum() >= options.min_points
            and _circle_on_plane(world[crossing], seed[0], normal, options).accepted
        ):
            return ShapeFit(reason="upper_ring_not_observed")
    return best


def _follows_circumference(offsets, normal, radius, width, options):
    xy = offsets @ np.column_stack(_basis(normal))
    near = xy[abs(np.linalg.norm(xy, axis=1) - radius) <= width]
    if len(near) < options.min_points:
        return False
    sectors, _ = np.histogram(
        np.arctan2(near[:, 1], near[:, 0]), bins=12, range=(-np.pi, np.pi)
    )
    return np.count_nonzero(sectors >= 2) * 30 >= options.min_circle_coverage_deg


def _openings_on_plane(world, valid, mask, plane, camera, options, cancel):
    origin, normal = plane
    height = (world - origin) @ normal
    band = options.distance_threshold_m
    level = valid & (abs(height) <= band)
    lip = mask & level
    lip |= binary_dilation(lip) & level
    lower = valid & (height < -band)
    clipped = mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
    # Keep openings separate even when their floors share the same height.
    enclosed = binary_fill_holes(lip) & ~lip
    void = ~lip if clipped else binary_fill_holes(mask) & ~lip
    regions, _ = label(void)
    boxes = find_objects(regions)
    adjacent = lower & binary_dilation(lip)
    for index in np.unique(regions[void & adjacent]):
        cancel()
        # Include a second lip pixel to measure the pitch across the boundary.
        box = tuple(slice(max(0, s.start - 2), s.stop + 2) for s in boxes[index - 1])
        region = regions[box] == index
        steps = _boundary_steps(lip[box], region & lower[box])
        boundary, pixels = _edge_samples(world[box], lip[box], steps)
        if len(boundary) < options.min_points:
            continue
        inward_half_steps = boundary - world[box][tuple(pixels.T)]
        fit = _fit_measured_circle(
            boundary,
            pixels,
            region.shape,
            plane,
            options,
            inward_half_steps=inward_half_steps,
        )
        if not fit.parameters:
            continue
        if not (enclosed[box] & region).any():
            # The mean pixel of a visible circular arc lies inside its projected
            # outline. A clipped exterior wall is on the other side of that arc.
            middle = np.rint(pixels.mean(axis=0)).astype(int)
            if not region[tuple(middle)]:
                continue
            radius = fit.parameters["radius"]
            radial = ray_plane_radius(
                world[box][region & lower[box]],
                camera,
                np.asarray(fit.parameters["center"]),
                normal,
            )
            if not (radial < radius).any():
                continue
        if fit.quality["angular_coverage_deg"] < 235:
            fit.warn("partial_circular_arc")
        yield fit


def _inner_ring_center(world, valid, mask, up, camera, options, cancel):
    band = options.distance_threshold_m
    elevation = world @ up
    neighbor = minimum_filter(
        np.where(valid, elevation, np.inf),
        footprint=generate_binary_structure(2, 1),
        mode="constant",
        cval=np.inf,
    )
    # Only boundary samples seed a lip plane. A large visible floor can
    # otherwise outnumber a one-pixel lip in every angular sector.
    edges = outer_boundary(mask, valid) | (mask & valid & (elevation > neighbor + band))
    contours, _ = label(edges, structure=np.ones((3, 3)))
    boxes = find_objects(contours)
    sizes = np.bincount(contours.ravel())[1:]
    planes = []
    fits = []
    for index in np.flatnonzero(sizes >= 6):
        cancel()
        box = boxes[index]
        points = world[box][contours[box] == index + 1]
        if any(
            np.all(abs((points - origin) @ normal) <= band / 2)
            for origin, normal in planes
        ):
            continue
        # At an oblique view a contour also samples the wall below the lip.
        # Use its upper arc to estimate the plane before tracing the opening.
        levels = points @ up
        keep = min(len(points), max(options.min_points // 2, len(points) // 4))
        top = np.partition(levels, -keep)[-keep]
        plane = _plane(points[levels >= top], up, band / 2)
        if plane is None or (camera - plane[0]) @ plane[1] <= 0:
            continue
        planes.append(plane)
        fits.extend(
            _openings_on_plane(world, valid, mask, plane, camera, options, cancel)
        )
    return _upper_center(fits, up, band, opening=True)


def _upper_center(fits, up, tolerance, *, opening=False):
    # A measured upper lip can be too incomplete to publish its center.
    # Its presence still rules out returning a deeper opening.
    partial_arc = {"partial_circular_arc", "insufficient_circle_angular_coverage"}
    candidates = [
        fit
        for fit in fits
        if fit.accepted
        or (opening and fit.warnings and partial_arc.issuperset(fit.warnings))
    ]
    if not candidates:
        if fits:
            return next((fit for fit in fits if fit.parameters), fits[0])
        return ShapeFit(
            reason="inner_ring_not_observed"
            if opening
            else "selected_face_not_observed"
        )
    highest = max(candidates, key=lambda fit: np.dot(fit.parameters["center"], up))
    center = np.asarray(highest.parameters["center"])
    for candidate in candidates:
        offset = center - candidate.parameters["center"]
        if offset @ up <= tolerance and np.linalg.norm(offset) > tolerance:
            return ShapeFit(
                reason="ambiguous_inner_rings" if opening else "ambiguous_upper_faces"
            )
    return highest


def _ring_center(world, valid, mask, up, camera, feature, options, cancel):
    if feature == "upper_inner_ring":
        return _inner_ring_center(world, valid, mask, up, camera, options, cancel)
    points = world[outer_boundary(mask, valid)]
    seed = rim(points, up, outer=True) if len(points) >= 7 else None
    higher = _raised_outer_rim(world, valid, mask, up, camera, seed, options, cancel)
    if higher is not None:
        return higher
    if seed is None:
        return ShapeFit(reason="circular_boundary_not_resolved")
    origin, normal, radius = seed
    delta = world - origin
    height = delta @ normal
    band = options.distance_threshold_m
    on_plane = mask & valid & (abs(height) <= band)
    view = (camera - origin) @ normal
    if view <= 0:
        # From below, an upper outer rim is visible only if a measured wall
        # extends down from its circumference. A flat underside has no wall.
        wall = delta[mask & valid & (height < -2 * band)]
        if not _follows_circumference(wall, normal, radius, band, options):
            return ShapeFit(reason="upper_ring_not_observed")
    # Higher material competes with the rim only if it follows its circumference.
    high = delta[mask & valid & (height > band)]
    if _follows_circumference(high, normal, radius, 2 * band, options):
        return ShapeFit(reason="upper_ring_not_observed")
    edge = _visible_outer_edge(on_plane, valid, mask, height, view, band)
    fit = _circle_on_plane(world[edge], origin, normal, options)
    if fit.parameters and fit.quality["angular_coverage_deg"] < 235:
        fit.warn("partial_circular_arc")
    return fit


def _edge_samples(world, on_face, steps):
    """Place a measured transition half a surface pixel beyond the last face sample."""
    samples, pixels = [], []
    for row, col, dr, dc in steps:
        previous = (
            np.clip(row - dr, 0, on_face.shape[0] - 1),
            np.clip(col - dc, 0, on_face.shape[1] - 1),
        )
        last = world[row, col]
        pitch = np.where(on_face[previous][:, None], last - world[previous], 0.0)
        samples.append(last + 0.5 * pitch)
        pixels.append(np.column_stack((row, col)))
    return np.concatenate(samples), np.concatenate(pixels)


def _fit_measured_circle(
    boundary, pixels, shape, plane, options, *, inward_half_steps=None
):
    """Fit the circle and check all connected edges for contradictory measurements."""
    origin, normal = plane
    fit = _circle_on_plane(
        boundary, origin, normal, options, inward_half_steps=inward_half_steps
    )
    if not fit.parameters:
        return fit
    center = np.asarray(fit.parameters["center"])
    radius = fit.parameters["radius"]
    tolerance = max(
        options.distance_threshold_m, options.max_relative_residual * radius
    )
    xy = (boundary - center) @ np.column_stack(_basis(normal))
    off_circle = abs(np.linalg.norm(xy, axis=1) - radius) > tolerance
    if not off_circle.any():
        return fit
    row, col = pixels[off_circle].T
    edge = np.zeros(shape, dtype=bool)
    edge[row, col] = True
    runs, _ = label(edge, structure=np.ones((3, 3)))
    sizes = np.bincount(runs[edge])
    run_ids = runs[row, col]
    offsets = xy[off_circle]
    if any(
        np.linalg.norm(np.ptp(offsets[run_ids == index], axis=0)) > 2 * tolerance
        for index in np.flatnonzero(sizes >= 3)
    ):
        fit.warn("noncircular_measured_edge")
    return fit


def _disk_center(world, valid, mask, on_face, face, camera, options):
    origin, normal, support = face
    band = options.distance_threshold_m
    height = (world - origin) @ normal
    # Resolve the start of a bevel when a coherent planar patch stands out
    # above depth noise. A few isolated mesh facets cannot define the disk.
    plane_band = band
    noise = _plane_noise_band(height, on_face, band)
    if noise is not None and noise < band:
        refined = _plane(world[on_face], normal, noise)
        if refined is not None:
            refined_height = (world - refined[0]) @ refined[1]
            flat = on_face & (abs(refined_height) <= noise)
            if flat.sum() >= options.min_inlier_ratio * on_face.sum():
                origin, normal = refined
                height, on_face, plane_band = refined_height, flat, noise
                support = support[abs((support - origin) @ normal) <= plane_band]
    camera_height = (camera - origin) @ normal
    # The cleaned cloud selects the plane. Its eroded perimeter would shrink
    # the radius, so measure the boundary on the original depth raster.
    steps = _outer_edge_steps(
        on_face, valid, mask, height, camera_height, plane_band, min_drop=band
    )
    boundary, pixels = _edge_samples(world, on_face, steps)
    if not len(boundary):
        return ShapeFit(reason="disk_perimeter_not_observed")
    fit = _fit_measured_circle(boundary, pixels, mask.shape, (origin, normal), options)
    if fit.parameters:
        radius = fit.parameters["radius"]
        center = np.asarray(fit.parameters["center"])
        offset = support - center
        radial_sq = np.einsum("ij,ij->i", offset, offset) - (offset @ normal) ** 2
        central = int((radial_sq < (0.5 * radius) ** 2).sum())
        fit.quality.update(surface_samples=len(support), central_face_support=central)
        if central < max(options.min_points // 3, len(support) / 8):
            fit.warn("disk_interior_not_observed")
        elif fit.accepted:
            below = valid & (height < -band)
            crossings = ray_plane_offsets(
                world[below], camera, center, normal
            ) @ np.column_stack(_basis(normal))
            if opening_covers_center(below, valid, crossings):
                fit.warn("disk_center_over_opening")
    return fit


def _rectangle_center(obs, mask, region, plane):
    face = observed_face(obs, region, object_mask=mask, plane=plane)
    if face is None:
        return ShapeFit(reason="rectangle_edges_not_observed")
    if face["clipped"] and not face["supported"].all():
        return ShapeFit(
            reason="rectangle_face_at_image_border",
            quality={
                "image_borders": face["image_borders"],
                "physical_edges": face["supported"].tolist(),
            },
        )
    if not face["supported"].all():
        return ShapeFit(reason="rectangle_edges_not_observed")
    return ShapeFit(
        accepted=True,
        parameters={
            "center": face["center"].tolist(),
            "side_lengths_m": sorted(face["sizes"].tolist(), reverse=True),
            "normal": plane[1].tolist(),
        },
        quality={
            "physical_edges": face["supported"].tolist(),
            "support_pixels": face["n_support"],
            "sample_pitch_m": face["pitch"],
            "adjacent_edge_samples": face["adjacent_edge_samples"],
            **({"image_borders": face["image_borders"]} if face["clipped"] else {}),
        },
    )


def _fits_on_plane(name, obs, mask, roi, face, options, cancel):
    world, valid, region = obs.world_xyz[roi], obs.valid_mask[roi], mask[roi]
    origin, normal, support = face
    tolerance = (
        options.distance_threshold_m if name == "disk_center" else PLANE_TOLERANCE_M
    )
    on_plane = region & valid & (abs((world - origin) @ normal) <= tolerance)
    components, _ = label(on_plane)
    counts = np.bincount(components.ravel())[1:]
    ids = np.flatnonzero(counts >= 25)
    matches = {}
    enclosed_components = set()
    for index in ids[np.argsort(-counts[ids], kind="stable")]:
        cancel()
        patch = components == index + 1
        if name == "rectangle_center":
            image_region = np.zeros_like(mask)
            image_region[roi] = patch
            fit = _rectangle_center(obs, mask, image_region, face[:2])
        else:
            if len(ids) > 1:
                support = object_points(
                    world,
                    valid,
                    patch,
                    obs.camera_to_world[:3, 2],
                    keep_small_faces=True,
                )
            fit = _disk_center(
                world,
                valid,
                region,
                patch,
                (origin, normal, support),
                obs.camera_to_world[:3, 3],
                options,
            )
        if fit.accepted and len(ids) > 1:
            # The same plane can also be visible through a hole in an occluder.
            # Count that patch once with the face that surrounds it.
            matches[index + 1] = fit
            holes = binary_fill_holes(patch) & ~patch
            enclosed_components.update(np.unique(components[holes]))
        else:
            yield fit
    for component, fit in matches.items():
        if component not in enclosed_components:
            yield fit


def fit_features(
    names: list[GeometryQuantity],
    feature: GeometryFeature,
    obs: GeometryObservation,
    mask: np.ndarray,
    cloud: np.ndarray,
    options: FitOptions,
    cancel: Callable[[], None],
) -> Iterator[tuple[GeometryQuantity, ShapeFit]]:
    """Fit each requested center on the selected object feature."""
    if len(cloud) < 25 and (feature not in RING_FEATURES or not mask.any()):
        for name in names:
            yield name, ShapeFit(reason="too_few_valid_object_points")
        return
    if feature == "supported_round_base":
        cancel()
        yield "base_center", fit_supported_base(obs, mask, cloud, options)
        return
    # A coplanar sample just outside the mask needs its own outward neighbor.
    box = find_objects(mask, max_label=1)[0]
    roi = tuple(slice(max(0, span.start - 2), span.stop + 2) for span in box)
    world, valid, region = obs.world_xyz[roi], obs.valid_mask[roi], mask[roi]
    up = obs.up_direction
    camera = obs.camera_to_world[:3, 3]
    if feature in RING_FEATURES:
        yield (
            "ring_center",
            _ring_center(world, valid, region, up, camera, feature, options, cancel),
        )
        return

    if feature != "upper_face":
        raise ValueError(f"Unsupported face feature {feature!r}")
    faces = [
        (origin, normal, support)
        for origin, normal, support in face_candidates(cloud, up, cancel)
        if (camera - origin) @ normal > 0
    ]
    for name in names:
        fits = [
            fit
            for face in faces
            for fit in _fits_on_plane(name, obs, mask, roi, face, options, cancel)
        ]
        yield name, _upper_center(fits, up, options.distance_threshold_m)


def feature_result(
    name: GeometryQuantity, obs: GeometryObservation, fit: ShapeFit
) -> dict[str, Any]:
    """Package the center and dimensions from one fit with their shared evidence."""
    params = fit.parameters
    field = {
        "value": params["center"] if fit.accepted else None,
        "definition": CENTER_DEFINITIONS[name],
        "unit": "m",
        "source_step": obs.step,
        "source_camera": obs.camera,
        "world_frame_id": obs.world_frame_id,
        "source_refs": dict(obs.source_refs),
        "evidence_status": "inferred" if fit.accepted else "unknown",
        "fit_accepted": fit.accepted,
        "warnings": fit.warnings,
        "quality": fit.quality,
        "reason": fit.reason,
    }
    if fit.accepted:
        if "normal" in params:
            field["normal_world"] = params["normal"]
        if "radius" in params:
            field["radius_m"] = params["radius"]
        elif "side_lengths_m" in params:
            field["side_lengths_m"] = params["side_lengths_m"]
    elif params:
        field["quality"] = {**fit.quality, "candidate_parameters": params}
    return field


def summarize_fits(data: dict, fields: dict) -> None:
    """Report the requested fits and their object-selection warnings."""
    object_warnings = data.get("warnings", [])
    warnings = object_warnings + [
        {"quantity": name, "reason": warning}
        for name, field in fields.items()
        for warning in field.get("warnings", ())
    ]
    missing = [
        name
        for name, field in fields.items()
        if not field.get("fit_accepted") or field.get("value") is None
    ]
    found = len(missing) < len(fields)
    if not found:
        status = "insufficient_evidence"
    elif missing:
        status = "partial"
    elif warnings:
        status = "warning"
    else:
        status = "usable"
    data.update(
        geometry=fields,
        found=found,
        status=status,
        warnings=warnings,
        evidence_status="inferred" if found else "unknown",
    )
    if missing:
        data["missing_geometry"] = missing
        data["reason"] = next(
            (fields[name]["reason"] for name in missing if fields[name].get("reason")),
            "requested_geometry_incomplete",
        )
