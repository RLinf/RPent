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

from toolkits.geometry.depth_foreground import local_depth_range
from toolkits.geometry.geometry import (
    FitOptions,
    ShapeFit,
    _basis,
    fit_circle,
    outer_boundary,
    ray_plane_radius,
)
from toolkits.geometry.measured_rectangle import PLANE_TOLERANCE_M, observed_face
from toolkits.geometry.rgbd_parts import face_candidates, object_points, rim
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


def _circle_on_plane(world, edge, origin, normal, options):
    """Fit observed boundary samples in the selected plane, with its upward normal."""
    points = world[edge]
    if len(points) > options.max_points:
        points = points[np.linspace(0, len(points) - 1, options.max_points, dtype=int)]
    offsets = (points - origin) @ normal
    fit = fit_circle(points - np.outer(offsets, normal), options)
    if fit.parameters:
        fit.parameters["normal"] = normal.tolist()
        fit.quality["radius"] = fit.parameters["radius"]
    return fit


def _visible_outer_edge(on_plane, valid, mask, height, camera_height, band):
    """Find where the object leaves the face plane or drops to the background."""
    exterior = valid & ~binary_fill_holes(on_plane)
    adjacent_step = binary_dilation(exterior & mask & (abs(height) > band))
    toward_camera = np.sign(camera_height) * height
    # Across a mask boundary, a coplanar neighbor or nearer occluder cannot
    # establish an edge. Compare the face pixel with its measured neighbors.
    background = minimum_filter(
        np.where(exterior & ~mask, toward_camera, np.inf),
        footprint=generate_binary_structure(2, 1),
        mode="constant",
        cval=np.inf,
    )
    return on_plane & valid & (adjacent_step | (toward_camera > background + band))


def _ring_center(world, valid, mask, cloud, up, camera, feature, options):
    outer = feature == "upper_outer_ring"
    if outer:
        silhouette = outer_boundary(mask, valid)
        points = world[silhouette]
    else:
        points = cloud
    seed = rim(points, up, outer=outer) if len(points) >= 7 else None
    if seed is None:
        return ShapeFit(reason="circular_boundary_not_resolved")
    origin, normal, radius = seed
    delta = world - origin
    height = delta @ normal
    band = options.distance_threshold_m
    on_plane = mask & valid & (abs(height) <= band)
    if outer:
        # Higher material competes with the rim only if it follows its circumference.
        high = delta[mask & valid & (height > 2 * band)] @ np.stack(
            _basis(normal), axis=1
        )
        near = high[abs(np.linalg.norm(high, axis=1) - radius) <= 2 * band]
        if len(near) >= options.min_points:
            angles = np.arctan2(near[:, 1], near[:, 0])
            sectors, _ = np.histogram(angles, bins=12, range=(-np.pi, np.pi))
            if np.count_nonzero(sectors >= 2) * 30 >= options.min_circle_coverage_deg:
                return ShapeFit(reason="upper_ring_not_observed")
        edge = _visible_outer_edge(
            on_plane, valid, mask, height, (camera - origin) @ normal, band
        )
    else:
        # An inner edge borders a measured drop inside the opening. Missing
        # depth inside a flat disk supplies no evidence of a hole.
        lower = valid & binary_dilation(on_plane) & (height < -band)
        if not any(edge.any() for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1])):
            lower &= binary_fill_holes(mask)
        inside = np.zeros_like(mask)
        inside[lower] = ray_plane_radius(world[lower], camera, origin, normal) < radius
        edge = on_plane & binary_dilation(inside)
        if not edge.any():
            return ShapeFit(reason="inner_ring_not_observed")
    fit = _circle_on_plane(world, edge, origin, normal, options)
    if fit.parameters and fit.quality["angular_coverage_deg"] < 235:
        fit.warn("partial_circular_arc")
    return fit


def _disk_center(world, valid, mask, on_face, face, camera, options):
    origin, normal, support = face
    band = options.distance_threshold_m
    height = (world - origin) @ normal
    camera_height = (camera - origin) @ normal
    # The cleaned cloud selects the plane. Its eroded perimeter would shrink
    # the radius, so measure the boundary on the original depth raster.
    edge = _visible_outer_edge(on_face, valid, mask, height, camera_height, band)
    if not edge.any():
        return ShapeFit(reason="disk_perimeter_not_observed")
    fit = _circle_on_plane(world, edge, origin, normal, options)
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
            # Rays through a visible hole reach a surface behind the fitted face.
            # Missing depth and material in front of it do not establish a hole.
            fit = _refine_shallow_disk(world, valid, mask, fit, camera, options) or fit
            center = np.asarray(fit.parameters["center"])
            near_center = ray_plane_radius(world[valid], camera, center, normal) <= min(
                band, 0.2 * fit.parameters["radius"]
            )
            behind = (
                np.sign(camera_height) * ((world[valid][near_center] - center) @ normal)
                < -band
            )
            if behind.sum() >= max(3, 0.75 * len(behind)):
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


def _refine_shallow_disk(world, valid, mask, fit, camera, options):
    """Return the enclosing rim fit at the measured center height, if connected."""
    if any(edge.any() for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1])):
        return
    radius = fit.parameters["radius"]
    center = np.asarray(fit.parameters["center"])
    normal = np.asarray(fit.parameters["normal"])
    band = options.distance_threshold_m
    material = mask & valid
    height = (world - center) @ normal
    low, high = np.quantile(height[material], [0.01, 0.99])
    if high - low > radius / 4:
        return
    edge = _visible_outer_edge(
        mask, valid, mask, height, (camera - center) @ normal, band
    )
    rim = _circle_on_plane(world, edge, center, normal, options)
    if (
        not rim.accepted
        or rim.quality["angular_coverage_deg"] < 330
        or rim.quality["inlier_ratio"] < 0.90
    ):
        return
    rim_center = np.asarray(rim.parameters["center"])
    rim_radius = rim.parameters["radius"]
    shift = np.linalg.norm(rim_center - center)
    if not (
        radius + max(shift, band) <= rim_radius <= 1.5 * radius and shift <= 2 * band
    ):
        return
    delta = world - rim_center
    central = material & (
        np.einsum("ijk,ijk->ij", delta, delta) - height**2 < (rim_radius / 2) ** 2
    )
    # A separate platform can have a concentric outline. Trace measured depth
    # across the face so a step or an occluder cannot supply the outer rim.
    smooth = material & (local_depth_range(height, material) <= band)
    components, _ = label(smooth)
    seeds = components[central & smooth & (abs(height) <= band)]
    if len(seeds) < options.min_points:
        return
    connected = components == np.bincount(seeds).argmax()
    middle = connected & central
    perimeter_share = float(binary_dilation(connected)[edge].mean())
    if middle.sum() < 0.9 * central.sum() or perimeter_share < 0.9:
        return
    rise = float(np.median(height[middle]))
    if abs(rise) > band:
        return
    # Translating the projection plane along its normal preserves radial residuals.
    rim.parameters["center"] = (rim_center + rise * normal).tolist()
    rim.quality.update(
        center_source="connected_shallow_rim",
        original_face_center=center.tolist(),
        original_face_quality=fit.quality,
        outer_center_on_plane=rim_center.tolist(),
        outer_radius_m=rim_radius,
        central_height_adjustment_m=rise,
        central_depth_pixels=int(middle.sum()),
        connected_perimeter_fraction=perimeter_share,
    )
    return rim


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
    for index in ids[np.argsort(-counts[ids], kind="stable")]:
        cancel()
        patch = components == index + 1
        if name == "rectangle_center":
            image_region = np.zeros_like(mask)
            image_region[roi] = patch
            yield _rectangle_center(obs, mask, image_region, face[:2])
        else:
            if len(ids) > 1:
                support = object_points(world, valid, patch, obs.camera_to_world[:3, 2])
            yield _disk_center(
                world,
                valid,
                region,
                patch,
                (origin, normal, support),
                obs.camera_to_world[:3, 3],
                options,
            )


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
    if len(cloud) < 25 and (feature != "upper_outer_ring" or not mask.any()):
        for name in names:
            yield name, ShapeFit(reason="too_few_valid_object_points")
        return
    if feature == "supported_round_base":
        cancel()
        yield "base_center", fit_supported_base(obs, mask, cloud, options)
        return
    # Keep a one-pixel halo for measured neighbors at the mask boundary.
    box = find_objects(mask, max_label=1)[0]
    roi = tuple(slice(max(0, span.start - 1), span.stop + 1) for span in box)
    world, valid, region = obs.world_xyz[roi], obs.valid_mask[roi], mask[roi]
    up = obs.up_direction
    camera = obs.camera_to_world[:3, 3]
    if feature in RING_FEATURES:
        yield (
            "ring_center",
            _ring_center(world, valid, region, cloud, up, camera, feature, options),
        )
        return

    if feature != "upper_face":
        raise ValueError(f"Unsupported face feature {feature!r}")
    faces = [
        (origin, normal, support)
        for origin, normal, support in face_candidates(cloud, up)
        if (camera - origin) @ normal > 0
    ]
    for name in names:
        best = None
        fits = (
            fit
            for face in faces
            for fit in _fits_on_plane(name, obs, mask, roi, face, options, cancel)
        )
        for fit in fits:
            if fit.accepted:
                best = fit
                break
            if best is None or (not best.parameters and fit.parameters):
                best = fit
        yield name, best or ShapeFit(reason="selected_face_not_observed")


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
