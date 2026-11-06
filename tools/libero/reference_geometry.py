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

"""Observable reference-point constructions used by the LIBERO test checklist."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tools.libero.geometry import (
    FitOptions,
    ShapeFit,
    _basis,
    _body_axes,
    _support_quality,
    fit_axisymmetric,
    fit_circle,
    fit_plane,
)

REFERENCE_PARTS = {
    "inner_rim_center": "inner_rim",
    "plate_center": "plate_surface",
    "lid_center": "lid",
    "rectangle_center": "rectangular_top",
    "body_segment_center": "bottle_segment",
}
REFERENCE_DESCRIPTIONS = {
    "inner_rim": "the visible INNER upper opening edge, excluding outer lip, handle and interior floor, of",
    "plate_surface": "the entire central flat circular supporting surface INSIDE the raised rim of",
    "lid": "the entire sealed circular top lid surface, excluding its raised rolled outer lip, of",
    "rectangular_top": "the entire main rectangular TOP face up to its four physical edges, excluding side faces, doors and printed-label-only crops, of",
    "bottle_segment": "a visible material point on the main cylindrical barrel, excluding shoulder, neck, cap and mouth, of",
}
REFERENCE_DEFINITIONS = {
    "inner_rim_center": "3d_center_in_observed_inner_upper_opening_edge_plane",
    "plate_center": "circular_support_surface_center_at_fitted_support_plane_height",
    "lid_center": "sealed_circular_lid_center_in_fitted_lid_plane",
    "rectangle_center": "center_of_complete_observed_main_rectangular_top_face",
    "body_segment_center": "axis_midpoint_of_observed_main_cylindrical_barrel_excluding_neck_and_mouth",
}


@dataclass
class ReferenceFit:
    """Fit input and raw support are saved independently for reproducibility."""

    fit: ShapeFit
    points: np.ndarray
    observed_points: np.ndarray
    boundary_mask: np.ndarray
    extraction: dict[str, Any] = field(default_factory=dict)


def _outer_boundary(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion, binary_fill_holes

    footprint = binary_fill_holes(mask)
    return mask & ~binary_erosion(footprint)


def _touches_frame(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
    )


def inner_rim_candidates(
    mask: np.ndarray,
    world: np.ndarray,
    anchor: np.ndarray | None,
    *,
    optical_depth: np.ndarray | None,
    options: FitOptions,
    image: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a material inner contour using holes or interior foreground jumps.

    The opening's center-pixel depth is never a candidate. Without an inner-edge
    cue this method abstains rather than renaming the outer rim as the inner rim.
    """
    from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, label

    holes, count = label(binary_fill_holes(mask) & ~mask)
    choices = []
    for index in range(1, count + 1):
        hole = holes == index
        if hole.sum() < 10:
            continue
        boundary = mask & binary_dilation(hole)
        points = world[boundary]
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < options.min_points:
            continue
        distance = (
            float(np.min(np.linalg.norm(points - anchor, axis=1)))
            if anchor is not None
            else np.inf
        )
        choices.append((distance, boundary))
    if choices and anchor is not None:
        choices.sort(key=lambda item: item[0])
        if choices[0][0] <= 3 * options.distance_threshold_m:
            if (
                len(choices) > 1
                and choices[1][0] - choices[0][0] < options.voxel_size_m
            ):
                return np.zeros_like(mask), {
                    "reason": "ambiguous_inner_hole_at_visual_anchor"
                }
            return choices[0][1], {"method": "mask_inner_hole_material_boundary"}
    if optical_depth is not None:
        inner = binary_erosion(mask, iterations=2)
        edge = np.zeros_like(mask)
        for axis in (0, 1):
            for shift in (-1, 1):
                neighbor = np.roll(optical_depth, shift, axis=axis)
                same_object = np.roll(mask, shift, axis=axis)
                edge |= (
                    inner
                    & same_object
                    & np.isfinite(neighbor)
                    & (neighbor - optical_depth > 2 * options.distance_threshold_m)
                )
        if edge.sum() >= options.min_points:
            return edge, {
                "method": "interior_foreground_depth_jump_excluding_outer_silhouette"
            }
    if image is not None and anchor is not None:
        interior = binary_erosion(mask, iterations=2)
        gray = np.asarray(image[..., :3], dtype=float).mean(axis=2) / 255
        strength = np.hypot(*np.gradient(gray))
        cutoff = (
            max(0.04, float(np.quantile(strength[mask], 0.8))) if mask.any() else 0.04
        )
        components, count = label(
            interior & (strength >= cutoff), structure=np.ones((3, 3))
        )
        choices = []
        for index in range(1, count + 1):
            boundary = components == index
            points = world[boundary]
            points = points[np.isfinite(points).all(axis=1)]
            if len(points) >= options.min_points:
                choices.append(
                    (float(np.min(np.linalg.norm(points - anchor, axis=1))), boundary)
                )
        choices.sort(key=lambda item: item[0])
        if choices and choices[0][0] <= 3 * options.distance_threshold_m:
            if (
                len(choices) > 1
                and choices[1][0] - choices[0][0] < options.voxel_size_m
            ):
                return np.zeros_like(mask), {
                    "reason": "ambiguous_inner_rgb_contour_at_visual_anchor"
                }
            return choices[0][1], {
                "method": "interior_rgb_contour_at_semantic_inner_edge_anchor",
                "gradient_cutoff": cutoff,
                "assumption": "visual anchor and its connected interior contour identify the inner upper opening edge",
            }
    return np.zeros_like(mask), {
        "reason": "inner_edge_not_resolved_from_outer_lip_and_floor"
    }


def _plane_mask(
    world: np.ndarray,
    mask: np.ndarray,
    cloud: np.ndarray,
    plane: ShapeFit,
    options: FitOptions,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    finite = np.isfinite(world).all(axis=2) & mask
    point, normal = (
        np.asarray(plane.parameters["point"]),
        np.asarray(plane.parameters["normal"]),
    )
    result = finite & (np.abs((world - point) @ normal) <= options.distance_threshold_m)
    if result.any():
        distance, _ = cKDTree(cloud).query(world[result])
        result[result] &= distance <= max(
            4 * options.distance_threshold_m, 3 * options.voxel_size_m
        )
    return result


def physical_plane_boundary(
    world: np.ndarray,
    face: np.ndarray,
    plane: ShapeFit,
    camera_origin: np.ndarray | None,
    options: FitOptions,
    *,
    reject_foreground: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Require a surface transition at a face edge; reject cuts/foreground covers."""
    boundary = _outer_boundary(face)
    supported = np.zeros_like(face)
    point, normal = (
        np.asarray(plane.parameters["point"]),
        np.asarray(plane.parameters["normal"]),
    )
    for axis in (0, 1):
        for shift in (-2, -1, 1, 2):
            neighbor = np.roll(world, shift, axis=axis)
            outside = ~np.roll(face, shift, axis=axis)
            finite = np.isfinite(neighbor).all(axis=2)
            transition = (
                np.abs((neighbor - point) @ normal) > options.distance_threshold_m
            )
            if reject_foreground:
                if camera_origin is None:
                    continue
                ray = neighbor - camera_origin
                denominator = ray @ normal
                fraction = np.divide(
                    float((point - camera_origin) @ normal),
                    denominator,
                    out=np.full(face.shape, np.nan),
                    where=np.abs(denominator) > 1e-12,
                )
                # s=1 is the observed neighbor; s>1 places it in front of the
                # extrapolated face, so it is an occluder, not a face edge.
                transition &= np.isfinite(fraction) & (fraction > 0) & (fraction < 1)
            supported |= boundary & outside & finite & transition
    supported[[0, -1], :] = False
    supported[:, [0, -1]] = False
    return supported, {
        "method": "observed_surface_transition",
        "reject_foreground": reject_foreground,
        "mask_touches_frame": _touches_frame(face),
        "n_boundary_pixels": int(boundary.sum()),
        "n_supported_boundary_pixels": int(supported.sum()),
    }


def fit_rectangle(
    points: np.ndarray,
    boundary_points: np.ndarray,
    options: FitOptions | None = None,
    *,
    plane: ShapeFit | None = None,
) -> ShapeFit:
    """Fit a rectangle only when four observed physical edges support its center."""
    from scipy.spatial import ConvexHull, QhullError

    options = options or FitOptions()
    plane = plane or fit_plane(points, options)
    result = ShapeFit(
        "rectangle",
        assumptions=[
            "one planar rectangular face",
            "all four physical edges and corner neighborhoods are observed",
        ],
    )
    if not plane.accepted:
        result.reason = plane.reason
        return result
    if len(boundary_points) < options.min_points:
        result.reason = "insufficient_physical_rectangle_boundary"
        return result
    origin = np.asarray(plane.parameters["point"])
    normal = np.asarray(plane.parameters["normal"])
    basis = np.column_stack(_basis(normal))
    xy = (points - origin) @ basis
    edges = (boundary_points - origin) @ basis
    try:
        hull = xy[ConvexHull(xy).vertices]
    except QhullError:
        result.reason = "degenerate_rectangle_support"
        return result
    vectors = np.roll(hull, -1, axis=0) - hull
    angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi / 2)
    angles = np.unique(np.round(angles, 5))
    if len(angles) > 128:
        angles = angles[np.linspace(0, len(angles) - 1, 128).astype(int)]
    candidates = []
    for angle in angles:
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        projected, boundary = xy @ rotation, edges @ rotation
        low, high = np.quantile(projected, [0.002, 0.998], axis=0)
        extent = high - low
        if min(extent) < 4 * options.distance_threshold_m:
            continue
        side_distances = np.column_stack(
            (
                np.abs(boundary[:, 0] - low[0]),
                np.abs(boundary[:, 0] - high[0]),
                np.abs(boundary[:, 1] - low[1]),
                np.abs(boundary[:, 1] - high[1]),
            )
        )
        nearest = side_distances.argmin(axis=1)
        residual = side_distances.min(axis=1)
        inliers = residual <= options.distance_threshold_m
        coverage = []
        for side in range(4):
            samples = boundary[inliers & (nearest == side), 1 if side < 2 else 0]
            coverage.append(
                float(np.ptp(samples) / extent[1 if side < 2 else 0])
                if len(samples) >= 6
                else 0.0
            )
        corners = np.array(
            [[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]]
        )
        corner_distance = np.array(
            [np.linalg.norm(boundary - corner, axis=1).min() for corner in corners]
        )
        if (
            min(coverage) < 0.65
            or max(corner_distance)
            > max(2 * options.distance_threshold_m, 0.08 * min(extent))
            or inliers.mean() < options.min_inlier_ratio
        ):
            continue
        candidates.append(
            (
                float(np.mean(residual[inliers])),
                rotation,
                low,
                high,
                inliers,
                residual,
                coverage,
                corners,
                corner_distance,
            )
        )
    if not candidates:
        result.reason = "four_complete_physical_edges_not_observed"
        result.quality = {
            "plane": plane.quality,
            "n_boundary_points": len(boundary_points),
        }
        return result
    _, rotation, low, high, inliers, residual, coverage, corners, corner_distance = min(
        candidates, key=lambda item: item[0]
    )
    result.accepted = True
    result.inlier_indices = np.flatnonzero(inliers)
    result.parameters = {
        "center": (origin + basis @ rotation @ ((low + high) / 2)).tolist(),
        "normal": normal.tolist(),
        "size": (high - low).tolist(),
        "corners": (corners @ rotation.T @ basis.T + origin).tolist(),
    }
    result.quality = _support_quality(residual, inliers, options)
    result.quality.update(
        edge_coverage=coverage,
        corner_distances_m=corner_distance.tolist(),
        plane=plane.quality,
    )
    return result


def fit_barrel_body(
    points: np.ndarray,
    anchor: np.ndarray | None,
    options: FitOptions,
) -> ShapeFit:
    """Seed the dominant cylinder at the visual barrel point, then expand support.

    Whole-bottle masks contain a shoulder and neck that are not one cylinder.
    An anchor-centred axial band supplies a hypothesis; acceptance still requires
    the normal consensus/coverage gates and a majority of the complete cloud.
    """
    if anchor is None or len(points) < options.min_points:
        return ShapeFit("cylinder", reason="main_barrel_material_anchor_required")
    candidates = []
    failed = ShapeFit("cylinder", reason="no_dominant_barrel_at_visual_anchor")
    for direction in _body_axes(points)[:2]:
        axial = (points - anchor) @ direction
        half_width = max(6 * options.distance_threshold_m, 0.18 * float(np.ptp(axial)))
        band = points[np.abs(axial) <= half_width]
        local = fit_axisymmetric(band, options)
        if not local.accepted or abs(local.parameters["radius_slope"]) > 0.12:
            continue
        origin = np.asarray(local.parameters["axis_point"])
        axis = np.asarray(local.parameters["axis_direction"])
        radius = local.parameters["radius_at_visible_midpoint"]
        offset = points - origin
        t = offset @ axis
        residual = np.abs(np.linalg.norm(offset - t[:, None] * axis, axis=1) - radius)
        support = np.flatnonzero(residual <= options.distance_threshold_m)
        if (
            len(support) < options.min_points
            or len(support) / len(points) < options.min_inlier_ratio
        ):
            continue
        fitted = fit_axisymmetric(points[support], options)
        if not fitted.accepted or abs(fitted.parameters["radius_slope"]) > 0.12:
            continue
        axis = np.asarray(fitted.parameters["axis_direction"])
        delta = anchor - np.asarray(fitted.parameters["axis_point"])
        if (
            abs(
                np.linalg.norm(delta - (delta @ axis) * axis)
                - fitted.parameters["radius_at_visible_midpoint"]
            )
            > 2 * options.distance_threshold_m
        ):
            continue
        fitted.inlier_indices = support[fitted.inlier_indices]
        fitted.quality["barrel_consensus_on_complete_mask"] = len(
            fitted.inlier_indices
        ) / len(points)
        fitted.quality["anchor_band_half_width_m"] = half_width
        candidates.append(fitted)
    return (
        max(candidates, key=lambda fit: len(fit.inlier_indices))
        if candidates
        else failed
    )


def fit_barrel_segment(
    points: np.ndarray,
    terminal_points: np.ndarray,
    anchor: np.ndarray | None,
    options: FitOptions | None = None,
    *,
    body: ShapeFit | None = None,
) -> ShapeFit:
    """Locate both barrel limits from radius transitions or certified object ends.

    An observed cloud's axial min/max alone never certifies an endpoint. Model
    coordinates and pre-labelled barrel bounds are evaluator-only data.
    """
    options = options or FitOptions()
    body = body or fit_barrel_body(points, anchor, options)
    result = ShapeFit(
        "bounded_barrel",
        assumptions=[
            "one dominant near-cylindrical main barrel",
            "barrel bounds are observed shoulder/base transitions or physical terminal edges",
            "no model-labelled bounds supplied to the agent",
        ],
    )
    if not body.accepted:
        result.reason = body.reason
        return result
    axis = np.asarray(body.parameters["axis_direction"])
    origin = np.asarray(body.parameters["axis_point"])
    radius = body.parameters["radius_at_visible_midpoint"]
    if abs(body.parameters["radius_slope"]) > 0.12:
        result.reason = "main_barrel_not_near_cylindrical"
        return result
    delta = points - origin
    axial = delta @ axis
    radial = np.linalg.norm(delta - axial[:, None] * axis, axis=1)
    width = max(2 * options.distance_threshold_m, float(np.ptp(axial)) / 48)
    bin_edges = np.arange(axial.min() - 1e-9, axial.max() + width, width)
    profile = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        samples = radial[(axial >= left) & (axial < right)]
        profile.append(
            float(np.quantile(samples, 0.8)) if len(samples) >= 4 else np.nan
        )
    profile = np.asarray(profile)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    expected = radius + body.parameters["radius_slope"] * centers
    stable = np.isfinite(profile) & (
        np.abs(profile - expected)
        <= max(0.5 * options.distance_threshold_m, 0.03 * radius)
    )
    seed_t = float((anchor - origin) @ axis) if anchor is not None else 0.0
    seed = int(np.argmin(np.abs(centers - seed_t)))
    if not stable[seed]:
        result.reason = "visual_anchor_not_on_stable_barrel_profile"
        return result
    low = high = seed
    while low > 0 and stable[low - 1]:
        low -= 1
    while high + 1 < len(stable) and stable[high + 1]:
        high += 1
    terminal_t = (
        (terminal_points - origin) @ axis if len(terminal_points) else np.empty(0)
    )
    bounds, evidence = [], []
    for index, side in ((low, -1), (high, 1)):
        bound = (
            bin_edges[index]
            if side < 0
            else min(bin_edges[index + 1], float(axial.max()))
        )
        outside = (
            np.arange(max(0, index - 4), index)
            if side < 0
            else np.arange(index + 1, min(len(profile), index + 5))
        )
        outside = outside[np.isfinite(profile[outside])]
        transition = (
            len(outside) >= 2
            and np.count_nonzero(
                profile[outside] < expected[outside] - options.distance_threshold_m
            )
            >= 2
        )
        terminal = np.count_nonzero(np.abs(terminal_t - bound) <= 1.5 * width) >= 6
        if transition:
            slope, intercept = np.polyfit(
                centers[outside], profile[outside] - expected[outside], 1
            )
            if abs(slope) > 1e-6:
                crossing = -intercept / slope
                if abs(crossing - bound) <= 2 * width:
                    bound = float(crossing)
        if not (transition or terminal):
            result.reason = "both_main_barrel_end_boundaries_not_observed"
            result.quality = {"profile_bin_width_m": width, "body": body.quality}
            return result
        bounds.append(float(bound))
        evidence.append("radius_transition" if transition else "physical_terminal_edge")
    if bounds[1] - bounds[0] < 4 * options.distance_threshold_m:
        result.reason = "main_barrel_segment_too_short"
        return result
    result.accepted = True
    result.parameters = {
        "center": (origin + np.mean(bounds) * axis).tolist(),
        "axis_direction": axis.tolist(),
        "axial_bounds": bounds,
        "endpoints": [(origin + value * axis).tolist() for value in bounds],
        "radius": radius,
    }
    result.inlier_indices = body.inlier_indices[
        (axial[body.inlier_indices] >= bounds[0])
        & (axial[body.inlier_indices] <= bounds[1])
    ]
    result.quality = {
        **body.quality,
        "profile_bin_width_m": width,
        "endpoint_evidence": evidence,
        "n_barrel_inliers": len(result.inlier_indices),
    }
    return result


def _barrel_boundary(
    world: np.ndarray,
    mask: np.ndarray,
    body: ShapeFit,
    camera_origin: np.ndarray | None,
    options: FitOptions,
) -> np.ndarray:
    """Certify outer-mask ends against context, excluding continuing barrel/occluders."""
    result = np.zeros_like(mask)
    if camera_origin is None or _touches_frame(mask):
        return result
    boundary = _outer_boundary(mask)
    origin, axis = (
        np.asarray(body.parameters["axis_point"]),
        np.asarray(body.parameters["axis_direction"]),
    )
    radius = body.parameters["radius_at_visible_midpoint"]
    relative = camera_origin - origin
    o = relative - (relative @ axis) * axis
    for dim in (0, 1):
        for shift in (-1, 1):
            neighbor = np.roll(world, shift, axis=dim)
            outside = ~np.roll(mask, shift, axis=dim)
            inside = np.roll(world, -shift, axis=dim)
            inward = world - inside
            # A longitudinal silhouette edge is not an axial barrel end.
            # Require an observed step inward along the fitted cylinder axis.
            end_facing = np.roll(mask, -shift, axis=dim) & (
                np.abs(inward @ axis) > 0.6 * np.linalg.norm(inward, axis=2)
            )
            ray = neighbor - camera_origin
            d = ray - (ray @ axis)[..., None] * axis
            a = np.sum(d * d, axis=2)
            b = 2 * (d @ o)
            c = float(o @ o - radius * radius)
            discriminant = b * b - 4 * a * c
            near = np.divide(
                -b - np.sqrt(np.maximum(0, discriminant)),
                2 * a,
                out=np.full(mask.shape, np.nan),
                where=a > 1e-12,
            )
            offset = neighbor - origin
            radial = np.linalg.norm(offset - (offset @ axis)[..., None] * axis, axis=2)
            physical = np.isfinite(neighbor).all(axis=2) & (
                np.abs(radial - radius) > 2 * options.distance_threshold_m
            )
            foreground = (
                (discriminant >= 0)
                & (near > 1)
                & (
                    near * np.linalg.norm(ray, axis=2) - np.linalg.norm(ray, axis=2)
                    > options.distance_threshold_m
                )
            )
            result |= boundary & outside & end_facing & physical & ~foreground
    return result


def fit_reference(
    points: np.ndarray,
    part: str,
    world: np.ndarray,
    mask: np.ndarray,
    anchor: np.ndarray | None,
    camera_origin: np.ndarray | None,
    options: FitOptions | None = None,
) -> ReferenceFit:
    """Fit a checklist reference without substituting a visible-fragment center."""
    options = options or FitOptions()
    empty = np.zeros_like(mask)
    if anchor is None or not np.isfinite(anchor).all():
        return ReferenceFit(
            ShapeFit(part, reason="visual_material_anchor_has_no_valid_depth"),
            points,
            points,
            empty,
        )
    if part == "inner_rim":
        fit = fit_circle(points, options, anchor=anchor, camera_origin=camera_origin)
        fit.assumptions.append("candidates are the resolved inner upper opening edge")
        return ReferenceFit(fit, points, points, mask)
    if part == "bottle_segment":
        body = fit_barrel_body(points, anchor, options)
        boundary = (
            _barrel_boundary(world, mask, body, camera_origin, options)
            if body.accepted
            else empty
        )
        fit = fit_barrel_segment(points, world[boundary], anchor, options, body=body)
        return ReferenceFit(
            fit,
            points,
            points,
            boundary,
            {"method": "barrel_radius_profile_and_observed_object_ends"},
        )
    plane = fit_plane(points, options, camera_origin=camera_origin)
    if not plane.accepted:
        return ReferenceFit(plane, points, points, empty)
    origin, normal = (
        np.asarray(plane.parameters["point"]),
        np.asarray(plane.parameters["normal"]),
    )
    if (
        anchor is not None
        and abs(float((anchor - origin) @ normal)) > 3 * options.distance_threshold_m
    ):
        return ReferenceFit(
            ShapeFit(
                "face", reason="selected_plane_does_not_contain_visual_face_anchor"
            ),
            points,
            points,
            empty,
        )
    face = _plane_mask(world, mask, points, plane, options)
    boundary, extraction = physical_plane_boundary(
        world,
        face,
        plane,
        camera_origin,
        options,
        reject_foreground=part == "rectangular_top",
    )
    observed = world[boundary]
    if extraction["mask_touches_frame"]:
        return ReferenceFit(
            ShapeFit("face", reason="face_boundary_clipped_by_image"),
            observed,
            observed,
            boundary,
            extraction,
        )
    if part == "rectangular_top":
        surface_points = points[
            np.abs((points - origin) @ normal) <= options.distance_threshold_m
        ]
        fit = fit_rectangle(surface_points, observed, options, plane=plane)
        return ReferenceFit(fit, observed, observed, boundary, extraction)
    projected = observed - ((observed - origin) @ normal)[:, None] * normal
    fit = fit_circle(
        projected,
        options,
        anchor=anchor,
        camera_origin=camera_origin,
        anchor_on_boundary=False,
    )
    fit.model = "circular_face"
    fit.assumptions = [
        "selected central support/lid face is planar and circular",
        "height comes from the selected face plane, not the raised outer rim",
    ]
    fit.quality["face_plane"] = plane.quality
    extraction["fit_input"] = "observed_face_boundary_projected_onto_fitted_face_plane"
    return ReferenceFit(fit, projected, observed, boundary, extraction)


def reference_fields(fits: dict[str, ShapeFit], requested: list[str]) -> dict[str, Any]:
    """Expose only accepted reference centers with their exact semantic definitions."""
    result = {}
    for name in requested:
        part = REFERENCE_PARTS[name]
        fit = fits.get(part)
        accepted = fit is not None and fit.accepted
        result[name] = {
            "value": fit.parameters["center"] if accepted else None,
            "definition": REFERENCE_DEFINITIONS[name],
            "unit": "m",
            "evidence_status": "inferred" if accepted else "unknown",
            "source_parts": [part],
            "assumptions": fit.assumptions if fit is not None else [],
            "reason": None
            if accepted
            else fit.reason
            if fit is not None
            else "reference_part_not_observed",
            "quality": {part: fit.quality} if fit is not None else {},
        }
    return result
