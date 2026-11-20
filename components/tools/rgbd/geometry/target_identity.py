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


"""Check the identity of the exact RGB-D region used for a measurement."""

import io
import re

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    find_objects,
    gaussian_filter,
    label,
)

from components.tools.rgbd.geometry.chromatic_region import described_hue
from components.tools.rgbd.geometry.depth_foreground import local_depth_range


def target_subject(query: str) -> str:
    """Separate the material to identify in a crop from its scene landmarks."""
    parts = re.split(
        r"\b(?:between|(?:(?:to|on)\s+(?:the\s+)?)?(?:left|right)\s+of"
        r"|next\s+to|in\s+front\s+of)\b",
        query,
        maxsplit=1,
        flags=re.I,
    )
    if len(parts) == 1:
        return query.strip()
    return re.sub(
        r"\s+(?:(?:that|which)\s+)?(?:is\s+)?(?:sitting|standing|lying|located|positioned|placed|is)$",
        "",
        parts[0].strip().rstrip(","),
        flags=re.I,
    ).strip()


def box_color(query):
    """Only an explicit single-colored cuboid supports a shape-and-color search."""
    if re.search(r"\b(cube|cuboid|block|box)\b", query.lower()):
        return described_hue(query)
    return None


def material_pixel(mask, anchor):
    """Select a measured interior pixel close to the visual proposal."""
    slices = find_objects(mask, max_label=1)[0]
    local = mask[slices]
    dist = distance_transform_edt(np.pad(local, 1))[1:-1, 1:-1]
    rr, cc = np.nonzero(local & (dist >= min(3, dist.max())))
    rr += slices[0].start
    cc += slices[1].start
    r, c = anchor if anchor is not None else (np.median(rr), np.median(cc))
    i = np.argmin((rr - r) ** 2 + (cc - c) ** 2)
    return [int(rr[i]), int(cc[i])]


def _views(rgb, mask, slices, *, full_scene=False):
    y, x = slices
    pad = max(6, round(max(y.stop - y.start, x.stop - x.start) * 0.30))
    # The first panel contains the measured component. A narrow halo restores
    # the native contour removed by the depth-edge filter.
    near = np.s_[max(0, y.start - 6) : y.stop + 6, max(0, x.start - 6) : x.stop + 6]
    isolate = binary_dilation(mask[near], iterations=2)
    tile = Image.fromarray(
        np.where(isolate[..., None], rgb[near], 155).astype(np.uint8)
    )
    box = (
        max(0, x.start - pad),
        max(0, y.start - pad),
        min(rgb.shape[1], x.stop + pad),
        min(rgb.shape[0], y.stop + pad),
    )
    context = Image.fromarray(rgb)
    if full_scene:
        width = max(2, round(max(rgb.shape[:2]) / 200))
        ImageDraw.Draw(context).rectangle(
            (x.start, y.start, x.stop - 1, y.stop - 1),
            outline=(0, 255, 255),
            width=width,
        )
    else:
        context = context.crop(box)
    result = Image.new("RGB", (640, 320), (155, 155, 155))
    for i, view in enumerate((tile, context)):
        result.paste(
            ImageOps.pad(view, (320, 320), color=(155, 155, 155)), (320 * i, 0)
        )
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def _isolated_view(rgb, mask, slices):
    y, x = slices
    roi = np.s_[max(0, y.start - 5) : y.stop + 5, max(0, x.start - 5) : x.stop + 5]
    material = mask[roi]
    isolated = Image.fromarray(
        np.where(material[..., None], rgb[roi], 155).astype(np.uint8)
    )
    image = ImageOps.pad(isolated, (320, 320), color=(155, 155, 155))
    rr, cc = np.indices(material.shape, dtype=np.int32)
    indices = (rr + roi[0].start) * mask.shape[1] + cc + roi[1].start + 1
    support = ImageOps.pad(
        Image.fromarray(np.where(material, indices, 0)),
        image.size,
        method=Image.Resampling.NEAREST,
        color=0,
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), np.asarray(support)


def _agrees(answer, word):
    return (
        isinstance(answer, str)
        and re.fullmatch(r"\s*" + word + r"[.!]?\s*", answer, re.I) is not None
    )


def _name_terms(text):
    # Retain nouns such as "soap" or "cheese". A yes/no answer primed with the
    # requested category can misidentify a visually similar unmarked object.
    return {
        w
        for w in re.findall(r"[^\W\d_]+", text.lower())
        if w not in {"a", "an", "the", "of"}
    }


def _named_with_context(rgb, region, slices, verify):
    """Recheck a nameless cutout with its narrow, original photographic surround."""
    y, x = slices
    margin = max(5, round(min(y.stop - y.start, x.stop - x.start) * 0.15))
    box = np.s_[
        max(0, y.start - margin) : y.stop + margin,
        max(0, x.start - margin) : x.stop + margin,
    ]
    tile = rgb[box].copy()
    outside = ~region[box]
    tile[outside] = np.rint(0.6 * tile[outside] + 0.4 * 155).astype(np.uint8)
    image = ImageOps.pad(Image.fromarray(tile), (320, 320), color=(155, 155, 155))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return _name_terms(verify(buf.getvalue(), None, isolated=True) or "")


def _footprint_labels(world, raised, region, axes, span):
    """Connect separated visible surfaces of one object in the support plane."""
    xy = world @ axes
    center = np.median(xy[region], axis=0)
    radius = min(1.2, max(0.30, 3 * span))
    pitch = max(0.0025, radius / 160, span / 28)
    local = raised & (np.max(abs(xy - center), axis=2) < radius)
    grid = np.zeros((int(2 * radius / pitch) + 2,) * 2, dtype=bool)
    bins = np.floor((xy[local] - center + radius) / pitch).astype(int)
    ij = tuple(bins.T)
    grid[ij] = True
    cell_labels, _ = label(binary_dilation(grid), structure=np.ones((3, 3), bool))
    labels = np.zeros(region.shape, dtype=np.int32)
    labels[local] = cell_labels[ij]
    return labels


def similar_instances(observation, region, support):
    """Find disjoint objects with comparable metric extents and RGB distributions."""
    if (
        support is None
        or observation.camera_to_world is None
        or observation.up_direction is None
    ):
        return
    from components.tools.rgbd.geometry.geometry import _basis

    world, up = observation.world_xyz, observation.up_direction
    cam = observation.camera_to_world
    depth = (world - cam[:3, 3]) @ cam[:3, 2]
    raised = observation.valid_mask & (world @ up > support + 0.002)
    labels, _ = label(
        raised & (local_depth_range(depth, observation.valid_mask) < 0.012)
    )
    axes = np.stack(_basis(up), axis=1)

    def signature(points, colors, hues):
        xy = points @ axes
        _, _, basis = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)
        extent = np.diff(np.quantile(xy @ basis.T, [0.01, 0.99], axis=0), axis=0)[0]
        height = np.ptp(np.quantile(points @ up, [0.01, 0.99]))
        palette = np.bincount(hues, minlength=10) / len(hues)
        return (
            np.r_[np.sort(extent), height],
            palette,
            np.quantile(colors, [0.2, 0.5, 0.8], axis=0),
        )

    rgb = observation.rgb
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=float)
    hue_bins = np.where(
        hsv[:, :, 1] > 0.22 * 255,
        1 + (((hsv[:, :, 0] * 360 / 255 + 20) % 360) / 40).astype(int),
        0,
    )
    count = int(region.sum())
    dims, palette, colors = signature(world[region], rgb[region], hue_bins[region])
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    contrast = np.abs(luma - gaussian_filter(luma, 1.25)) / (luma + 12)

    def texture(mask, raster):
        interior = binary_erosion(mask, iterations=2)
        return np.quantile(raster[interior], 0.75) if interior.any() else None

    initial_texture = texture(region, contrast)
    center_xy = np.median(world[region] @ axes, axis=0)
    checked = region.copy()
    for components in (labels, _footprint_labels(world, raised, region, axes, dims[1])):
        regions = find_objects(components)
        sizes = np.bincount(components.ravel())
        for i in np.flatnonzero(
            (sizes >= max(100, 0.20 * count)) & (sizes <= 2.5 * count)
        ):
            if i == 0:
                continue
            box = regions[i - 1]
            other = components[box] == i
            if np.any(other & checked[box]):
                continue
            extents, pigments, appearance = signature(
                world[box][other], rgb[box][other], hue_bins[box][other]
            )
            other_texture = texture(other, contrast[box])
            if initial_texture is not None and other_texture is not None:
                low, high = sorted((initial_texture, other_texture))
                if high > 0.12 and high > 2.5 * max(low, 0.015):
                    continue
            if (
                np.mean(abs(appearance - colors)) > 18
                or np.sum(abs(palette - pigments)) / 2 > 0.18
            ):
                continue
            close_shape = (
                np.all(abs(extents - dims) <= 0.004 + 0.32 * dims)
                and abs(
                    extents[0] / max(extents[1], 0.001) - dims[0] / max(dims[1], 0.001)
                )
                <= 0.20
            )
            if not close_shape:
                # An occluder can truncate one horizontal extent of an otherwise
                # compatible object. Require a measured foreground depth boundary.
                span_ok = np.all(extents <= 0.004 + 1.32 * dims) and np.all(
                    extents >= [0.35, 0.42, 0.62] * dims
                )
                separated = (
                    np.linalg.norm(
                        np.median(world[box][other] @ axes, axis=0) - center_xy
                    )
                    >= 0.65 * dims[1]
                )
                if not (span_ok and separated):
                    continue
                y, x = box
                roi = np.s_[
                    max(0, y.start - 5) : y.stop + 5, max(0, x.start - 5) : x.stop + 5
                ]
                local = components[roi] == i
                ring = (
                    binary_dilation(local, iterations=4)
                    & ~binary_dilation(local, iterations=2)
                    & observation.valid_mask[roi]
                )
                near = distance_transform_edt(
                    ~local, return_distances=False, return_indices=True
                )
                occluded = depth[roi][tuple(near)] - depth[roi] > 0.025
                if (
                    not ring.any()
                    or np.mean(occluded[ring]) < 0.16
                    or np.mean(abs(appearance - colors)) > 12
                ):
                    continue
            checked[box] |= other
            yield components == i, not close_shape


def verify_region(
    observation,
    mask,
    query,
    grounder,
    data,
    anchor,
    *,
    compare_peers=True,
    require_material_point=False,
    matched_instance=False,
):
    """Return explicit identity evidence before publishing measured coordinates."""
    region = mask & observation.valid_mask
    n = int(region.sum())
    evidence = {"accepted": False, "pixels": n}
    if n < 100:
        return {**evidence, "reason": "insufficient_resolved_target_pixels"}
    slices = find_objects(region, max_label=1)[0]
    extent = [s.stop - s.start for s in slices]
    if min(extent) < 9:
        return {**evidence, "reason": "visible_target_too_narrow"}
    if anchor is not None and min(extent) < 32:
        rr, cc = np.nonzero(region)
        gap = np.min(np.hypot(rr - anchor[0], cc - anchor[1]))
        if gap > max(3, 0.16 * min(extent)):
            if n < 250 or gap > min(12, max(6, 0.75 * min(extent))):
                return {
                    **evidence,
                    "reason": "small_candidate_disagrees_with_visual_point",
                }
            require_material_point = True
    verify = getattr(grounder, "verify", None)
    if verify is None:
        return {**evidence, "reason": "local_identity_evidence_unavailable"}
    subject = target_subject(query)
    relational = subject != query.strip()
    if not subject:
        return {**evidence, "reason": "target_description_missing"}
    context_query = (
        f"{query} (evaluate the cyan-outlined candidate in the right panel)"
        if relational
        else query
    )
    answer = verify(
        _views(observation.rgb, region, slices, full_scene=relational), context_query
    )
    if answer is None:
        return {**evidence, "reason": "local_identity_evidence_unavailable"}
    data["calls"]["identity"] = data["calls"].get("identity", 0) + 1
    paired = _agrees(answer, "yes")
    if not paired and (
        relational
        or matched_instance
        or not _agrees(answer, "no")
        or n < 1500
        or min(extent) < 32
        or anchor is None
    ):
        return {**evidence, "reason": "local_model_did_not_confirm_target"}
    bare_name = compare_peers and not relational and len(query.split()) <= 4
    category_phrase = (
        re.split(
            r"\b(?:with|without|whose|which|that)\b", subject, maxsplit=1, flags=re.I
        )[0]
        if relational
        else subject
    )
    class_words = re.findall(r"[^\W\d_]+", category_phrase.lower())
    appearance_only = (
        compare_peers
        and not bare_name
        and not relational
        and bool(class_words)
        and not re.search(
            r"[,;]|\b(with|without|whose|which|that|on|in|near|beside|behind|under|excluding)\b",
            query,
            re.I,
        )
    )
    if (
        not paired
        or require_material_point
        or relational
        or bare_name
        or appearance_only
        or (compare_peers and n < 700)
    ):
        # A nearby object in the context panel can explain a positive answer.
        # Check that the measured material alone still identifies this target.
        isolated, material = _isolated_view(observation.rgb, region, slices)
        if bare_name or appearance_only or relational:
            data["calls"]["identity"] += 1
            label = verify(isolated, None, isolated=True)
            terms = _name_terms(label or "")
            if (
                paired
                and n >= 700
                and terms in ({"unknown"}, {"unclear"}, {"unidentifiable"})
            ):
                data["calls"]["identity"] += 1
                terms = _named_with_context(observation.rgb, region, slices, verify)
                require_material_point = True
            required_terms = _name_terms(query) if bare_name else set(class_words[-1:])
            if (
                not required_terms
                or not required_terms <= terms
                or terms
                & {
                    "unknown",
                    "maybe",
                    "possibly",
                    "or",
                }
            ):
                return {**evidence, "reason": "independent_object_name_not_confirmed"}
        if appearance_only or (not bare_name and (compare_peers or not paired)):
            data["calls"]["identity"] += 1
            answer = verify(isolated, subject, isolated=True)
            if not _agrees(answer, "yes"):
                return {
                    **evidence,
                    "reason": "target_not_identifiable_on_measured_material",
                }
        if not paired:
            # Recheck a conflicting context verdict using only observed material.
            rr, cc = np.nonzero(region)
            if np.min(np.hypot(rr - anchor[0], cc - anchor[1])) > 3:
                return {**evidence, "reason": "conflicting_target_identity"}
        if not paired or require_material_point:
            point = grounder.ground(isolated, subject)
            data["calls"]["grounding"] += 1
            if (
                not point.found
                or point.image_size != (320, 320)
                or point.point_xy is None
            ):
                return {**evidence, "reason": "isolated_material_not_grounded"}
            x, y = point.point_xy
            index = (
                material[min(319, round(y)), min(319, round(x))]
                if (0 <= x < 320 and 0 <= y < 320)
                else 0
            )
            if not index:
                return {
                    **evidence,
                    "reason": "isolated_point_outside_measured_material",
                }
            evidence["isolated_point_confirmed"] = True
            evidence["material_pixel_rc"] = list(
                map(int, divmod(int(index) - 1, region.shape[1]))
            )
    if re.search(r"\b(cup|mug|bowl|ramekin|goblet)\b", subject, re.I):
        opening = _opening_evidence(observation, region)
        evidence["opening"] = opening
        if not opening["resolved"]:
            return {**evidence, "reason": "open_container_cavity_not_observed"}
        if not matched_instance and _other_plain_vessel(
            observation,
            region,
            data.get("depth_selection", {}).get("support_height_m"),
            query,
        ):
            return {**evidence, "reason": "another_compatible_open_container"}
    if compare_peers:
        support = data.get("depth_selection", {}).get("support_height_m")
        for other, obscured in similar_instances(observation, region, support):
            if obscured:
                return {**evidence, "reason": "another_occluded_compatible_instance"}
            other_box = find_objects(other, max_label=1)[0]
            data["calls"]["identity"] += 1
            answer = verify(
                _views(observation.rgb, other, other_box, full_scene=relational),
                context_query,
            )
            if not _agrees(answer, "no"):
                return {**evidence, "reason": "another_visually_compatible_instance"}
    if relational:
        evidence["relation_verified"] = True
    return {**evidence, "accepted": True, "reason": "local_model_confirmed_candidate"}


def _opening_evidence(obs, region):
    """Check rays through an estimated lip against actual optical depths below its plane."""
    from components.tools.rgbd.geometry.geometry import _basis
    from components.tools.rgbd.geometry.rgbd_parts import object_points, rim

    if (
        obs.up_direction is None
        or obs.camera_to_world is None
        or obs.intrinsics is None
    ):
        return {"resolved": False}
    cam = obs.camera_to_world
    cloud = object_points(obs.world_xyz, obs.valid_mask, region, cam[:3, 2])
    fit = rim(cloud, obs.up_direction) if len(cloud) >= 25 else None
    if fit is None or fit["coverage"] < 190:
        return {"resolved": False}
    axes = np.stack(_basis(fit["normal"]), axis=1)
    grid = np.linspace(-0.72, 0.72, 15)
    disk = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    disk = disk[np.linalg.norm(disk, axis=1) < 0.72]
    cv = (fit["center"] + fit["radius"] * disk @ axes.T - cam[:3, 3]) @ cam[:3, :3]
    pixels = cv @ obs.intrinsics.T
    xy = np.rint(pixels[:, :2] / pixels[:, 2:]).astype(int)
    h, w = region.shape
    visible = (
        (cv[:, 2] > 0)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < w)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < h)
    )
    xy, cv = xy[visible], cv[visible]
    rc = (xy[:, 1], xy[:, 0])
    valid = obs.valid_mask[rc]
    clearance = (obs.optical_depth()[rc] - cv[:, 2])[valid]
    threshold = max(0.0025, 0.09 * fit["radius"])
    fraction = float(np.mean(clearance > threshold)) if len(clearance) else 0.0
    return {
        "resolved": len(clearance) >= 25 and fraction >= 0.55,
        "interior_samples": len(clearance),
        "fraction_below_rim": fraction,
        "required_clearance_m": threshold,
    }


def _other_plain_vessel(obs, region, support, query):
    """Find a second measured opening with the same appearance under a plain vessel description."""
    attributes = {
        "cup",
        "mug",
        "bowl",
        "goblet",
        "ramekin",
        "ceramic",
        "glass",
        "metal",
        "plastic",
        "round",
        "circular",
        "open",
        "hollow",
        "opening",
        "rim",
        "with",
        "an",
        "a",
        "the",
        "and",
        "pale",
        "light",
        "dark",
        "colored",
        "coloured",
        "plain",
        "small",
        "large",
        "red",
        "blue",
        "green",
        "yellow",
        "pink",
        "purple",
        "brown",
        "white",
        "black",
        "gray",
        "grey",
    }
    if support is None or not _name_terms(query) <= attributes:
        return False
    depth = obs.optical_depth()
    raised = obs.valid_mask & (obs.world_xyz @ obs.up_direction > support + 0.002)
    components, _ = label(raised & (local_depth_range(depth, obs.valid_mask) < 0.012))
    tone = np.median(obs.rgb[region], axis=0)
    for i, box in enumerate(find_objects(components), 1):
        if box is None:
            continue
        local = components[box] == i
        if local.sum() < 120 or np.any(local & region[box]):
            continue
        other_tone = np.median(obs.rgb[box][local], axis=0)
        if abs(np.mean(tone) - np.mean(other_tone)) > 30:
            continue
        if (
            np.max(
                abs(tone / max(tone.sum(), 1) - other_tone / max(other_tone.sum(), 1))
            )
            > 0.085
        ):
            continue
        other = components == i
        if _opening_evidence(obs, other)["resolved"]:
            return True
    return False


def need_observation(data, reason):
    """Withhold any target location when the selected region lacks identity evidence."""
    for decision in data.get("identity_evidence", {}).values():
        decision.pop("material_pixel_rc", None)
    data.update(
        status="need_observation",
        reason=reason,
        found=False,
        evidence_status="unknown",
        world_xyz=None,
    )
    for key in (
        "pixel_rc",
        "surface_point_xyz",
        "visible_bounds_xyz",
        "color_depth_evidence",
    ):
        data.pop(key, None)
    if data.get("required_geometry"):
        data["missing_geometry"] = list(data["required_geometry"])
