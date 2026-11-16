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

"""Measure whole-object features after one native point and one verified object mask."""

import numpy as np

from toolkits.geometry import target_identity
from toolkits.geometry.measured_rectangle import observed_face, rectangle_value
from toolkits.geometry.package_identity import package_description, select_package
from toolkits.geometry.rgbd_parts import (
    cuboid_vertices,
    measure,
    object_points,
)

FEATURES = {
    "inner_rim_center": "3d_center_in_observed_inner_upper_opening_edge_plane",
    "plate_center": "circular_support_surface_center_at_fitted_support_plane_height",
    "lid_center": "sealed_circular_lid_center_in_fitted_lid_plane",
    "rectangle_center": "center_of_complete_observed_main_rectangular_top_face",
    "body_segment_center": "axis_midpoint_of_observed_main_cylindrical_barrel_excluding_neck_and_mouth",
    "inner_rim_radius": "inner_opening_radius_at_the_observed_lip_ridge",
    "rim_extrema": "inner_lip_extrema_left_right_near_far_in_the_projected_camera_basis",
    "top_corners": "top_face_corners_far_left_far_right_near_left_near_right",
    "cuboid_corners": "upright_cuboid_top_then_bottom_corners_on_an_observed_contacting_plane",
    "top_size": "long_and_short_edge_lengths_of_the_rectangular_top_face",
    "object_height": "visible_vertical_extent_along_the_provider_up_direction",
    "height_above_support": "highest_observed_object_surface_above_a_nearby_horizontal_support_plane",
    "supported_height": "upper_surface_height_above_an_observed_contacting_support_plane",
}


def run(
    observation,
    data,
    grounder,
    segmenter,
    cancel,
    *,
    evidence=None,
    direction_camera=None,
    projected_rc=None,
):
    output_names = data.get("required_geometry", [])
    requested = list(output_names)
    if "cuboid_corners" in requested:
        requested = list(
            dict.fromkeys(
                [k for k in requested if k != "cuboid_corners"]
                + ["top_corners", "supported_height"]
            )
        )
    image = observation.png_bytes()
    h, w = observation.rgb.shape[:2]
    native = projected_rc
    if native is None:
        selected = grounder.ground(image, data["query"])
        data["calls"]["grounding"] += 1
        if selected.found and selected.point_xy is not None:
            col, row = selected.point_xy
            if selected.image_size != (w, h) or not (0 <= row < h and 0 <= col < w):
                raise ValueError("grounding raster identity or pixel bounds mismatch")
            native = [min(h - 1, round(row)), min(w - 1, round(col))]
    if native is not None:
        row, col = native
        data["pixel_rc"] = list(native)
        if evidence is not None and observation.valid_mask[tuple(native)]:
            # Only use the proposal to check the same surface in a synchronized view.
            evidence["anchor_xyz"] = observation.world_xyz[tuple(native)]
    cloud = None
    candidates = {}
    regions = {}
    faces = {}
    decisions = {}
    reason = "target_not_identified"
    subject = target_identity.between_subject(data["query"])
    relational = subject != data["query"]
    # Color/label shortcuts do not verify the relation to the landmarks.
    hue = None if relational else target_identity.box_color(subject)
    box_search = hue is not None
    package = None if relational else package_description(subject)
    if package is not None:
        proposal, reason = select_package(
            observation, data["query"], package, native, grounder, data, cancel
        )
        if proposal is not None:
            (
                regions["package"],
                candidates["package"],
                faces["package"],
                decisions["package"],
            ) = proposal
    elif box_search:
        from toolkits.geometry.chromatic_region import unique_colored_box

        mask, reason = unique_colored_box(
            observation,
            hue,
            cube="cube" in data["query"].lower().split(),
        )
        if mask is not None:
            regions["color_depth"] = mask
            candidates["color_depth"] = object_points(
                observation.world_xyz,
                observation.valid_mask,
                mask,
                observation.camera_to_world[:3, 2],
            )
            data["color_depth_evidence"] = {"color": hue, "search_scope": reason}
    elif native is not None:
        reason = "no_consistent_object_mask"
        camera = observation.camera_to_world
        from toolkits.geometry.depth_foreground import (
            anchored_plane,
            described_lightness,
            foreground,
        )

        appearance = described_lightness(subject)

        mask, depth_info = foreground(
            observation.world_xyz,
            observation.valid_mask,
            observation.up_direction,
            camera,
            [row, col],
            rgb=observation.rgb,
            lightness=appearance,
        )
        data["depth_selection"] = depth_info
        if any(
            k
            in (
                "top_size",
                "top_corners",
                "rectangle_center",
                "lid_center",
                "plate_center",
                "supported_height",
                "height_above_support",
            )
            for k in requested
        ):
            plane = anchored_plane(
                observation.world_xyz,
                observation.valid_mask,
                observation.up_direction,
                camera,
                [row, col],
            )
            if plane is not None:
                regions["anchor_plane"] = plane
                candidates["anchor_plane"] = observation.world_xyz[plane]
                data["plane_touches_frame"] = bool(
                    plane[0].any()
                    or plane[-1].any()
                    or plane[:, 0].any()
                    or plane[:, -1].any()
                )
        if mask is not None:
            regions["depth"] = mask
            candidates["depth"] = object_points(
                observation.world_xyz, observation.valid_mask, mask, camera[:3, 2]
            )
    if (
        not box_search
        and package is None
        and native is not None
        and data.get("pixel_rc") is not None
        and any(k in ("rectangle_center", "top_corners", "top_size") for k in requested)
    ):
        from toolkits.geometry.chromatic_region import colored_region, described_hue

        camera = observation.camera_to_world
        hue = described_hue(subject)
        if camera is not None and hue is not None:
            optical = (observation.world_xyz - camera[:3, 3]) @ camera[:3, 2]
            chromatic = colored_region(
                observation.rgb,
                optical,
                observation.valid_mask,
                data["pixel_rc"],
                hue,
                world=observation.world_xyz,
                up=observation.up_direction,
            )
            if chromatic is not None:
                mask, core, color_evidence = chromatic
                regions["color_depth"] = mask
                candidates["color_depth"] = object_points(
                    observation.world_xyz, observation.valid_mask, mask, camera[:3, 2]
                )
                data["color_depth_evidence"] = {
                    **color_evidence,
                    "interior_pixel_rc": core,
                }
            # A nearby uncolored surface cannot substitute for the requested object.
            candidates = {k: v for k, v in candidates.items() if k == "color_depth"}
            regions = {k: v for k, v in regions.items() if k in candidates}
            if not candidates:
                reason = "described_color_has_no_unambiguous_depth_contour"

    # Associate a projected surface with the same material before verifying its identity.
    if projected_rc is not None:
        for source, mask in list(regions.items()):
            if not mask[tuple(projected_rc)]:
                regions.pop(source)
                candidates.pop(source, None)
                decisions.pop(source, None)
                reason = "visual_hint_disagrees_with_region"

    def confirmed(source):
        if source not in decisions:
            cancel()
            matched = False
            if relational and evidence is not None and "source_points" in evidence:
                from toolkits.geometry.multiview import surfaces_overlap

                points = observation.world_xyz[regions[source] & observation.valid_mask]
                matched = surfaces_overlap(
                    evidence["source_points"], points[:: max(1, len(points) // 2500)]
                )
            decision = target_identity.verify_region(
                observation,
                regions[source],
                subject if matched else data["query"],
                grounder,
                data,
                None if source == "color_depth" else native,
                compare_peers=not (box_search or matched),
                matched_instance=matched,
                require_material_point=matched
                or (
                    source == "color_depth"
                    and reason == "occluded_box_only_visible_material"
                ),
            )
            if matched and decision["accepted"]:
                decision["matched_source_material"] = True
            decisions[source] = decision
        return decisions[source]["accepted"]

    def record_region(region):
        if evidence is not None:
            coords = observation.world_xyz[region & observation.valid_mask]
            if len(coords):
                evidence["points"] = coords[:: max(1, len(coords) // 2500)]
                evidence["observation"] = observation

    if not requested:
        for source in (
            "package",
            "color_depth",
            "depth",
            "anchor_plane",
            "sam_fallback",
        ):
            if (
                source == "sam_fallback"
                and segmenter is not None
                and not box_search
                and package is None
            ):
                # A second mask can fix a poor depth component, not an ambiguous identity.
                if any(
                    d["reason"]
                    not in {
                        "local_model_did_not_confirm_target",
                        "insufficient_resolved_target_pixels",
                        "visible_target_too_narrow",
                        "open_container_cavity_not_observed",
                    }
                    for d in decisions.values()
                ):
                    break
                cancel()
                prompt = (
                    {"point": native}
                    if native is not None
                    else {"text_prompt": data["query"]}
                )
                found = segmenter.segment(image, **prompt)
                data["calls"]["segment"] += 1
                if found.found:
                    mask = np.asarray(found.mask)
                    if mask.dtype != np.bool_ or mask.shape != (h, w):
                        raise ValueError("Object mask does not match RGB.")
                    if (
                        native is not None
                        and mask[tuple(native)]
                        and observation.valid_mask[tuple(native)]
                    ):
                        regions[source] = mask
                    elif (
                        native is None
                        and projected_rc is None
                        and (mask & observation.valid_mask).any()
                    ):
                        # Verify foreground material from an independent text proposal.
                        regions[source] = mask
                if source in regions and (
                    observation.up_direction is None
                    or observation.camera_to_world is None
                ):
                    regions.pop(source)
                if source in regions:
                    if data.get("depth_selection", {}).get("support_height_m") is None:
                        from toolkits.geometry.depth_foreground import foreground

                        anchor = target_identity.material_pixel(regions[source], native)
                        _, info = foreground(
                            observation.world_xyz,
                            observation.valid_mask,
                            observation.up_direction,
                            observation.camera_to_world,
                            anchor,
                        )
                        data["depth_selection"] = info
                    # Whole-scene peer checks need a measurable support.
                    if data.get("depth_selection", {}).get("support_height_m") is None:
                        regions.pop(source)
            if source not in regions or not confirmed(source):
                continue
            region = regions[source] & observation.valid_mask
            pixel = decisions[source].get(
                "material_pixel_rc"
            ) or target_identity.material_pixel(region, native)
            points = observation.world_xyz[region]
            surface = observation.world_xyz[tuple(pixel)].tolist()
            data.update(
                status="usable",
                found=True,
                pixel_rc=pixel,
                surface_point_xyz=surface,
                world_xyz=surface,
                evidence_status="measured",
                identity_evidence=decisions,
            )
            data["calls"]["back_project"] += 1
            if data["measurement"] == "visible_region":
                data.update(
                    world_xyz=np.median(points, axis=0).tolist(),
                    visible_bounds_xyz=[
                        points.min(axis=0).tolist(),
                        points.max(axis=0).tolist(),
                    ],
                )
            record_region(region)
            return region
        data["identity_evidence"] = decisions
        target_identity.need_observation(
            data, next((d["reason"] for d in decisions.values()), reason)
        )
        return None
    fields = {}
    for name in requested:
        cancel()
        value = None
        quality = {}
        warnings = []
        cam = observation.camera_to_world
        order = (
            ("depth",)
            if name
            not in (
                "rectangle_center",
                "top_size",
                "top_corners",
                "height_above_support",
            )
            else ("depth", "anchor_plane")
        )
        if name in ("rectangle_center", "top_size", "top_corners") and data.get(
            "plane_touches_frame"
        ):
            order = ("anchor_plane", *order)
        if "package" in candidates:
            order = ("package", *order)
        if "color_depth" in candidates:
            order = ("color_depth", *order)
        for source in order:
            if source not in candidates:
                continue
            if (
                source == "color_depth"
                and reason == "occluded_box_only_visible_material"
                and name != "object_height"
            ):
                quality = {"reason": "complete_box_face_not_observed"}
                continue
            cloud = candidates[source]
            if len(cloud) < 25:
                continue
            measured_rectangle = name in ("rectangle_center", "top_size", "top_corners")
            if measured_rectangle:
                if source not in faces:
                    faces[source] = observed_face(observation, regions[source], cloud)
                face = faces[source]
                if face is None or not face["supported"].all():
                    quality = {"reason": "rectangle_edges_not_observed"}
                    continue
                if face["clipped"] and (
                    not face["wall_extended"] or name != "rectangle_center"
                ):
                    quality = {"reason": "full_face_limits_outside_view"}
                    continue
            if name in ("supported_height", "height_above_support"):
                from toolkits.geometry.depth_foreground import nearby_support_height

                z0 = data.get("depth_selection", {}).get("support_height_m")
                if observation.up_direction is None:
                    quality = {"reason": "support_plane_not_observed"}
                    continue
                low, high = np.quantile(
                    cloud @ observation.up_direction, [0.002, 0.998]
                )
                if z0 is None or high - z0 < 0.006:
                    z0 = nearby_support_height(
                        observation.world_xyz,
                        observation.valid_mask,
                        regions[source],
                        observation.up_direction,
                        high,
                    )
                if z0 is None:
                    quality = {"reason": "support_plane_not_observed"}
                    continue
                if (name == "supported_height" and abs(low - z0) > 0.007) or high <= z0:
                    quality = {"reason": "contact_with_support_not_observed"}
                    continue
                value = float(high - z0)
                quality = {
                    "support_height_m": z0,
                    "observed_contact_gap_m": float(low - z0),
                    "warnings": ["resting_on_observed_support_assumed"]
                    if name == "supported_height"
                    else ["distance_to_support_plane"],
                }
            elif measured_rectangle:
                basis = direction_camera if direction_camera is not None else cam
                value = rectangle_value(face, name, basis)
                quality = {
                    "physical_edges": face["supported"].tolist(),
                    "support_pixels": face["n_support"],
                    "sample_pitch_m": face["pitch"],
                    "wall_extended": face["wall_extended"],
                }
                if face["clipped"]:
                    warnings.append("clipped_roof_center_uses_observed_side_wall")
                if value is None:
                    quality["reason"] = "camera_relative_corner_order_unresolved"
            else:
                basis = direction_camera if direction_camera is not None else cam
                value, quality = measure(
                    cloud,
                    observation.up_direction,
                    name,
                    None if basis is None else basis[:3, 0],
                    None if basis is None else -basis[:3, 2],
                )
            if value is None:
                continue
            # Single horizontal supports get mistaken for rims when a point selects a tabletop.
            if (
                name in ("inner_rim_center", "inner_rim_radius", "rim_extrema")
                and observation.up_direction is not None
            ):
                vertical = np.ptp(
                    np.quantile(cloud @ observation.up_direction, [0.01, 0.99])
                )
                if vertical < 0.003:
                    value = None
                    quality = {"reason": "flat_region_has_no_resolved_open_lip"}
                    continue
            quality["region_source"] = source
            if source == "color_depth":
                quality["color_depth_evidence"] = data["color_depth_evidence"]
                warnings.append("region_selected_by_described_color_and_depth")
            if not confirmed(source):
                value = None
                quality = {"reason": decisions[source]["reason"]}
                continue
            if quality.get("coverage", 360) < 235:
                warnings.append("partial_circular_arc")
            if source == "anchor_plane" and data.get("plane_touches_frame"):
                warnings.append("object_mask_touches_image_border")
            if source == "depth" and data["depth_selection"]["anchor_distance_px"] > 3:
                warnings.append("grounded_pixel_adjacent_to_depth_component")
            if source == "depth" and data["depth_selection"].get(
                "selected_by_relative_lightness"
            ):
                warnings.append("neighbor_selected_by_described_lightness")
            if name == "object_height":
                warnings.append("visible_vertical_extent_only")
            warnings.extend(quality.get("warnings", []))
            data["segment_mode"] = source
            break
        fields[name] = {
            "value": value,
            "definition": FEATURES[name],
            "unit": "m",
            "source_step": observation.step,
            "source_camera": observation.camera,
            "world_frame_id": observation.world_frame_id,
            "source_refs": dict(observation.source_refs),
            "evidence_status": "inferred" if value is not None else "unknown",
            "fit_accepted": value is not None and not warnings,
            "warnings": warnings,
            "quality": quality,
            "reason": None if value is not None else quality.get("reason", reason),
        }
    if "cuboid_corners" in output_names:
        top, support = fields["top_corners"], fields["supported_height"]
        level = support["quality"].get("support_height_m")
        vertices = cuboid_vertices(
            top["value"], observation.up_direction, level, support["value"]
        )
        warnings = (
            list(
                dict.fromkeys(
                    [
                        *top["warnings"],
                        *support["warnings"],
                        "vertical_sides_to_observed_support_assumed",
                    ]
                )
            )
            if vertices is not None
            else []
        )
        fields["cuboid_corners"] = {
            **top,
            "value": vertices,
            "definition": FEATURES["cuboid_corners"],
            "warnings": warnings,
            "quality": {
                "support_height_m": level,
                "top": top["quality"],
                "contact": support["quality"],
            },
            "evidence_status": "inferred" if vertices is not None else "unknown",
            "fit_accepted": False,
            "reason": None
            if vertices is not None
            else "upright_top_and_contacting_support_required",
        }
        fields = {name: fields[name] for name in output_names}
    count = sum(f["value"] is not None for f in fields.values())
    alerts = [
        {"quantity": k, "reason": w} for k, f in fields.items() for w in f["warnings"]
    ]
    data.update(
        geometry=fields,
        warnings=alerts,
        found=bool(count),
        definition="object_mask_rgbd_geometry",
        evidence_status="inferred" if count else "unknown",
        status="need_observation"
        if not count
        else "partial"
        if count < len(fields)
        else "warning"
        if alerts
        else "usable",
    )
    if count < len(fields):
        data["missing_geometry"] = [k for k, f in fields.items() if f["value"] is None]
    data["identity_evidence"] = decisions
    if not count:
        if evidence is not None:
            for source in regions:
                if confirmed(source):
                    record_region(regions[source])
                    break
        reason = next(
            (d["reason"] for d in decisions.values() if not d["accepted"]),
            next((f["reason"] for f in fields.values() if f["reason"]), reason),
        )
        target_identity.need_observation(data, reason)
    elif data.get("segment_mode") in regions:
        record_region(regions[data["segment_mode"]])
        data["pixel_rc"] = decisions[data["segment_mode"]].get(
            "material_pixel_rc"
        ) or target_identity.material_pixel(
            regions[data["segment_mode"]] & observation.valid_mask, native
        )
        return regions[data["segment_mode"]]
    return None
