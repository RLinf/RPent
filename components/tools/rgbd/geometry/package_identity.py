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

"""Identify a described multicolour package from its measured material and a blind label reading."""

import io
import re

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import (
    find_objects,
    label,
    map_coordinates,
)

from components.tools.rgbd.geometry.chromatic_region import HUES
from components.tools.rgbd.geometry.depth_foreground import (
    foreground,
    local_depth_range,
)
from components.tools.rgbd.geometry.measured_rectangle import observed_face
from components.tools.rgbd.geometry.rgbd_parts import object_points
from components.tools.rgbd.geometry.target_identity import (
    _isolated_view,
    _name_terms,
)


def package_description(query):
    """Extract colour and product terms only from simple, explicit package descriptions."""
    if re.search(
        r"[,;]|\b(with|near|behind|beside|under|without|excluding)\b", query, re.I
    ):
        return None
    words = _name_terms(query)
    containers = {"box", "carton", "package", "packet"}
    colors = words & (HUES.keys() | {"white", "black"})
    nouns = (
        words
        - containers
        - colors
        - {"rectangular", "square", "small", "large", "and", "colored", "coloured"}
    )
    return (
        (colors, nouns) if words & containers and len(colors) >= 2 and nouns else None
    )


def _label_images(obs, mask, box, face):
    """Present original material first, then rectify its measured top in reading orientations."""
    isolated, _ = _isolated_view(obs.rgb, mask, box)
    yield isolated
    if face is None:
        return
    order = np.argsort(face["sizes"])[::-1]
    long, short = face["sizes"][order]
    axes = face["axes"][:, order]
    w = 300
    h = max(12, round(w * short / long))
    yy, xx = np.mgrid[:h, :w]
    plane = (
        face["center"]
        + ((xx / (w - 1) - 0.5) * long)[..., None] * axes[:, 0]
        + ((yy / (h - 1) - 0.5) * short)[..., None] * axes[:, 1]
    )
    cv = (plane - obs.camera_to_world[:3, 3]) @ obs.camera_to_world[:3, :3]
    uv = cv @ obs.intrinsics.T
    if (uv[:, :, 2] <= 0).any():
        return
    rc = [uv[:, :, 1] / uv[:, :, 2], uv[:, :, 0] / uv[:, :, 2]]
    sampled = map_coordinates(
        mask.astype(np.uint8), rc, order=0, mode="constant"
    ).astype(bool)
    if sampled.mean() < 0.72:
        return
    pixels = np.stack(
        [
            map_coordinates(channel, rc, order=1)
            for channel in np.moveaxis(obs.rgb, -1, 0)
        ],
        axis=2,
    )
    pixels[~sampled] = 155
    im = ImageOps.autocontrast(Image.fromarray(pixels), cutoff=1)
    # Preserve the handedness of the actual photographed text.
    if (rc[1][0, -1] - rc[1][0, 0]) * (rc[0][-1, 0] - rc[0][0, 0]) - (
        rc[0][0, -1] - rc[0][0, 0]
    ) * (rc[1][-1, 0] - rc[1][0, 0]) < 0:
        im = ImageOps.mirror(im)
    for angle in (0, 180):
        view = ImageOps.pad(im.rotate(angle), (352, 256), color=(155, 155, 155))
        buf = io.BytesIO()
        view.save(buf, format="PNG")
        yield buf.getvalue()


def _candidate_faces(obs, cc, pigments):
    """Bind each label to the footprint actually measured, even if two packages touch."""
    for i, box in enumerate(find_objects(cc), 1):
        if box is None:
            continue
        parts = [cc[box] == i]
        while parts:
            local = parts.pop()
            if not 100 <= local.sum() <= 0.18 * cc.size:
                continue
            rr, co = np.nonzero(local)
            if min(np.ptp(rr), np.ptp(co)) < 8:
                continue
            fractions = {k: float(np.mean(p[box][local])) for k, p in pigments.items()}
            if min(fractions.values()) < 0.045 or sum(fractions.values()) < 0.47:
                continue
            region = np.zeros_like(cc, dtype=bool)
            region[box] = local
            cloud = object_points(
                obs.world_xyz, obs.valid_mask, region, obs.camera_to_world[:3, 2]
            )
            face = observed_face(obs, region, cloud) if len(cloud) >= 25 else None
            if face is None or face["fill"] < 0.82:
                yield region, box, cloud, None, fractions
                continue
            uv = (obs.world_xyz[box] - face["center"]) @ face["axes"]
            margin = max(0.0015, 3 * face["pitch"])
            inside = np.all(abs(uv) <= face["sizes"] / 2 + margin, axis=2)
            rest = local & ~inside
            if rest.sum() > max(100, 0.035 * local.sum()):
                labels, n = label(rest)
                parts.extend(
                    labels == k
                    for k in range(1, n + 1)
                    if np.count_nonzero(labels == k) >= 100
                )
                local = local & inside
                if local.sum() < 100:
                    continue
                region[box] = local
                cloud = object_points(
                    obs.world_xyz, obs.valid_mask, region, obs.camera_to_world[:3, 2]
                )
                fractions = {
                    k: float(np.mean(p[box][local])) for k, p in pigments.items()
                }
                if min(fractions.values()) < 0.045 or sum(fractions.values()) < 0.47:
                    continue
            material_box = find_objects(region, max_label=1)[0]
            yield region, material_box, cloud, face, fractions


def select_package(obs, query, description, anchor, grounder, data, cancel):
    """Return one uniquely identified package. A matching palette alone gives no identity."""
    verify = getattr(grounder, "verify", None)
    if (
        verify is None
        or obs.camera_to_world is None
        or obs.intrinsics is None
        or obs.up_direction is None
    ):
        return None, "package_identity_evidence_unavailable"
    colors, nouns = description
    h, w = obs.valid_mask.shape
    _, info = foreground(
        obs.world_xyz,
        obs.valid_mask,
        obs.up_direction,
        obs.camera_to_world,
        anchor or (h // 2, w // 2),
    )
    support = info.get("support_height_m")
    if support is None:
        return None, "support_plane_not_observed"
    optical = obs.optical_depth()
    jump = local_depth_range(optical, obs.valid_mask)
    cc, _ = label(
        obs.valid_mask
        & (obs.world_xyz @ obs.up_direction > support + 0.002)
        & (jump < 0.012)
    )
    hsv = np.asarray(Image.fromarray(obs.rgb).convert("HSV"), dtype=float) / 255
    hue, sat, v = np.moveaxis(hsv, -1, 0)
    pigments = {}
    for color in colors:
        if color == "white":
            pigments[color] = (sat < 0.28) & (v > 0.36)
        elif color == "black":
            pigments[color] = v < 0.20
        else:
            pigments[color] = (
                (np.abs((hue * 360 - HUES[color] + 180) % 360 - 180) < 33)
                & (sat > 0.26)
                & (v > 0.18)
            )
    found = []
    unknown = 0
    for region, box, cloud, face, fractions in _candidate_faces(obs, cc, pigments):
        if face is None:
            unknown += 1
            continue
        reads = []
        identified = False
        different = False
        for view in _label_images(obs, region, box, face):
            cancel()
            answer = verify(view, None)
            data["calls"]["identity"] = data["calls"].get("identity", 0) + 1
            terms = _name_terms(answer or "")
            reads.append(answer)
            if (
                terms
                & {"unknown", "unclear", "maybe", "possibly", "cannot", "unreadable"}
                or not terms
            ):
                continue
            if nouns <= terms:
                identified = True
                break
            if (
                len(
                    terms
                    - {
                        "box",
                        "package",
                        "product",
                        "text",
                        "blue",
                        "white",
                        "label",
                        "unidentified",
                    }
                )
                >= 1
            ):
                different = True
        if identified:
            found.append(
                (
                    region,
                    cloud,
                    face,
                    {
                        "accepted": True,
                        "reason": "blind_product_name_on_measured_package",
                        "label_readings": reads,
                        "pigment_fractions": fractions,
                        "pixels": int(region.sum()),
                    },
                )
            )
        elif not different:
            unknown += 1
    if len(found) == 1 and not unknown:
        data["depth_selection"] = info
        return found[0], "blind_product_name_on_measured_package"
    if len(found) > 1:
        return None, "multiple_matching_packages"
    return None, "package_label_or_uniqueness_unresolved"
