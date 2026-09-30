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
"""Generate destination proposals from aligned RGB and measured world maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from robots.libero.onjev.config import OneJevConfig


@dataclass(frozen=True)
class Region:
    """An observed surface proposal, without simulator-derived semantics."""

    id: str
    bbox: tuple[int, int, int, int]
    pixel: tuple[int, int]
    xyz: tuple[float, float, float]
    rim_z: float
    width_xy: tuple[float, float]
    valid_pixels: int
    source: str

    def as_dict(self) -> dict:
        """Describe the proposal and its measurement provenance."""
        return asdict(self)


def in_workspace(xyz: np.ndarray, config: OneJevConfig) -> bool:
    """Check finite XYZ against the explicit robot workspace bounds."""
    bounds = np.asarray(config.workspace).reshape(3, 2)
    return bool(
        np.isfinite(xyz).all()
        and np.all(xyz >= bounds[:, 0])
        and np.all(xyz <= bounds[:, 1])
    )


def propose_regions(
    rgb: np.ndarray,
    world: np.ndarray,
    *,
    receptacle: str,
    eef: np.ndarray,
    config: OneJevConfig,
) -> tuple[list[Region], float]:
    """Propose visible receptacle interiors with classical image/depth rules.

    These are geometric hypotheses. OneJev must compare their marked RGB
    regions with the destination named in the instruction. A configured ROI
    is an explicit operator prior, not a simulator object location.
    """
    rgb = np.asarray(rgb)
    world = np.asarray(world, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or world.shape != rgb.shape:
        raise ValueError("OneJev requires aligned HxWx3 RGB and world-map artifacts")
    bounds = np.asarray(config.workspace).reshape(3, 2)
    valid = np.isfinite(world).all(axis=2)
    valid &= ((world >= bounds[:, 0]) & (world <= bounds[:, 1])).all(axis=2)
    if np.count_nonzero(valid) < 100:
        raise ValueError("Too few valid RGB-D workspace pixels")
    z = world[..., 2]
    # The dominant measured horizontal height supplies a support-plane hypothesis.
    counts, edges = np.histogram(
        z[valid], bins=np.arange(bounds[2, 0], bounds[2, 1] + 0.005, 0.005)
    )
    index = int(np.argmax(counts))
    plane_z = float((edges[index] + edges[index + 1]) / 2)
    h, w = valid.shape
    roi_mask = np.ones((h, w), dtype=bool)
    if config.target_roi is not None:
        r0, c0, r1, c1 = config.target_roi
        roi_mask[:] = False
        roi_mask[int(r0 * h) : int(r1 * h), int(c0 * w) : int(c1 * w)] = True
        mask = valid & roi_mask
        source = "operator_rgb_roi+measured_depth"
    else:
        color = rgb.astype(float) / 255
        brightness = color.mean(axis=2)
        saturation = color.max(axis=2) - color.min(axis=2)
        if receptacle in {"plate", "tray"}:
            color_mask = (brightness > 0.55) & (saturation < 0.30)
            min_height = 0.004
        else:
            color_mask = brightness < 0.60
            min_height = 0.012
        mask = valid & color_mask & (z > plane_z + min_height) & (z < plane_z + 0.20)
        # Remove the robot's vicinity using observed EEF state, not body geometry.
        mask &= np.linalg.norm(world[..., :2] - eef[:2], axis=2) > 0.055
        mask = ndimage.binary_closing(mask, iterations=2)
        mask = ndimage.binary_fill_holes(mask) & valid
        source = "rgb_color+depth_height+connected_components"
    labels, count = ndimage.label(mask)
    regions = []
    for label in range(1, count + 1):
        component = labels == label
        if np.count_nonzero(component) < max(40, h * w * 0.001):
            continue
        rows, cols = np.where(component)
        samples = world[component]
        xy_min, xy_max = np.quantile(samples[:, :2], [0.05, 0.95], axis=0)
        size = xy_max - xy_min
        if np.any(size < config.region_min_width) or np.any(
            size > config.region_max_width
        ):
            continue
        # Sample an interior, away from the visible component boundary.
        distances = ndimage.distance_transform_edt(component)
        interior = component & (distances >= max(2.0, float(distances.max()) * 0.55))
        interior_samples = world[interior]
        if len(interior_samples) < 10:
            continue
        # Front walls dominate image-space medians. Derive the placement XY
        # from the measured footprint, rather than aiming at that visible wall.
        center_xy = (xy_min + xy_max) / 2
        inner_xy = np.all(np.abs(samples[:, :2] - center_xy) <= size * 0.30, axis=1)
        support = samples[inner_xy]
        if len(support) < 10:
            support = interior_samples
        xyz = np.array(
            [
                center_xy[0],
                center_xy[1],
                float(np.quantile(support[:, 2], 0.25)),
            ]
        )
        if not in_workspace(xyz, config):
            continue
        # The marker names the image proposal; XYZ is a footprint-center
        # hypothesis and need not coincide with a directly observed surface.
        pixel = (int((rows.min() + rows.max()) / 2), int((cols.min() + cols.max()) / 2))
        regions.append(
            Region(
                id="",
                bbox=(
                    int(rows.min()),
                    int(cols.min()),
                    int(rows.max()) + 1,
                    int(cols.max()) + 1,
                ),
                pixel=(int(pixel[0]), int(pixel[1])),
                xyz=tuple(float(v) for v in xyz),
                rim_z=float(np.quantile(samples[:, 2], 0.95)),
                width_xy=tuple(float(v) for v in size),
                valid_pixels=len(interior_samples),
                source=source + "+footprint_center+inner_depth_quartile",
            )
        )
    regions.sort(
        key=lambda region: (-region.width_xy[0] * region.width_xy[1], region.pixel)
    )
    return [
        Region(**{**asdict(region), "id": f"r{index}"})
        for index, region in enumerate(regions[: config.max_regions])
    ], plane_z


def mark_regions(rgb: np.ndarray, regions: list[Region]) -> np.ndarray:
    """Mark public RGB proposals so option descriptions have visible anchors."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    for region in regions:
        r0, c0, r1, c1 = region.bbox
        row, col = region.pixel
        draw.rectangle((c0, r0, c1 - 1, r1 - 1), outline=(255, 40, 40), width=2)
        draw.ellipse((col - 3, row - 3, col + 3, row + 3), fill=(255, 40, 40))
        draw.text(
            (c0 + 2, r0 + 2),
            region.id,
            fill=(255, 255, 0),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    return np.asarray(image)
