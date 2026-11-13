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

"""Read and render a configured Nav2 map without starting ROS or devices."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from robots.lekiwi.runtime_config import load_mapping


def encode_png(image: Image.Image) -> np.ndarray:
    """Encode a bounded image for the existing NumPy RPC transport."""
    image = image.copy()
    image.thumbnail((1024, 1024))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return np.frombuffer(output.getvalue(), dtype=np.uint8).copy()


def render_map(map_yaml, page: dict, navigation: dict, *, radius_m=None) -> dict:
    """Overlay place labels, the selected node and a fresh robot pose on a grid."""
    path = Path(map_yaml).expanduser().resolve()
    config = load_mapping(path)
    resolution = float(config["resolution"])
    origin = np.asarray(config["origin"], dtype=float)
    occupied, free = float(config["occupied_thresh"]), float(config["free_thresh"])
    negate = config.get("negate", 0)
    if (
        not math.isfinite(resolution)
        or resolution <= 0
        or origin.shape != (3,)
        or not np.isfinite(origin).all()
        or not 0 <= free < occupied <= 1
        or negate not in (0, 1)
    ):
        raise ValueError("Invalid occupancy map metadata")
    if config.get("mode", "trinary") != "trinary":
        raise ValueError("Map viewing currently requires a trinary occupancy map")
    with Image.open(path.parent / config["image"]) as source:
        rgba = np.asarray(source.convert("RGBA"))
    probability = rgba[:, :, :3].mean(axis=2) / 255
    if not negate:
        probability = 1 - probability
    pixels = np.full(probability.shape, 205, dtype=np.uint8)
    pixels[probability > occupied] = 0
    pixels[probability < free] = 255
    pixels[rgba[:, :, 3] < 255] = 205
    image = Image.fromarray(pixels).convert("RGB")
    width, height = image.size
    cosine, sine = math.cos(origin[2]), math.sin(origin[2])

    def pixel(pose):
        dx, dy = pose[0] - origin[0], pose[1] - origin[1]
        return (
            (cosine * dx + sine * dy) / resolution - 0.5,
            height - 0.5 - (-sine * dx + cosine * dy) / resolution,
        )

    center = page.get("center")
    crop = (0, 0, width, height)
    if radius_m is not None:
        cx, cy = pixel(center["pose"])
        radius = radius_m / resolution
        crop = (
            max(0, math.floor(cx - radius)),
            max(0, math.floor(cy - radius)),
            min(width, math.ceil(cx + radius + 1)),
            min(height, math.ceil(cy + radius + 1)),
        )
        if crop[0] >= crop[2] or crop[1] >= crop[3]:
            raise ValueError("Requested node region does not overlap the occupancy map")
    image = image.crop(crop)
    original_size = image.size
    image.thumbnail((1024, 1024), Image.Resampling.NEAREST)
    sx, sy = image.width / original_size[0], image.height / original_size[1]
    draw = ImageDraw.Draw(image)

    def displayed(pose):
        x, y = pixel(pose)
        return (x - crop[0] + 0.5) * sx - 0.5, (y - crop[1] + 0.5) * sy - 0.5

    def inside(point):
        return 0 <= point[0] < image.width and 0 <= point[1] < image.height

    candidates = {n["node_id"]: n for n in page["nodes"]}
    if center is not None:
        candidates[center["node_id"]] = center
    markers = []
    for node in candidates.values():
        x, y = displayed(node["pose"])
        if not inside((x, y)):
            continue
        markers.append(dict(node, image_pixel=[x, y]))
        selected = center is not None and node["node_id"] == center["node_id"]
        label = node.get("label", "")
        if selected:
            label = f"{label} ({node['node_id']})" if label else str(node["node_id"])
        elif not label:
            continue
        color = "#df7300" if selected else "#1766cc"
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.text(
            (x + 5, y - 5),
            label,
            fill=color,
            stroke_width=1,
            stroke_fill="white",
        )

    robot = {
        "valid": False,
        "reason": navigation.get("reason", "localization_unavailable"),
    }
    pose = navigation.get("pose")
    if navigation.get("ready") and pose is not None:
        x, y = displayed(pose)
        robot = {
            "valid": True,
            "pose": pose,
            "source_age_s": navigation.get("source_age_s"),
            "visible": inside((x, y)),
        }
        if robot["visible"]:
            angle = pose[2] - origin[2]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#008040")
            draw.line(
                (x, y, x + 18 * math.cos(angle), y - 18 * math.sin(angle)),
                fill="#008040",
                width=3,
            )
            draw.text(
                (x + 6, y + 6),
                "robot",
                fill="#008040",
                stroke_width=1,
                stroke_fill="white",
            )

    return {
        "frame_id": "map",
        "map_kind": "static_occupancy",
        "resolution_m": resolution,
        "origin": origin.tolist(),
        "map_size_pixels": [width, height],
        "crop_pixels": list(crop),
        "image_size_pixels": list(image.size),
        "nodes": markers,
        "nodes_truncated": page["nodes_truncated"],
        "robot": robot,
        "center_node_id": center["node_id"] if center else None,
        "legend": {
            "black": "occupied",
            "white": "free",
            "gray": "unknown",
            "blue": "place label",
            "orange": "selected node and ID",
            "green": "robot and heading",
        },
        "image_png": encode_png(image),
    }
