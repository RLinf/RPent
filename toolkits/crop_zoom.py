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

"""Inspect a region of a recorded image while retaining its source coordinates."""

import io
from typing import Annotated

from PIL import Image, ImageDraw
from pydantic import Field

from rpent.tools import ToolResult, tool
from toolkits.observation import (
    GeometryObservation,
    ImageSelection,
    Index,
    ObservationProvider,
    Size,
)

Pixel = Annotated[list[Index], Field(min_length=2, max_length=2)]
MAX_CROP_EDGE = 1024
_MARK_COLOR = (255, 220, 0)


@tool(readonly=True, exclude=("observations",))
def crop_zoom(
    camera: Annotated[str, Field(min_length=1, max_length=128)],
    image_shape: Annotated[list[Size], Field(min_length=2, max_length=2)],
    box_rc: Annotated[list[Index], Field(min_length=4, max_length=4)],
    step: Annotated[int, Field(ge=-1)] = -1,
    zoom: Annotated[int, Field(strict=True, ge=1, le=8)] = 2,
    grid_step: Size | None = None,
    marks_rc: Annotated[list[Pixel], Field(min_length=1, max_length=16)] | None = None,
    *,
    observations: ObservationProvider,
) -> ToolResult:
    """Magnify part of a recorded camera image to inspect small details.

    The PNG goes in the image result. Its [row, column] pixel maps to the
    original image by dividing each index by zoom with floor rounding, then
    adding the top and left of source_box_rc. Use the returned camera, step
    and image_shape for subsequent full-frame tools.
    With marks, label_boxes_rc lists each number's half-open box in the PNG,
    in mark order. A null box means the number could not fit without overlap.
    The mark's coordinates remain available. Increase zoom or use fewer marks.

    Args:
        camera: Name of the recorded camera.
        image_shape: [height, width] of the full frame used to select the box.
        box_rc: [top, left, bottom, right] on that frame, bottom and right exclusive.
        step: Recorded step, or -1 for the latest record.
        zoom: Integer magnification of the native crop, 2 by default. Choose a
            tighter box or lower zoom if an output edge would exceed 1024 pixels.
        grid_step: Optional grid spacing in original image pixels, measured
            from the native crop's top-left corner.
        marks_rc: Up to 16 points inside the box on the declared full frame,
            as [row, column] pairs. Marks must map to distinct native pixels.
            Numbering starts at 1 in the given order.
        observations: Provider for recorded camera observations.
    """
    try:
        box = ImageSelection(image_shape=image_shape, box_rc=box_rc)
        points = (
            ImageSelection(image_shape=image_shape, points_rc=marks_rc)
            if marks_rc is not None
            else None
        )
        top, left, bottom, right = box_rc
        if any(
            not (top <= row < bottom and left <= col < right)
            for row, col in marks_rc or ()
        ):
            raise ValueError("Each mark must lie inside box_rc on the declared image.")
        obs = observations.get_observation(camera=camera, step=step)
        if obs.camera != camera or (step >= 0 and obs.step != step):
            raise ValueError("Provider returned a different camera or step.")
        return _render_crop(obs, box, points, zoom=zoom, grid_step=grid_step)
    except (ValueError, LookupError, OSError, RuntimeError) as exc:
        return ToolResult(data={"status": "error", "error": str(exc)})


def _render_crop(
    obs: GeometryObservation,
    box: ImageSelection,
    points: ImageSelection | None,
    *,
    zoom: int,
    grid_step: int | None,
) -> ToolResult:
    _, native_box = box.native(obs)
    top, left, bottom, right = native_box
    width, height = (right - left) * zoom, (bottom - top) * zoom
    if max(width, height) > MAX_CROP_EDGE:
        raise ValueError(
            f"Crop output would be {height}x{width} pixels. "
            f"Choose a smaller box or zoom for a maximum edge of {MAX_CROP_EDGE} pixels."
        )

    image = Image.fromarray(obs.rgb[top:bottom, left:right])
    image = image.resize((width, height), Image.Resampling.NEAREST)
    if grid_step is not None:
        draw = ImageDraw.Draw(image, "RGBA")
        spacing = grid_step * zoom
        grid_color = (32, 180, 255, 140)
        for row in range(0, height, spacing):
            draw.line((0, row, width - 1, row), fill=grid_color)
        for col in range(0, width, spacing):
            draw.line((col, 0, col, height - 1), fill=grid_color)

    marks = []
    label_data = {}
    if points is not None:
        native_points, _ = points.native(obs)
        if len(set(native_points)) != len(native_points):
            raise ValueError(
                "Two marks map to the same native pixel. Choose distinct pixels."
            )
        centers = [
            ((row - top) * zoom + zoom // 2, (col - left) * zoom + zoom // 2)
            for row, col in native_points
        ]
        for number, ((row, col), (crop_row, crop_col)) in enumerate(
            zip(native_points, centers, strict=True), start=1
        ):
            marks.append(
                {
                    "number": number,
                    "source_point_rc": [row, col],
                    "crop_point_rc": [crop_row, crop_col],
                }
            )

        boxes = _draw_labels(ImageDraw.Draw(image), centers, width=width, height=height)
        label_data["label_boxes_rc"] = [
            list(box) if box is not None else None for box in boxes
        ]
        if None in boxes:
            label_data["label_hint"] = (
                "Some numbers could not fit. Increase zoom or use fewer marks."
            )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return ToolResult(
        data={
            "status": "cropped",
            "camera": obs.camera,
            "step": obs.step,
            "image_shape": list(obs.rgb.shape[:2]),
            "source_box_rc": list(native_box),
            "crop_shape": [height, width],
            "zoom": zoom,
            "grid_step": grid_step,
            "marks": marks,
            **label_data,
        },
        images=[buffer.getvalue()],
    )


def _draw_labels(
    draw: ImageDraw.ImageDraw,
    centers: list[tuple[int, int]],
    *,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int] | None]:
    occupied = [(r, c, r + 1, c + 1) for r, c in centers]
    boxes: list[tuple[int, int, int, int] | None] = []
    labels = []
    for number, (row, col) in enumerate(centers, start=1):
        draw.ellipse(
            (col - 3, row - 3, col + 3, row + 3), fill=_MARK_COLOR, outline="black"
        )
        label = str(number)
        left, top, right, bottom = draw.textbbox((0, 0), label, stroke_width=1)
        w, h = right - left, bottom - top
        box = None
        if w <= width and h <= height:
            # Try a small neighborhood in a fixed order, including clipped edges.
            for dx, dy in (
                (4, -h // 2),
                (-w - 4, -h // 2),
                (-w // 2, -h - 4),
                (-w // 2, 4),
                (4, -h - 4),
                (-w - 4, -h - 4),
                (4, 4),
                (-w - 4, 4),
                (-2 * w - 4, -h - 4),
                (-w - 4, -2 * h - 4),
                (w + 4, -h // 2),
                (-2 * w - 4, -h // 2),
                (-w // 2, h + 4),
            ):
                x = min(max(0, col + dx), width - w)
                y = min(max(0, row + dy), height - h)
                if any(
                    y < b and t < y + h and x < r and l < x + w
                    for t, l, b, r in occupied
                ):
                    continue
                box = (y, x, y + h, x + w)
                occupied.append(box)
                labels.append((label, x - left, y - top))
                break
        boxes.append(box)

    for (row, col), box in zip(centers, boxes, strict=True):
        if box is not None:
            top, left, bottom, right = box
            x = min(max(col, left), right - 1)
            y = min(max(row, top), bottom - 1)
            draw.line((col, row, x, y), fill=_MARK_COLOR)
    for label, x, y in labels:
        draw.text((x, y), label, fill=_MARK_COLOR, stroke_width=1, stroke_fill="black")
    for row, col in centers:
        draw.point((col, row), fill=_MARK_COLOR)
    return boxes
