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

"""Planner image coordinates over unchanged native RGB-D observations."""

from __future__ import annotations

import io
import math

from PIL import Image

IMAGE_SIZES = (1024, 512, 256)
NATIVE_SIZE = 1024


def resize_image(data: bytes, size: int) -> bytes:
    """Resize one planner image without changing the recorded native image."""
    with Image.open(io.BytesIO(data)) as image:
        if image.size == (size, size):
            return data
        output = io.BytesIO()
        image.resize((size, size), Image.Resampling.LANCZOS).save(output, format="PNG")
        return output.getvalue()


def scale_box(box: list[int], source: int, target: int) -> list[int]:
    """Map half-open rectangle edges outward, retaining their covered region."""
    return [
        max(
            0,
            min(
                target,
                math.floor(value * target / source)
                if index < 2
                else math.ceil(value * target / source),
            ),
        )
        for index, value in enumerate(box)
    ]


def native_pixel(pixel: list[int], size: int) -> list[int]:
    """Map a displayed pixel centre into the unchanged native depth image."""
    if any(value < 0 or value >= size for value in pixel):
        raise ValueError(f"pixel outside {size}x{size} planner image")
    return [
        min(NATIVE_SIZE - 1, math.floor((value + 0.5) * NATIVE_SIZE / size))
        for value in pixel
    ]
