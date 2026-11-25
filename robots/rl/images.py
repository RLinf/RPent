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


def map_box(
    box: list[int], source: tuple[int, int], target: tuple[int, int]
) -> list[int]:
    """Map validated half-open rectangle edges outward."""
    return [
        (math.floor if index < 2 else math.ceil)(
            value * target[index % 2] / source[index % 2]
        )
        for index, value in enumerate(box)
    ]


def map_pixel(
    pixel: list[int], source: tuple[int, int], target: tuple[int, int]
) -> list[int]:
    """Map a validated displayed pixel centre to the native observation."""
    return [
        min(to - 1, math.floor((value + 0.5) * to / from_))
        for value, from_, to in zip(pixel, source, target)
    ]
