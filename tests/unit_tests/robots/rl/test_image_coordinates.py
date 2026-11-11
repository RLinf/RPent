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

"""Display coordinates remain tied to the native depth frame."""

import io

import pytest
from PIL import Image

from robots.rl.images import native_pixel, resize_image, scale_box


@pytest.mark.parametrize("size", [1024, 512, 256])
def test_display_resize_and_native_pixel_centre(size):
    stream = io.BytesIO()
    Image.new("RGB", (1024, 1024)).save(stream, format="PNG")
    resized = resize_image(stream.getvalue(), size)
    assert Image.open(io.BytesIO(resized)).size == (size, size)
    assert native_pixel([0, size - 1], size) == [
        1024 // size // 2,
        1024 - (1024 // size + 1) // 2,
    ]
    assert scale_box([0, 0, size, size], size, 1024) == [0, 0, 1024, 1024]
    with pytest.raises(ValueError, match="outside"):
        native_pixel([size, 0], size)


def test_half_open_boxes_round_outward():
    assert scale_box([1, 2, 5, 7], 1024, 256) == [0, 0, 2, 2]
    assert scale_box([0, 0, 2, 2], 256, 1024) == [0, 0, 8, 8]
