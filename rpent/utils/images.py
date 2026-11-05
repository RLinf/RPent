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

"""Frame encoding shared by the services that accept an image over RPC."""

from __future__ import annotations

import io

import imageio.v2 as imageio
import numpy as np


def encode_rgb_png(image: np.ndarray) -> bytes:
    """Encode an RGB camera frame as PNG bytes.

    Service requests carry one PNG. Camera frames are ``uint8`` RGB; accepting
    any other layout here would either fail deep inside the encoder or reach the
    service as a different image than the caller saw.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"image must be RGB [H,W,3], got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"image must be a uint8 array, got {array.dtype}")
    buffer = io.BytesIO()
    imageio.imwrite(buffer, array, format="png")
    return buffer.getvalue()
