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

"""Measure local depth discontinuities in calibrated RGB-D samples."""

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


def local_depth_range(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return the measured 3x3 depth range at each valid pixel."""
    if valid.all():
        return maximum_filter(depth, 3) - minimum_filter(depth, 3)
    high = maximum_filter(np.where(valid, depth, -np.inf), 3)
    low = minimum_filter(np.where(valid, depth, np.inf), 3)
    spread = high - low
    spread[~valid] = np.nan
    return spread
