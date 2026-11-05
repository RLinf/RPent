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

"""Cosmos Policy native predictions to platform execution actions."""

from typing import Any

import numpy as np

from rpent.robots.components.am_contracts import WAMPrediction


def decode_libero(
    result: dict[str, Any], *, predict_future: bool = False
) -> WAMPrediction:
    """The official get_action already unnormalizes LIBERO actions."""
    return WAMPrediction(
        actions=np.asarray(result["actions"], dtype=np.float32),
        future_observation=result.get("future_image_predictions")
        if predict_future
        else None,
        value=result.get("value_prediction") if predict_future else None,
    )
