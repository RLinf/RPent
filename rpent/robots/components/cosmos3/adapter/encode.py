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

"""Map physical LIBERO observations to RLinf's Cosmos3 adapter."""

from typing import Any

import numpy as np


def encode_libero(request: dict, *, policy: Any) -> dict:
    """Let RLinf compose images and construct the native action request."""
    # Physical views are vertically flipped; RLinf expects a 180-degree rotation.
    observation = {
        "main_images": np.ascontiguousarray(request["images"]["primary"][:, ::-1])[
            None
        ],
        "wrist_images": np.ascontiguousarray(request["images"]["wrist"][:, ::-1])[None],
        "task_descriptions": [request["instruction"]],
    }
    payload, state = policy.build_request(observation, mode="eval")
    return {"payload": payload, "state": state}
