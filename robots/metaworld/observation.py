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

"""Physical observations without model normalization or privileged task state."""

import numpy as np

from rpent.robots.components.am_contracts import AMObservation


def physical_observation(obs: dict) -> AMObservation:
    """Expose top-left RGB, world XYZ in metres and normalized gripper opening."""
    return {
        "images": {"primary": np.ascontiguousarray(obs["rgb"])},
        "state": {
            "eef_position": np.asarray(obs["eef_position"]),
            "gripper_opening": np.asarray([obs["gripper_opening"]]),
        },
        "instruction": obs["instruction"],
    }
