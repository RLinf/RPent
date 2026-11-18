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

"""Native Sawyer control identity for action-model adapters."""

from rpent.robots.components.am_contracts import AMControlSpec

# At 80 Hz, XYZ commands in [-1, 1] request world-frame deltas scaled by
# 0.01 m. Wrist orientation is fixed; +1 closes and -1 opens the gripper.
METAWORLD_CONTROL = AMControlSpec(
    embodiment="metaworld_sawyer_4d",
    action_space="metaworld.cartesian.delta.v1",
    action_schema=["delta_x", "delta_y", "delta_z", "gripper"],
    action_type=None,
)
