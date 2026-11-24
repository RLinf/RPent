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

"""RoboTwin EEF16 values the LingBot-VLA runtime requires.

The robot slice reads the same values when it negotiates the server's runtime
identity, so they live with the model instead of the robot.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoboTwinModelSpec:
    """Values required by the LingBot EEF runtime."""

    policy_name: str
    robot_config_relpath: str
    norm_stats: str
    qwen_base: str
    camera_order: tuple[str, ...]
    state_layout: str
    action_layout: str
    use_length: int


MODEL_SPEC = RoboTwinModelSpec(
    policy_name="robotwin_eef",
    robot_config_relpath="configs/robot_configs/robotwin_eef.yaml",
    norm_stats="norm_stats/robotwin_eef.json",
    qwen_base="qwen_base",
    camera_order=("cam_high", "cam_left_wrist", "cam_right_wrist"),
    state_layout="eef16",
    action_layout="eef16",
    use_length=50,
)


def vla_runtime_contract() -> dict[str, object]:
    """Return the identity required from a LingBot RoboTwin server.

    Only the hard contract fields that break inference input/output when
    mismatched are kept; descriptive behaviour fields live with the
    implementation (facade/transport), not in the runtime-equality check.
    """
    return {
        "runtime": "lingbotvla",
        "policy_name": MODEL_SPEC.policy_name,
        "camera_order": list(MODEL_SPEC.camera_order),
        "state_layout": MODEL_SPEC.state_layout,
        "action_layout": MODEL_SPEC.action_layout,
        "use_length": MODEL_SPEC.use_length,
    }
