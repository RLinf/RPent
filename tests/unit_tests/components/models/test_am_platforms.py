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

"""Robots own the wire contracts; capabilities keep matching copies."""

from components.sim.libero import control as libero_control
from components.sim.robotwin import control as robotwin_control
from components.sim.robotwin import observation as robotwin_observation
from robots.yam import contracts as yam_contracts
from rpent.components.models import platforms as am_platforms


def test_robot_and_capability_copies_agree():
    assert libero_control.LIBERO_OSC == am_platforms.LIBERO_OSC
    assert robotwin_control.ROBOTWIN_QPOS == am_platforms.ROBOTWIN_QPOS
    assert robotwin_control.ROBOTWIN_EEF == am_platforms.ROBOTWIN_EEF
    assert robotwin_control.ROBOTWIN_CONTROLLERS == am_platforms.ROBOTWIN_CONTROLLERS
    assert robotwin_observation.ROBOTWIN_CAMERAS == am_platforms.ROBOTWIN_CAMERAS
    assert yam_contracts.YAM_VLA_CONTRACT == am_platforms.YAM_VLA_CONTRACT


def test_capability_copies_are_independent_of_the_robot_modules():
    # A capability may not import a robot integration, so the two sides are
    # separate objects that the drift test above keeps equal.
    assert libero_control.LIBERO_OSC is not am_platforms.LIBERO_OSC
    assert robotwin_control.ROBOTWIN_QPOS is not am_platforms.ROBOTWIN_QPOS
    assert robotwin_observation.ROBOTWIN_CAMERAS is not am_platforms.ROBOTWIN_CAMERAS


def test_controllers_are_indexed_by_action_space():
    assert set(am_platforms.ROBOTWIN_CONTROLLERS) == {
        am_platforms.ROBOTWIN_QPOS["action_space"],
        am_platforms.ROBOTWIN_EEF["action_space"],
    }
    assert (
        am_platforms.ROBOTWIN_CONTROLLERS[am_platforms.ROBOTWIN_EEF["action_space"]]
        is am_platforms.ROBOTWIN_EEF
    )


def test_yam_robot_contract_is_a_copy_of_the_capability_contract():
    assert yam_contracts.vla_runtime_contract() == am_platforms.YAM_VLA_CONTRACT
    assert yam_contracts.vla_runtime_contract() is not am_platforms.YAM_VLA_CONTRACT
    assert am_platforms.YAM_VLA_CONTRACT["camera_order"] == ["top", "left", "right"]
    assert am_platforms.YAM_VLA_CONTRACT["state_layout"] == "qpos14"
    assert am_platforms.YAM_VLA_CONTRACT["action_layout"] == "qpos14"
    assert am_platforms.YAM_VLA_CONTRACT["action_dim"] == 14
