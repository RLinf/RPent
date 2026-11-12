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

"""Real requests to existing BEHAVIOR Env/VLA services; no process ownership."""

from __future__ import annotations

import os
import unittest
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from robots.behavior.motion import R1ProMotionModel
from tests.e2e_tests.common import publish_check

if TYPE_CHECKING:
    from tests.e2e_tests.behavior.scenario import BehaviorScenario


def test_environment_component(behavior_scenario: BehaviorScenario) -> None:
    _, result = behavior_scenario.environment
    publish_check("behavior_environment_component", result)


def test_vla_component(behavior_scenario: BehaviorScenario) -> None:
    publish_check("behavior_vla_component", behavior_scenario.vla)


class TestBehaviorMotionModel(unittest.TestCase):
    def setUp(self):
        if os.environ.get("RPENT_RUN_BEHAVIOR_INTEGRATION") != "1":
            self.skipTest(
                "set RPENT_RUN_BEHAVIOR_INTEGRATION=1 and OMNIGIBSON_DATA_PATH"
            )
        self.model = R1ProMotionModel()
        self.state = np.zeros(61)
        self.state[24:26] = [0.02, 0.03]
        self.state[49:51] = [0.035, 0.045]

    def tearDown(self):
        self.model.close()

    def test_zero_pose_and_sourced_eef_frame(self):
        expected = {
            "left": [-0.07951818, 0.25199935, 0.65659],
            "right": [-0.07951906, -0.25200065, 0.65659],
        }
        for arm in ("left", "right"):
            pose = self.model.forward(arm, self.state)
            np.testing.assert_allclose(pose["position"], expected[arm], atol=1e-10)
            np.testing.assert_allclose(
                Rotation.from_quat(pose["quaternion_xyzw"]).as_matrix(),
                np.diag([-1, 1, -1]),
                atol=1e-12,
            )
            self.assertEqual(pose["frame"], "robot_base")

    def test_torso_joint3_uses_urdf_negative_axis_once(self):
        self.state[55] = 0.3
        pivot = np.asarray([-0.079032, 0.0001005, 1.04265])
        old = np.asarray([-0.07951818, 0.25199935, 0.65659])
        rotation = Rotation.from_euler("y", -0.3)
        pose = self.model.forward("left", self.state)
        np.testing.assert_allclose(
            pose["position"], pivot + rotation.apply(old - pivot), atol=1e-10
        )
        np.testing.assert_allclose(
            Rotation.from_quat(pose["quaternion_xyzw"]).as_matrix(),
            rotation.as_matrix() @ np.diag([-1, 1, -1]),
            atol=1e-12,
        )
