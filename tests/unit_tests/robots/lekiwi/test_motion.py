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

"""Timed drive semantics and signed turns from measured angular velocity."""

import numpy as np
import pytest

from robots.lekiwi.motion import Motion


@pytest.mark.parametrize("angle", [0, 30, -30, 360, -360])
def test_turn_reaches_signed_angle_from_slow_feedback(angle):
    motion = Motion(np.array([0.0, 0.0, 30.0]), angle_deg=angle)
    velocity = np.zeros(3)
    for tick in range(1200):
        now = tick * 0.05
        motion.sample(velocity, now)
        action, done = motion.command(now)
        assert not action[:2].any()
        assert abs(action[2]) <= 30
        velocity = action * 0.6
        if done:
            break
    else:
        pytest.fail("turn did not converge")
    assert abs(angle - motion.estimated_angle_deg) <= 2
    if angle:
        assert now > abs(angle) / 30


def test_drive_duration_starts_with_first_command():
    motion = Motion(np.array([0.1, 0, 0]), duration_s=1)
    assert not motion.command(5)[1]
    assert not motion.command(5.9)[1]
    assert motion.command(6)[1]
    assert "estimated_angle_deg" not in motion.result(6)


@pytest.mark.parametrize("stamp", [0, -0.1])
def test_out_of_order_feedback_is_not_integrated(stamp):
    motion = Motion(np.array([0.0, 0.0, 30.0]), angle_deg=90)
    motion.sample(np.array([0, 0, 10]), 0)
    with pytest.raises(RuntimeError, match="feedback"):
        motion.sample(np.array([0, 0, 10]), stamp)
    assert motion.estimated_angle_deg == 0


def test_rotation_integrates_the_full_sampling_interval():
    motion = Motion(np.array([0.0, 0.0, 30.0]), angle_deg=90)
    motion.sample(np.array([0, 0, 10]), 0)
    motion.sample(np.array([0, 0, 20]), 0.3)
    assert motion.estimated_angle_deg == pytest.approx(4.5)
    assert motion.feedback_valid
