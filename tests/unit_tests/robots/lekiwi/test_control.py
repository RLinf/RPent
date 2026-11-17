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

"""Wheel-feedback integration without starting hardware or a control thread."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from robots.lekiwi.control import EnvControl


@pytest.fixture
def odometry(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("robots.lekiwi.control.time.monotonic", lambda: clock[0])
    raw = Mock(config=SimpleNamespace(step_frequency=10))
    control = EnvControl(None, raw, max_linear=1, max_angular=1, command_timeout=0.3)

    def sample(stamp, velocity):
        clock[0] = stamp
        raw.get_velocity_sample.return_value = {
            "velocity": velocity,
            "sampled_at": stamp + 1000,
            "sampled_monotonic": stamp,
        }
        control._update_sample()
        return control.get_odometry()

    return control, sample, clock


def test_odometry_uses_feedback_and_reads_do_not_integrate(odometry):
    control, sample, clock = odometry
    assert control.get_odometry()["valid"] is False
    sample(10, [0.2, -0.1, 0])
    pose = sample(10.5, [0.2, -0.1, 0])
    assert pose["valid"]
    assert pose["x_m"] == pytest.approx(0.1)
    assert pose["y_m"] == pytest.approx(-0.05)
    assert pose["yaw_deg"] == 0
    assert control._target == [0, 0, 0]
    assert sample(10.5, [9, 9, 9]) == pose
    clock[0] = 10.6
    assert control.get_odometry()["x_m"] == pose["x_m"]
    control.raw.stop.assert_not_called()


def test_turn_keeps_translation_in_task_start_frame(odometry):
    control, sample, _ = odometry
    sample(10, [0, 0, 0])
    sample(10.5, [0, 0, 180])
    assert sample(11, [0, 0, 0])["yaw_deg"] == pytest.approx(90)
    sample(11.5, [0.2, 0, 0])
    pose = sample(12, [0, 0, 0])
    assert pose["x_m"] == pytest.approx(0, abs=1e-12)
    assert pose["y_m"] == pytest.approx(0.1)
    control.reset_odometry()
    sample(12.5, [0.2, 0, 0])
    pose = sample(13, [0, 0, 0])
    assert pose["x_m"] == pytest.approx(0.1)
    assert pose["y_m"] == pytest.approx(0)
    assert pose["yaw_deg"] == 0


@pytest.mark.parametrize("bad_stamp", [9.9, 12])
def test_missing_or_reversed_feedback_stays_invalid_until_reset(odometry, bad_stamp):
    control, sample, _ = odometry
    sample(10, [0.2, 0, 0])
    pose = sample(bad_stamp, [0.2, 0, 0])
    assert not pose["valid"]
    assert "x_m" not in pose
    assert not sample(max(10, bad_stamp) + 0.1, [0, 0, 0])["valid"]
    control.reset_odometry()
    assert control.get_odometry()["valid"]
    assert control.get_odometry()["x_m"] == 0


@pytest.mark.parametrize("unavailable", ["stale", "fault", "closed"])
def test_unavailable_feedback_omits_pose(odometry, unavailable):
    control, sample, clock = odometry
    sample(10, [0, 0, 0])
    if unavailable == "stale":
        clock[0] = 12
    elif unavailable == "fault":
        control._fault = "wheel read failed"
    else:
        control._closing.set()
    pose = control.get_odometry()
    assert not pose["valid"]
    assert "x_m" not in pose
    assert "y_m" not in pose
    assert "yaw_deg" not in pose
