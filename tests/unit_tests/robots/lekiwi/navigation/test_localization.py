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

"""Localization freshness and invalid observations, without ROS or hardware."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robots.lekiwi.navigation.localization import Localization


def test_service_startup_retries_and_can_stop(monkeypatch):
    from robots.lekiwi.navigation import localization as module

    client = Mock()
    client.call.side_effect = [ConnectionError("loading map"), {"ok": True}]
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    module.wait_for_localization(client, lambda: False)
    assert client.call.call_count == 2
    client.call.assert_called_with("healthz", timeout_s=1)
    client.reset_mock()
    module.wait_for_localization(client, lambda: True)
    client.call.assert_not_called()


def test_service_startup_timeout_preserves_failure():
    from robots.lekiwi.navigation.localization import wait_for_localization

    client = Mock()
    client.call.side_effect = ConnectionError("map loading failed")
    with pytest.raises(TimeoutError, match="did not become ready") as failure:
        wait_for_localization(client, lambda: False, timeout_s=0)
    assert isinstance(failure.value.__cause__, ConnectionError)


def test_ros_odometry_filters_frames_and_interpolates_capture_pose():
    state = Localization({"T_base_camera": np.eye(4).tolist()})

    def message(nanosec, x, y, yaw):
        return SimpleNamespace(
            header=SimpleNamespace(
                frame_id="odom", stamp=SimpleNamespace(sec=10, nanosec=nanosec)
            ),
            child_frame_id="base_link",
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=x, y=y),
                    orientation=SimpleNamespace(
                        x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)
                    ),
                )
            ),
        )

    first = message(200_000_000, 1, 2, math.radians(170))
    first.header.frame_id = "map"
    state.receive_odometry(first)
    first.header.frame_id = "odom"
    first.child_frame_id = "camera"
    state.receive_odometry(first)
    assert not state.odometry

    first.child_frame_id = "base_link"
    state.receive_odometry(first)
    state.receive_odometry(message(400_000_000, 3, 4, math.radians(-170)))
    np.testing.assert_allclose(
        state.odom_at(10.3), [[-1, 0, 2], [0, -1, 3], [0, 0, 1]], atol=1e-12
    )
    for stamp in (10.1, 10.5):
        with pytest.raises(ValueError, match="No bracketing fresh odometry"):
            state.odom_at(stamp)


def fix(stamp):
    return {
        "stamp_ns": round(stamp * 1e9),
        "clock_domain": "ros",
        "map_frame_id": "map",
        "camera_frame_id": "camera_optical_frame",
        "camera_calibrated": True,
        "status": "localized",
        "T_map_camera": np.eye(4).tolist(),
    }


@pytest.fixture
def localization():
    state = Localization(
        {
            "T_base_camera": np.eye(4).tolist(),
            "localization_max_age": 2,
            "max_correction_translation": 0.5,
            "max_correction_yaw_deg": 30,
        }
    )
    for stamp in (9.9, 10.1, 10.3):
        state.add_odometry(stamp, 0, 0, 0)
    assert state.accept(fix(10), now=10.1)
    return state


@pytest.mark.parametrize(
    "kind", ["unmatched", "duplicate", "expired", "missing_odometry"]
)
def test_rejected_observation_does_not_renew_or_invalidate_good_fix(localization, kind):
    result = fix(10.2)
    if kind == "unmatched":
        result["status"] = "lost"
    elif kind == "duplicate":
        result["stamp_ns"] = 10_000_000_000
    elif kind == "expired":
        result["stamp_ns"] = 7_000_000_000
    else:
        result["stamp_ns"] = 10_500_000_000
    assert not localization.accept(result, now=10.6)
    assert localization.ready(10.6)
    assert localization.accepted_stamp == 10
    assert not localization.ready(12.1)
    assert localization.accept(fix(10.2), now=10.7)
    assert localization.ready(12.1)


@pytest.mark.parametrize("kind", ["wrong_frame", "correction_jump"])
def test_invalid_identity_or_jump_requires_a_new_good_fix(localization, kind):
    result = fix(10.2)
    if kind == "wrong_frame":
        result["map_frame_id"] = "other_map"
    else:
        result["T_map_camera"][0][3] = 2
    assert not localization.accept(result, now=10.3)
    assert not localization.ready(10.3)
    assert localization.accept(fix(10.2), now=10.4)
    assert localization.ready(10.4)
