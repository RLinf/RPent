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

"""RPent/RLinf interface checks using real drivers and mocked device I/O.

These CPU-only checks do not validate physical motion or stopping behavior.
"""

from __future__ import annotations

import os
import time
from unittest.mock import Mock

import numpy as np
import pytest
import yaml


@pytest.fixture
def rlinf_devices(monkeypatch, tmp_path):
    if os.environ.get("RPENT_RUN_LEKIWI_INTEGRATION") != "1":
        pytest.skip("set RPENT_RUN_LEKIWI_INTEGRATION=1 with RLinf main installed")

    from rlinf.robotics.parts.cameras.uvc import UVCCamera
    from rlinf.robotics.parts.mobility import lekiwi

    from robots.lekiwi.detect import build_yaml_config
    from robots.lekiwi.runtime_config import load_runtime_config

    # Keep real RLinf robot composition and drivers; replace only device I/O.
    raw_velocity = dict.fromkeys(lekiwi.WHEEL_ORDER, 0)
    bus = Mock(is_connected=True)
    bus.sync_write.side_effect = lambda register, values, **kw: raw_velocity.update(
        values
    )
    bus.sync_read.side_effect = lambda register, names, **kw: dict(raw_velocity)
    bus.write.side_effect = lambda register, name, value, **kw: (
        raw_velocity.update({name: value}) if register == "Goal_Velocity" else None
    )
    open_bus = Mock(return_value=bus)
    monkeypatch.setattr(lekiwi, "make_wheel_bus", open_bus)

    frame = np.full((48, 64, 3), [15, 30, 90], dtype=np.uint8)
    camera_device = Mock()
    monkeypatch.setattr(UVCCamera, "_open", lambda self: camera_device)
    monkeypatch.setattr(UVCCamera, "_release", lambda self, device: device.release())

    def read_frame(self):
        time.sleep(0.01)
        return True, frame.copy()

    monkeypatch.setattr(UVCCamera, "_read_frame", read_frame)
    serial = "/dev/v4l/by-id/usb-test-video-index0"
    monkeypatch.setattr(UVCCamera, "discover", classmethod(lambda cls: {serial}))
    path = tmp_path / "lekiwi.yaml"
    path.write_text(yaml.safe_dump(build_yaml_config("/dev/ttyACM0", [serial])))
    return load_runtime_config(path), bus, open_bus, camera_device, frame, serial


def test_env_uses_rlinf_robot_interfaces(rlinf_devices):
    from robots.lekiwi.env_server import LeKiwiEnvBackend

    cfg, bus, open_bus, camera, frame, serial = rlinf_devices
    backend = LeKiwiEnvBackend(cfg)
    try:
        assert backend.get_env_meta()["action_dim"] == 3
        observation = backend.get_observation()
        assert observation["main_images"].shape == (128, 128, 3)
        np.testing.assert_array_equal(
            observation["main_images"][0, 0], frame[0, 0, ::-1]
        )
        assert backend.get_camera_meta()["cameras"]["front"]["serial_number"] == serial
        result = backend.drive([0.1, 0.05, 5.0], duration_s=0.2)
        assert result["status"] == "succeeded", result
        assert result["stop_confirmed"]
        assert result["feedback_valid"]
        assert backend.reset()["ok"]
        np.testing.assert_allclose(backend.get_robot_state()["velocity"], 0)
    finally:
        backend.close_env()
    open_bus.assert_called_once_with("/dev/ttyACM0")
    bus.disconnect.assert_called_once()
    camera.release.assert_called_once()


def test_camera_only_and_detection_use_rlinf_interfaces(rlinf_devices):
    from robots.lekiwi.detect import discover_cameras, probe_camera
    from robots.lekiwi.env_server import LeKiwiCameraBackend

    cfg, _, open_bus, camera, frame, serial = rlinf_devices
    assert discover_cameras() == [serial]
    assert probe_camera(serial) == {
        "serial": serial,
        "ok": True,
        "width": 64,
        "height": 48,
    }
    backend = LeKiwiCameraBackend(cfg)
    try:
        np.testing.assert_array_equal(
            backend.get_observation()["main_images"], frame[..., ::-1]
        )
        assert backend.get_camera_meta()["cameras"]["front"]["camera_type"] == "uvc"
    finally:
        backend.close_env()
    open_bus.assert_not_called()
    assert camera.release.call_count == 2
