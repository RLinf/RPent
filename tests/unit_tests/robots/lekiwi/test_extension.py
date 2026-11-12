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

"""Config and lifecycle checks over public RLinf robot parts."""

from pathlib import Path

import pytest
import yaml

from robots.lekiwi.runtime_config import load_runtime_config

EXAMPLE_CONFIG = Path(__file__).parents[4] / "robots/lekiwi/config/example.yaml"


def test_config_keeps_native_device_ids_and_applies_defaults(tmp_path):
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw["robot"] = {"serial_port": "/dev/serial/by-id/wheels"}
    raw.pop("drive")
    serial = "/dev/v4l/by-id/usb-camera-video-index0"
    raw["cameras"]["devices"]["front"]["serial"] = serial
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_runtime_config(path)
    assert cfg.robot.serial_port == "/dev/serial/by-id/wheels"
    assert cfg.robot.wheel_radius == 0.05
    assert cfg.cameras.front.serial == serial
    assert cfg.main_camera == "front"
    assert cfg.drive.step_frequency == 10
    assert cfg.drive.frame_size == [128, 128]
    assert "env" not in cfg and "cluster" not in cfg


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_rank", 7),
        ("disable_validate", True),
        ("camera_serials", ["unused-camera"]),
        ("camera_type", "realsense"),
    ],
)
def test_unused_robot_config_rejected(tmp_path, field, value):
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw["robot"][field] = value
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=f"robot: unknown key.*{field}"):
        load_runtime_config(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cruise_altitude", 1),
        ("step_frequency", 0),
        ("max_linear_velocity", float("nan")),
        ("camera_max_age", -1),
        ("frame_size", [0, 128]),
        ("frame_size", [128.5, 128]),
    ],
)
def test_invalid_drive_config_rejected(tmp_path, field, value):
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw["drive"][field] = value
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_runtime_config(path)


def test_requires_single_main_camera(tmp_path):
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw["cameras"]["devices"]["rear"] = {"serial": "rear", "main": True}
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="exactly one camera"):
        load_runtime_config(path)


def test_direct_robot_ownership_and_observation(robot_parts):
    backend, robot = robot_parts.build()
    assert robot.connected
    assert robot.build_kwargs["port"] == "/dev/ttyACM0"
    assert backend.get_observation()["main_images"].shape == (128, 128, 3)
    meta = backend.get_camera_meta()
    assert meta["observation_camera_map"] == {"main": "front"}
    assert meta["cameras"]["front"]["camera_type"] == "uvc"
    backend.close_env()
    backend.close_env()
    assert robot.releases == 1
    assert not robot.connected
    assert robot.cameras["front"].releases == 1


@pytest.mark.parametrize("failure", ["connect", "camera", "feedback"])
def test_startup_failure_releases_robot(robot_parts, monkeypatch, failure):
    from robots.lekiwi.env_server import LeKiwiEnvBackend

    def fail(*args, **kwargs):
        raise OSError("startup failed")

    if failure == "connect":
        monkeypatch.setattr(robot_parts.Robot, "connect", fail)
    elif failure == "camera":
        monkeypatch.setattr(robot_parts.Camera, "is_ready", lambda *a, **k: False)
    else:
        monkeypatch.setattr(LeKiwiEnvBackend, "_read_velocity", fail)
    with pytest.raises((OSError, RuntimeError)):
        robot_parts.build()
    assert robot_parts.instances[-1].releases == 1
    assert not robot_parts.instances[-1].connected
