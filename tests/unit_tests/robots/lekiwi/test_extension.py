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

"""Offline tests for LeKiwi config loading and gym registration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest
import yaml

from robots.lekiwi.runtime_config import load_device_config, load_runtime_config

EXAMPLE_CONFIG = Path(__file__).parents[4] / "robots/lekiwi/config/example.yaml"


def test_lekiwi_uses_rpent_owned_robot_config(fake_rlinf_realworld_modules):
    runtime = load_runtime_config(EXAMPLE_CONFIG)
    cfg = runtime.rlinf

    assert cfg.env.eval.init_params.id == "RPentLeKiwiDriveEnv-v1"
    assert cfg.env.eval.main_image_key == "front"
    assert cfg.env.eval.override_cfg.camera_names == {"<front-camera-serial>": "front"}
    assert cfg.env.eval.override_cfg.enable_camera_player is False
    assert cfg.env.eval.override_cfg.step_frequency == 10.0
    # Camera timing follows RLinf defaults unless the site YAML overrides it.
    assert "camera_max_age" not in cfg.env.eval.override_cfg
    hardware = cfg.cluster.node_groups[0].hardware.configs[0]
    assert hardware["port"] == "/dev/ttyACM0"
    assert hardware["camera_serials"] == ["<front-camera-serial>"]
    assert hardware["camera_type"] == "webcam"
    assert hardware["wheel_radius"] == 0.05
    assert cfg.cluster.node_groups[0].hardware["type"] == "LeKiwi"
    assert cfg.cluster.component_placement.env.node_group == "lekiwi"
    assert runtime.controller == {
        "drive_max_duration_s": 10.0,
        "hold_max_ttl_s": 2.0,
    }


@pytest.mark.parametrize(
    "section, key, value, error",
    [
        ("drive", "cruise_altitude", 1.0, "unknown key"),
        ("robot", "wheel_radius", 0, "wheel_radius"),
        ("robot", "base_radius", float("inf"), "base_radius"),
        ("robot", "max_raw", True, "max_raw"),
        ("robot", "max_raw", -1, "max_raw"),
        ("robot", "max_raw", 1.5, "max_raw"),
    ],
)
def test_lekiwi_runtime_config_rejects_invalid_settings(
    fake_rlinf_realworld_modules, tmp_path, section, key, value, error
):
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw[section][key] = value
    config = tmp_path / "lekiwi.yaml"
    config.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match=error):
        load_runtime_config(config)


def test_lekiwi_runtime_config_requires_single_main_camera(
    fake_rlinf_realworld_modules, tmp_path
):
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw["cameras"]["devices"]["rear"] = {
        "serial": "rear-serial",
        "type": "webcam",
        "main": True,
    }
    config = tmp_path / "lekiwi.yaml"
    config.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="exactly one camera"):
        load_runtime_config(config)


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("video2", "/dev/video2"),
        ("usb-ZED-video-index0", "/dev/v4l/by-id/usb-ZED-video-index0"),
        ("/dev/video2", "/dev/video2"),
        ("/run/cameras/zed", "/run/cameras/zed"),
    ],
)
def test_camera_config_preserves_device_aliases_and_resolves_calibration(
    fake_rlinf_realworld_modules, tmp_path, device, expected
):
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    camera = raw["cameras"]["devices"]["front"]
    camera["serial"] = device
    camera.pop("type")
    raw["perception"]["calibration"] = "calibration/zed.conf"
    config = tmp_path / "robot.yaml"
    config.write_text(yaml.safe_dump(raw))

    runtime = load_runtime_config(config)
    devices = load_device_config(config)
    assert runtime.rlinf.env.eval.main_image_key == "front"
    assert (
        runtime.rlinf.cluster.node_groups[0].hardware.configs[0].camera_type == "webcam"
    )
    assert devices["device"] == expected
    assert devices["serial"] == camera["zed_serial"]
    assert devices["calibration"] == str(tmp_path / "calibration/zed.conf")


def test_rpent_lekiwi_registration_exists(fake_rlinf_realworld_modules):
    from robots.lekiwi.rpent_env import register_rpent_lekiwi_env

    register_rpent_lekiwi_env()
    assert gym.spec("RPentLeKiwiDriveEnv-v1") is not None


def test_rpent_lekiwi_env_accepts_override_cfg_dict(
    fake_rlinf_realworld_modules,
):
    # RealWorldEnv creates envs with an override_cfg dict, not a config
    # object; the wrapper must ride LeKiwiDriveEnv's dict conversion rather
    # than LeKiwiEnv's config-object constructor. This failed on the robot
    # with "unexpected keyword argument 'override_cfg'".
    from robots.lekiwi.rpent_env import RPentLeKiwiEnv

    env = RPentLeKiwiEnv(
        {
            "enable_camera_player": False,
            "step_frequency": 10.0,
            "camera_names": {"cam-0": "front"},
        },
        None,
        None,
        0,
    )
    assert env is not None


def test_camera_metadata_reports_every_camera(fake_rlinf_realworld_modules):
    from robots.lekiwi.rpent_env import RPentLeKiwiEnv

    env = RPentLeKiwiEnv.__new__(RPentLeKiwiEnv)
    env.config = SimpleNamespace(is_dummy=False, frame_size=(128, 128))
    env._camera_metadata = {"front": {"source_stamp": 1.0}}
    env._cameras = [
        SimpleNamespace(
            name="front",
            camera_info=SimpleNamespace(
                serial_number="cam-1",
                camera_type="webcam",
                resolution=(128, 128),
            ),
        )
    ]

    metadata = env.get_camera_metadata()

    assert metadata["source"] == "webcam"
    assert metadata["cameras"]["front"] == {
        "serial_number": "cam-1",
        "camera_type": "webcam",
        "resolution": [128, 128],
        "observation_resolution": [128, 128],
        "source_stamp": 1.0,
    }


@pytest.mark.parametrize("mode, resets", [("navigation", 0), ("manual", 1)])
def test_runtime_attaches_to_navigation_without_reset(
    monkeypatch, tmp_path, mode, resets
):
    from unittest.mock import Mock

    from robots.lekiwi import robot_spec

    rpc = Mock()
    rpc.call.return_value = {"control_mode": mode}
    monkeypatch.setattr(robot_spec, "make_rpc_client", lambda _: rpc)
    monkeypatch.setattr(robot_spec, "try_wait_server", lambda *args, post_fn: post_fn())
    daemons, kwargs = robot_spec._init_runtime(
        SimpleNamespace(env_endpoint="http://robot", task_id=0, dashboard=False),
        tmp_path,
        Mock(),
        None,
    )
    assert daemons == []
    assert kwargs["env"].meta["control_mode"] == mode
    assert (
        sum(call.args[0] == "env.reset" for call in rpc.call.call_args_list) == resets
    )
