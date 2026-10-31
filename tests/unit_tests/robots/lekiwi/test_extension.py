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
import numpy as np
import pytest
import yaml

from robots.lekiwi.runtime_config import load_runtime_config

EXAMPLE_CONFIG = Path(__file__).parents[4] / "robots/lekiwi/config/example.yaml"


def test_lekiwi_uses_rpent_owned_robot_config(fake_rlinf_realworld_modules):
    runtime = load_runtime_config(EXAMPLE_CONFIG)
    cfg = runtime.rlinf

    assert cfg.env.eval.init_params.id == "RPentLeKiwiDriveEnv-v1"
    assert cfg.env.eval.main_image_key == "front"
    assert cfg.env.eval.override_cfg.camera_names == {"<front-camera-serial>": "front"}
    assert cfg.env.eval.override_cfg.enable_camera_player is False
    assert cfg.env.eval.override_cfg.step_frequency == 10.0
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


def test_lekiwi_runtime_config_rejects_unknown_drive_key(
    fake_rlinf_realworld_modules, tmp_path
):
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw["drive"]["cruise_altitude"] = 1.0
    config = tmp_path / "site.yaml"
    config.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="unknown key"):
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
    config = tmp_path / "site.yaml"
    config.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="exactly one camera"):
        load_runtime_config(config)


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


def test_live_observation_delegates_to_rlinf_env(fake_rlinf_realworld_modules):
    from robots.lekiwi.rpent_env import RPentLeKiwiEnv

    env = RPentLeKiwiEnv.__new__(RPentLeKiwiEnv)
    observation = {"frames": object(), "state": object()}
    env._get_observation = lambda: observation

    assert env.get_live_observation() is observation


def test_measured_velocity_reads_base_part(fake_rlinf_realworld_modules):
    from robots.lekiwi.rpent_env import RPentLeKiwiEnv

    env = RPentLeKiwiEnv.__new__(RPentLeKiwiEnv)
    env.config = SimpleNamespace(is_dummy=False)
    env._state_parts = SimpleNamespace(
        get_observation=lambda: {"base": {"velocity": [0.1, 0.0, 5.0]}}
    )

    np.testing.assert_allclose(
        env.get_measured_velocity(), np.array([0.1, 0.0, 5.0], dtype=np.float32)
    )


def test_measured_velocity_is_zero_in_dummy_mode(fake_rlinf_realworld_modules):
    from robots.lekiwi.rpent_env import RPentLeKiwiEnv

    env = RPentLeKiwiEnv.__new__(RPentLeKiwiEnv)
    env.config = SimpleNamespace(is_dummy=True)

    np.testing.assert_array_equal(
        env.get_measured_velocity(), np.zeros(3, dtype=np.float32)
    )


def test_stop_resets_base_wheels(fake_rlinf_realworld_modules):
    from robots.lekiwi.rpent_env import RPentLeKiwiEnv

    env = RPentLeKiwiEnv.__new__(RPentLeKiwiEnv)
    env.config = SimpleNamespace(is_dummy=False)
    base = SimpleNamespace(reset=lambda: calls.append("reset"))
    calls: list[str] = []
    env._base = base

    env.stop()

    assert calls == ["reset"]


def test_camera_metadata_reports_every_camera(fake_rlinf_realworld_modules):
    from robots.lekiwi.rpent_env import RPentLeKiwiEnv

    env = RPentLeKiwiEnv.__new__(RPentLeKiwiEnv)
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

    assert metadata["source"] == "rlinf_lekiwi_env"
    assert metadata["cameras"]["front"] == {
        "serial_number": "cam-1",
        "camera_type": "webcam",
        "resolution": [128, 128],
    }
