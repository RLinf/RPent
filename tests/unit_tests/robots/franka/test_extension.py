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

"""Offline tests for Franka config loading and gym registration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from robots.franka.runtime_config import load_runtime_config


def test_franka_uses_rpent_owned_robot_config(fake_rlinf_realworld_modules):
    config_path = Path(__file__).parents[4] / "robots/franka/config/example.yaml"
    runtime = load_runtime_config(config_path, task_description="test task")
    cfg = runtime.rlinf

    assert config_path.is_file()
    assert cfg.env.eval.init_params.id == "RPentFrankaEnv-v1"
    assert cfg.env.eval.override_cfg.task_description == "test task"


def test_rpent_franka_registration_exists(fake_rlinf_realworld_modules):
    from robots.franka.rpent_env import register_rpent_franka_env

    register_rpent_franka_env()
    assert gym.spec("RPentFrankaEnv-v1") is not None


def test_live_camera_observation_refreshes_rlinf_camera_cache(
    fake_rlinf_realworld_modules,
):
    from robots.franka.rpent_env import RPentFrankaEnv

    env = RPentFrankaEnv.__new__(RPentFrankaEnv)
    env.config = SimpleNamespace(is_dummy=False)
    state = object()
    frames = {"wrist_1": np.ones((4, 5, 3), dtype=np.uint8)}
    depths = {"wrist_1": np.ones((4, 5), dtype=np.float32)}
    calls = []

    def read_robot():
        calls.append("read")
        return state

    env._read_robot = read_robot
    env._get_observation = lambda: {"frames": frames, "depths": depths}

    assert env.get_live_camera_observation() == (frames, depths)
    assert env._franka_state is state
    assert calls == ["read"]


def test_camera_metadata_matches_current_rlinf_crop(
    fake_rlinf_realworld_modules,
):
    from robots.franka.rpent_env import RPentFrankaEnv

    info = SimpleNamespace(
        name="wrist_1",
        serial_number="camera-1",
        camera_type="realsense",
        resolution=(640, 480),
        crop_region=None,
        enable_depth=True,
    )
    camera = SimpleNamespace(
        camera_info=info,
        depth_scale=0.001,
        get_color_intrinsics=lambda: {
            "width": 640,
            "height": 480,
            "fx": 600.0,
            "fy": 600.0,
            "ppx": 320.0,
            "ppy": 240.0,
        },
    )
    env = RPentFrankaEnv.__new__(RPentFrankaEnv)
    env._cameras = {"wrist_1": camera}
    env.config = SimpleNamespace(enable_camera_depth=True)
    env.observation_space = {
        "frames": {"wrist_1": SimpleNamespace(shape=(128, 128, 3))}
    }

    metadata = env.get_camera_metadata()

    assert metadata["depth_unit"] == "m"
    assert metadata["cameras"]["wrist_1"]["crop_bounds_xyxy"] == [80, 0, 560, 480]
    assert metadata["cameras"]["wrist_1"]["output_resolution"] == [128, 128]
    assert metadata["cameras"]["wrist_1"]["intrinsic_K"] == [
        [160.0, 0.0, 64.0],
        [0.0, 160.0, 64.0],
        [0.0, 0.0, 1.0],
    ]
