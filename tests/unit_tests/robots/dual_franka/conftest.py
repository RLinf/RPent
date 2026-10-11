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

"""Shared fixtures for offline dual-Franka tests."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robots.dual_franka.runtime_config import DEFAULT_CONFIG
from robots.franka.runtime_config import set_robot_config_path


class FakeEnv:
    def __init__(self):
        self.meta = {}
        self.resets = 0
        self.moves = []
        self.reset_result = {"ok": True}
        self.fail_observation = False

    def reset(self):
        self.resets += 1
        return self.reset_result

    def get_observation(self):
        if self.fail_observation:
            raise RuntimeError("camera offline")
        return {
            "main_images": np.full((8, 8, 3), self.resets, dtype=np.uint8),
            "extra_view_images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
            "main_depths": np.ones((8, 8), dtype=np.float32),
            "extra_view_depths": np.ones((2, 8, 8), dtype=np.float32),
            "d455_images": np.zeros((8, 8, 3), dtype=np.uint8),
            "d455_depths": np.ones((8, 8), dtype=np.float32),
        }

    def get_robot_state(self):
        return {
            "left_arm": {"tcp_pose": [0.5, 0, 0.5, 0, 0, 0, 1]},
            "right_arm": {"tcp_pose": [0.5, 0, 0.5, 0, 0, 0, 1]},
        }

    def get_camera_meta(self):
        return {"cameras": {}}

    def move_delta(self, arm, delta):
        self.moves.append((arm, list(delta)))
        return {"ok": True}


@pytest.fixture
def fake_env():
    return FakeEnv()


@pytest.fixture
def dual_franka_run(tmp_path, dual_franka_robot_config, fake_env):
    from robots.dual_franka import robot_spec
    from rpent.cli import main

    calls = []

    def init_runtime(*args):
        calls.append(args)
        return [], {"env": fake_env, "model": None, "task_description": "test"}

    run = SimpleNamespace(
        env=fake_env,
        spec=replace(robot_spec.get_robot_spec(), init_runtime=init_runtime),
        runtime_calls=calls,
        argv=[
            "rpent",
            "--robot",
            "dual_franka",
            "--planner",
            "codex",
            "--explore",
            "--auto-merge-memory",
            "--robot-config",
            str(dual_franka_robot_config),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--output-dir",
            str(tmp_path / "run"),
        ],
    )
    parser = main._build_argparser()
    run.spec.add_cli_args(parser, use_dashboard=False)
    run.args = parser.parse_args(run.argv[1:])
    return run


@pytest.fixture
def dual_franka_robot_config(tmp_path: Path) -> Iterator[Path]:
    """Return the packaged config with calibration paths redirected to fixtures."""
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    fixtures = Path(__file__).with_name("fixtures")
    config["perception"]["calibration"] = {
        "base_camera": str(fixtures / "third_to_right_base_calib_eye_on_base.yaml"),
        "d455_camera": str(fixtures / "d455_to_right_base_eye_on_base.yaml"),
    }
    path = tmp_path / "dual_franka.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    try:
        yield path
    finally:
        set_robot_config_path(None)
