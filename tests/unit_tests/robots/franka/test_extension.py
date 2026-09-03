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

"""Offline tests for Franka environment discovery and config loading."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest

from robots.franka import get_robot_spec
from robots.franka.runtime_config import load_runtime_config
from rpent.robots.base import enumerate_robots
from rpent.robots.base import get_robot_spec as resolve_robot_spec


def test_franka_extension_is_discoverable():
    assert "franka" in enumerate_robots()
    spec = resolve_robot_spec("franka")
    assert spec.name == "franka"
    assert get_robot_spec().name == spec.name


def test_franka_uses_rpent_owned_robot_config():
    pytest.importorskip("rlinf.envs.realworld.franka.franka_env")
    config_path = Path(__file__).parents[4] / "robots/franka/config/example.yaml"
    runtime = load_runtime_config(config_path, task_description="test task")
    cfg = runtime.rlinf

    assert config_path.is_file()
    assert cfg.env.eval.init_params.id == "RPentFrankaEnv-v1"
    assert cfg.env.eval.override_cfg.task_description == "test task"


def test_rpent_franka_registration_exists():
    pytest.importorskip("rlinf.envs.realworld.franka.franka_env")
    from robots.franka.rpent_env import register_rpent_franka_env

    register_rpent_franka_env()
    assert gym.spec("RPentFrankaEnv-v1") is not None
