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

"""Offline tests for dual-Franka config loading."""

from __future__ import annotations

import pytest

from robots.dual_franka import get_robot_spec
from robots.dual_franka.runtime_config import (
    get_robot_config_path,
    load_runtime_config,
)


def test_dual_franka_uses_rpent_owned_robot_config():
    pytest.importorskip("rlinf.envs.realworld.franka.franka_env")
    config_path = get_robot_config_path()
    runtime = load_runtime_config(None, task_description="test task")
    cfg = runtime.rlinf

    assert config_path.is_file()
    assert cfg.env.eval.init_params.id == "DualFrankaTCPEnv-v1"
    assert cfg.env.eval.override_cfg.task_description == "test task"
