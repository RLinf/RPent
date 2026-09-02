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

"""Contract tests guarding the single-Franka config keys against RLinf."""

from __future__ import annotations

import dataclasses

from robots.franka.runtime_config import ENV_DEFAULTS


def test_env_defaults_keys_are_valid_rlinf_fields():
    """RPent's override keys must not drift from RLinf's dataclass fields."""
    from rlinf.envs.realworld.franka.franka_env import FrankaRobotConfig

    valid = {field.name for field in dataclasses.fields(FrankaRobotConfig)}
    unknown = sorted(set(ENV_DEFAULTS) - valid)
    assert not unknown, f"ENV_DEFAULTS keys not in FrankaRobotConfig: {unknown}"
