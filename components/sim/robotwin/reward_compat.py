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

"""Temporary compatibility for a bug in the pinned RoboTwin runtime."""

from __future__ import annotations

import importlib
from typing import Any


def install_native_reward_compat(task_name: str) -> bool:
    """Adapt the legacy ``place_fan`` reward constructor when required."""
    if task_name != "place_fan":
        return False

    task_module = importlib.import_module("robotwin.envs.place_fan")
    reward_factory = task_module.Reward
    if getattr(reward_factory, "_rpent_place_fan_compat", False):
        return False

    class PlaceFanRewardCompat(reward_factory):
        """Build the intended serial reward from the legacy arguments."""

        _rpent_place_fan_compat = True

        def __new__(
            cls,
            *,
            subtasks: list[Any],
            transition_rewards: list[float],
        ) -> Any:
            del cls
            return reward_factory.build_top(
                {
                    "type": "Serial",
                    "subtasks": subtasks,
                    "transition_rewards": transition_rewards,
                }
            )

    task_module.Reward = PlaceFanRewardCompat
    return True
