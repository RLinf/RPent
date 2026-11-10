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

"""Real requests to existing BEHAVIOR Env/VLA services; no process ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.e2e_tests.common import publish_check

if TYPE_CHECKING:
    from tests.e2e_tests.behavior.scenario import BehaviorScenario


def test_environment_component(behavior_scenario: BehaviorScenario) -> None:
    _, result = behavior_scenario.environment
    publish_check("behavior_environment_component", result)


def test_vla_component(behavior_scenario: BehaviorScenario) -> None:
    publish_check("behavior_vla_component", behavior_scenario.vla)
