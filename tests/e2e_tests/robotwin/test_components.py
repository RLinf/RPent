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

from tests.e2e_tests.common import publish_check
from tests.e2e_tests.robotwin.scenario import RoboTwinScenario


def test_environment_component(robotwin_scenario: RoboTwinScenario) -> None:
    _, _, result = robotwin_scenario.environment
    publish_check("robotwin_environment_component", result)


def test_lingbot_component(robotwin_scenario: RoboTwinScenario) -> None:
    _, result = robotwin_scenario.lingbot
    publish_check("lingbot_component", result)
