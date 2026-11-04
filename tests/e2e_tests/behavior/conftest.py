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

"""Opt-in fixtures that borrow operator-owned BEHAVIOR services."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.e2e_tests.behavior.scenario import BehaviorScenario


@pytest.fixture(scope="session")
def behavior_scenario() -> Iterator[BehaviorScenario]:
    if os.environ.get("RPENT_RUN_BEHAVIOR_INTEGRATION") != "1":
        pytest.skip(
            "set RPENT_RUN_BEHAVIOR_INTEGRATION=1 to use live borrowed services"
        )
    from tests.e2e_tests.behavior.scenario import BehaviorScenario, runtime_args

    scenario = BehaviorScenario(runtime_args())
    try:
        yield scenario
    finally:
        scenario.close()
