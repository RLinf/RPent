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

import pytest

from tests.e2e_tests.common import publish_check
from tests.e2e_tests.libero.scenario import LiberoScenario


def test_policy_chain(libero_scenario: LiberoScenario) -> None:
    if libero_scenario.variant != "pro":
        pytest.skip("shared LIBERO policy chain is covered by LIBERO-PRO")
    publish_check("libero_policy_chain", libero_scenario.policy_chain)
