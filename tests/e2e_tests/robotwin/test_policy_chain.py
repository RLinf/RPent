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

import json

import pytest

from tests.e2e_tests.common import (
    ScriptedToolCall,
    publish_check,
    run_scripted_policy_chain,
)
from tests.e2e_tests.robotwin.scenario import RoboTwinScenario


def test_policy_chain(robotwin_scenario: RoboTwinScenario) -> None:
    publish_check("robotwin_policy_chain", robotwin_scenario.policy_chain)


@pytest.mark.timeout(1200)
def test_wam_policy_chain(wam_argv, tmp_path, record_property) -> None:
    result = run_scripted_policy_chain(
        robot="robotwin",
        robot_argv=wam_argv,
        output_dir=tmp_path / "chain",
        action=ScriptedToolCall("wam_act", {"max_chunks": 2}),
        action_count_field="executed_steps",
        use_memory=False,
    )
    assert result["status"] == "passed"
    states = json.loads((tmp_path / "chain" / "states.json").read_text())["steps"]
    verdict = states[-1]["result"]
    record_property("native_success", verdict["episode_status"]["eval_success"])
    record_property("executed_steps", verdict["executed_steps"])
