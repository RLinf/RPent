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

"""Single-Franka task messages preserve constraints and literal task text."""

import pytest

from robots.franka import prompt_bundle
from robots.franka.tasks import FRANKA_TASKS


@pytest.mark.parametrize("task_id", [0, 1])
@pytest.mark.parametrize("endpoint", [None, "http://policy"])
def test_current_task_and_unchanged_vla_guidance(task_id, endpoint):
    task = FRANKA_TASKS[task_id]
    variables = {
        "task_name": task.name,
        "instruction": task.instruction + " {{literal}}",
        "success_criteria": task.success_criteria,
        "constraints": "\n".join(task.constraints),
        "endpoint": endpoint,
    }
    system = prompt_bundle.system_prompt(variables)
    user = prompt_bundle.user_prompt(variables)
    assert task.instruction + " {{literal}}" in user
    assert task.success_criteria in user
    assert all(constraint in user for constraint in task.constraints)
    assert "request_scene_reset" not in system
    assert "request_operator_verdict" not in system
    assert (
        "Inspect view_env_state before motion and after every mutating tool call"
        in system
    )
    assert (
        "Finish only when the success evidence is visible and consistent with state"
        in system
    )
    assert "open the gripper before calling vla_grasp near contact" in system
    assert "Call view_env_state with step 0" in system
    assert "Call view_env_state with step 0" not in user
