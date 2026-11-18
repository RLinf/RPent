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

"""Preserve task-specific data and existing general dual-Franka policies."""

import pytest

from robots.dual_franka import prompt_bundle
from robots.dual_franka.tasks import DUAL_FRANKA_TASKS


@pytest.mark.parametrize("mode", ["eval", "explore"])
def test_task_switch_preserves_constraints_and_policy_scope(mode):
    for task_id, task in DUAL_FRANKA_TASKS.items():
        variables = {
            "task_id": task_id,
            "task_name": task.name,
            "instruction": task.instruction + " {{literal}}",
            "setup": task.setup,
            "success_criteria": task.success_criteria,
            "constraints": "\n".join(task.constraints),
            "mode": mode,
            "recipe_tag": f"dual_franka_t{task_id}" if task_id == 0 else task.name,
            "memory_dir": f"/memory/task{task_id}",
            "memory_inbox": f"/inbox/task{task_id}",
            "output_dir": f"/run/task{task_id}",
            "session_number": 2,
            "session_max": 3,
            "explore_attempts_per_session": 5,
        }
        system = prompt_bundle.system_prompt(variables)
        user = prompt_bundle.user_prompt(variables)
        assert task.instruction + " {{literal}}" in user
        assert task.success_criteria in user
        assert task.setup in user
        assert all(constraint in user for constraint in task.constraints)
        # The existing task2/general-policy tension is intentionally retained.
        assert "Do not move an arm or held object merely to improve" in system
        assert "Use VLA segment tools for contact-rich motion" in system
        assert "requested/effective prompts" in system
        assert "request_scene_reset" in system
        if mode == "explore":
            assert f"/memory/task{task_id}" in system
            assert f"exploration notes: /inbox/task{task_id}" in user
            assert f"failed-attempt archives: /run/task{task_id}/attempts/" in user
            assert "Any subsequent physical action invalidates the verdict" in system
            assert "After operator-confirmed success" in system
            assert "robot's authoritative success confirmation" in system
            assert "write /run/task" in system
            assert "The runner exports motion commands" in system
            assert "scope: task-family" in system
            assert "attempt_s<session>_a<attempt>_failed.json" in system
        assert "request_scene_reset before acting" not in user
