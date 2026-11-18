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

"""Current task and policy selection remain visible in final YAM prompts."""

import pytest

from robots.yam import prompt_bundle
from robots.yam.tasks import TASK_INSTRUCTIONS


@pytest.mark.parametrize("mode", ["eval", "explore"])
@pytest.mark.parametrize("vla_enabled", [False, True])
def test_task_switch_binds_current_context(mode, vla_enabled):
    for task_id in (0, 1, 2):
        variables = {
            "task_id": task_id,
            "instruction": TASK_INSTRUCTIONS[task_id] + " {{literal}}",
            "seed": task_id + 10,
            "mode": mode,
            "policy_backend": (
                "pi05" if vla_enabled and task_id != 0 else "primitives-only"
            ),
            "memory_dir": f"/memory/task{task_id}",
            "memory_inbox": f"/inbox/task{task_id}",
            "recipe_tag": f"yam_t{task_id}",
            "output_dir": f"/run/task{task_id}",
            "session_number": 2,
            "session_max": 3,
        }
        system = prompt_bundle.system_prompt(variables)
        user = prompt_bundle.user_prompt(variables)
        assert TASK_INSTRUCTIONS[task_id] + " {{literal}}" in user
        assert f"seed label: {task_id + 10}" in user
        assert f"/memory/task{task_id}" in system
        assert ("A trained YAM qpos14 VLA is connected" in system) == (
            vla_enabled and task_id != 0
        )
        assert ("pi05_act is not available" in system) == (
            not vla_enabled or task_id == 0
        )
        assert ("session 2 of 3" in system) == (mode == "explore")
        if mode == "explore":
            assert "bounded failure lessons may be merged" in system
            assert "Real-robot explore" in system
            assert "## Exploration memory contract" not in system
            assert "attempt_s<session>" not in system
            assert "do not invent a second action log" in system
            assert "scope: task-family" in system
            assert "regime: real" in system
        else:
            assert "read-only" in system
        assert "Only fresh env eval_success=true" in system
        assert "Verify env eval_success before finish" in user
