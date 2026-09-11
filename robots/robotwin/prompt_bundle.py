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

"""RoboTwin prompt bundle assembly."""

from __future__ import annotations

from collections.abc import Mapping

from robots.robotwin.prompts import system as system_parts
from robots.robotwin.prompts import user as user_parts
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    if (variables or {}).get("mode") == "explore":
        from robots.robotwin.prompts.explore import system_prompt as explore_prompt

        return explore_prompt(variables)
    return {
        "ROLE": system_parts.ROLE,
        "READ ORDER": (
            "Read view_env_state(step=0) and its head image before acting. "
            "Read {{memory_dir}}/MEMORY.md when present, then list "
            "{{memory_dir}}/task_only and read only current-task pairs named "
            "robotwin_{{task_name}}_s<seed>.json and "
            "robotwin_{{task_name}}_s<seed>_recipe.jsonl. Prefer seed-0 evidence "
            "when available. Missing local memory is allowed. Re-ground all "
            "geometry in current observations."
            if (variables or {}).get("memory_profile") == "local"
            else system_parts.READ_ORDER
        ),
        "CLEAN-TO-RANDOMIZED TRANSFER": system_parts.TRANSFER,
        "ACCURACY-FIRST LOOP": system_parts.ACCURACY_LOOP,
        "CONDITIONAL TASK-FAMILY PLAYBOOKS": system_parts.TASK_FAMILIES,
        "VLA AND PRIMITIVE CONTROL": system_parts.CONTROL,
        "PERCEPTION": system_parts.PERCEPTION,
        "RUNTIME": system_parts.RUNTIME,
        "BUDGET AND SUCCESS": system_parts.BUDGET_AND_SUCCESS,
        "MODE": system_parts.USER_MODE,
    }


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    if (variables or {}).get("mode") == "explore":
        return "\n\n".join(
            (
                "## Task cell\n\n" + user_parts.CELL,
                "## Begin\n\n" + user_parts.BEGIN,
            )
        )
    return {
        "CELL": user_parts.CELL,
        "BEGIN": user_parts.BEGIN,
    }


__all__ = ["system_prompt", "user_prompt"]
