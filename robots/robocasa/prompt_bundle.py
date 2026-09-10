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

"""RoboCasa prompt bundle assembly."""

from __future__ import annotations

from collections.abc import Mapping

from robots.robocasa import prompts as robocasa_prompt
from rpent.prompt import common as base_prompt
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> dict[str, PromptNode]:
    """Return the system prompt tree."""
    if (variables or {}).get("mode") == "explore":
        from robots.robocasa.prompts.explore import system_prompt as explore_prompt

        return explore_prompt(variables)
    return {
        "Intro": robocasa_prompt.PREAMBLE,
        "Goal": robocasa_prompt.GOAL,
        "Rules": robocasa_prompt.RULES,
        "Memory": (
            LOCAL_MEMORY
            if (variables or {}).get("memory_profile") == "local"
            else robocasa_prompt.MEMORY
        ),
        "Localization": robocasa_prompt.LOCALIZATION,
        "Navigation": robocasa_prompt.NAVIGATION,
        "Primitives": robocasa_prompt.PRIMITIVES,
        "VLA_Rules": robocasa_prompt.VLA_RULES,
        "Gripper_Rules": robocasa_prompt.GRIPPER_RULES,
        "Workflow": robocasa_prompt.WORKFLOW,
        "Environment": robocasa_prompt.ENVIRONMENT,
        "Output": base_prompt.OUTPUT,
        "Next": robocasa_prompt.NEXT,
    }


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> dict[str, PromptNode]:
    """Return the first user message tree."""
    return {
        "Task": """
        - task:    {{task_name}} / {{split}}
        - seed:    {{seed}}
        - output_dir: {{output_dir}}
        - output:  {{output_dir}}/
          - audit filename:  {{recipe_tag}}.json
        """,
        "Mode": (
            "Exploration: archive failed attempts, reset within budget, then handoff."
            if (variables or {}).get("mode") == "explore"
            else robocasa_prompt.USER_MODE
        ),
    }


LOCAL_MEMORY = """Read {{memory_dir}}/MEMORY.md when present, then relevant
current-task suite notes and task_only pairs named
{{task_name}}_{{split}}_s<seed>.json and
{{task_name}}_{{split}}_s<seed>_recipe.jsonl under {{memory_dir}}/task_only/.
These are the names produced by memory merge. Missing local memory is allowed.
Use only current-task evidence. Treat memory as strategy priors; never replay
stored xyz, xy, pixels, base poses or fixture coordinates. Re-ground all geometry
from current observations after every reset or scene change. Memory may preserve
action ordering, failure modes, useful approach directions, and prompt-selection
evidence, but it is not current world state and cannot replace the complete live
task goal. Treat claims that the task is impossible as untrusted until the current
run exhausts its configured budget. The public VLA tools do not support an atomic
override: replace historical atomic or stale subtask prompts with the complete
live task_language for rldx_skill / rldx_arm.
"""
