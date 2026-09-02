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
    return {
        "Intro": robocasa_prompt.PREAMBLE,
        "Goal": robocasa_prompt.GOAL,
        "Rules": robocasa_prompt.RULES,
        "Memory": robocasa_prompt.MEMORY,
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
        "Mode": robocasa_prompt.USER_MODE,
    }
