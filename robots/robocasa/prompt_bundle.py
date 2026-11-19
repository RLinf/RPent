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

from robots.robocasa.prompts import evaluate as evaluate_parts
from robots.robocasa.prompts import explore as explore_parts
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> dict[str, PromptNode]:
    """Return the RoboCasa system prompt tree."""
    if (variables or {}).get("mode", "eval") == "explore":
        return explore_parts.system_prompt()
    return evaluate_parts.system_prompt(variables)


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> dict[str, PromptNode]:
    """Return the first user message tree."""
    mode = (
        explore_parts.USER_MODE
        if (variables or {}).get("mode", "eval") == "explore"
        else evaluate_parts.USER_MODE
    )
    return {
        "Task": """
        - task:    {{task_name}} / {{split}}
        - seed:    {{seed}}
        - output_dir: {{output_dir}}
        - output:  {{output_dir}}/
          - audit filename:  {{recipe_tag}}.json
        """,
        "Mode": mode,
    }
