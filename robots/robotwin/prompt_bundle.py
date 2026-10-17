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

from robots.robotwin.prompts import evaluate as evaluate_parts
from robots.robotwin.prompts import explore as explore_parts
from robots.robotwin.prompts import user as user_parts
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    """Return the RoboTwin system prompt for the selected run mode."""
    if (variables or {}).get("policy_kind") == "wam":
        return {
            "ROLE": "Control the RoboTwin robot using wam_act and scripted tools.",
            "WORKFLOW": "Inspect view_env_state at step 0. Call wam_act for one chunk, "
            "then inspect the new state and images before continuing. Omit prompt to use "
            "the native task; supply a grounded subtask only when needed. "
            "Stop acting when the episode ends. Use native eval_success to judge full-task "
            "success, and call finish with the observed outcome.",
        }
    if (variables or {}).get("mode", "eval") == "explore":
        return explore_parts.system_prompt()
    return evaluate_parts.system_prompt(variables)


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    if (variables or {}).get("policy_kind") == "wam":
        return {
            "TASK": "Inspect view_env_state for the native task and current scene, then use wam_act."
        }
    prompt = {
        "CELL": user_parts.CELL,
        "BEGIN": user_parts.BEGIN,
    }
    if (variables or {}).get("mode", "eval") == "explore":
        prompt["EXPLORE MODE"] = explore_parts.USER_MODE
    return prompt


__all__ = ["system_prompt", "user_prompt"]
