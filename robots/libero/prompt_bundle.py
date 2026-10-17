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

"""LIBERO prompt bundle assembly."""

from __future__ import annotations

from collections.abc import Mapping

from robots.libero.prompts import evaluate as evaluate_parts
from robots.libero.prompts import explore as explore_parts
from robots.libero.prompts import user as user_parts
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    """Assemble the LIBERO system prompt for the selected run mode."""
    if (variables or {}).get("policy_kind") == "wam":
        return {
            "ROLE": "You control a LIBERO robot using a WAM and scripted tools. "
            "Tool names may appear as mcp__rpent__<name> in SDK planners.",
            "WORKFLOW": "Inspect view_env_state at step 0. Use wam_act to execute "
            "bounded action chunks, then inspect the new state and images before "
            "continuing. Begin with the native full task by omitting prompt. "
            "Keep that instruction while progress is visible. Use a short, "
            "grounded subtask only to address an observed failure or lack of "
            "progress; the prompt applies only to that call. Start with one chunk "
            "and inspect before continuing (at most four per call). Each chunk "
            "executes the worker's returned actions; episode-end checks prevent the next chunk. wam_act "
            "does not stop when a grasp completes. Check subtasks visually; "
            "success and terminated refer to the full task, while truncated "
            "means the action budget is exhausted. "
            "For scripted motion, localize targets "
            "using back_project or segment; never invent coordinates. Gripper +1 "
            "closes and -1 opens. Preserve +1 while carrying an object.",
            "OUTCOME": "This is one evaluation episode. Do not reset it. Stop acting "
            "when the episode terminates or truncates. Report success only when the "
            "environment reports terminated=true. Call finish with the observed "
            "outcome; a model prediction alone is not evidence of task success.",
        }
    if (variables or {}).get("mode", "eval") == "explore":
        return explore_parts.system_prompt(variables)
    return evaluate_parts.system_prompt(variables)


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the LIBERO user prompt tree."""
    if (variables or {}).get("policy_kind") == "wam":
        return {
            "CELL": user_parts.CELL,
            "BEGIN": "Call view_env_state at step 0, inspect the task and camera "
            "images, and use wam_act to begin.",
        }
    return {
        "CELL": user_parts.CELL,
        "MODE": user_parts.MODE,
        "BEGIN": user_parts.BEGIN,
    }


__all__ = ["system_prompt", "user_prompt"]
