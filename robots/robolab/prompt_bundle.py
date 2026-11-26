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

"""Task prompts for the RoboLab DROID environment."""

from __future__ import annotations

from collections.abc import Mapping

from rpent.prompt.utils import PromptNode


def system_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Describe the registered control and observation tools."""
    has_policy = (variables or {}).get("policy_backend", "pi05") == "pi05"
    return {
        "ROLE": "Control one RoboLab DROID simulation episode using the registered RPent tools. Follow the current task language and preserve completed subgoals.",
        "OBSERVE": "Start with view_env_state(step=0). Inspect the current external and wrist camera images before acting. Recheck them after each action; policy motion alone does not establish task success.",
        "CONTROL": (
            "Use pi05_act for a short policy chunk, or execute_action for one native 8-value DROID joint-position action. "
            if has_policy
            else "Use execute_action for one native 8-value DROID joint-position action. "
        )
        + "The environment reports success and step-limit termination. Stop acting once the episode ends.",
        "RESULT": "Use the latest view_env_state status to decide whether the benchmark task succeeded. Report uncertainty if the environment has not returned a terminal result.",
    }


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Supply the selected benchmark task and seed."""
    del variables
    return {
        "TASK": "RoboLab task: {{task}}; seed: {{seed}}.",
        "BEGIN": "Call view_env_state(step=0), read the task language, then choose the first short action.",
    }
