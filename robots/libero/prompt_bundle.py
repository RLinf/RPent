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

from robots.libero.prompts import explore as explore_parts
from robots.libero.prompts import local_eval as local_eval_parts
from robots.libero.prompts import system as system_parts
from robots.libero.prompts import user as user_parts
from rpent.context.prompt_utils import Numbered, PromptNode


def system_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the LIBERO system prompt for the selected run mode."""
    if (variables or {}).get("mode", "eval") == "explore":
        return explore_parts.system_prompt()
    if (variables or {}).get("memory_profile", "hf") == "local":
        return local_eval_parts.system_prompt()
    return {
        "ROLE AND EVALUATION": system_parts.ROLE_AND_EVALUATION,
        "PROVEN LEVERS & LESSONS — libero_10_task seed-0 sweep solved 9/10 (READ THIS)": (
            system_parts.PROVEN_LEVERS
        ),
        "RUNTIME": system_parts.RUNTIME,
        "YOUR GOAL": system_parts.GOAL,
        "RULES (NON-NEGOTIABLE)": system_parts.RULES,
        "LOCALIZATION — how to get an object's world xyz WITHOUT GT coords": (
            system_parts.LOCALIZATION
        ),
        "FIRST-STEP ALGORITHM — agentview = IDENTITY, wrist = GEOMETRY": (
            system_parts.PERCEPTION_ALGORITHM
        ),
        "WORKFLOW": Numbered(system_parts.WORKFLOW_STEPS),
        "KEY HYPERPARAMETERS": system_parts.KEY_HYPERPARAMETERS,
        "OUTPUT DISCIPLINE": system_parts.OUTPUT_DISCIPLINE,
    }


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the LIBERO user prompt tree."""
    return {
        "CELL": user_parts.CELL,
        "MODE": user_parts.MODE,
        "BEGIN": user_parts.BEGIN,
    }


__all__ = ["system_prompt", "user_prompt"]
