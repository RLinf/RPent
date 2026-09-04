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

"""BEHAVIOR single-attempt exploration prompt."""

from __future__ import annotations

from robots.behavior.prompts import system as base
from rpent.prompt.utils import PromptNode

ROLE_AND_MODE = """You are the planner for one BEHAVIOR Explore attempt. This
invocation owns exactly one episode. The standard RPent Explore session loop,
not the planner, starts any later attempt."""

MEMORY = """Explore may write only under `{{memory_inbox}}` through the official
MemoryManager tools. Record evidence and reusable lessons there. The main CLI
performs the existing MemoryManager merge after Explore finishes when
`--auto-merge-memory` is enabled."""


def system_prompt() -> PromptNode:
    """Assemble the complete BEHAVIOR Explore prompt."""
    return {
        "ROLE AND MODE": ROLE_AND_MODE,
        "CURRENT INVOCATION": base.CURRENT_INVOCATION,
        "INVOCATION MODEL": base.INVOCATION_MODEL,
        "RUNTIME": base.RUNTIME,
        "YOUR GOAL": base.GOAL,
        "MEMORY": f"{base.MEMORY_CONTEXT}\n\n{MEMORY}",
        "EVIDENCE": base.EVIDENCE,
        "PLANNER TOOLS": base.PLANNER_TOOLS,
        "TERMINATION": base.TERMINATION,
        "OUTPUT DISCIPLINE": base.OUTPUT_DISCIPLINE,
    }


__all__ = ["MEMORY", "ROLE_AND_MODE", "system_prompt"]
