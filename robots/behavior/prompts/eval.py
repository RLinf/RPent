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

"""BEHAVIOR single-attempt evaluation prompt."""

from __future__ import annotations

from robots.behavior.prompts import system as base
from rpent.prompt.utils import PromptNode

ROLE_AND_EVALUATION = """You are the planner for one BEHAVIOR evaluation
episode. The invocation controls only that episode. Pursue the exact task while
remaining honest about official success."""

MEMORY = """Memory access is read-only in Eval. Read only relevant material
through the official MemoryManager tools; do not write to the corpus."""


def system_prompt() -> PromptNode:
    """Assemble the complete BEHAVIOR Eval prompt."""
    return {
        "ROLE AND EVALUATION": ROLE_AND_EVALUATION,
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


__all__ = ["MEMORY", "ROLE_AND_EVALUATION", "system_prompt"]
