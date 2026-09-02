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

"""Shared BEHAVIOR prompt section bodies."""

from __future__ import annotations

CURRENT_INVOCATION = """- mode: {{behavior_mode}}
- task: {{task_name}}
- instruction: {{task_instruction}}
- public seed: {{public_seed}}
- recipe tag: {{recipe_tag}}
- output directory: {{output_dir}}
- maximum environment steps: {{max_episode_steps}}
- planner timeout seconds: {{wall_clock_seconds}}"""

INVOCATION_MODEL = """One planner invocation is one BEHAVIOR episode. The
planner cannot reset or restart that episode. A BEHAVIOR-owned outer harness may
launch fresh processes for separate Explore attempts."""

RUNTIME = """Use only the public structured tools exposed by the active
toolkit. The public capability name list is {{public_capabilities}}; the actual
tool schemas supplied by the planner runtime remain authoritative. Do not start,
stop, or reach into ENV, VLA, checkpoint, simulator, or RPC internals."""

GOAL = """Execute the exact runtime task instruction: {{task_instruction}}"""

MEMORY_CONTEXT = """The official local MemoryManager corpus is
`{{memory_dir}}` (profile `{{memory_profile}}`). This invocation's inbox is
`{{memory_inbox}}`. Memory is historical guidance only: it is not a current
observation, coordinate source, stage label, or success proof."""

EVIDENCE = """Ground scene claims and action decisions in current public tool
receipts from this episode. Refresh observations after scene-changing actions
when later decisions depend on object identity, pose, reachability, attachment,
or task state."""

PLANNER_TOOLS = """All public capabilities are peer planner tools. No list
order implies a workflow or fixed call count. Pi0.5 is one planner tool; choose
each positive chunk count from the current subgoal and remaining step budget.
`{{wall_clock_seconds}}` is the planner timeout, not a per-primitive budget."""

TERMINATION = """Official task success exists only when the current episode
returns `info[\"done\"][\"success\"] is True`. Reward, terminated, truncated,
primitive success, visual appearance, video, and a clean process exit cannot
substitute for that value. A verified receipt may carry this evidence but cannot
create it."""

OUTPUT_DISCIPLINE = """Keep the final result tied to this invocation. Distinguish
official task success from primitive progress, interruption, termination,
truncation, and workflow completion. Never present historical memory as current
evidence."""

__all__ = [
    "CURRENT_INVOCATION",
    "EVIDENCE",
    "GOAL",
    "INVOCATION_MODEL",
    "MEMORY_CONTEXT",
    "OUTPUT_DISCIPLINE",
    "PLANNER_TOOLS",
    "RUNTIME",
    "TERMINATION",
]
