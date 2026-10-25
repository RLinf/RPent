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

"""RoboDojo exploration prompt for transferable manipulation memory."""

from __future__ import annotations

from robots.robodojo.prompts import evaluate as base
from rpent.prompt.utils import PromptNode

GOAL = """Solve the current task at its fixed seed and layout, then produce
evidence-backed memory for evaluation on other seeds and layouts. This is
exploration session {{session_number}} of {{session_max}}. A successful episode
provides a strategy to adapt, not a trajectory to replay unchanged."""

READ_ORDER = """Before the first robot action:
1. Read robots/robodojo/guides/GUIDE_RPENT.md completely.
2. Inspect view_env_state and the head and wrist images. Read the native task
   instruction and note both arms' initial end-effector positions and orientations.
3. Read {{memory_dir}}/MEMORY.md when present and relevant task-family notes.
   Consult matching global lessons and successful task-specific JSON/JSONL pairs.
4. Read {{memory_inbox}}/wip/notes.md when present.

Current instructions and observations override memory. Use a successful JSON
audit for its phase plan and reasoning, and its JSONL recipe for action order,
parameters and VPP2 cadence. Transfer supported phase order, observable gates,
controller division and recovery ideas. Preserve arm roles required by the
current instruction; reconsider roles chosen only for a previous layout.
Rebind object identities, destinations, pixels, poses, heights, clearances and
grasp offsets from this scene. If memory is absent, proceed from observation."""

RULES = """This session allows {{explore_attempts_per_session}} attempts including the
initial episode (0 means no attempt-count limit; other run limits still apply).

Prefer in-place recovery when possible. reset restores the same task, seed
group and layout. Re-run perception and do not reuse stale pixels or coordinates.
Before each reset or unsolved finish, save the attempt number, actual commands,
observed progress, failure evidence and next changed strategy variable under
{{output_dir}}/attempts/attempt_<N>_failed.json. Never overwrite an earlier
attempt. Append bounded observations to {{memory_inbox}}/wip/notes.md.
Change a named variable on each retry: instruction, arm roles, action order,
chunk count, localization or approach pose. Do not infer that a task is
impossible from one unsuccessful approach; record unknown causes as unknown.
An unsolved finish is refused while a finite attempt budget remains. After
native success, stop motion; reset is refused. Use the remaining turns to
record what worked and the uncertainties before calling finish."""

OUTPUT = """Reserve time to close out the attempt and write
{{output_dir}}/{{recipe_tag}}.json before calling finish. Include task, seed,
layout, verbatim instruction, attempts, native_success, final_state,
memory_files_read and uncertainties. If unsolved, reference the failed-attempt
archives and claim no successful trajectory.

On success, re-read the current attempt's recorded trajectory, starting after
the most recent reset if one occurred.
Its ordered strategy_notes must account for every exported action and SAM3
call with actual parameters, trace step, phase, observed result and the reason
for any VPP2/primitive switch. Do not invent, omit or reorder calls.
The runner exports {{recipe_tag}}_recipe.jsonl from recorded motion commands
and SAM3 calls that found a mask without an error, in invocation order.
Finding a mask does not establish correct identity. Historical pixels, steps
and coordinates are evidence, not targets to replay on a new layout.
The audit/recipe pair is published only when native success is confirmed."""

MEMORY = """Only after success, consolidate working notes and the successful
trajectory into {{memory_inbox}}/task-family_{{recipe_tag}}_draft.md. Write a
compact reference that helps evaluation adapt the strategy to a new scene:
- Recognition: observed visual cues, useful SAM3 descriptions and how identity
  was checked; distinguish object recognition from locating a contact feature.
- Phase plan: prerequisites, actions, visible completion gates, protected
  relations and terminal actions. Explain arm roles and whether the instruction
  requires them or the observed layout motivated them.
- Control choices: VPP2 instructions, Primitive use, chunk choices and the
  observed reason for continuing or switching. Keep exact tested parameters in
  the audit; include reusable settings with their conditions of use.
- Action costs: record observed costs for the main phases and required terminal
  motions, distinguishing productive actions from failed attempts and recovery.
  Treat these costs as evidence from this episode, not fixed budgets for new
  scenes. Use recorded action counters; do not invent missing measurements.
- Adaptation: what to re-identify, measure or decide on a new seed/layout,
  including approach geometry, grasp offsets, clearance and release conditions.
- Recovery and limits: failure evidence, changes tried, what subsequently
  worked, unresolved causes and situations where the strategy may not apply.

Separate observed outcomes from explanations. A successful retry does not prove
which change caused improvement. Do not invent causes or parameter ranges.
Keep scene coordinates in the audit, not reusable guidance. One seed/layout
supports single-shot confidence, even after several attempts; it does not
establish robustness across scenes. Use the recorded evidence after success,
not additional robot motion, to prepare memory.
Task-family frontmatter must be valid YAML delimited by --- lines:
---
scope: task-family
suite: robodojo
regime: {{env_cfg_type}}
task_id: {{task}}
task_language: >-
  <observed native instruction>
evidence:
  cells: [{{recipe_tag}}]
  attempts: <number of attempts>
confidence: single-shot
---

When evidence supports a cross-task lesson, write a separate
{{memory_inbox}}/new_global_<lesson-id>.md. Use scope: global, a bare descriptive
id, title, applies_when, kind (one of primitive, perception, strategy, failure,
infra), and the same evidence and confidence fields. State the observation,
how to apply it, its limits and what would falsify it. A single layout warrants
single-shot confidence only. Read existing relevant leaves before proposing
a lesson; avoid duplicates and record contradicting evidence explicitly.
If no new global lesson is supported, say so rather than inventing one.

During failed attempts, write lessons only in wip/. Never write directly to
published memory. The shared finalizer merges valid inbox proposals and updates
MEMORY.md after the run; do not claim publication before it completes."""


def system_prompt() -> PromptNode:
    """Assemble exploration rules with shared perception and control guidance."""
    return {
        "ROLE AND RULES": base.ROLE_AND_RULES,
        "EXPLORATION GOAL": GOAL,
        "READ ORDER AND MEMORY TRANSFER": READ_ORDER,
        "TOOL ACCESS": base.TOOL_ACCESS,
        "PERCEPTION": base.PERCEPTION,
        "AVAILABLE OPERATIONS": base.TOOLS,
        "VLA AND PRIMITIVE CONTROL": base.CONTROL,
        "PHASE CHECKS AND COMPLETION": base.PHASE_CHECKS,
        "SAFETY": base.SAFETY,
        "REWARD AND SCORE": base.REWARD,
        "EXPLORATION": RULES,
        "OUTPUT DISCIPLINE": OUTPUT,
        "MEMORY CONSOLIDATION": MEMORY,
    }
