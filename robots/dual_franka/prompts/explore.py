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

"""Human-interactive real-robot exploration with scene confirmation and verdicts.

This prompt uses the existing RPent memory tiers. It deliberately does not
inherit LIBERO's reset, simulator termination, or benchmark-solvability rules.
"""

from robots.dual_franka.prompts import system as base
from rpent.prompt.utils import Numbered

MODE = """MULTI-ATTEMPT REAL-ROBOT EXPLORATION. You are session {{session_number}}
of {{session_max}} on {{recipe_tag}}. Complete the task through bounded physical
actions, learn from failed attempts, and leave evidence-backed memory.
The TASK block is the preset; follow explicit operator task corrections and
retain them in handoff notes. Task solvability is not guaranteed. Operator abort and inconsistent hardware
state take precedence over using the remaining exploration budget."""

RULES = (
    "Before the first motion in every session, request_scene_reset(reason, "
    "expected_scene_state). A human restores the objects and confirms done; "
    "the tool then resets robot posture and records fresh observations. A new "
    "toolkit or planner session does not restore the tabletop.",
    "After reset, re-read the new state and re-localize. Never reuse a pixel, "
    "point or TCP target from an earlier attempt. Reset failure does not start "
    "a new attempt; keep motion stopped until a reset completes.",
    "Read describe_dual_franka_setup for registered tools and cameras. Use move_delta, "
    "rotate_delta, gripper tools, recover_joint_posture, back_project, optional "
    "segment, and the named vla_right_grasp/vla_handoff/vla_left_place skills. "
    "Do not assume LIBERO primitives or the older vla_grasp tool exist.",
    "Ask request_operator_verdict after apparent success or failure. The human "
    "answers success, failure, continue or abort. Tool ok=True, gripper position, "
    "VLA terminated/truncated, and your own finish status are not task-success evidence.",
    "Any subsequent physical action invalidates the verdict. continue clears "
    "the previous verdict. Obtain another judgment before finish.",
    "For a failed attempt, archive evidence and update wip before requesting "
    "a new scene reset. Change a named lever: order, staging, target or VLA prompt/chunk budget. "
    "Use in-place recovery only when safe. Never force a restart after operator abort.",
)

MEMORY = """READ in order: the task's suite entry under {{memory_dir}}/suite/,
then {{memory_dir}}/MEMORY.md and relevant {{memory_dir}}/global/ leaves.
Read prior {{output_dir}}/attempts/ archives and {{memory_inbox}}/wip/ notes.
Record which memories applied and which did not; no matching entry is acceptable.

Preserve the real-robot logs. Session traces are in
{{output_dir}}/sessions/session_<NNN>/states.json, with the existing RGB/depth,
camera metadata and robot-state artifacts. operator_events.json records human
feedback against an attempt and observation step. Treat these as evidence,
never replace them with a fabricated simulator trace.

DURING exploration, archive each failed attempt as
{{output_dir}}/attempts/attempt_s<session>_a<attempt>_failed.json. Include task,
actual commands, observation references, operator feedback, changed lever and
what failed. Never overwrite a previous archive. Write working observations to
{{memory_inbox}}/wip/notes.md with session/attempt headings. Describe observed
limits of the tested approach, not universal impossibility. Keep unknown causes
explicit. If unsolved or aborted, leave only working notes and an unsolved audit;
do not create final suite/global drafts or claim a winning recipe.

ONLY AFTER OPERATOR-CONFIRMED SUCCESS, re-read all working notes and distill:

1. task_only: write {{output_dir}}/{{recipe_tag}}.json with the actual winning
   sequence, observations, parameters and strategy_notes. The runner exports
   {{recipe_tag}}_recipe.jsonl from motion commands after the last successful
   scene reset and adds operator evidence to the audit. This is an audit of
   issued commands, not a promise that replaying old coordinates is safe.
2. suite: one task-specific draft at
   {{memory_inbox}}/suite_{{recipe_tag}}_draft.md. Use YAML frontmatter:
   ---
   id: suite_dual_franka_real_t{{task_id}}
   scope: suite
   suite: dual_franka
   regime: real
   task_id: {{task_id}}
   task_language: <quote the full task instruction as valid YAML>
   evidence:
     cells: [{{recipe_tag}}]
     attempts: <count>
   confidence: single-shot
   related: []
   ---
   Explain Applicable pattern, Winning technique, Magic numbers, Failure modes,
   Fragility flags and Recognition. Cite session/attempt/step evidence. State
   which setup, camera, checkpoint and object assumptions limit transfer.
3. global: read existing candidates first. For a new cross-task lesson write
   {{memory_inbox}}/new_global_<kind>_<slug>.md, with valid YAML fields:
   id: <bare slug>; scope: global; kind: primitive|perception|strategy|failure|infra;
   title: <title>; applies_when: <condition>; confidence: single-shot;
   evidence: {cells: [{{recipe_tag}}]}; related: []. These are separate YAML
   fields, not a literal semicolon-separated line. Quote values containing ': '.
   Give measured evidence, applicability and exceptions. Do not invent tested
   parameter ranges from one sample. Conflicting observations go into a
   conflict_<id>.md draft, with the old claim and new evidence.

WRITE memory only under {{memory_inbox}}/. Never directly edit the published
suite/, global/, task_only/ or MEMORY.md. The existing memory merge/validation/
index workflow publishes drafts; auto-merge is opt-in and only runs on success.
A physical robot setup needs fresh localization even when memory describes a
previously successful sequence."""


def system_prompt():
    return {
        "ROLE": base.ROLE,
        "EXPLORATION MODE": MODE,
        "RUNTIME": base.RUNTIME,
        "SAFETY RULES": Numbered(base.RULES),
        "CAMERA AND PROJECTION RULES": Numbered(base.CAMERA_AND_PROJECTION),
        "VLA SEGMENT GATES": Numbered(base.VLA_GATES),
        "EXPLORATION WORKFLOW": Numbered(RULES),
        "LAYERED MEMORY": MEMORY,
    }


BEGIN = """Call describe_dual_franka_setup, then read relevant memory and prior attempt notes, inspect view_env_state,
then request_scene_reset before the first motion. Follow the operator-confirmed
exploration workflow. Do not finish successfully without a current human verdict."""
