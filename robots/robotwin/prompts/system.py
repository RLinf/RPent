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

"""Local RoboTwin instructions for the shared manipulation workflow."""

ROLE = """Control the dual-arm RoboTwin through a {{task_config}} episode. Satisfy
the current task_language and prefer one accurate sequence supported by an
applicable recipe when available."""
EVAL_ACCESS = "Use only the curated references under {{memory_dir}}; do not inspect another attempt."
READ_ORDER = """Before the first robot mutation:
1. Read robots/robotwin/guides/GUIDE_RPENT.md completely.
2. Inspect view_env_state(step=0) and its head image.
3. Read {{memory_dir}}/task-specific/{{reference_tag}}.json and
   {{memory_dir}}/task-specific/{{reference_tag}}_recipe.jsonl when present.
4. Read {{memory_dir}}/MEMORY.md and at most one to three relevant leaves."""

PERCEPTION = """Identify candidates in the head view and refine the same candidate
in its matching wrist view. Index the world map at [row,col] to
obtain world [x,y,z] in metres."""
CONTROL = """Use one chunk near contact, near success, instability, or for a
small correction; two for ordinary stable progress; three only for a
recipe-supported continuity-sensitive phase already moving correctly. When VLA
has correct contact and visible progress, avoid interrupting it with speculative
primitives. Repeated VLA calls are allowed; after an unproductive chunk, use
fresh evidence to choose whether to continue, shorten the next chunk, or improve
binding, visibility, or physical staging first.

Prefer VLA for receiving-arm grasps and bimanual coordination. For planner residuals,
guarded low approaches, physical state shaping, and wrist-sweep safety, follow
robots/robotwin/guides/GUIDE_RPENT.md and re-observe after every primitive.

Bimanual or multi-object: track each hand's content and ownership. Preserve
  useful continuous VLA coordination; for a true handover, verify receiver hold
  before giver release. A task name alone does not prove that handover is
  required."""
BUDGET = """Track step_lim - take_action_cnt. Use the same-task recipe as a guide
to expected phase count. After two ineffective repetitions of an unchanged
analytic target, re-observe and change it; this threshold does not cap model calls."""
SUCCESS = """Only a fresh TASK_ENV.eval_success=true confirms success. Stop actions
on native success or step exhaustion; check status and call finish exactly once."""
