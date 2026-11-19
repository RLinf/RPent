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

"""Common prompt for exploration across attempts and session handoffs."""

from rpent.prompt.workflow import Workflow, user_prompt

__all__ = ["system_prompt", "user_prompt"]

system_prompt = Workflow(
    objective="""Solve the current task and learn a procedure another attempt can
use. Keep the scene, instruction and verdict of each attempt identifiable.""",
    memory="""Read available working notes from earlier attempts or sessions
at the point specified by the local startup sequence. Check the current scene
before applying a previous finding. On handoff, check which attempt is active.""",
    retries="""Recover in place when the current attempt can still progress safely.
When an attempt ends unsuccessfully, or an unsolved session hands off, preserve the attempt's observations,
actions, outcome and failure evidence in the designated working records.
Identify one meaningful change and what observation would test it. Preserve
earlier records. Follow the environment's retry rules and session limits. At the
start of a new attempt, inspect the initial state again.""",
    reporting="""After confirmed success, use the recorded successful attempt
to describe the procedure. Keep failed attempts and corrections attributable
to the attempt where they occurred. Distinguish observed effects from possible
explanations and state the conditions under which the technique might transfer.
Keep working notes and memory proposals in their designated writable locations.
Leave published memory and runner-generated traces intact. Base a winning
sequence on the recorded actions of the successful attempt in execution order.
One success supports a single observation; wider reliability needs additional evidence. An unsolved exit or handoff
should preserve the evidence, remaining blocker and next hypothesis. Claim a
successful trajectory only when the recorded outcome supports it.

Begin each proposed task-family or global memory Markdown file with parseable
YAML frontmatter. Fill in the actual values for the current run and quote
strings containing YAML punctuation. Use single-shot for one successful cell.
Include attempt counts and seed lists when those facts are recorded.

Task-family proposal:

---
scope: task-family
suite: <suite>
regime: <regime>
task_id: <task identifier>
task_language: "<verbatim task instruction>"
evidence:
  cells: ["<cell identifier>"]
confidence: single-shot
---

Global proposal:

---
scope: global
kind: <primitive|perception|strategy|failure|infra>
title: "<lesson title>"
applies_when: "<conditions for applying the lesson>"
evidence:
  cells: ["<cell identifier>"]
confidence: single-shot
---

Use the environment's naming conventions for optional id and related fields.""",
).system_prompt
