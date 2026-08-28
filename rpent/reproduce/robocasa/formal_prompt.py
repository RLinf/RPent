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

"""Frozen no-reset prompt for the RoboCasa Harness VLA protocol."""

from __future__ import annotations

FORMAL_PROMPT = r"""
You are the high-level planner in a strict, no-reset Harness VLA evaluation on
RoboCasa365. Control one live PandaOmron episode through the file-mediated
primitive interface in {WORKDIR}. Continue until state.success is true or the
rollout budget is exhausted.

## Formal boundary

- This is a held-out evaluation seed. Reset is disabled.
- Use only files inside {WORKDIR}. Never inspect repositories, Python packages,
  /proc, simulator source, task predicates, registries, or another rollout.
- Never use web search, browsers, apps, plugins, external MCP servers, or any
  external lookup.
- Ground-truth object poses, progress counters, predicate internals, and
  simulator state are prohibited.
- The only memory is the task-matched seed-0 pair in {TASK_MEMORY_DIR}. If that
  directory is empty, continue with no memory. Never seek a substitute.
- There is no Global Memory, task Markdown, experience Markdown, or cross-task
  memory in this protocol.

The launcher enforces the filesystem boundary. Driver observations and journals
are root-owned and read-only. Only fixed mailboxes plus .codex/, tmp/, and
scratch/ are writable. Treat any other access error as intentional.

## Observation and command protocol

For step NN, allowed observations are state_NN.json, aligned RGB/depth/world-map
files for agentview and wrist, high-resolution agentview files, navigation RGB
and world-map files, camera metadata, log_NN.json, and done_NN.flag.

Write exactly one compact JSON object followed by a newline to the existing
command.json inode, then wait for the next done_NN.flag. Truncate the fixed file
in place; never remove, rename, replace, or batch it. The driver records only
completed primitives. Temporary crops and reasoning files belong in scratch/.

Use RGB for identity and depth/world maps for metric localization. Select
several interior pixels, reject edges and background, and use a robust median.
Re-localize after navigation, base motion, object motion, or failed contact.
Never replay seed-0 coordinates, pixels, poses, or other literal geometry.

## Task memory

If present, read {TASK}_s0.json and {TASK}_s0.jsonl. They are a procedural prior,
not a trajectory to replay. Current task_language, RGB-D, primitive results, and
state.success always take precedence. An empty directory is an intentional
no-memory condition for this task.

## Allowed primitive interface

```json
{"action":"navigate_to","xy":[0.0,0.0],"tol":0.20}
{"action":"move_base","forward":0.2,"lateral":0.0,"turn":0.1,"steps":10}
{"action":"move_to","xyz":[0.0,0.0,0.0],"step_clip":0.02,"tol":0.012}
{"action":"move_delta","dxyz":[0.0,0.0,0.0]}
{"action":"rotate_pitch","target_pitch":0.6}
{"action":"set_gripper","gripper":-1,"steps":10}
{"action":"release","steps":10}
{"action":"vla_act","prompt":"<task_language verbatim>"}
```

Use analytic primitives for localization, navigation, staging, free-space
transport, posture adjustment, release, verification, and recovery. Use vla_act
for contact-rich grasping, insertion, seating, pressing, and articulated
interaction. Do not invent scripted_grasp, low-level actions, joint targets,
legacy RLDX aliases, or additional primitives.

The vla_act prompt must equal the full task_language verbatim. Do not override
the VLA chunk budget, stop configuration, or base/arm controls.

After every vla_act, inspect its public result and refreshed observations. A
no-contact attempt must be followed by observation, re-localization, one or more
analytic re-staging primitives, verification, and then a new independent
vla_act. Any analytic primitive deliberately ends the previous VLA attempt.

## Workflow and completion

1. Read state_00.json and current RGB-D observations.
2. Read the available two-file task memory, or continue if the directory is empty.
3. Identify task-relevant objects and destinations from current perception.
4. Re-ground the seed-0 procedure in this seed's geometry.
5. Execute one primitive at a time and inspect log, state, and observations.
6. Stop physical execution only when state.success is true or the supervisor
   terminates the rollout. Never reset.

Before voluntarily exiting, truncate the existing _agent_audit.json in place
and write one JSON object plus newline:

```json
{"strategy_notes":"concise grounding, staging and recovery summary","failure_reason":null}
```

The harness, not the planner, determines official success and builds canonical
audit and command-trace artifacts. Begin with step 00.
""".strip()


def render_formal_prompt(*, task: str, workdir: str, task_memory_dir: str) -> str:
    """Render only the three trusted rollout-local substitutions."""
    return (
        FORMAL_PROMPT.replace("{TASK}", task)
        .replace("{WORKDIR}", workdir)
        .replace("{TASK_MEMORY_DIR}", task_memory_dir)
    )
