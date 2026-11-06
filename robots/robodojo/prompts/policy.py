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

"""Backend-specific policy instructions shared by evaluation and exploration."""

POLICY_PROMPTS = {
    "vpp2": {
        "policy_name": "VPP2",
        "policy_tools": """vpp2_act(instruction, chunks=1) executes the supplied instruction. Describe the
visible object, action, destination or goal relation, preferably in English.
Include required arm roles, dependencies and what the other hand must preserve.
Keep the current phase consistent with the complete task; do not add unrelated
actions. VPP2 predicts 32 dual-arm targets and executes the first 24 per chunk,
then replans for each additional chunk. You inspect the result after the call.
History includes preceding Primitive and VPP2 actions; changing instructions
does not start a new episode. If history is unavailable after uncertain
execution, do not retry on stale feedback: inspect the scene and use fresh
analytic actions or follow the current mode's reset rules.""",
        "policy_role": """- VPP2: grasping, re-grasping, receiver grasp, bimanual coordination, insertion,
  articulation, tool use and motion requiring sustained contact.""",
        "policy_contact": "Use VPP2 for uncertain contact geometry rather than inventing poses.",
        "policy_retry": """clarify the
instruction""",
        "policy_action_counts": "24, 48 and 72",
        "retry_strategy": """Change a named variable on each retry: instruction, arm roles, action order,
chunk count, localization or approach pose.""",
        "memory_control": "VPP2 instructions",
    },
    "awomo05": {
        "policy_name": "Awomo05",
        "policy_tools": """awomo05_act(chunks=1) executes the original full task instruction using dual-arm
joint targets. It has no instruction parameter and cannot accept a rewritten
instruction or a phase-specific subtask. Each chunk executes 8 native actions,
then replans from real feedback. You inspect the result after the call.
History includes preceding Primitive and Awomo05 actions and persists across
calls; switching controllers does not start a new episode. A completed chunk
means commands executed, not task success. If execution is uncertain or model
history synchronization is lost, stop all motion, including Primitive actions,
and follow the current mode's reset or finish rules. Do not reset model history
independently of the environment.""",
        "policy_role": """- Awomo05: continue the original full-task policy when current observations
  support useful progress toward the task. A phase plan helps you choose when
  to call, inspect or interrupt the policy; it cannot be sent as a subtask.""",
        "policy_contact": """For uncertain contact geometry, consider continuing the original
full-task policy when observations support it; do not invent contact poses or
assume Awomo05 can be instructed to perform only the next phase.""",
        "policy_retry": """change the
chunk count""",
        "policy_action_counts": "8, 16 and 24",
        "retry_strategy": """Change a named variable on each retry: Primitive action order, policy/Primitive
switching point, chunk count, localization, approach pose or arm roles where
the native task instruction permits a choice.""",
        "memory_control": """the original full task instruction, observed phase and
  state at Awomo05 takeover""",
    },
}
