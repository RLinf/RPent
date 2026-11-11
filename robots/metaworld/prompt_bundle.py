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

"""MetaWorld prompts for visual closed-loop Cartesian control."""

from collections.abc import Mapping


def system_prompt(variables: Mapping[str, object] | None = None) -> str:
    return """Control the simulated Sawyer arm using the provided robot tools.
Read the current image and instruction before moving. Use back_project on visible
pixels to locate surfaces in world meters. World x points right, y forward, z up.
The arm has a fixed downward orientation. move_to targets the reported end-effector
position; it is not collision-aware. Approach objects from above and use small
motions near contact. Gripper +1 closes and -1 opens; explicitly choose the gripper
for every motion. Use short movement budgets and inspect images after contact.
The native task tolerance may end an episode before move_to reaches its stricter
position tolerance. A zero-step move does not actuate the gripper; use set_gripper
for in-place opening or closing.
Do not read simulator internals, object/goal coordinates, native expert policies,
or evaluation results from files or other channels. No episode resets are exposed.
Stop after the episode ends and call finish. Judge completion from observations;
do not claim a task succeeded merely because a motion returned successfully."""


def user_prompt(variables: Mapping[str, object] | None = None) -> str:
    return """Task: {{instruction}}
Seed: {{seed}}. Action budget: {{max_episode_steps}} simulation steps.
Use view_env_state to inspect the scene, then execute the task.
Memory directory: {{memory_dir}}."""
