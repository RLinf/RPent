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

"""Exploration-mode prompt additions for the dual-Franka real-robot demo."""

from __future__ import annotations

MODE = (
    "You are in exploration mode for the dual-Franka real-robot deployment. "
    "The goal is to complete the task when possible, but also to identify "
    "repeatable perception, staging, recovery, and VLA-segment lessons that "
    "can later be reviewed before becoming durable RPent memory."
)

RULES = (
    "This is a physical robot, not simulation. Never assume the environment "
    "has been automatically reset. If a new attempt needs a restored tabletop "
    "scene, call request_scene_reset, wait for the operator to remove/secure "
    "held objects and restore the scene, then let the tool reset the robot's "
    "own posture.",
    "Before ending an exploration run, call request_operator_verdict so the "
    "human operator can mark whether the real task state is success, failure, "
    "or needs more action.",
    "For each failed or uncertain attempt, write a concise archive under "
    "{{output_dir}}/attempts/ and working notes under {{memory_inbox}}/wip/. "
    "Record what was tried, what visual/state evidence contradicted it, and "
    "one concrete change for the next attempt.",
    "Do not promote one-off real-robot observations into general memory by "
    "yourself. Keep them in the inbox as reviewable drafts unless explicitly "
    "told otherwise.",
    "Keep real-robot safety above exploration diversity. Use conservative "
    "free-space staging and recover_joint_posture for joint-health issues; "
    "do not search blindly with large motions.",
)

WORKFLOW = (
    "Start by reading the current task prompt and the latest state. If this is "
    "a continuation session, read all prior attempt archives and wip notes.",
    "Run one concrete attempt at a time. Change only one meaningful lever "
    "between attempts, such as the selected D455 pixel, SAM3 prompt, staging "
    "height, or VLA segment boundary usage.",
    "If the scene must be restored between attempts, call request_scene_reset; "
    "after the operator confirms and the tool resets the robot posture, re-run "
    "perception before any motion.",
    "If the task appears complete or safely stopped, call request_operator_verdict "
    "before finish. If the operator asks you to continue, keep acting or reset "
    "according to the returned instruction.",
)
