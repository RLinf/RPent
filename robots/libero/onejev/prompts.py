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
"""Concise instructions for choosing among computed LIBERO actions."""

INSTRUCTIONS = (
    "Choose the feasible action that best advances the stated task using the RGB views, "
    "measured geometry, robot state and recent action results. Red rectangles labeled "
    "r0, r1, ... are geometric proposals, not identified objects. Match the proposed "
    "region to the destination in the task visually. A pick success is only a motion "
    "and gripper heuristic; inspect the views for evidence that the named object is "
    "held. A task may name several objects sharing one destination; each grasp "
    "targets only one object. active_object is the current holding hypothesis. "
    "Per-object placement counts record attempts, not confirmed success. Use the "
    "new views to check placement and avoid grasping an object already correctly "
    "placed. After release, clear the receptacle with the open gripper before the "
    "next grasp. Candidate arguments are fixed and cannot be changed. Probabilities rank "
    "these actions; they are not task-success estimates."
)

CONTACT_INSTRUCTIONS = (
    "Choose a bounded Pi0.5 contact action for the drawer named in the task. "
    "Compare the current public agentview and wrist RGB images with recent "
    "actions before repeating contact. A contact action's success field means "
    "official task termination, not visual progress. Use finish if the drawer "
    "cannot be identified or another attempt is unsafe. Candidate arguments "
    "are fixed and cannot be changed."
)
