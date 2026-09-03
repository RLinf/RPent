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
"""Visual-grounding queries used at each stage of task-card replay.

The object noun comes from the card. These templates add only the spatial or
visual context needed by each localization stage.
"""

PROMPTS = {
    # Opening survey from the fixed camera.
    "survey": "{object}",
    # Close refinement from the wrist camera with the arm above the object.
    "refine": "the center of the {object} directly below the gripper",
    # Recovery when a point falls outside the object's surface.
    "recover": "a point on the body surface of the {object}, not the table behind it",
    # Identity confirmation using visible text on the object.
    "confirm": "the printed label on the {object}",
    # Top face used to estimate placement height.
    "top": "the top surface of the {object}",
    # Object currently held by the gripper.
    "held": "the body of the {object} held in the gripper",
    # Resting position after release.
    "settled": "the {object} resting on the table",
}


def build(situation: str, obj: str) -> str:
    return PROMPTS[situation].format(object=obj)


#: Localization situation associated with each stage of a recipe.
SCHEDULE = """
recipe action                     situation   camera      what it answers
--------------------------------- ----------- ----------- ---------------------
(episode start)                   survey      agentview   coarse position
move_to above object, before pick  refine      wrist       precise position
  if refine lands off the object   recover     wrist       a point on the body
  if two objects could be confused confirm     wrist       which one this is
place destination, before descent  top         wrist       surface height
set_gripper after pi0_pick         held        wrist       offset from gripper
release                            settled     wrist       where it landed
"""
