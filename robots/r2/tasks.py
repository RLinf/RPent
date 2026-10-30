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

"""Task definitions for the R2 RPent extension."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class R2Task:
    """One real-robot task presented to the planner."""

    name: str
    instruction: str
    success_criteria: str
    constraints: tuple[str, ...]


R2_TASKS = {
    0: R2Task(
        name="primitive_smoke_test",
        instruction=(
            "Inspect the current state, tilt the head to look at the area in front "
            "of the robot, then move each arm once by a small offset, rotate one "
            "wrist slightly, open and close one gripper, and return home."
        ),
        success_criteria=(
            "Every primitive returns ok and the recorded state and camera images "
            "agree with the commanded change."
        ),
        constraints=(
            "Keep translations at or below 0.03 m per call.",
            "Keep rotations at or below 0.2 rad per call.",
            "Stop immediately if images or state indicate unsafe motion.",
        ),
    ),
    1: R2Task(
        name="tabletop_pick_place",
        instruction=(
            "Pick up the object on the table in front of the robot and place it "
            "about 0.15 m to the side of where it started."
        ),
        success_criteria=(
            "The object is visibly lifted between the gripper fingers, then rests "
            "on the table at the new place with the gripper open and clear of it."
        ),
        constraints=(
            "Locate the object with segment or back_project before moving toward it.",
            "Hover 0.10 m above a grasp point, then descend in steps of 0.05 m or less.",
            "Do not infer a grasp from gripper opening alone; confirm it in the images.",
        ),
    ),
}


def get_r2_task(task_id: int) -> R2Task:
    """Return a task by numeric ID with a clear error for unknown IDs."""
    try:
        return R2_TASKS[int(task_id)]
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"unknown R2 task {task_id!r}; choices are {sorted(R2_TASKS)}"
        ) from exc
