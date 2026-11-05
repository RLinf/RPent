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

"""Task definitions for the LeKiwi RPent extension."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeKiwiTask:
    """One real-robot task presented to the planner."""

    name: str
    instruction: str
    success_criteria: str
    constraints: tuple[str, ...]


LEKIWI_TASKS = {
    0: LeKiwiTask(
        name="drive_smoke_test",
        instruction=(
            "Inspect the current state and camera view, then exercise one "
            "conservative forward drive, one conservative rotation, and stop."
        ),
        success_criteria=(
            "Every requested primitive returns a usable result and the camera "
            "view changes in a way that agrees with the commanded motion."
        ),
        constraints=(
            "Keep each drive at or below 0.1 m/s and 30 deg/s.",
            "Keep each drive at or below 2 seconds.",
            "Call stop and inspect the camera view after every drive.",
            "Stop immediately if the camera view or velocity reading looks wrong.",
        ),
    ),
    1: LeKiwiTask(
        name="room_survey",
        instruction=(
            "Survey the room: drive short bounded segments, reading the front "
            "camera after each, and report what is visible around the base."
        ),
        success_criteria=(
            "The report describes distinct, consistent observations from "
            "several camera views taken from different positions."
        ),
        constraints=(
            "Keep each drive at or below 0.2 m/s and 45 deg/s.",
            "Keep each drive at or below 3 seconds.",
            "Inspect the camera view after every drive before moving again.",
            "Avoid obstacles: stop and turn when the view shows anything close.",
        ),
    ),
}


def get_lekiwi_task(task_id: int) -> LeKiwiTask:
    """Return a task by numeric ID with a clear error for unknown IDs."""
    try:
        return LEKIWI_TASKS[int(task_id)]
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"unknown LeKiwi task {task_id!r}; choices are {sorted(LEKIWI_TASKS)}"
        ) from exc
