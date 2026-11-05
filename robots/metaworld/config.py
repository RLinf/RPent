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

"""Lightweight supported-task and launch configuration."""

TASKS = {
    "reach-v3": "Move the gripper to the visible red goal marker.",
    "push-v3": "Push the puck to the visible goal marker on the table.",
    "pick-place-v3": "Pick up the puck and place it at the visible goal marker.",
    "door-open-v3": "Pull the handle to open the door.",
    "drawer-open-v3": "Pull the drawer open using its handle.",
    "drawer-close-v3": "Push the drawer closed.",
    "button-press-topdown-v3": "Press the button down from above.",
    "peg-insert-side-v3": "Insert the peg into the side-facing hole.",
    "window-open-v3": "Slide the window open using its handle.",
    "window-close-v3": "Slide the window closed using its handle.",
}
CAMERAS = ("corner", "corner2", "corner3", "topview", "behindGripper", "gripperPOV")


def validate(task: str, seed: int, max_episode_steps: int, camera: str) -> None:
    """Reject unsupported launch selections before loading the simulator."""
    if task not in TASKS:
        raise ValueError(f"unsupported MetaWorld task: {task!r}")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if camera not in CAMERAS:
        raise ValueError(f"unsupported camera: {camera!r}")
