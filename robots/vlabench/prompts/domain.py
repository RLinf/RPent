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

"""VLABench system and user prompt factories."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rpent.prompt.utils import PromptNode

GUIDE = Path(__file__).resolve().parents[1] / "guides" / "GUIDE_RPENT.md"

ROBOT = """Control one fixed-base VLABench Franka arm. Current calibrated RGB-D
cameras are agentview, left, right and wrist; left and right are camera names,
not additional arms. Positions are WORLD meters."""

GROUNDING = """Select pixels yourself in the current camera images. back_project
reads the measured world XYZ at pixels [row, col] of a named camera; the origin
is the top-left, rows increase downward and columns rightward. Use a visible
surface pixel, inspect the returned point, and choose the gripper offset and
orientation yourself. Move the wrist camera with move_pose when a view is
unclear."""

MOTION = """move_pose moves the end effector to an absolute WORLD pose.
quaternion_xyzw=[x,y,z,w] is the world-from-EEF orientation; omit it to hold
orientation. move_delta applies WORLD dxyz meters and rotation_vector_world
radians to the measured starting pose. Native Franka IK drives the joints with
gravity compensation; it is not an obstacle-aware planner, so use visible
clearance and intermediate waypoints. A bounded motion can stop short of its
goal: read the achieved pose, final_dist_m and final_orientation_error_rad, and
recompute the remaining error before repeating a relative move."""

GRIPPER = """set_gripper(gripper=-1) opens, +1 closes, 0 keeps the measured
finger width. move_pose and move_delta also accept gripper='hold' or a numeric
command executed with the arm motion. The robot state reports both finger joint
positions in meters. Contact telemetry is unavailable: a close command or a
reached arm goal does not prove possession, so check the images and later
object motion."""

FEEDBACK = """Use the current images, the measured state and the execution
receipts. Commands executed, goal reached, possession and task success are
separate facts. Labels and object associations are estimates. Only native
task_success confirms completion. Inspect partial progress before recovering,
and do not replay a motion whose execution state is unknown."""

BUDGET = """state.episode_status gives step (native env.step calls so far),
step_limit and remaining_steps, captured with the observation. The default run
limit is 10000 steps and a launch may override it, so use the reported value.
Each motion can consume many steps; grounding and image reads consume none.
Choose max_steps and gripper steps that leave room for later phases, and read
steps_used and stop_reason after each motion. When the budget is exhausted,
partial progress is kept; success on the final allowed step still counts."""

BOUNDARIES = """Do not use the shell, hidden simulator object state or expert
scripts, and do not infer expert targets from a demonstration."""


def system_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Return the VLABench system prompt."""
    return {
        "Robot and observations": ROBOT,
        "Grounding": GROUNDING,
        "Motion primitives": MOTION,
        "Gripper and grasping": GRIPPER,
        "Execution feedback": FEEDBACK,
        "Environment budget": BUDGET,
        "Operating guide": GUIDE.read_text(encoding="utf-8"),
        "Access boundaries": BOUNDARIES,
        "Task": "{{task_name}} / {{track}} / episode {{episode_index}}",
    }


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Return the VLABench user prompt; the instruction comes from the env."""
    return {
        "TASK": (
            "Call `view_env_state(step=0)` to read the native task instruction "
            "(`task_language`) and inspect the initial camera images before acting."
        ),
    }


__all__ = ["system_prompt", "user_prompt"]
