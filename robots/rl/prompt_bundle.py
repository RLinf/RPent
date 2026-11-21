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

"""Compose one shared robot contract with the selected workflow instructions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

OBSERVATION_MODES = ("adaptive",)

BASE = """Robot and frames
- One Franka arm with a parallel-jaw gripper. Its base stands at the -x edge of the workspace and faces +x; +y is the robot's left and +z is up. The work surface is horizontal.
- pose.xyz is the grasp point between the finger pads, in meters. pose.orientation_rpy is the absolute grip-site orientation [roll,pitch,yaw] in radians, fixed-axis XYZ, R=Rz(yaw)@Ry(pitch)@Rx(roll); [pi,0,yaw] points its +z approach axis down. opening is the normalized jaw gap, 1 open and 0 closed, and finger_gap_m is that gap in meters.
Execution
- move_to servos toward one absolute target at 20 Hz, at most 0.03 m and 0.12 rad per control step, and returns once within 0.004 m and 0.02 rad, when progress stalls, or after 300 steps (15 s). Its reply is the measured outcome: reached, partial with a reason, or error. Omitted targets keep their measured values and the jaw target persists.
- set_gripper drives the jaws toward an opening and reports the measured one; closing stops at contact, which is not proof of a grasp.
- Motion is a straight line with no collision avoidance. Approach objects from above, keep the grasp point above surfaces and other objects, and move at most about 0.10 m per call.
Method
- Observe before acting and after every motion; the measured state and the latest observation step are the truth, never the command you sent.
- A grasp cycle: hover above the object, align the yaw, descend, close, read the opening, lift, carry, lower, open, then retreat straight up before moving aside.
- The environment judges success from object poses. After placing, release and retreat so the object settles; call finish once the task state is visibly complete."""

GRIPPER = """Gripper geometry
- The grasp point lies between the finger pads, 9.7 cm beyond the wrist flange along the approach axis. The pads reach about 1.2 cm above and 0.4 cm below it and the fingertips about 0.9 cm below it, so keep the grasp point at least 1.5 cm above the surface an object stands on.
- The jaws close along the grip-site x-axis. With orientation_rpy=[pi,0,yaw], that closing line has world-xy heading yaw. axis_yaw describes a visible surface's major axis, not a commanded gripper orientation.
- Fully open, the jaws span about 0.08 m (opening 1.0 is a gap of about 0.078 m); each pad is 16 mm wide and 16 mm tall. An object wider than 0.075 m along the closing axis cannot be grasped there.
- Grasp near mid-height so the pads cover the object's sides: descend until the grasp point is 1 to 2 cm below top_z_m for low objects, and to mid-height for tall ones.
- After closing, an opening of 0.03 or less (gap under 3 mm) means the jaws met nothing; a held object leaves an opening of roughly 0.08 to 0.6 (gap 6 to 50 mm) with status blocked. Reopen before retrying. The jaw target persists through later moves, so a held object stays held until set_gripper opens."""

CAMERA = """Cameras
- agentview is fixed on the far side of the workspace, looking back toward the robot and downward. The robot base appears near the top of its image; down the image is +x, toward the camera, and right is +y, the robot's left. Its vertical field of view is 45 degrees.
- wrist rides with the gripper, 9.7 cm above the grasp point along the approach axis and 5 cm to its side, looking along that axis. Use the displayed wrist image for pixel directions: its camera frame is different from both the grip-site frame and the fixed external camera. Its vertical field of view is 75 degrees.
- Pixels are [row, col] from the top-left corner. A pixel alone fixes only a ray; RGB-D measurements report the depth-measured world point of the visible surface, so world_xyz lies on the object's top or facing side rather than its center, and top_z_m is its highest visible point.
- Use pixels, masks and poses from the same observation step, and take a new observation after every motion."""

ENVIRONMENT_WORKFLOWS = {
    "libero": (
        "episode",
        "episode_image",
        "zap_turn",
        "zap_episode",
        "zap_subtask",
        "pear",
    ),
    "robocasa": ("episode",),
    "robotwin": ("episode",),
}


def validate_workflow(environment: str, workflow: str) -> None:
    """Reject unsupported combinations before any simulator is started."""
    if environment not in ENVIRONMENT_WORKFLOWS:
        raise ValueError(f"unsupported simulator: {environment!r}")
    if workflow not in ENVIRONMENT_WORKFLOWS[environment]:
        raise ValueError(
            f"{environment} does not support --workflow {workflow}; "
            f"choose {' or '.join(ENVIRONMENT_WORKFLOWS[environment])}"
        )


WORKFLOWS = ENVIRONMENT_WORKFLOWS["libero"]


PROMPT_VARIANTS = ("simple", "base", "base_gripper", "base_gripper_camera")
_SECTIONS = {
    "simple": (),
    "base": (BASE,),
    "base_gripper": (BASE, GRIPPER),
    "base_gripper_camera": (BASE, GRIPPER, CAMERA),
}


def system_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Return the adaptive tool contract and the selected prompt sections."""
    variables = variables or {}
    if variables.get("observation_mode", "adaptive") != "adaptive":
        raise ValueError("rl supports only adaptive observations")
    environment = str(variables.get("environment", "libero"))
    workflow = str(variables.get("workflow", "episode"))
    validate_workflow(environment, workflow)
    from robots.rl.contracts import validate_perception_options

    perception = str(variables.get("perception_mode", "single"))
    validate_perception_options(
        workflow,
        perception,
        int(variables.get("max_observe_calls", 8)),
        int(variables.get("observe_concurrency", 8)),
    )
    variant = str(variables.get("prompt", "simple"))
    if variant not in _SECTIONS:
        raise ValueError(
            f"unknown prompt variant {variant!r}; expected one of {PROMPT_VARIANTS}"
        )
    prompts = Path(__file__).parent / "prompts"
    general = variables.get("world_cord", "base") == "general"
    if workflow == "pear":
        sections = [(prompts / "pear.md").read_text().strip()]
        sections.append(
            (prompts / ("libero_general.md" if general else "libero.md"))
            .read_text()
            .strip()
            .replace(
                "Segmentation reports visible surfaces, not object centres or guaranteed grasp poses. ",
                "",
            )
        )
        sections.append(
            "Reason from the current observations, then call exactly one action tool."
            if variables.get("reasoning", True)
            else "Reasoning is disabled. Call exactly one action tool directly. Do not output reasoning, explanations, or any other text."
        )
        return "\n\n".join(sections)
    sections = [
        (prompts / name).read_text().strip()
        for name in (
            "shared.md",
            f"{environment}_general.md"
            if general and environment in {"libero", "robotwin", "robocasa"}
            else f"{environment}.md",
            "episode_observe.md" if perception == "batch" else f"{workflow}.md",
        )
    ]
    if perception == "batch":
        if environment != "libero":
            from robots.rl.backends import RoboCasaPrimitives, RoboTwinPrimitives

            camera = {
                "robocasa": RoboCasaPrimitives,
                "robotwin": RoboTwinPrimitives,
            }[environment].cameras[0]
            sections[-1] = sections[-1].replace("agentview", camera)
        sections.append(
            f"Each observe accepts at most {variables.get('max_observe_calls', 8)} suboperations. Independent operations are scheduled with concurrency at most {variables.get('observe_concurrency', 8)}."
        )
    if (environment == "libero" and workflow in {"episode", "episode_image"}) or (
        environment in {"robotwin", "robocasa"} and workflow == "episode" and general
    ):
        if variables.get("action_mode", "single") == "batch":
            if environment == "robocasa":
                sections.append(
                    "Action interface: use execute_actions for move_to and set_gripper, including single-item lists. move_to and set_gripper are not standalone tools; move_base and move_lift remain standalone tools. Maximum list length is "
                    + str(variables.get("max_actions_per_call", 3))
                    + ". Items execute in order and stop on partial/error/success/truncation. Read every measured result before deciding the next batch; remaining items after a stop did not execute."
                )
            else:
                sections.append(
                    "Action interface: use execute_actions for every physical action, including single-item lists. move_to and set_gripper are not standalone tools. Maximum list length is "
                    + str(variables.get("max_actions_per_call", 3))
                    + ". Items execute in order and stop on partial/error/success/truncation. Read every measured result before deciding the next batch; remaining items after a stop did not execute."
                )
        sections.append(
            "The request may omit old reasoning and camera images according to configured history retention. Omitted images are unavailable: do not pretend to see their pixels. Tool calls and measured text results remain available, subject to conversation compaction."
        )
    if environment == "libero":
        sections.extend(_SECTIONS[variant])
        if general:
            sections = [
                section.replace("grip_site", "grasp TCP").replace(
                    "grip-site", "grasp TCP"
                )
                for section in sections
            ]
            sections = [
                section.replace(
                    "The grasp point lies between the finger pads, 9.7 cm beyond the wrist flange along the approach axis. The pads reach about 1.2 cm above and 0.4 cm below it and the fingertips about 0.9 cm below it, so keep the grasp point at least 1.5 cm above the surface an object stands on.",
                    "The grasp TCP is the measured midpoint of the two finger-pad geometry centres. Keep clearance for the full pads and fingertips when approaching a support surface; the TCP is not the lowest part of the gripper.",
                ).replace(
                    "9.7 cm above the grasp point along the approach axis and 5 cm to its side",
                    "mounted above and beside the grasp TCP",
                )
                for section in sections
            ]
            if CAMERA in _SECTIONS[variant]:
                sections[-1] = (
                    "Cameras: agentview is fixed and wrist moves with the gripper. Use the displayed image orientation and current observation step; do not infer pixel directions from a different camera or robot pose."
                )
    return "\n\n".join(sections)


def user_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Introduce the task and initial state that planners append."""
    text = "The task and the measured initial state follow."
    if (variables or {}).get("perception_mode", "single") == "batch":
        text += " image_sources lists the current camera image IDs; these are references, not displayed pictures. Use observe to view them or measure them, and copy returned image_id/mask_id values into source."
    return text
