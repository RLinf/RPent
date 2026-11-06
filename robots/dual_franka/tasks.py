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

"""Task definitions for the dual-Franka RPent extension."""

from __future__ import annotations

from robots.franka.tasks import FrankaTask

# Named VLA tools pass this instruction to the policy unchanged.
CLEAN_DESK_VLA_PROMPT = (
    "I am currently performing a desk organizing task. I will use my right hand "
    "to hand bowls, plates, cups, chopsticks, and a spoon to my left hand in "
    "order, and then put them into a basket."
)

OPERATOR_FINISH_CONSTRAINT = (
    "The agent must not finish the episode by itself. Call finish only after an "
    "operator_verdict.status of success or failure is observed."
)


DUAL_FRANKA_TASKS = {
    0: FrankaTask(
        name="primitive_smoke_test",
        instruction=(
            "Inspect the current state, then for each arm in turn exercise one "
            "conservative translation, one conservative rotation, and the gripper "
            "controls."
        ),
        success_criteria=(
            "Every requested per-arm primitive returns a usable result and the "
            "synchronized camera/state snapshots agree with the commanded change."
        ),
        constraints=(
            "Move exactly one arm per motion call; never assume a 'both' mode.",
            "Keep translation commands at or below 0.02 m per call.",
            "Keep rotation commands at or below 0.15 rad per call.",
            "Do not call a VLA skill during this diagnostic task.",
            "Stop immediately if images or state indicate unsafe motion.",
        ),
    ),
    1: FrankaTask(
        name="clean_desk_dirty_clean_sorting_agent_vla",
        vla_instruction=CLEAN_DESK_VLA_PROMPT,
        instruction=(
            "Complete a dual-arm collaborative tabletop object-storage task "
            "in this object category order: bowls, plates, "
            "cup, chopsticks, then spoon. Finish the entire current category "
            "before targeting anything from the next category. Within any "
            "category that has multiple colored source objects, use this "
            "color order: process green before blue. Do not let "
            "dirty/clean status change the grasp order. The only added logic "
            "is placement-time dirty/clean sorting for bowls and plates: before "
            "grasping each bowl or plate, classify whether it is dirty or "
            "clean. Dirty bowls and dirty plates go into the metal wire "
            "basket/frame; clean bowls and clean plates go into the "
            "black-outside, white-inside cardboard box. The cup still goes "
            "into the cardboard box, and the chopsticks and spoon still go "
            "inside the cup after the cup is stable in the cardboard box. "
            "After the right hand successfully grasps an object, call "
            "vla_handoff directly from the confirmed post-grasp state. Before "
            "calling vla_left_place, localize the required target container "
            "and horizontally align the left-held object above that target."
        ),
        setup=(
            "This is a dirty/clean sorting variant of the clean-desk task, "
            "using category order bowls, plates, cup, chopsticks, spoon and "
            "green-before-blue order within each category. "
            "The base camera and fourth-view D455 are RealSense RGBD cameras, "
            "but this dirty/clean sorting task exposes D455 metric "
            "localization tools. D455 aligned depth can be back-projected "
            "into the right_base frame using the calibrated extrinsics. "
            "The D455 is a fixed external camera; moving either arm or a held "
            "object does not move the camera viewpoint and must not be used as "
            "an active-vision strategy. "
            "right_base is the shared world coordinate frame exposed to the agent: "
            "camera points, left/right TCP positions, and rule-based base-frame "
            "deltas are all represented in right_base. "
            "The black-outside, white-inside cardboard box is on the +y side "
            "of the adjacent metal wire basket/frame. Both containers are "
            "valid in this task: the metal wire basket/frame is the required "
            "destination for dirty bowls and dirty plates, while the cardboard "
            "box is the required destination for clean bowls and clean plates. "
            "A dirty bowl or plate is marked by a visible egg tart or egg-tart "
            "foil cup/tray inside or on the concave eating surface; a clean "
            "bowl or plate has no egg tart or egg-tart foil cup/tray marker. "
            "Use the contrast between the black exterior and white interior as "
            "the cardboard-box inside/outside boundary, and use the metal "
            "wires/rim as the metal basket boundary. "
            "The VLA checkpoint is the deployment-aligned dual-arm tcp-rot6d "
            "clean-desk policy; named VLA tools keep using its fixed training "
            "instruction, while the planner decides the dirty/clean destination."
        ),
        success_criteria=(
            "All target objects have been processed in the required category "
            "order. All dirty bowls and dirty plates are inside the metal wire "
            "basket/frame. All clean bowls, clean plates, and the cup are "
            "inside the black-outside, white-inside cardboard box. The "
            "chopsticks and spoon are inside the cup after the cup is stable "
            "inside the cardboard box. Stop for operator feedback if a "
            "dirty/clean label is uncertain, if an object was placed in the "
            "wrong container, or if visual evidence does not support safe "
            "continuation."
        ),
        constraints=(
            "Call describe_dual_franka_setup before acting.",
            "Call view_env_state before the first action only if no fresh primitive snapshot is already available. After any primitive that returns a snapshot, inspect that returned snapshot directly; never call view_env_state immediately afterward unless the primitive failed, returned no snapshot, an operator changed the scene, or a specific historical step is needed.",
            "Enforce this grasp order without exception: first category order bowls -> plates -> cup -> chopsticks -> spoon, then within any category that has multiple colored source objects, green before blue. Always act on the earliest unfinished category. While any object from that category remains in the source workspace, do not localize, approach, grasp, or call a VLA grasp skill for any later category. Within the current category, do not localize, approach, grasp, or call a VLA grasp skill for a blue object while a green object from the same category remains in the source workspace.",
            "Dirty/clean status must not reorder grasping. For bowls and plates, first choose the next object by the category/color order, then classify that chosen object as dirty or clean, and only use the dirty/clean label to choose the placement destination before vla_left_place.",
            "Classify dirty vs clean before each bowl or plate attempt. A dirty bowl or plate has a visible egg tart or egg-tart foil cup/tray inside or on the concave eating surface; a clean bowl or plate has no egg tart or egg-tart foil cup/tray marker. If the egg-tart marker is occluded, ambiguous, outside the dish, or cannot be judged in the latest D455 image, do not move the robot; request operator feedback.",
            "An egg tart or egg-tart foil cup/tray counts as a dirty marker only if it is visibly inside or on the concave eating surface of that specific bowl or plate. A nearby egg tart or foil tray on the table, in another object, or outside the dish must not make the dish dirty.",
            "Build a brief D455 localization and sorting table before moving in each category. For each visible candidate in the current category, list object type, color if visible, dirty_clean_label for bowls/plates, egg_tart_evidence for bowls/plates, required_destination, D455 evidence, projected point if used, and status.",
            "When SAM3 is available, use segment with camera='d455' on the D455 image to segment the current object or required placement target before falling back to a manual pixel. Use short phrases, inspect the returned mask overlay yourself, and retry with a point prompt, more specific text prompt, or manual D455 pixel whenever the mask or median marker is not visibly on the intended target.",
            "Use back_project only on a pixel well inside the visible material of the intended object or placement container. Do not select the table, background, rim edge, air above the object, another object, a container wall, metal wire, or a shared container boundary. Inspect the returned annotated image; if the marker is not on the intended target, retry or stop.",
            "When exact spatial relation is uncertain, use segment or back_project again instead of judging metric x/y/z offsets by 2D RGB appearance. After a segmentation/projection is verified, trust the returned right_base xyz and TCP-to-point deltas as the main metric evidence for rule-based correction moves.",
            "Treat right_base as the only world coordinate frame for agent reasoning. For both left and right arms, move_delta and rotate_delta use right_base/world deltas; do not convert left-arm coordinates yourself.",
            "For bowls, do not use rule-based move_delta to pre-align or move above the bowl before vla_right_grasp. This avoids pushing the learned VLA policy out of distribution. After D455 confirms the intended bowl identity and dirty/clean label, call vla_right_grasp directly from the current robot state.",
            "For plates, the cup, chopsticks, and the spoon, rule-based pre-grasp staging is allowed only when clearly needed: move the right TCP to a safe pre-grasp pose about 10cm above the projected target surface point in right_base/world coordinates, without touching the object.",
            "Call vla_right_grasp directly after bowl identity and dirty/clean classification are confirmed. For non-bowl objects, call vla_right_grasp only after the latest right-arm move_delta reports a reached/effective target near the intended object, unless the object is already safely staged. Leave max_chunks unset unless running a cautious diagnostic.",
            "Use VLA segment tools for contact-rich motion: vla_right_grasp for the right-arm grasp segment, vla_handoff for bimanual transfer, and vla_left_place for left-arm placement.",
            "After a grasp VLA, inspect the returned snapshot. Continue only if the right gripper appears to hold the intended object. Do not judge lifted/held status from 2D image appearance alone. If it is unclear whether the object has lifted or remains on the table, call back_project with camera='d455' on the best visible held-object surface when possible. If only a source/table candidate is visible, treat that projection as a source-candidate check, not proof of failure. Compare returned right_base z primarily with the source/table height or pre-grasp source projection. For wide objects such as plates, projected center/rim x/y can be far from the right TCP while still held; do not mark failure from TCP x/y distance alone. If a verified projection of the intended object or held-object surface is higher than the source/table height or pre-grasp source projection, treat the grasp as successful and proceed to vla_handoff. Mark failure only when a verified projection shows the same intended object still at source/table height and gripper/image evidence shows the right gripper is empty or not supporting it. If the object is occluded or identity/source is ambiguous, mark the grasp status uncertain and request operator feedback instead of opening or retrying. If the wrong object is grasped or identity is uncertain, stop for operator feedback.",
            "After a confirmed right-hand grasp, call vla_handoff directly. Do not call move_delta, rotate_delta, open_gripper, close_gripper, or any other rule-based primitive to prepare or reposition either arm before handoff; vla_handoff owns the complete bimanual approach and transfer from the post-grasp state.",
            "Before vla_left_place, confirm vla_handoff has ended and the left gripper holds the intended object. If the held bowl or plate was classified dirty, localize an interior placement point inside the open metal wire basket/frame. If it was classified clean, localize a white interior floor/cavity placement point at the cardboard-box center, still clearly inside the box. For the cup, localize the same cardboard-box placement area. For chopsticks or the spoon, localize the target cup after the cup is stable inside the cardboard box. Stage the left-held object with safe left-arm horizontal x/y move_delta only when staging is needed: use the projected target x/y, keep the current left TCP z with delta_z=0 by default, do not move to the projected z coordinate, and do not add a large vertical clearance. Call vla_left_place only if the latest staging move succeeded and reports a reached/effective target. If the staging move fails, is uncertain, or does not reach the target, do not call vla_left_place.",
            "The metal wire basket/frame is the target only for dirty bowls and dirty plates. The black-outside, white-inside cardboard box is the target only for clean bowls and clean plates. Do not count dirty objects in the cardboard box or clean objects in the metal basket as success.",
            "Advance to the next category only after the latest inline D455 image confirms that every object in the current category has reached its required destination, or that no such object remains available in the source workspace after a careful visual check.",
            "The D455 is a fixed external camera. Do not move either arm or held object merely to improve camera visibility. Arm motion is allowed only for task execution or safety: free-space approach/staging, VLA handoff/placement/grasp execution, or joint recovery.",
            "After every VLA or rule-based action, inspect joint_health in the returned snapshot. If either arm is warning or critical, especially right-arm q1 positive / q3 negative accumulation, call recover_joint_posture before continuing more VLA chunks; it records each gripper's open/closed state, re-commands that state before and after joint reset so closed grippers stay clamped, resets both arms' joints, then returns both TCPs near their pre-recovery right_base poses.",
            OPERATOR_FINISH_CONSTRAINT,
            "If depth is missing, projection is uncertain, the intended object is occluded, dirty/clean status is uncertain, or success is ambiguous, do not infer metric deltas from RGB alone; stop for operator feedback instead of opening a gripper or blindly retrying.",
        ),
    ),
    2: FrankaTask(
        vla_instruction=CLEAN_DESK_VLA_PROMPT,
        name="clean_desk_dirty_clean_sorting_two_cups_explore_candidate",
        instruction="Complete the dirty/clean tabletop sorting task with exactly two cups. "
        "Process bowls, plates, then both cups; finish each "
        "category before starting the next. Within a category, process green "
        "before blue when applicable. Classify bowls using chicken nuggets "
        "inside them, and plates and cups using an egg-tart foil tray marker: "
        "marked objects are dirty and go into "
        "the metal wire basket/frame; unmarked objects are clean and go into "
        "the black-outside, white-inside cardboard box. Handle both cups "
        "individually, not as one object. The right hand grasps each object, "
        "transfers it to the left hand, and the left hand places "
        "it at its classified destination. Only bowls, plates, and the two "
        "cups are task objects. There are no chopsticks or spoons to process.",
        setup="This sorting scene contains bowls, plates, and exactly "
        "two cups in total. Do not assume either cup's classification from its "
        "color or from the other cup. Chicken nuggets inside a bowl mark "
        "that bowl as dirty; the bowl marker is not an egg tart or foil tray. "
        "An egg-tart foil tray visibly inside a cup or on a plate's concave "
        "eating surface marks that specific cup or plate as dirty. Absence of the marker counts "
        "as clean only when that object's interior is sufficiently visible. "
        "The adjacent metal wire basket/frame receives dirty objects; the "
        "black-outside, white-inside cardboard box receives clean objects. "
        "The cardboard box is on the +y side of the metal basket in right_base. "
        "Use D455 as the primary observation and calibrated metric localization "
        "source. D455 is fixed; wrist cameras provide RGB evidence. All exposed "
        "TCP positions, projected points, and base-frame motion deltas use "
        "right_base/world. Named VLA skills retain the deployed clean-desk "
        "policy instruction. Choosing a destination in planner text does not "
        "retarget that policy: its demonstrated left-placement destination is "
        "the metal basket. For a clean object's cardboard-box destination, "
        "use a suitable recorded rule-based placement or plan direct-control "
        "placement from current perception; changing the VLA prompt is not "
        "a substitute for those actions.",
        success_criteria="All bowls, plates, and both cups have been individually classified "
        "and placed: every dirty object is inside the metal wire basket/frame "
        "and every clean object is inside the cardboard box's white interior. "
        "Both cups are accounted for separately at their correct destinations. "
        "No target object remains at source or in a gripper. "
        "Uncertain classification, a missing cup, or a wrong destination does "
        "not count as success. Final success requires operator confirmation.",
        constraints=(
            "Call describe_dual_franka_setup before acting.",
            "Call view_env_state before the first action only if no fresh primitive snapshot is already available. After any primitive that returns a snapshot, inspect that returned snapshot directly; never call view_env_state immediately afterward unless the primitive failed, returned no snapshot, an operator changed the scene, or a specific historical step is needed.",
            "Process bowls -> plates -> both cups. Both cups must be completed. Within a category, green precedes blue when applicable; dirty/clean status must not reorder grasping.",
            "Classify each bowl, plate, and cup from current evidence before choosing its placement destination. Chicken nuggets inside a bowl mark it dirty. An egg-tart foil tray inside a cup or on a plate's eating surface marks that cup or plate dirty. A marker nearby or in another object does not classify the chosen object. An occluded interior alone is not evidence of cleanliness.",
            "Choose available perception tools and camera views as useful. Inspect segmentation overlays and projected markers before using their coordinates; invalid depth is not a valid metric measurement.",
            "Treat right_base as the only world coordinate frame for agent reasoning. For both left and right arms, move_delta and rotate_delta use right_base/world deltas; do not convert left-arm coordinates yourself.",
            "Choose VLA skills, direct control, or a combination for grasping, handoff, placement, and contact-rich adjustments. Contact is not restricted to VLA, and direct-control actions are not restricted to free-space staging. Verify the actual grasp, transfer, and placement outcomes from returned images and state.",
            "Judge motion accuracy by what the next manipulation needs, not by eliminating every residual error. If the actual pose and visual evidence support continuing, proceed. Do not repeatedly micro-adjust or request feedback solely to satisfy a numerical tolerance. Choose whether and how much to correct from the current object, destination, clearance, and observed outcome; no fixed correction size or mandatory alignment is imposed. This does not waive actual controller faults or protective aborts, and action completion is not proof of task success.",
            "Before placement, verify that the left gripper holds the intended object. Choose its destination: dirty bowls, plates, and cups go to the metal basket; clean bowls, plates, and cups go to the cardboard box.",
            "Do not count dirty objects in the cardboard box or clean objects in the metal basket as success. Track the two cups separately by current visual identity and classification; placing one does not complete the cup category.",
            "Advance to the next category only after the latest inline D455 image confirms that every object in the current category has reached its required destination, or that no such object remains available in the source workspace after a careful visual check.",
            "D455 is fixed, but arm/object motion can improve visibility or change the background. Choose observation motions, direction, distance, and strategy from the scene; no small-motion requirement or fixed adjustment sequence is imposed.",
            OPERATOR_FINISH_CONSTRAINT,
            "Missing depth or uncertain localization, classification, or placement alone does not require stopping for operator feedback. Use available perception, RGB evidence, motion, and retries to resolve uncertainty autonomously. Request assistance for a concrete blocker requiring human intervention; stop for actual unsafe motion or a control anomaly. Final operator evaluation and scene-reset confirmation remain required.",
            "vla_right_grasp ends after right gripper closure plus lift; vla_handoff ends after right gripper opening plus the configured release delay; vla_left_place ends after left gripper opening plus lift. These boundaries only mean a skill segment ended, not that the intended object was successfully grasped, transferred, or placed. Verify the intended object and physical outcome from the returned images and gripper state before continuing; use projection evidence when needed.",
        ),
    ),
}


def get_dual_franka_task(task_id: int) -> FrankaTask:
    """Return a task by numeric ID with a clear error for unknown IDs."""
    try:
        return DUAL_FRANKA_TASKS[int(task_id)]
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"unknown dual-Franka task {task_id!r}; choices are "
            f"{sorted(DUAL_FRANKA_TASKS)}"
        ) from exc
