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

# Deployment task registry note:
# tasks 1 and 3 below intentionally mirror the old PhysicalAgent clean-desk
# reproduction prompts rather than trying to be generic dual-Franka examples.
# They encode the current table layout, object order, D455-centric evidence
# policy, and named VLA segment boundaries so the new RPent runner can be
# compared against previous live logs.  Before an upstream/general PR, keep
# these as optional demo tasks and avoid treating their object/container rules
# as part of the robot-wide prompt or tool contract.
#
# Keep this stable with the instruction used by the deployed clean-desk
# checkpoint. Named VLA tools use it directly instead of allowing the planner
# to accidentally paraphrase the policy conditioning text.
CLEAN_DESK_VLA_PROMPT = (
    "I am currently performing a desk organizing task. I will use my right hand "
    "to hand bowls, plates, cups, chopsticks, and a spoon to my left hand in "
    "order, and then put them into a basket."
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
        name="clean_desk_dual_franka_agent_vla",
        instruction=(
            "Complete a dual-arm collaborative tabletop object-storage task. "
            "The right hand is responsible for grasping, and the left hand is "
            "responsible for placing. After the right hand grasps an object, it "
            "must hand the object over to the left hand by calling vla_handoff "
            "directly from the confirmed post-grasp state, without rule-based "
            "pose preparation or repositioning of either arm. Rule-based primitives "
            "may only stage the grippers through clearly separated free space; "
            "physical bimanual interaction, handoff contact, grasp closure, "
            "and placement contact must be handled by named VLA skill tools. "
            "Use the right hand to process object categories in this strict "
            "order: bowls, plates, cup, chopsticks, then spoon. Finish the "
            "entire current category before targeting anything from the next "
            "category. Within any category that has multiple colored source "
            "objects, process the green object before the blue object; do not "
            "target blue while green from the same category remains in the "
            "source workspace. A category is finished only when the latest visual "
            "evidence confirms that all of its objects have reached their "
            "required destination, or that no object from that category "
            "remains available in the source workspace after a careful visual "
            "check. Operate on only one object at a time. After the right hand successfully "
            "grasps an object, hand it over to the left hand; the handoff is "
            "completed by the VLA. After the left hand has stably received the "
            "object, place it according to these rules: put the bowls, plates, "
            "and cup inside the black-outside, white-inside cardboard box; insert the two chopsticks and the spoon "
            "into the cup; before inserting the chopsticks and spoon, ensure "
            "the cup is already stably inside the cardboard box. Avoid dropping, "
            "colliding with, or tipping over objects. Continue to the next "
            "object only after confirming that the right-hand grasp, the "
            "bimanual handoff, and the left-hand placement have all succeeded."
        ),
        setup=(
            "The base camera and fourth-view D455 are RealSense RGBD cameras, "
            "but this clean-desk agent task exposes only the D455 "
            "back-projection tool. D455 aligned depth can be back-projected "
            "into the right_base frame using the calibrated extrinsics. "
            "The D455 is a fixed external camera; moving either arm or a held "
            "object does not move the camera viewpoint and must not be used as "
            "an active-vision strategy. "
            "right_base is the shared world coordinate frame exposed to the agent: "
            "camera points, left/right TCP positions, and rule-based base-frame "
            "deltas are all represented in right_base. The wrist Lumos "
            "cameras currently provide RGB only to the agent; do not rely on "
            "wrist depth. On the right side of the table there are two adjacent "
            "container-like regions: a metal wire basket/frame region and a "
            "black-outside, white-inside cardboard box. They are visually close, "
            "but the metal wire basket/frame is not the placement target; only "
            "the white cardboard-box interior is the target. In right_base/world "
            "coordinates, the cardboard box is on the +y side of the metal wire "
            "basket/frame. Use the contrast between the black outer surface and "
            "white inner surface as the visual boundary between inside and "
            "outside of the box. "
            "The VLA checkpoint is the deployment-aligned "
            "dual-arm tcp-rot6d clean-desk policy."
        ),
        success_criteria=(
            "All target objects are inside the white interior of the black-outside, "
            "white-inside cardboard box, and the spoon and chopsticks are inside "
            "the cup. The final cardboard-box state should "
            "visually be like that"
            "the plates are stacked together and standing on their sides in "
            "the cardboard box; the cups are stacked or nested together; the bowls are "
            "stacked or nested together; the chopsticks and spoon are "
            "inserted into the cup. The cardboard box and all objects should remain "
            "stable without tipping. The agent should make purposeful progress "
            "without blind centimeter-scale search. Rule-based motion should "
            "only stage arms with clear free-space margins; VLA should handle "
            "grasp/place/handoff/contact-rich segments. Finish only after "
            "visual feedback or operator verdict supports success or safe "
            "stopping. An object in the metal wire basket/frame region, touching "
            "the metal wires, on the table near the containers, or ambiguous "
            "between the metal frame and cardboard box does not count as "
            "inside the cardboard box."
        ),
        constraints=(
            "Call describe_dual_franka_setup before acting.",
            "Call view_env_state before the first action only if no fresh primitive snapshot is already available. After any primitive that returns a snapshot, inspect that returned snapshot directly; never call view_env_state immediately afterward unless the primitive failed, returned no snapshot, an operator changed the scene, or a specific historical step is needed.",
            "Enforce this category gate without exception: bowls -> plates -> cup -> chopsticks -> spoon. Always act on the earliest unfinished category. While any object from that category remains in the source workspace, do not localize, approach, grasp, or call a VLA grasp skill for any later category. Within a category that has both green and blue source objects, complete the green object before localizing, approaching, grasping, or calling a VLA grasp skill for the blue object.",
            "Container disambiguation is mandatory: the right-side metal wire basket/frame and the black-outside, white-inside cardboard box are adjacent, but only the white interior of the cardboard box is the valid placement target. Never place target objects into the metal wire basket/frame. Never count objects in the metal wire basket/frame, touching the metal wires, on the table near the containers, on the black exterior surface, or ambiguous between the two regions as inside_the_cardboard_box; mark them uncertain or failed.",
            "For bowls, plates, and the cup, choose a placement target on the white interior floor/cavity inside the cardboard box in right_base/world coordinates, about 0.5cm in the +y direction from the box center, while still clearly inside the white interior and away from the adjacent metal wire basket/frame, rim, and box walls. If the box center is uncertain, choose the visible white interior point that best matches this approximate +y offset without moving onto an edge. Use the black exterior vs white interior contrast as the inside/outside boundary cue; the selected placement pixel must be clearly on the white interior, not on the black exterior, rim, table, metal frame, or outside of the box. For chopsticks or the spoon, choose the target cup instead.",
            "The D455 is a fixed external camera. Do not move an arm, raise an arm, lift a held object, or change the scene merely to improve camera visibility. Arm motion is allowed only for task execution or safety: free-space approach/staging, VLA handoff/placement/grasp execution, or joint recovery. If D455 visibility is insufficient, mark the state uncertain or request operator feedback; do not perform active-vision motions.",
            "Advance to the next category only after the latest inline D455 image confirms that every object in the current category has reached its required destination, or that no such object remains available in the source workspace. Bowls, plates, and the cup must be inside the cardboard box's white interior; chopsticks and the spoon must be inside the cup, with the cup already stable inside the cardboard box.",
            "Before acting on a category, build a brief localization table in your reasoning from the latest inline D455 image. For each visible candidate in the current category, list its D455 evidence, the chosen interior pixel if projected, the resulting right_base xyz, and status: source_workspace, inside_cardboard_box, in_cup, or uncertain. Keep this table updated after each primitive result. Do not switch objects or categories without reconciling the table against the latest returned snapshot.",
            "When SAM3 is available, use segment with camera='d455' on the D455 image to segment the current object or placement target before falling back to a manual pixel. SAM3 text grounding is phrase-sensitive: prefer short color/object/relation phrases, and for the clean-desk box use 'white interior of the black cardboard box' or 'cardboard box' rather than over-specific wording like 'white interior floor'. If a text prompt returns a very low score, retry a shorter/rephrased prompt or point prompt rather than lowering min_score blindly. If SAM3 is unavailable or the mask overlay is wrong, use back_project with camera='d455' only for D455 pixels instead of guessing right_base/world directions from RGB alone. Choose a concrete target name and a pixel well inside visible target material, away from its silhouette. Never select image-space air/background above the object.",
            "After every SAM3 segmentation or D455 projection, inspect its returned overlay/annotated D455 image yourself and verify that the mask or marker center is visibly inside the intended target material, not on the table, container wall, rim, background, occluder, shared boundary, or a different object. selection_valid=true is necessary but not sufficient; retry with a point prompt, more specific text prompt, or a new interior pixel whenever the returned diagnostic image looks wrong.",
            "When exact spatial relation is uncertain, use segment or back_project again instead of judging metric x/y/z offsets by 2D RGB appearance. After a segmentation/projection is verified, trust the returned right_base xyz and TCP-to-point deltas as the main metric evidence for rule-based correction moves.",
            "Treat right_base as the only world coordinate frame for agent reasoning. For both left and right arms, move_delta and rotate_delta use right_base/world deltas; do not convert left-arm coordinates yourself.",
            "Rule-based move_delta, rotate_delta, open_gripper, and close_gripper commands must specify exactly one arm: left or right. Use rule-based primitives only for free-space approach or placement staging with clear margins; do not use them to prepare a handoff pose.",
            "Use VLA segment tools for contact-rich motion: vla_right_grasp for the right-arm grasp segment, vla_handoff for bimanual transfer, and vla_left_place for left-arm placement.",
            "After a grasp VLA, do not judge lifted/held status from 2D image appearance alone. Check the right gripper state and latest inline D455 image. If it is unclear whether the intended object has lifted or remains on the table, call back_project with camera='d455' on the best visible held-object surface when possible. If only a source/table candidate is visible, treat that projection as a source-candidate check, not proof of failure. Compare returned right_base z primarily with the source/table height or pre-grasp source projection. For wide objects such as plates, projected center/rim x/y can be far from the right TCP while still held; do not mark failure from TCP x/y distance alone. If a verified projection of the intended object or held-object surface is higher than the source/table height or pre-grasp source projection, treat the grasp as successful and proceed to vla_handoff. Mark failure only when a verified projection shows the same intended object still at source/table height and gripper/image evidence shows the right gripper is empty or not supporting it. If the object is occluded or identity/source is ambiguous, mark the grasp status uncertain and request operator feedback instead of opening or retrying.",
            "After the latest returned snapshot confirms that the right hand has successfully grasped the intended object, call vla_handoff directly. Do not call move_delta, rotate_delta, open_gripper, close_gripper, or any other rule-based primitive to prepare or reposition either arm before handoff; vla_handoff owns the complete bimanual approach and transfer from the post-grasp state.",
            "Before calling vla_left_place, enforce this placement gate: first confirm vla_handoff has ended and the left gripper is holding the object; then use the latest inline D455 image to localize the required placement target; prefer segment with camera='d455' for the target region when SAM3 is available, otherwise call back_project with camera='d455' on an interior pixel of that placement target; use only safe left-arm free-space move_delta commands to horizontally align the left-held object over the projected right_base target. Use the projected target x/y only; keep the current left TCP z with delta_z=0 by default, do not move to the projected z coordinate, and do not add a large vertical clearance. Only use a tiny safety lift if the current carried object is visibly below the rim or at collision risk. Call vla_left_place only if the latest staging move succeeded and reports a reached/effective target.",
            "Named VLA skills may end early through semantic boundary rules: grasp ends after right gripper closure plus lift, handoff ends after the right gripper opens and the configured release delay elapses, and place ends after the left gripper opens and then lifts about 10cm. This only means the current skill segment ended; inspect images and grippers before judging whether it succeeded.",
            "When calling a named grasp, handoff, or placement VLA skill, usually leave max_chunks unset so the server gives the skill enough budget and stops it at the semantic boundary. Override max_chunks only for a cautious diagnostic run.",
            "After every VLA or rule-based action, inspect joint_health in the returned snapshot. If either arm is warning or critical, especially right-arm q1 positive / q3 negative accumulation, call recover_joint_posture before continuing more VLA chunks; it records each gripper's open/closed state, re-commands that state before and after joint reset so closed grippers stay clamped, resets both arms' joints, then returns both TCPs near their pre-recovery right_base poses.",
            "If depth is missing, projection is uncertain, the intended object is occluded, or success is ambiguous, do not infer metric deltas from RGB alone; stop for operator feedback instead of opening a gripper or blindly retrying.",
        ),
    ),
    3: FrankaTask(
        name="clean_desk_dirty_clean_sorting_agent_vla",
        instruction=(
            "Complete a dual-arm collaborative tabletop object-storage task "
            "using the same object category order as task 1: bowls, plates, "
            "cup, chopsticks, then spoon. Finish the entire current category "
            "before targeting anything from the next category. Within any "
            "category that has multiple colored source objects, use the same "
            "color order as task 1: process green before blue. Do not let "
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
            "using the same category order and same within-category color "
            "order as task 1. "
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
            "Enforce the same grasp order as task 1 without exception: first category order bowls -> plates -> cup -> chopsticks -> spoon, then within any category that has multiple colored source objects, green before blue. Always act on the earliest unfinished category. While any object from that category remains in the source workspace, do not localize, approach, grasp, or call a VLA grasp skill for any later category. Within the current category, do not localize, approach, grasp, or call a VLA grasp skill for a blue object while a green object from the same category remains in the source workspace.",
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
            "Before vla_left_place, confirm vla_handoff has ended and the left gripper holds the intended object. If the held bowl or plate was classified dirty, localize an interior placement point inside the open metal wire basket/frame. If it was classified clean, localize a white interior floor/cavity placement point at the cardboard-box center, still clearly inside the box. For the cup, localize the same cardboard-box placement area. For chopsticks or the spoon, localize the target cup after the cup is stable inside the cardboard box. Stage the left-held object with safe left-arm horizontal x/y move_delta only: use the projected target x/y, keep the current left TCP z with delta_z=0 by default, do not move to the projected z coordinate, and do not add a large vertical clearance. Call vla_left_place only if the latest staging move succeeded and reports a reached/effective target.",
            "The metal wire basket/frame is the target only for dirty bowls and dirty plates. The black-outside, white-inside cardboard box is the target only for clean bowls and clean plates. Do not count dirty objects in the cardboard box or clean objects in the metal basket as success.",
            "Advance to the next category only after the latest inline D455 image confirms that every object in the current category has reached its required destination, or that no such object remains available in the source workspace after a careful visual check.",
            "The D455 is a fixed external camera. Do not move either arm or held object merely to improve camera visibility. Arm motion is allowed only for task execution or safety: free-space approach/staging, VLA handoff/placement/grasp execution, or joint recovery.",
            "After every VLA or rule-based action, inspect joint_health in the returned snapshot. If either arm is warning or critical, especially right-arm q1 positive / q3 negative accumulation, call recover_joint_posture before continuing more VLA chunks; it records each gripper's open/closed state, re-commands that state before and after joint reset so closed grippers stay clamped, resets both arms' joints, then returns both TCPs near their pre-recovery right_base poses.",
            "If depth is missing, projection is uncertain, the intended object is occluded, dirty/clean status is uncertain, or success is ambiguous, do not infer metric deltas from RGB alone; stop for operator feedback instead of opening a gripper or blindly retrying.",
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
