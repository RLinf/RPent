## Camera and projection rules

1. Read describe_dual_franka_setup or view_env_state for available_camera_views before choosing a camera name. The tool schema does not enumerate views because real deployments may register additional cameras in the robot config.
2. Use back_project only for pixels selected from the named camera image. Choose a pixel well inside visible material of the named target object or target container, away from silhouettes, rims, walls, wires, occluders, and background. Never project image-space air above an object.
3. When SAM3 is available, use segment on a registered localization camera to segment visible target objects or destination regions before manual pixel projection. Inspect the returned mask overlay yourself; trust the returned right_base point only when the mask and median marker cover the intended target.
4. SAM3 text prompts are phrase-sensitive. Prefer short color/object/relation phrases. If a text prompt returns a very low score, retry a shorter/rephrased prompt or point prompt rather than lowering min_score.
5. After every projection, inspect the returned annotated camera image. The marker center must be visibly inside the intended target material; selection_valid=true is necessary but not sufficient. Retry a more central interior pixel whenever the marker looks wrong.
6. Treat right_base as the only world coordinate frame for agent reasoning. After a verified projection, trust the returned right_base xyz and TCP-to-point deltas as the primary metric evidence; do not override them with unaided 2D RGB distance guesses.
7. Do not move an arm or held object merely to improve camera visibility. If visibility is insufficient, mark the state uncertain or stop for operator feedback instead of performing active-vision motions.

## VLA segment gates

1. Use VLA segment tools for contact-rich motion: vla_right_grasp, vla_handoff, and vla_left_place.
2. vla_right_grasp ends after right gripper closure plus lift; vla_handoff ends after right gripper opening plus a configured release delay; vla_left_place ends after left gripper opening plus lift. These boundaries only mean a segment ended; verify physical success from images, gripper widths/open flags, and projection geometry before continuing.
3. After a grasp VLA, do not decide lifted/held status from 2D appearance alone. Check right gripper state and the latest inline image. If held status is unclear, project the best visible held-object surface when possible and compare right_base z primarily with the source/table height or pre-grasp source projection.
4. If a verified projection of the intended object or held-object surface is higher than source/table height, treat the grasp as successful and proceed to vla_handoff. Mark failure only when verified projection shows the same object still at table height and gripper/image evidence shows the right gripper is empty or not supporting it.
5. After a confirmed right-hand grasp, call vla_handoff directly. Do not use move_delta, rotate_delta, open_gripper, close_gripper, or any other rule-based primitive to prepare or reposition either arm for handoff; the VLA owns the bimanual approach, relative pose adjustment, left-hand grasp, right-hand release, and collision-aware transfer.
6. Before vla_left_place, first verify that vla_handoff has ended and the left gripper holds the intended object. If the left hand is not clearly holding the object, do not stage for placement.
7. For placement staging, project a clearly valid target pixel inside the required destination, then use projected x/y only. Keep the current left TCP z with delta_z=0 by default; do not move to the projected z coordinate or add large vertical clearance. Let vla_left_place handle descent, release, and post-release lift.

## Registered tools

Read describe_dual_franka_setup for registered tools and cameras. Use move_delta,
rotate_delta, gripper tools, recover_joint_posture, back_project, optional segment,
and the named vla_right_grasp/vla_handoff/vla_left_place skills. Do not assume
LIBERO primitives or the older vla_grasp tool exist.
