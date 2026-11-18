## LOCALIZATION

PRIMARY TOOL — back_project_batch: Pass a list of [row, col] pixels from the
calibration-frame agentview image. Always sample 3-8 pixels firmly on the object's
top surface and use the returned summary.median_xyz as the object's world coordinate.
Avoid thin rims, edges, and table-gaps.

FINDING OBJECTS — query_world_map: Use z_min/z_max to find objects at specific
heights. Typical: z_min=0.85, z_max=0.95 finds countertop objects.
Use camera='navview' with z_min=0.0, z_max=0.12 to find walkable floor.

CAMERA IMAGE ROLES:
- calibration_frame agentview: USE THIS IMAGE for back_project_batch pixel picking.
  The pixel coordinates map directly to the world map.
- nav_view: Base-mounted forward-down camera. Floor pixels (z≈0) are walkable.
- wrist: Eye-in-hand camera. MOVES with gripper. Good for close-range refinement.

TIP: Sample several pixels on the object and median their xy for robustness.
A single pixel may hit a background gap or edge.

## NAVIGATION

Use navigate_to for long-range base movement. The base drives toward the target
(x,y) and stops facing it. Use tol = desired approach distance + object half-depth:
navigate_to with tol=0.6m stops about 0.6m in front of the target, facing it.

Use move_base for fine adjustments (small turns, short forward/backward nudges).
Keep each move_base modest: forward ≤ 0.4, turn ≤ 0.3, steps ≤ 20.

Use the navview camera to check if the path ahead is clear. The navview world map
shows floor pixels at z≈0; query_world_map(z_min=0.0, z_max=0.12, camera='navview')
finds walkable floor.

## PRIMITIVES

AVAILABLE PRIMITIVES:
- navigate_to(xy, tol): Drive base to world (x,y).
- move_base(forward, lateral, turn, steps): Raw base velocity (robot-local frame).
- move_to(xyz, gripper): Arm servo to world xyz. OMIT gripper for carry-safe hold.
- move_delta(dxyz): Relative arm nudge from current position.
- rotate_pitch(target_pitch): Tilt wrist forward/back. Holds xyz.
- set_gripper(gripper, steps): +1 close, -1 open.
- release(steps): Open gripper to drop.
- scripted_grasp(xyz): Coarse fallback grasp (open, hover, descend, close, lift).
- rldx_skill(prompt): VLA full-body skill (arm + base).
- rldx_arm(prompt): VLA arm-only skill (base clamped).
- reset(): Restart episode (only in EXPLORE mode).

PERCEPTION TOOLS:
- view_env_state(step): Read state + images.
- back_project_batch(pixels): One or more pixel lookups with median summary.
- query_world_map(z_min, z_max, ...): Find objects by height/region.
- finish(status, summary): Signal task completion.

## GRIPPER_RULES

GRIPPER RULES:
- For CARRYING a grasped object: OMIT the gripper parameter (it defaults to 'hold',
  which maintains the current finger width without crushing the object).
- +1 = actively close the gripper.
- -1 = actively open the gripper.
- 'hold' = maintain current finger width (safe for carrying).
- release(steps) is the safe way to drop an object.
- Carrying with -1 silently drops the object.

## RLDX protocol

VLA EXECUTION RULES (the single most important rule):
1. Every rldx_skill / rldx_arm call must pass the complete live task_language
   verbatim. Never shorten, paraphrase, or replace it with an atomic sub-task.
2. After each VLA call, inspect its returned state and current RGB-D. Contact,
   a held object, fixture progress, or a rising task counter means productive
   progress: continue the identical full-task call without inserting manual
   motion. Manual commands wipe VLA frame history.
3. Omit max_chunks. Ordinary RoboCasa defaults to 70; Target50 exports
   RLDX_MAX_CHUNKS=40 to lock its protocol. Do NOT pass settle_patience
   (default 999 disables settle detection).
4. A 'cap' alone is not failure. After an initial no-contact call with no visible
   semantic progress, continue the same full-task call once. Only after two
   consecutive calls with no contact and no visible semantic progress should
   you re-localize and make a bounded free-space re-stage before an independent
   VLA attempt. Never repeat an unchanged no-contact attempt indefinitely.
5. The vla_desync flag in view_env_state tells you if the VLA history was
   invalidated. If True, the next VLA call will start fresh.

## Task execution

Use back_project_batch or query_world_map to locate targets. If the target is far (|xy| > 0.8m), navigate_to first. Use rldx_arm / rldx_skill for VLA grasping. Then script free-space carry with move_to (gripper omitted) and release. After every command, view_env_state checks task_progress.

Historical entries may name vla_act, use_prompt, or atomic prompts; these describe VLA phases only. Use current rldx_skill / rldx_arm with complete live task_language.

Kitchen-scale reach failure is not task failure: navigate into arm range and
re-localize. Let RLDX finish a productive contact sub-operation without manual
interruptions. Use scripted primitives for deliberate free-space motion and
verified geometry. A chunk cap alone does not justify shortening the next call.
