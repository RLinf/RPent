# RPent LIBERO Global Memory

This is the single index for the complete Global Memory. Read the core rules first, then open the reusable manipulation pattern and the 2-3 related memories that best match the current scene. Apply the calibration block below only when its scene gate matches the current observations. Every link below points directly to a leaf file.

## Low-table grocery-scene calibration

**Applicability gate:** Use this calibration only when the initial robot EEF height is approximately `0.26 m`, the visible table surface is near world `z = 0`, and the scene contains a low-table grocery basket layout. Otherwise, use the calibration that matches the current scene. Re-localize the basket and every target object from the latest images before applying any reference height.

| Object profile | `pre_pos_z` | `carry_z` | Release guidance | Held-object offset guidance |
|---|---:|---:|---|---|
| Flat box or can | `0.13` | `0.22` | Descend to `release_z=0.16` | Measure from the current images; do not assume an offset. |
| Tall bottle, including salad dressing, ketchup, milk, and BBQ sauce | `0.16` | `0.30` | Release at the carry height; do not descend inside the basket footprint. | A matching grasp often leaves the bottle about `0.03 m` behind the EEF in y, so an initial target may place the EEF about `+0.03 m` past the basket center. Confirm and refine visually. |
| Orange-juice bottle exception | `0.16` | `0.30` | Release at the carry height; do not descend inside the basket footprint. | The grasp is usually near-centered. Start at the basket center instead of adding the tall-bottle `+0.03 m` y compensation, then verify visually. |

- For slippery tall bottles, `pi0_pick(..., lift_thresh=0.08)` allows the grasp action to complete a larger post-descent ascent. Treat its return value as provisional and confirm the actual hold from the latest wrist and agent images.
- After a verified grasp, use `set_gripper(gripper=+1, steps=8)` between transport stages to maintain the hold, especially for cylindrical cans.
- If a tall bottle bottom contacts the basket rim, continued OSC descent can stall and displace the basket. Keep the bottle high and release from `carry_z=0.30` instead of pushing down into the basket.
- Related guidance: [[feedback_read_image_before_decide]], [[feedback_pi0_false_positive_lift]], [[feedback_pi0_pick_full_prompt]], and [[feedback_staged_held_object_transport]].

## Core operating rules

- [no-pi0-end-to-end](feedback_no_pi0_end_to_end.md) - STRICT_HYBRID_GUIDE Rule 1 forbids pi0_end_to_end — Pi0 only does the gripper grasp; LLM must do all motion planning and release. Recovery must remain inside the current episode.
- [feedback-read-image-before-decide](feedback_read_image_before_decide.md) - Inspect the latest view_driver_state images before every non-trivial decision; states.json alone turns the agent into a control tuner instead of a spatial reasoner.
- [failure-forensics-render-images](feedback_failure_forensics.md) - After an action sequence fails, inspect the relevant view_driver_state images and compare what happened with what was expected; do not tune numbers blindly.
- [pi0-chunks-egl-crash](feedback_pi0_chunks_egl_crash.md) - In the libero hybrid REPL driver, Pi0 invocations with max_chunks >= ~50 reliably crash the libero env subprocess with EOFError / EGL_NOT_INITIALIZED; keep pi0_pick calls <= ~25 chunks per invocation.
- [move-pose-covarying-reach](feedback_move_pose_covarying_reach.md) - move_pose primitive (co-varying xyz+pitch+yaw) threads the cabinet-front singularity that decoupled move_to walls at; reach-blocked drawer cells are reachable
- [worker-reads-memory-snapshot](feedback_worker_reads_memory_snapshot.md) - RPent workers read the reviewed in-repository memory snapshot at resources/libero/memory; update it deliberately before evaluation.
- [region-ranges-table-frame](feedback_region_ranges_table_frame.md) - Prefer direct visual localization; when a destination is fixture-relative, transform only geometry estimated from current images through the observed fixture pose.

## General feedback and control knowledge

- [bowl-eef-y-offset](feedback_bowl_eef_y_offset.md) - A rim-hook bowl grasp can leave the bowl center behind the EEF; estimate the current held-object offset and compensate relative to the visually localized target.
- [completion-sensitive-multi-object-ordering](feedback_completion_sensitive_multi_object_ordering.md) - For similar objects sharing one target, choose order before execution so partial completion does not hide the remaining object, then keep later carries above the occupied target.
- [cook-region-offset](feedback_cook_region_offset.md) - Visually localize the usable burner surface instead of aiming at the center of the whole stove platform or a remembered coordinate.
- [gripper-ctrl-is-finger-position](feedback_gripper_ctrl_is_finger_position.md) - RPent high-level primitives use gripper=+1 to close or hold and gripper=-1 to open; verify the grasp with supported gripper-opening telemetry and current images.
- [handle-bar-grasp-orientation](feedback_handle_bar_grasp_orientation.md) - Roll the gripper so its pads close across the short axis of a horizontal handle, then verify the grasp from images and gripper opening.
- [rotate-pitch-orientation-control](feedback_rotate_pitch_orientation_control.md) - Use rotate_pitch to tilt the gripper around world X for low-clearance insertion, then co-vary position and pitch with move_pose through the opening.
- [liberopro-driver-patch](feedback_liberopro_driver_patch.md) - RPent LIBERO environment setup must resolve get_benchmark through the LIBERO_TYPE-aware RLinF benchmark module, not base libero
- [feedback-long-horizon-cell-timeout-1200](feedback_long_horizon_cell_timeout_1200.md) - Complex LIBERO cells need CELL_TIMEOUT_S=1200 so long recovery sequences can finish and write their audit
- [max-episode-steps-libero](feedback_max_episode_steps_libero.md) - LiberoEnv inherits robosuite's per-episode step counter; when the --max-episode-steps cap is hit (e.g. 600), the env enters terminated state and any further env.step throws 'executing action in terminated episode' ValueError that kills the worker. For long hybrid sessions use 5000+.
- [feedback_osc_push_mujoco_nan](feedback_osc_push_mujoco_nan.md) - Long OSC push (high max_steps through sustained contact) can trigger MuJoCo NaN at QACC DOF9; close drawers with short capped pushes or pi0_doubled, never one long push
- [pi0-delivery-service](feedback_pi0_delivery_service.md) - LLM is delivery for Pi0, not replacement. Walk full prompt-escalation ladder before LLM-scripted pick. One sub-instr failure is NOT enough evidence to bail.
- [pi0-false-positive-lift](feedback_pi0_false_positive_lift.md) - Treat pi0_pick success as provisional: EEF ascent plus gripper closure can still be an empty grasp, so confirm with supported diagnostics and current wrist/agentview images.
- [pi0-pick-full-prompt](feedback_pi0_pick_full_prompt.md) - For elevated picks under LIBERO-Pro (stove z≈0.93, cabinet-top z≈1.13, drawer interior), use the full active task language as pi0_pick.prompt and verify the grasp visually.
- [pi0-pre-pos-can-hurt](feedback_pi0_pre_pos_can_hurt.md) - Choose Pi0's start pose from current visibility: home pose for isolated familiar objects, low object-relative pre-positioning only for clutter-driven grounding ambiguity.
- [render-skip-env-step](feedback_render_skip_in_env_step.md) - Historical RLinF render-skip patch avoided EGL accumulation by disabling camera rendering during OSC primitives. Current RPent retains a cached-frame fallback in robots/libero/env_server.py; revalidate render toggling before porting the old optimization.
- [rotate-wrist-yaw-sign](feedback_rotate_wrist_yaw_sign.md) - `robots/libero/tools.py:rotate_wrist` once rotated in the OPPOSITE direction from commanded because `as_euler('zyx')[0]` returns the negative of world yaw for gripper-down configs (R[2,2]≈-1). Fixed by using `atan2(R[1,0], R[0,0])` directly. For pose-control yaw extraction in libero/robosuite, NEVER use scipy euler decomposition when gripper points down — go through the rotation matrix's first column.
- [sampled-fixture-pose-relocalization](feedback_sampled_fixture_pose_relocalization.md) - Re-localize moved or rotated fixtures from current images and depth before targeting their visible compartments, cavities, or surfaces.
- [scripted-pick-limits](feedback_scripted_pick_limits.md) - Fully scripted descent plus gripper close is geometrically unreliable in LIBERO for small and medium objects because the fingers miss by millimeters
- [single-target-prompt-grounding-limits](feedback_single_target_prompt_grounding_limits.md) - Similar motion under different prompts in a single-target scene does not prove language grounding; keep exact prompts and verify the selected object in clutter.
- [staged-held-object-transport](feedback_staged_held_object_transport.md) - Carry a verified held object with lift, horizontal transport, and descent stages while compensating the observed object-to-EEF offset.
- [stove-turnoff-strict](feedback_stove_turnoff_strict.md) - Stove on/off tasks require a real, visible knob rotation; verify the physical change from images and the official task signal.
- [swap-perturbs-fixtures](feedback_swap_perturbs_fixtures.md) - Layout changes may relocate whole fixtures; identify the requested fixture and localize its placement surface from the current images.
- [thin-handle-drawer-closure](feedback_thin_handle_drawer_closure.md) - Close a shallow slide drawer by verifying object fit, maintaining low front-panel contact, and using short capped motions instead of one sustained push.

## Reusable manipulation patterns

- [Can and small-package basket placement](can_and_small_package_basket_placement.md) - Reusable shared-basket recipe for a rigid can and a small package when a single closed-loop instruction outperforms manual placement.
- [Closed-loop cup back-compartment placement](closed_loop_cup_back_compartment_placement.md) - Reusable closed-loop recipe for carrying a cup into a rear caddy cavity when manual placement is rim-limited.
- [Dual-moka obstacle-aware stove placement](dual_moka_obstacle_aware_stove_placement.md) - Reusable dual-vessel stove recipe using cap-grip, tilt-extend placement, and a front-arc carry around the first placed pot.
- [Dual-object anchor and relative placement](dual_object_anchor_and_relative_placement.md) - Reusable plate-centered recipe combining a mug-rim grasp, a box grasp, and relative-side placement.
- [Fragile-box and rigid-can basket placement](fragile_box_and_rigid_can_basket_placement.md) - Reusable mixed-rigidity basket recipe for a fragile flat box and a rigid can in a cluttered scene.
- [Horizontal bottle drawer insertion and close](horizontal_bottle_drawer_insertion_and_close.md) - Reusable narrow-drawer recipe that rotates a long bottle horizontally, inserts with a tilted pose, and finishes with a hooked close.
- [Identity-conditioned dual-mug tilted placement](identity_conditioned_dual_mug_tilted_placement.md) - Reusable identity-aware two-mug recipe using tilted placement to reach opposing lateral plates around a distractor.
- [Mixed can-box basket recovery](mixed_can_box_basket_recovery.md) - Reusable mixed-shape basket recipe with can-delivery recovery and tilt-descend placement for a flat box.
- [Mixed-control dual-mug plate assignment](mixed_control_dual_mug_plate_assignment.md) - Reusable identity-aware two-mug placement pattern using manual control for one plate and closed-loop delivery for the harder side.
- [Moka placement with deferred stove activation](moka_placement_with_deferred_stove_activation.md) - Reusable stove recipe that places a fragile moka pot before activating the knob to avoid orientation drift.
- [Occluded thin-box basket placement](occluded_thin_box_basket_placement.md) - Reusable shared-basket recipe for two thin packages that begin partially occluded by distractors.
- [Ordered dual-can basket placement](ordered_dual_can_basket_placement.md) - Reusable shared-basket recipe for two rigid cans with obstacle-aware carry and object-specific delivery behavior.
- [Relative-side-first anchor placement](relative_side_first_anchor_placement.md) - Reusable two-object recipe that places the tight relative-side target before the anchor-surface object.
- [Single-moka burner-contact placement](single_moka_burner_contact_placement.md) - Reusable single-vessel stove recipe that requires base-to-burner contact rather than visual overlap alone.
- [Single-rollout drawer insert and close](single_rollout_drawer_insert_and_close.md) - Reusable articulated-container pattern that keeps open, insert, and close inside one continuous contact rollout.
- [Stove activation then handle-grasped pan placement](stove_activation_then_handle_grasped_pan_placement.md) - Reusable stove recipe that activates the knob before carrying a stable handle-grasped pan to the burner.
- [Tall-item-first basket placement](tall_item_first_basket_placement.md) - Reusable shared-basket ordering rule that places a tall unstable bottle before a small box.
- [Upright book back-compartment placement](upright_book_back_compartment_placement.md) - Reusable upright-book recipe that uses held-object overhang to place into a rear caddy compartment despite an OSC descent wall.

## Memory guidance

- [RPent LIBERO global memory](README.md) - Global Memory entry.
