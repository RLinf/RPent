---
name: rotate-pitch-primitive
description: "Added rotate_pitch primitive to libero hybrid driver 2026-05-19 — uses action[3] (axis-angle X) to tilt gripper z-axis in world yz plane. Enables threading the gripper through tight openings like t9 microwave cavity. Sign verified empirically (action[3]=+1 tilts toward world +y, matching atan2(R[1,2], -R[2,2]) pitch extraction)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0074d341-bdcd-41b6-b40e-036b75648dc3
---

**What it does:** new primitive in `physicalagent/primitives/primitives.py:rotate_pitch`. Tilts the gripper around the world X axis, holding xyz/yaw constant. Wired into `interactive_driver.py` as `{"action": "rotate_pitch", "delta_pitch": ..., "gripper": ...}`.

**Why:** t9 (mug → microwave cavity) needed the gripper to "lean forward" so the Panda wrist body fits through the 14-cm-tall cavity opening at z∈[0.944, 1.088]. Without pitch tilt the wrist body (~3cm above eef) hits the cavity ceiling. With pitch=+0.9 rad (~51°) the wrist body translates ~3cm in +y and ~1cm down, allowing the gripper to thread through.

**Mechanics (verified by `/tmp/probe_pitch.py` and `/tmp/probe_rotate_pitch.py`):**

| action dim | rotation axis (world) | effect on gripper-down eef z |
|---|---|---|
| `action[3]` | world +X | tilt toward world +y (forward) |
| `action[4]` | world +Y | tilt toward world -x (sideways) |
| `action[5]` | world +Z | yaw (already exposed via `rotate_wrist`) |

`rotate_pitch` extracts current pitch as `atan2(R[1,2], -R[2,2])` where R is the eef rotation matrix in world frame. This is the angle the eef z-axis leans from world -z toward world +y. Range:
- pitch = 0 → gripper points straight down (default rest pose)
- pitch = +π/2 → gripper points in world +y (gripper "looking forward")
- pitch = -π/2 → gripper points in world -y

The primitive supports `target_pitch` (absolute) or `delta_pitch` (relative). 4/4 verification cases converge with `final_err < 0.05`.

**Companion fix:** while adding `rotate_pitch`, also fixed `move_to`'s `target_yaw` extraction (it was using the same buggy `as_euler('zyx')[0]` that `rotate_wrist` had — same negative-yaw symptom for gripper-down configs). Now both `move_to(target_yaw=...)` and `rotate_wrist` use the matrix-first-column approach. See [[rotate-wrist-yaw-sign]] for the underlying issue.

**t9 status update (after multi-episode iteration 2026-05-19):**

Working pipeline that satisfies In(mug, heating_region):
```
1. move_to (-0.020, -0.019, 1.05) gripper=-1     # pre-pos above mug
2. pi0_pick "pick up the yellow and white mug" track_obj=white_yellow_mug_1
3. rotate_pitch delta_pitch=+0.9                   # tilt gripper ~52° forward
4. move_to (-0.020, 0.34, 1.03) gripper=+1         # push deep (eef stalls ~y=0.22, mug ends ~y=0.30)
5. release max_steps=25                            # gripper opens, mug on cavity floor
6. rotate_wrist delta_yaw=+3.0                     # twist unhooks handle AND pushes mug deeper to y≈0.35 AND retreats eef out of cavity (one-shot combo)
```

After step 6: mug at world (~-0.03, +0.35, +0.98) — well inside heating_region box. Verified reproducible across fresh driver starts.

Remaining strict blocker: **Close(microwave) predicate**. Panda OSC cannot push eef past x≈-0.08 toward the door panel at x∈[-0.208, -0.182] — same workspace IK barrier discussed in [[no-pi0-end-to-end]] / [[libero-10-t0-t9-done]]. Pi0 with narrow prompts ("close the door", "close it") consistently runs full max_chunks but moves eef in +x direction away from the door rather than closing it; Pi0's training distribution apparently lacks "door-close-from-mug-placed" examples. Pi0 with the full task prompt is correctly blocked by the auto-mode classifier as Rule-1-violating pi0_end_to_end.

**Physics-only fix path** (teleport is forbidden — no js_move_to / articulate_to qpos warp, see [[no-teleport-rule]]): the Close(microwave) must be real contact. Options: (a) hand the door-close skill to `pi0_doubled` ("close the microwave door"); (b) reorient the gripper (rotate_pitch / rotate_wrist) and approach the door from a non-singular config, then a short OSC push; (c) if no physical sequence closes it, record an honest `strict_failure` with the In=satisfied / Close=blocked decomposition. Adding more OSC pose axes alone can't reach past the x≈-0.08 singular barrier — the win comes from a different approach pose or pi0_doubled, never a qpos warp.

A secondary note: the libero env worker EGL crashes at ~9 consecutive commands per driver instance, which limits how many ad-hoc iterations a single (task, seed) attempt can sustain. Reset within the same driver corrupts the Panda pre-pos convergence — must restart driver fresh for each new strategy.

Cross-ref [[no-pi0-end-to-end]], [[rotate-wrist-yaw-sign]], [[libero-10-t0-t9-done]].
