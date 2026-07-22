---
name: move-pose-covarying-reach
description: move_pose primitive (co-varying xyz+pitch+yaw) threads the cabinet-front singularity that decoupled move_to walls at; reach-blocked drawer cells are reachable
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 173e389d-6fb4-4726-99ec-b7c6d73b9da0
---

The "Panda OSC can't reach the cabinet-front-low pose" verdict (microwave t9, wooden_cabinet
bottom/middle handles) is an **orientation-coupled singularity of the decoupled `move_to` servo**,
NOT a kinematic limit. `move_to` holds gripper-down and servos straight, driving the wrist into a
singularity that walls the eef at z≈1.04 / y≈-0.03 reaching forward — yet the same servo reaches
z=0.92 at y=+0.12, and pi0 (which co-varies orientation) reaches even farther-forward low handles.

**Why:** a parallel-jaw reach to a forward-low cabinet face needs the wrist to TILT during the reach
(like pi0's learned trajectory). Holding a fixed gripper-down orientation makes the Jacobian
rank-deficient there → the position servo stalls.

**How to apply:** use the new **`move_pose`** primitive (added 2026-05-27 to
`robots/libero/tools.py`; OSC control). It servos `action[:3]` (xyz) AND
`action[3]`(pitch)/`action[5]`(yaw) SIMULTANEOUSLY
each env.step toward the `xyz` target plus `target_pitch` and `target_yaw`.
Call the current structured tool directly:
`move_pose({"xyz":[x,y,z],"target_pitch":-0.7,"target_yaw":null,"gripper":-1,
"step_clip":0.016,"pitch_step":0.05,"tol":0.012,"max_steps":180})`.

Recipe that reached the goal-t0 wooden_cabinet **bottom handle** (z=0.946, which move_to walls ~8cm
above): warm-up `move_to (0.04,0.12,0.92) → (0.045,-0.03,0.93)` (gets the arm into a low config),
THEN `move_pose(≈(0.05,-0.135,0.95), pitch -0.7..-1.0)` approaching from the RIGHT. Reached eef
(0.048,-0.123,0.956); pi0 pre-positioned there got the pinch-point to z=0.944 = the bar.

⚠ **Caveats:** (1) config-sensitive — the SAME target converges to z0.95 or stalls at z1.03 depending
on the warm-up/approach; approach from the right + warm-up low first. (2) Target only REACHABLE points
with modest `max_steps`; targeting past the boundary makes it sustain-push → QACC NaN worker crash
([[feedback_osc_push_mujoco_nan]]). (3) For closing on an object, set `tol:0.0` so it servos-in-place
while the gripper closes (avoids the early-exit-before-close bug) and counters set_gripper's pose drift.

For a low cabinet handle, reaching the bar does not by itself solve the pull:
the finger pads still need the correct bar-straddling orientation and sustained
contact can destabilize the simulation. See [[feedback_handle_bar_grasp_orientation]]
and [[feedback_thin_handle_drawer_closure]].
