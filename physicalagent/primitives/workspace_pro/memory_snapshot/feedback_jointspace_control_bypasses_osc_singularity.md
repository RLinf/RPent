---
name: jointspace-control-bypasses-osc-singularity
description: "Joint-space IK + PD torque control (MuJoCo) reaches cabinet-front handles where the OSC operational-space servo walls; the cabinet 'unreachable' verdict is an OSC limitation, not kinematic. Needs a collision-free PATH planner (RRT/CHOMP) for the grasp."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 173e389d-6fb4-4726-99ec-b7c6d73b9da0
---

The recurring "OSC can't reach the cabinet-front-low handle" wall (goal_task_t0 bottom, goal_swap_t0
relocated middle, 10_swap_t3, microwave t9) is a **limitation of the OSC operational-space position
servo** (`move_to`/`move_pose`), NOT a kinematic limit. OSC inverts the Jacobian; near the
cabinet-front singularity it walls (eef y≈-0.08 gripper-down, ~-0.10 even co-varying). pi0 reaches it
by co-varying orientation, but pi0 is position/prompt-blind (won't target relocated/bottom drawers).

**Why / the fix (user-suggested, 2026-05-27 — "use non-parametric motion planning"):** control in
JOINT/CONFIGURATION space instead. Build it with **MuJoCo's own kinematics** (mujoco 3.5.0 is installed;
pinocchio/ompl/roboticstoolbox are NOT) — no extra deps:
- **IK** = damped least squares on the 7 arm joints: `mj_jacSite` → `dq = Jᵀ(JJᵀ+λ²I)⁻¹·err` (err =
  [pos(3), rotvec(3)]; λ≈0.05–0.08; **weight orientation fully, not ×0.5, or it converges to a rolled
  config**). Iterate on `d.qpos` scratch with `mj_forward` (save/restore qpos+qvel around it).
- **Control** = joint-space PD **torque** with gravity comp: arm actuators are torque motors
  (robot0_torq_j1..7, ±80/±12 N·m); `tau = Kp(q*-q) - Kd·qd + d.qfrc_bias[arm_dof]`, clip, set
  `d.ctrl[0:7]`, `mj_step`. Gripper = 2 POSITION actuators `d.ctrl[7:8]` (open (0.04,-0.04), close (0,0)).
  **Track a smoothly interpolated joint trajectory** (q0→q* over N steps), not one big jump (saturates/
  overshoots). Kp≈[300×4,150,90,50].
- **Collision check** offline: set q, `mj_forward`, scan `d.contact` for gripper/robot0–cabinet pairs.

**RESULT (goal_swap_t0, relocated middle drawer at world (-0.247,-0.152,1.015)):** joint-space control
REACHES the cabinet front (eef y≈-0.10 from front, y≈-0.157 from above) where OSC walls at y-0.08; IK finds
0-contact grasp configs AT the handle; and in one run it **physically opened the middle drawer (qpos -0.16,
< -0.14 open threshold)** — a real solve. Finger-sep must be ACROSS the bar (world-Y): grab orientation
`R_x(-0.45)·Rdown·R_z(π/2)` (the R_z(π/2) roll flips finger-sep from along-bar to across-bar; verify via
pad_collision geom positions).

**2026-05-27 RRT-Connect added** (`workspace_pro/jointspace_experiments/rrt_planner.py`, committed
29a8296): joint-space RRT-Connect with MuJoCo `d.contact` as the collision oracle (count only
gripper/robot–cabinet penetration deeper than ~4mm; save/restore qpos+qvel around every check).
It plans collision-free C-space paths and the PD tracks them (give the final tight insertion ~1000
track + 500 settle steps or it stalls 0.4 rad short). RESULT: with the gripper pointing **-y** (body
stays in front of the cabinet → collision-free at the bar) + vertical finger-sep, the stack **reaches
and grasps the relocated middle handle bar** (grip width ≈ bar thickness). BUT the drawer still won't
open — a fundamental **grasp-vs-PULL geometry conflict** for the stacked-handle-near-panel:
- gripper pointing **-y** (vertical pinch): collision-free + grabs the bar, but a +y pull **slides the
  bar out** along the finger axis (can't drag the drawer).
- gripper-down / pitched **front/back pinch** (the grip that CAN pull +y): the body **collides** with
  the stacked top handle / cabinet and stalls ~4cm short of the bar (can't reach it).
**2026-05-27 (cont.) — grasp-vs-pull conflict traced to a GEOMETRY incompatibility (definitive).**
Exhaustively tested every gripper orientation for the relocated middle bar (vertical pinch, front/back
pinch, gripper-down, pitched, gripper-pointing-y, compliant force-drives). Root cause: the handle bar
sits only **~0.8 cm from the drawer face**, wedged between the top & bottom handles. A parallel-jaw
grasp that RESISTS a +y pull needs the back finger BEHIND the bar (front/back pinch) — but the back
finger doesn't fit the 0.8 cm gap (presses through the face) AND the gripper-down body collides with
the stacked top handle; a compliant drive into it gets DEFLECTED (eef shoved to x-0.30). The only
collision-free grip (gripper pointing -y, vertical pinch) holds the bar but a +y pull SHEARS it out.
So it is NOT a reach/planning problem (solved) — it is a parallel-jaw-gripper-vs-wedged-handle grasp
incompatibility. pi0 can't substitute (position-blind for swap_t0; won't-target-bottom for task_t0).
Conclusion: goal_swap_t0 / goal_task_t0 (and 10_swap_t3) need a learned/compliant grasp policy (what
pi0 uses in-distribution) or a non-parallel-jaw strategy; not solvable with OSC / pi0 / IK+RRT+PD as-is.

So RRT/IK/PD solve the REACH + a collision-free grasp, but not the PULL. pi0/human use compliant/dynamic
manipulation. Possible fix not yet tried: open an adjacent drawer first to reshape the column (but the
open drawer box then overhangs). NOT a clean check_success=True yet for goal_swap_t0.

**Remaining gap = a collision-free PATH to the grasp config.** Straight joint interpolation from home
hits the cabinet (contact stalls y-0.10) or the stacked top handle (descent stalls z1.12). The grasp config
is reachable+collision-free but the PATH isn't — need RRT (joint space, MuJoCo as collision oracle) or
CHOMP/trajectory-opt (the user's other suggestions). ⚠ BUG that cost time: after `env.reset()` the MjData
is recreated — RE-BIND `m,d=env.sim.model._model, env.sim.data._data` AFTER any reset or check_success reads
a stale data object. Reference scripts: `workspace_pro/jointspace_experiments/js_swap_clean.py` (+ _final, +
jointspace_swap_t0). Not yet integrated into the primitive library (user said don't rush). See
[[move-pose-covarying-reach]] [[goal-pert-physical-redo]].
