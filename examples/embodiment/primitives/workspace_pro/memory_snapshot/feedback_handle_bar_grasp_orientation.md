---
name: handle-bar-grasp-orientation
description: "To grip a horizontal drawer-handle bar with a parallel-jaw gripper, roll the gripper so the pads straddle the bar ACROSS its axis (z), not along it; verify via pad_collision geom_xpos"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 173e389d-6fb4-4726-99ec-b7c6d73b9da0
---

A drawer handle is a horizontal BAR (its long axis along world-x). A parallel-jaw gripper grips by
closing two pads toward each other — that close axis must cross the bar, not run along it.

- **Wrong:** gripper points -y with pads separating along world-x → both pads land at the two ENDS
  of the bar and closing just slides them along its length → empty close (width ≈ 0.002).
- **Right:** ROLL the gripper 90° about its approach axis so the pads separate along world-z
  (one above, one below the bar). Closing pinches the bar's z-faces (width ≈ bar thickness 0.029).
  Orientation matrix that worked (gripper points -y, rolled): `Rg = [[1,0,0],[0,0,-1],[0,1,0]] @
  Rz(90°)`.

**Why:** I had the un-rolled orientation for many attempts; the close always came up empty and I
mis-blamed reach/geometry. The fix is the roll.

**How to apply:** ALWAYS verify the grasp orientation offline before driving: IK the eef onto the
bar, then read `d.geom_xpos` of `gripper0_finger1_pad_collision` and `..._finger2_pad_collision`.
The two pads should differ in **z** (straddling the bar) and share x,y ≈ the bar. If they differ in
x, you're aligned along the bar — roll 90°. Then pull the drawer open by servoing the eef +y
(wooden_cabinet drawers open toward +y; is_open at qpos < -0.14). Found solving goal_swap_t0 —
see [[goal-pert-physical-redo]], [[gripper-ctrl-is-finger-position]].
