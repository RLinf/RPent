# Physics-only redo recipes (reference for a following agent)

These OVERWRITE the original teleport recipes (set_object_pose / articulate_to / js_move_to /
carry_object are all forbidden — see memory `no-teleport-rule`). Every recipe below is real physics:
joint-space damped-LS IK + PD torque control (+ RRT-Connect) done OFFLINE, or pi0_pick/pi0_doubled
for VLA contact skills. Per-cell audit JSONs are overwritten in `results_10_pert/` and
`results_goal_pert/`. Scripts live in `workspace_pro/jointspace_experiments/`.

## Status table

| cell | task | status | how |
|---|---|---|---|
| **goal_swap_t0** | open middle drawer (relocated cabinet) | ✅ SOLVED | joint-space IK+PD+RRT, `js_swap_RRT.py` |
| **10_swap_t8** | both moka pots on stove | ✅ SOLVED | joint-space IK+PD, `js_t8.py` |
| 10_lan_t3 | bowl→bottom drawer→close | ✅ SOLVED (earlier) | pi0_doubled close + scripted carry |
| goal_task_t0 | open bottom drawer | ❌ BLOCKED | gripper hand can't fit at the low bar (geometric) |
| 10_task_t3 | bottle→bottom drawer→close | ⚠️ In✓ / Close✗ | In via scripted grasp; close = wall |
| 10_swap_t3 | bowl→bottom drawer→close | ❌ BLOCKED | shares the close wall |

## Reusable techniques (the load-bearing lessons)

1. **Joint-space IK+PD bypasses the OSC singularity.** MuJoCo damped-LS IK (`dq = Jᵀ(JJᵀ+λ²I)⁻¹·err`,
   λ≈0.05) + joint PD torque with gravity comp (`tau = clip(Kp(q*-q) - Kd·qd + d.qfrc_bias[arm], ±FMAX)`,
   Kp≈[360×4,190,120,65], FMAX≈80). Reaches cabinet handles where OSC `move_to` walls.
2. **Gripper convention (raw actuator):** CLOSE = `grip 0.0`, OPEN = `grip 0.04`. `grip=1.0` CLAMPS to
   OPEN — the bug that faked a reachability wall. Verify a grasp via finger-width ≈ object thickness.
3. **Grip a horizontal handle bar:** ROLL the gripper 90° so the pads straddle the bar ACROSS its axis
   (in z), `Rg=[[1,0,0],[0,0,-1],[0,1,0]]@Rz(90°)`. Verify with `pad_collision` geom_xpos: pads differ
   in z = right; differ in x (along bar) = empty close. Pull/push the eef to move the drawer.
4. **IK seed matters:** the rolled at-bar config only converges from the HOME seed; chain Cartesian
   seeds (front→bar) and compute it BEFORE moving the arm (pure kinematics).
5. **RRT collision oracle must check ALL robot-vs-environment** penetrations (table/bowl/plate/cabinet/
   bottle), not just the target fixture, or it plans the arm through clutter.
6. **Reach a far/raised target** (e.g., the stove burner ~0.88 m, beyond the gripper-DOWN reach ~x0.16):
   use position-only IK (free orientation) so the gripper tilts forward; place objects so the tilted
   arm doesn't sweep already-placed ones (place the far one first, near one last).
7. **Narrow tall object (wine bottle r0.022):** UPRIGHT joint-space grasp slips; knock it FLAT first
   (push it over, or let pi0 knock it) then grasp the LYING body center — holds (gw~0.05).

## SOLVED recipes

### goal_swap_t0 — open the relocated cabinet's middle drawer  (`js_swap_RRT.py`)
HANDLE bar = (-0.247,-0.155,1.015). The wine bottle (-0.196,-0.064) is the ONLY blocker.
1. Compute at-bar config qsol: IK from HOME seed, rolled orientation, chain y 0.10→-0.155. err 0.0000, collision-free.
2. PD-pick the bottle (gripper-down, close=grip0.0, eef y = bottle_y-0.02 to cancel +2 cm drift), lift, carry to (0.18,0.22,1.18), drop.
3. RRT-Connect post-bottle→qsol (free once bottle gone).
4. Close grip=0.0 on the bar (width 0.029).
5. Pull +y in 0.018 steps → middle qpos −0.161. **CHECK_SUCCESS=True.**

### 10_swap_t8 — both moka pots on the stove  (`js_t8.py`)
Stove ALREADY ON (button qpos 0.96 ≥ 0.5). cook_region center x0.215 is beyond gripper-down reach; burner raised z0.927.
1. Grasp each moka at the UPPER body (gripper-down, descend body_z+0.03, close grip=0.0) — reliable for both.
2. Lift straight up.
3. Place via FREE-ORIENTATION (position-only IK) so the gripper tilts forward onto the raised far burner.
4. Place moka_2 FIRST at x0.21,y0.09; moka_1 LAST at front-left x0.155,y0.0 (so neither knocks the other; >x0.25 overshoots).
Result: both in cook_region (0.166,0.017)+(0.245,0.101). **CHECK_SUCCESS=True.**

## BLOCKED / PARTIAL (don't repeat the dead ends)

### goal_task_t0 — open the bottom drawer  (BLOCKED, geometric)
NO collision-free grasp config exists for the bottom bar (z0.946, 4.6 cm above the table). Probe (rolled
grasp, pitch {0,0.3,0.6,0.9} × 120 seeds, clutter removed): min collisions never < 3. The gripper HAND
can't fit — horizontal → hand/forearm into the table; tilted → hand into the drawer faces above. User
accepted as a genuine block. **Do not retry** (it's robot/gripper-vs-cabinet geometry, not tuning).

### 10_task_t3 — bottle in bottom drawer + close  (In✓, Close✗)
Drawer is ALREADY OPEN (qpos −0.144). **In works** (`js_t3.py`): knock the bottle flat → scripted grasp
the lying body center (gw 0.049) → carry → lay in bottom_region (−0.011,0.176,0.922). **Close is the wall:**
Close needs qpos > 0.005 (fully seated); the thin handle (z0.946) glances when pushed + shields the face
panel; pushing above (z1.0) misses, z1.035 hits the closed middle drawer → scripted push maxes at −0.035
and can't seat the last ~4 cm; pi0_doubled close is defeated by the P1 perturbation (does the original task).
**Open problem for a following agent:** reliably SEAT this drawer (qpos>0.005) without pi0.

### 10_swap_t3 — bowl in bottom drawer + close  (BLOCKED)
Shares the white_cabinet bottom-drawer CLOSE wall above. The bowl is short (no jam) and easier to grasp,
so In is likely achievable; the CLOSE is the shared blocker. Cabinet relocated to (-0.11,-0.293).
