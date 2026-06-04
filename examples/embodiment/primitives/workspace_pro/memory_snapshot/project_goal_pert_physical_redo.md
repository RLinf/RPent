---
name: goal-pert-physical-redo
description: "results_goal_pert 14 teleport cells redone physics-only: 12 SOLVED 2026-05-26 + goal_swap_t0 SOLVED 2026-05-27 via joint-space IK+RRT (13/14); goal_task_t0 bottom-drawer in progress"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a54aab4-1492-40e1-bb85-3bd7146d3399
---

The 14 `results_goal_pert` cells that used teleport (`articulate_to`×12, `carry_object`×2)
were redone physics-only after teleport removal ([[no-teleport-rule]]). Outcome 12/14 SOLVED:

**SOLVED (libero_terminated=True, no teleport):**
- **t7 base/lan/swap (TurnOn) + task (TurnOff)** — `pi0_doubled` knob turn: `pi0_pick` prompt
  "Turn on/off the stove", `track_obj=null`, `gripper_closed_thresh=0`, `lift_thresh=99` so the
  pick-break never fires. Burner glows red / goes dark. Turn-OFF works too but is **stochastic**
  (flow-matching) — needed ~2 tries. Replaces `articulate_to flat_stove_1_button`.
- **t0 base/lan (Open middle drawer)** — `pi0_doubled` "open the middle drawer of the cabinet"
  (the TRAINING-DISTRIBUTION phrasing; the PRO `:language` "Open the middle layer of the drawer"
  makes Pi0 miss the cabinet). Pi0 pulls the handle ~25-35 chunks.
- **t3 base/lan/swap/task (In bowl/cream_cheese, top_region)** — STRICT/pi0_doubled. Pi0 opens
  the top drawer AND grasps the object in ONE call (full task prompt + `track_obj` cut at +0.06
  lift; pick `libero_terminated=False` => Pi0 did NOT place, Rule-1 OK). Then LLM `set_gripper`
  clamp + lift z=1.18 + travel over cabinet + descend into the slid-out drawer (~eef 0.02,-0.10,
  1.08); In fires during the descent. **Flaky** (Pi0 open-amount + grasp vary) — used a retry loop.
- **t5 lan/swap (On plate_1, stove_front_region)** — Pi0 picks the plate (`track_obj=plate_1` cut),
  LLM `set_gripper` clamp + slow OSC carry (step_clip 0.015) to (-0.05,0.21) + descend + release.
  Replaces `carry_object`/`js_move_to` (physics-only OSC carry, like base t5).

**PHYSICS-BLOCKED (regime=strict_failure_physical, libero_terminated=False) — the 2 cells teleport faked:**
- **t0 swap** Open(middle_region) — P2: cabinet relocated far-left (world x=-0.25). Pi0 is
  position-blind: across 6 episodes drives to the HABITUAL center (eef~0.03,-0.13) and grabs air.
  Scripted OSC reaches in front of the cabinet (eef y~-0.05) but STALLS ~0.10 m short of the handle
  (Panda OSC singularity at the cabinet front).
- **t0 task** Open(bottom_region) — P1: goal flips middle->bottom. Pi0 (SFT'd on "middle" for this
  scene) opens the MIDDLE drawer instead (descends to z~1.015) across 6 episodes. Scripted bottom-
  handle pull blocked: 3 handles stack at the same xy (vertical descent hits the upper handles),
  horizontal reach-in stalls (IK singularity), plate blocks the front corridor.

Key reusable facts: wooden_cabinet drawers slide world **+y** to open (qpos 0→-0.16); handle z =
top 1.085 / middle 1.015 / bottom 0.946 at xy~(0.034,-0.147); the Panda OSC `move_to` **cannot**
thread the cabinet-front-low reach (only Pi0's learned trajectory can) — same barrier as microwave.
Tools left in /tmp: introspect.py (mujoco geometry via OffScreenRenderEnv), run_cell.py / retry_cell.sh
(launch+replay+stitch), stitch_goal.py. See [[no-teleport-rule]] [[libero-10-pert-physical-redo]].

**2026-05-27 RE-VERIFICATION of goal_task_t0 (Open bottom drawer) — STILL BLOCKED, now quantified.**
Re-attempted physics-only (4 fresh episodes); confirms strict_failure_physical with hard numbers
(audit `goal_task_t0_s0.json` → `reverification_2026_05_27`):
- **Goal is NOT infeasible.** Offline probe: `Open(bottom)` == `WoodenCabinet.is_open` == bottom_level
  qpos < **-0.14** (damping=50). Forcing qpos=-0.15 + 40 zero-steps HOLDS at -0.1496, check_success=True.
  So the barrier is applying the open FORCE, not the goal.
- **pi0 opens MIDDLE → then TOP, NEVER bottom** (lowest/least-salient handle). Tried: from home, from a
  middle-handle pre-pose, and after the middle was already open — it never targets the bottom. Visual
  prior dominates the prompt. (My above-stack low pre-pos just makes pi0 lift away.)
- **OSC hard reach floor at the cabinet front ≈ z 1.025.** Gripper-down descent stalls ~z1.10 (top-handle
  collision + IK). Pitched-horizontal (rotate_pitch -1.45, fingers→-y) stalls ~z1.025; a 140-step
  step_clip=0.04 hard push does NOT break below z1.025. The bottom handle (z=0.946) is **~8 cm below the
  reachable floor** → no scripted move_to orientation reaches it.
- From ABOVE the bottom handle is blocked whether upper drawers are CLOSED (handle stack same xy) or OPEN
  (open drawer BOX overhangs at z~0.99-1.04); the plate blocks the front corridor.
Verdict: genuine **controller-reachability** dead-end (pi0 won't target bottom; OSC can't reach it) — same
class as goal_swap_t0 / 10_swap_t3 (all cabinet-front drawer ops). Probe at /tmp/probe_goal_task_t0.py,
/tmp/probe2.py. The NEW reusable number is the **OSC cabinet-front floor z≈1.025 (even pitched)**.

**2026-05-27 SESSION 2 (user pushed: "it IS physically reachable, keep trying") — REACHABILITY PROVEN,
new `move_pose` primitive built, but the contact-PULL still blocked. Big update:**
- **Reach is NOT the wall.** Bottom handle z=0.946 IS kinematically reachable: pi0's libero_90 training
  scene `KITCHEN_SCENE1_open_the_bottom_drawer` (identical robot, same eef home) has the bottom handle at
  (-0.006,-0.21,0.946) — even FARTHER forward than goal-t0's (0.042,-0.132,0.946) — and pi0 opens it.
  So the earlier "OSC floor z≈1.025" is an **orientation-coupled singularity** of the decoupled `move_to`
  (holds gripper-down + servos straight), NOT a kinematic limit. Proof: at y=+0.12 the same servo reaches
  z=0.92 cleanly; the wall only appears reaching forward (y→neg) while holding gripper-down.
- **NEW PRIMITIVE `move_pose`** (added to primitives.py + interactive_driver.py, physics-only OSC,
  Rule-compliant): servos xyz AND pitch/yaw SIMULTANEOUSLY each step. Co-variation **threads the
  cabinet-front singularity** that decoupled move_to+rotate_pitch cannot. Recipe that reaches the bottom
  handle: warm-up [move_to (0.04,0.12,0.92) → (0.045,-0.03,0.93)] to get a low arm config, THEN
  move_pose(target≈(0.05,-0.135,0.95), target_pitch −0.7..−1.0, approach from the RIGHT). Best eef reached
  (0.048,-0.123,0.956); pi0 (pre-positioned there) got the pinch-point to z=0.944 = the bar. ⚠ move_pose is
  config-sensitive (same target gives z0.95 or z1.03 depending on warm-up); target only REACHABLE points
  with modest max_steps or it sustain-pushes a wall → MuJoCo NaN crash.
- **Still blocked on the PULL**, three compounding reasons: (a) **pi0 policy-lock** — never pulls the bottom
  drawer (tried from home / middle-pose / after opening upper / pre-positioned AT the bar; it dips to z0.944
  then lifts/wanders to x0.22, never executes the +y pull). (b) **Pinch geometry** — the parallel-jaw pinch
  point is the eef site; scripted move_pose walls the eef ~1cm short of the bar so jaws close in FRONT of it
  → empty grab. Only pi0 puts the pinch-point AT the bar, and pi0 won't pull. (c) **Sim-crash on contact** —
  every firm scripted contact with the handle/panel at the cabinet-front-low crashes the worker (QACC NaN →
  EOFError, see [[feedback_osc_push_mujoco_nan]]); the crash-free +y pull is unusable because engaging the
  bar first needs the crash-prone -y/-z contact. Note: Panda finger separation axis still unconfirmed
  (yaw=0 vs 90 both gave empty grabs) — worth an offline finger-geom probe next.
  Probe /tmp/probe_scene1.py compares the two scenes. Audit `goal_task_t0_s0.json`.session2_movepose_reachability.

**2026-05-27 SESSION 3 — goal_task_t0 (bottom drawer) CONFIRMED genuinely BLOCKED (user accepted, "就这样吧").**
Re-tested with the joint-space IK+PD+RRT toolkit that just SOLVED swap_t0. Result: UNLIKE swap_t0 (which had
a clean collision-free at-bar config and was only bug-masked), the bottom drawer has **NO collision-free grasp
config at all**. Exhaustive offline probe (rolled grasp, pitch ∈ {0,0.3,0.6,0.9}, 120 seeds each, PEN=-0.006,
**plate+bowl teleported away for the test** so clutter is ruled out): min blockers never < 3. The gripper HAND
cannot fit at the bottom bar (z=0.946, only 4.6cm above the table): horizontal puts hand+forearm (link5/link7)
into the **table_collision**; tilting up to clear the table drives the **hand into the drawer faces above**
(wooden_cabinet g11/g12/g16/g18/g20/g22). Clutter (plate at 0.046,-0.014; bowl) was a red herring — the
cabinet-stack+table geometry is the wall. So goal_pert physics redo final tally = **13/14** (swap_t0 solved,
task_t0 blocked). Probes /tmp/truth_probe.py (clutter-free), /tmp/pitch_probe.py.

**2026-05-27 SESSION 3 — goal_swap_t0 SOLVED physics-only (CHECK_SUCCESS=True, mid qpos -0.161).**
Abandoned the OSC interactive_driver entirely; solved OFFLINE with **joint-space damped-LS IK + PD
torque control + RRT-Connect** (script `workspace_pro/jointspace_experiments/js_swap_RRT.py`, renders
`result_paper/goal_fail_renders/swap_t0_SOLVE.mp4`). The "OSC singularity / unreachable" verdict was
WRONG — it was FOUR stacked bugs masking a simple task (user was right: "it's a very simple task"):
  1. **Gripper-close clamp bug** — the two finger actuators have ctrlrange [0,0.04] / [-0.04,0]; ctrl =
     target finger POSITION. So `grip=0.0` = CLOSED, `grip=0.04` = OPEN, and `grip=1.0` clamps to 0.04 =
     OPEN. Every prior "close on the bottle / closed-gripper ram" used grip=1.0 = silently OPEN → never
     grabbed/knocked anything. See [[gripper-ctrl-is-finger-position]].
  2. **Bottle pick failed** (open + ~2cm +y eef-drift) so the wine bottle NEVER cleared and kept
     blocking the gripper BODY at y≈-0.064. Fix: close=grip0.0 + target bot_y-0.02 → grabs+lifts it; drop
     it far front-right. The bottle WAS the only blocker (user insisted) — once removed, the straight-line
     path to the bar is collision-free.
  3. **Grasp orientation** — must ROLL the gripper 90° so the pads straddle the bar ACROSS its axis (z),
     not along it. Verify with `geom_xpos` of `gripper0_finger{1,2}_pad_collision`: wrong = pads differ in
     x (along the bar) → empty close; right = pads at z 0.97 / 1.06 (above/below the bar). Orientation:
     `Rg = [[1,0,0],[0,0,-1],[0,1,0]] @ Rz(90°)` (points -y, rolled). See [[handle-bar-grasp-orientation]].
  4. **IK seed sensitivity** — the rolled at-bar IK only converges from the HOME config seed (err 0.0000);
     from a post-motion pose it diverges (err 0.18 → garbage pose). Compute qsol from qhome FIRST (pure
     kinematics), chain seeds front→bar [0.10,0.05,0,-0.05,-0.10,-0.13,-0.155], THEN do the bottle removal,
     THEN RRT(post-bottle_q → qsol). The at-bar config is collision-free + limit-free.
Recipe: rolled gripper, bar HANDLE=(-0.247,-0.155,1.015); close grip=0.0 gave width 0.029 (real bar grip,
not empty); pull +y in 0.018 increments → mid qpos 0→-0.161 (<-0.14). Kp=[360×4,190,120,65], FMAX 80.
**Reusable**: joint-space IK+PD bypasses the OSC singularity (already in [[jointspace-control-bypasses-osc-singularity]]);
the RRT collision oracle MUST check ALL robot-vs-environment penetrations (table/bowl/plate/cabinet/bottle),
not just the target fixture, else it plans the arm through clutter.
