---
name: libero-10-pert-physical-redo
description: "results_10_pert teleport cells redone physics-only. 2026-05-27 UPDATE: lan_t3 drawer-close SOLVED; swap_t8 reach-block DISPROVEN (sim-fragility-limited, not reach)"
metadata: 
  node_type: memory
  type: project
  originSessionId: bc1d8704-c816-4268-9976-2a28696eeecf
---

The 8 `results_10_pert` cells that had used `articulate_to` teleport were redone with
physics-only primitives after teleport was removed ([[no-teleport-rule]]). Outcome:

**SOLVED (libero_terminated=True, no teleport):**
- `10_lan_t2`, `10_swap_t2` — stove: pi0_doubled knob turn + scripted moka pick + OSC carry/release.
- `10_task_t8` — "left moka → stove": same, with ULTRA-slow 0.08 m-hop carry (long y-traverse slips otherwise).
- `10_lan_t8` — "both mokas → stove": two pick+carry+place cycles onto the cook.

**PHYSICS-BLOCKED (honest strict_failure, regime=strict_failure_physical):**
- `10_lan_t3`, `10_swap_t3`, `10_task_t3` — drawer: open ✓ + bowl-in-drawer ✓ + partial
  close ✓, but `WhiteCabinet.is_close` needs joint **qpos > 0** (pushed past flush). Pi0-close
  stalls at qpos<0 (~60–70%); a firm scripted OSC push to reach qpos>0 repeatedly crashes the
  MuJoCo worker (EOFError, collision blowup at the cabinet front, eef z≲1.02). ~6 episodes, 2 crashes.
- `10_swap_t8` — "both mokas → stove" (swapped): TurnOn ✓ (burner red), but the swap-perturbed
  cook_region is **past the Panda OSC reach** (empty-gripper forward reach maxes x=0.182 at
  burner height; burner sits ~x≥0.25). No physical arm motion reaches it; a sustained shove
  crashes the sim (QACC NaN).

**RETRY (2026-05-26, user asked for more attempts, ~15+ more episodes):** bowl-into-drawer is
now SOLVED reliably — place at the CAVITY FRONT (y~0.11) with the bowl hanging below the eef,
descend eef to z~1.04, release (drops to floor z~0.92); placing too far back (y~0.18) perches the
bowl on the rear wall at z~1.08. But the t3 **full drawer close to qpos>0 remains blocked** (pi0
stalls short, scripted push + front-repositioning crash the worker, and the worker dies from
EGL/resource buildup at ~25-30 cmds — fewer than this task needs). swap_t8 reach confirmed
unreachable. The 4 blocked cells stand as honest strict_failure_physical.

The 4 blocked cells are exactly the ones the teleport originally faked (articulate_to OVERSHOOT
for the qpos>0 drawer close; set_object_pose for the past-reach cook). Audits + recipes in
`physicalagent/primitives/workspace_pro/results_10_pert/`. Lessons in [[no-teleport-rule]],
[[stove-turnoff-strict]], [[scripted-pick-limits]], [[feedback_osc_push_mujoco_nan]].

**2026-05-27 SESSION — two of the four "blocked" cells re-examined and largely overturned:**

1. **`10_lan_t3` drawer-close — SOLVED (libero_terminated=True, regime=pi0_doubled).** Offline
   probe proved qpos>0 is a STABLE physical state (set joint qpos=+0.005/+0.009, it HOLDS; the
   WhiteCabinet drawer is near-frictionless, damping 0.0005, NO collision seat). So the close was
   never physically impossible — it stalls only because the closing FORCE stops (pi0 quits at
   visual-flush ~qpos-0.03..-0.09; OSC move_to is rank-deficient in +y at the cabinet-front
   singularity, hard wall at eef y~0.157). **Winning recipe:** pick bowl → place at drawer CAVITY
   CENTER (y~0.16, eef descend to ~1.03, release; bowl lands on the floor z~0.92) → `pi0_doubled`
   "close the bottom drawer" (gets near flush) → OSC FRONT-APPROACH push (back out, descend IN
   FRONT of the handle eef~(0,0.04,0.95), push +y; advances drawer to ~qpos-0.025) → back out gently
   → `pi0_doubled` "close the bottom drawer" AGAIN (joint-space reaches eef y~0.20 PAST the OSC wall
   and rams the last bit to qpos>0). Bowl proxy: qpos = bowl_y - 0.31 (bowl rides drawer 1:1);
   need bowl_y > 0.31. Recipe saved `recipe_10_lan_t3_s0.jsonl`.
   - **task_t3 NOT transferable**: object is wine_bottle (~0.18 m), jams the shallow drawer at
     ~qpos-0.07 (too long to seat past flush) — apparent infeasibility of the perturbed goal.
   - **swap_t3 NOT transferable**: cabinet RELOCATED far-right + rotated 180° (closes -y); pi0 is
     position-blind (won't close it, so the "pi0 finish" trick is gone) and OSC -y push walls at
     eef y~-0.017. Momentum strikes got to ~qpos-0.075 then the handle passes the wall. Still blocked.

2. **`10_swap_t8` reach-block DISPROVEN.** The cook_region is a BOX (site (0.178,0.066), half-size
   0.075) spanning x[0.103,0.253]; eef reaches x~0.17 — WELL INSIDE. (Prior "x>0.22, unreachable"
   used the burner BODY xpos, not the predicate box.) Demonstrated: TurnOn (pi0, burner red), both
   mokas graspable (precise scripted grab, gripper 0.085 vs moka 0.079 = align to <2mm), single moka
   placed ON the burner in the box. NEW blocker = SIM-STABILITY of the full 2-moka sequence (~18-20
   cmds): the SECOND placement-descend reliably crashes (CUMULATIVE — same xy works as an early cmd,
   crashes as the ~18th; EGL/contact budget) + gripper≈moka-width tips/drags the placed moka out of
   the box on retreat. Auto-retry (relaunch-on-crash, /tmp/auto_swap_t8.sh) never completed a full
   run. Solvable in principle, sim-fragility-limited — NOT reach-limited.

**2026-05-27 SESSION 2 — two more "blocked" cells re-examined, both OVERTURNED as solvable:**

1. **`10_task_t3` (wine bottle in drawer + close) — feasibility OVERTURNED (was "geometrically infeasible").**
   Goal `(And Close(white_cabinet bottom) In(wine_bottle, bottom_region))`; WhiteCabinet is_close needs qpos>0.
   The bottom_region box is 0.20m wide (x) × 0.15 deep (y) × only **0.06m tall (z)**. Bottle ~0.16m long
   UPRIGHT can't fit (jams at qpos-0.07) — that was the only orientation prior sessions tried. **Laid FLAT
   along the drawer width (x), the bottle fits (0.045 dia < 0.06)**: offline (probe_t3_flat.py) set bottle
   horizontal in the drawer + qpos=+0.005 + 40 zero-steps → holds, **check_success()=True**. Solution path =
   pick bottle → rotate horizontal → place flat in open drawer → close white_cabinet to qpos>0 (the lan_t3
   close recipe). Physical execution NOT yet done. Audit `10_task_t3_s0.json`.reverification_2026_05_27_feasibility.

2. **`10_swap_t8` (both mokas → stove) — reach DISPROVEN again + KEY: stove is ALREADY ON.**
   `flat_stove_1_button` qpos=**0.96** in the init (≥0.5 turn_on threshold) → TurnOn already satisfied. Goal
   reduces to placing BOTH mokas on cook_region — **NO pi0 knob-turn, fully scriptable**. cook box x[0.14,0.29]
   y[-0.024,0.126], On needs |dx|,|dy|<0.075 + moka_center_z in (cook_z-0.0025, cook_z+0.1025) + CONTACT.
   Offline both-mokas-placed → check_success=True (probe_t8_v2.py). **moka_2 (side pos) grasped+carried+placed
   On the cook SUCCESSFULLY** (technique: accurate HIGH pre-pos → STRAIGHT vertical descent for <2mm xy align →
   firm close 24-26 steps grip_sum~-0.008 → slow staged carry w/ re-clamps → release at eef z~1.07). **Blocked
   on moka_1** (starts far-forward y0.258): descent DRIFTS +x ~6mm & WALLS z~1.04 (far-forward reach limit);
   even at 0.8mm align the close was EMPTY (moka_1 body/handle presents differently → needs per-moka YAW); a
   marginal grip SLIPS on carry & the moka TIPS (z0.937, then ungraspable); + CUMULATIVE NaN crash on big arm
   moves from extended configs. Fixes for next time: grasp far-forward moka_1 FIRST (fresh home config),
   per-moka yaw for a body grip, ultra-slow short re-clamped carries, avoid singular poses. Audit
   `10_swap_t8_s0.json`.reverification_2026_05_27_v2. Confirms "sim-fragility-limited, auto-retry never completed."

Method lesson worth keeping: for a frictionless slide-joint whose `is_close` needs qpos>0, alternate
pi0 (joint-space, threads the cabinet-front singularity) with a scripted OSC front-approach push; the
drawer accumulates progress and a final pi0 close rams past flush. Diagnose feasibility offline by
writing the joint qpos and stepping with zero action (does it hold >0?). See [[feedback_osc_push_mujoco_nan]].
