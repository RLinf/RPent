---
name: libero-10-t0-t9-done
description: "libero_10 t0-t9 2026-05-19 in primitives/results_all_10/ — t0/t1/t4/t5/t6/t7/t8 strict, t2/t3 pi0_doubled, 0 pi0_end_to_end (Rule 1). ⚠ t9's Close(microwave) used articulate_to TELEPORT (now removed, see [[no-teleport-rule]]) — that solve is RETRACTED; t9 needs a physical door-close (pi0_doubled / OSC push)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0074d341-bdcd-41b6-b40e-036b75648dc3
---

Solved seed-0 rollouts for libero_10 tasks 0-5 in
`examples/embodiment/primitives/results_all_10/{t0,t1,t2,t3,t4,t5}_s0.json` plus
`all_rows.json` (2026-05-19).

**Why:** user asked to apply [[STRICT_HYBRID_GUIDE]] to libero_10 tasks 1-3 and
build the libero_10/long audit corpus (results_all_10/ was empty before this
session). The prior REPL session in WORK_DONE.md had executed t1-t9 but never
persisted them as audit JSONs.

**How to apply (geometry that's stable across sessions):**
- **t2 stove cook_region (KITCHEN_SCENE3):** flat_stove origin at world
  ~(-0.20, +0.20, ~0.93). Burner body is at `flat_stove`-local (0.15, 0, 0)
  with site half-size (0.075, 0.075, 0.0025) — so cook_region world center
  ≈ (-0.05, +0.20, ~0.94). Drop moka with **eef_z = 1.07** for clean release.
- **t3 white_cabinet bottom drawer (KITCHEN_SCENE4):** cabinet base at world
  ~(0, +0.30, 0.91). Bottom drawer is `cabinet_bottom` body sliding on
  +y joint, range [-0.16, 0.01]. When fully open at qpos=-0.16, the
  bottom_region site sits at world (0.003, +0.151, 0.957) — interior bounds
  x∈[-0.099, +0.105], y∈[+0.075, +0.227], z∈[+0.927, +0.987]. Drop bowl at
  **eef = (-0.045, +0.149, +1.00)** clears the cabinet's bottom-drawer ceiling
  (z≈+0.989) and centers the bowl in the cavity.
- **t3 drawer close:** LLM scripted push from eef at (-0.045, -0.06, 0.97)
  toward y=+0.40 closed it ~95%; final ~5% needs Pi0 — `pi0_pick("close the
  drawer", max_chunks=20, lift_thresh=99, gripper_closed_thresh=99)` fires
  libero_term=True.
- **bowl rim-hook grasp** (akita_black_bowl_1): Pi0 typically grips via rim
  hook giving offset_x ≈ +0.04, offset_z ≈ -0.03 from eef, and gripper qpos
  sum near 0 (fingers closed empty, bowl hanging on outer rim). Bowl still
  tracks lift reliably — verify by raising eef 15cm and checking bowl_z
  follows.

- **t5 desk_caddy back compartment (STUDY_SCENE1):** the bddl init_region
  midpoint is NOT the fixture world position. init_region says caddy center
  is at world (~-0.20, -0.14), but the **actual back_contain world center is
  (-0.411, -0.168, 1.03)** — ~17cm off in x. Reason: bddl init_region defines
  a RANGE; the fixture is sampled within it but not at midpoint, and yaw=π
  rotation flips the sign of the local→world offset for back_contain.
  Practically: do NOT trust init_region midpoint for fixtures; **re-derive
  the target by observing where Pi0 carries the object in a separate
  exploratory episode** (or render the post-pick image, or read the
  unwrapped XML). On 2026-05-19 t5 was solved strict by targeting eef
  (-0.400, -0.151, 1.17) for a book grasp with offset_z=-0.103 — book
  settled at (-0.415, -0.155, 1.018) inside the 5.6cm-wide slot,
  libero_term=True on release. See [[no-pi0-end-to-end]] for why the
  earlier pi0_end_to_end audit was redone.
- **t6 plate_right_region (LIVING_ROOM_SCENE6):** "right of plate" in the
  task name maps to bddl +y direction (image-LEFT under agentview), NOT
  image-right. Region world x∈[0.10, 0.20], y∈[+0.05, +0.15]. Drop pudding
  at center (0.15, 0.10, eef z≈0.51) — libero_term fires on release.
- **t8 two moka pots → stove (KITCHEN_SCENE8):** cook_region centre world
  (-0.05, -0.20), 15×15cm (x∈[-0.125,+0.025], y∈[-0.275,-0.125]). Pick
  ORDER matters — pick **moka_2 (the closer one at init y=+0.04) FIRST**,
  not moka_1 at the back. When moka_1 is picked first and placed, Pi0
  refuses to grasp moka_2 on the second call (sees one moka on stove and
  treats task as done — hovers without closing). Use opposite-corner
  layout, eef drop z=1.075-1.078, step_clip=0.01 for descent. **High-altitude
  carry z=1.30 between pick and place is mandatory** — at z=1.20 the eef
  wrist brushed the already-placed moka and knocked it off the stove.
- **t9 mug → microwave (KITCHEN_SCENE6) — In SOLVED, Close RETRACTED:**
  The In(mug, heating_region) part is physical and stands: `rotate_pitch
  +0.9` so the wrist body fits through the 14-cm cavity opening (else OSC
  stalls at eef y≈0.22, mug y≈0.29), then push, release, and `rotate_wrist
  delta_yaw=+3.0` (unhooks the handle, retreats eef, pushes the mug deeper
  to y≈0.33 clear of the door swing). ⚠ The Close(microwave) part used
  `articulate_to` (joint qpos TELEPORT) + a js_move_to retreat — **both
  removed** ([[no-teleport-rule]]), so that solve is RETRACTED. To close the
  door physically: hand it to `pi0_doubled` ("close the microwave door"), or
  reorient and OSC-push from a non-singular pose; if no physical sequence
  works, record In=satisfied / Close=blocked as an honest strict_failure.
  See [[rotate-pitch-primitive]].

**Regime by task (current):**
- t0: strict (Pi0 picks soup + tomato_sauce; LLM all places).
- t1: strict (Pi0 picks cream_cheese + butter; LLM all places).
- t2: pi0_doubled (Pi0 turns knob in call 1, picks moka in call 2; LLM places).
- t3: pi0_doubled (Pi0 picks bowl in call 1, closes drawer in call 2; LLM
  places + scripts partial drawer push).
- t4: strict (Pi0 picks both mugs; LLM places on plate_1 / plate_2).
- t5: strict (Pi0 picks book; LLM places into corrected caddy back_contain
  coords — redone from prior forbidden pi0_end_to_end attempt).
- t6: strict (Pi0 picks porcelain mug + chocolate pudding; LLM places on
  plate and at plate_right_region).
- t7: strict (Pi0 picks soup + cream cheese; LLM places into basket with
  +x offset to avoid stacking).
- t8: strict (Pi0 picks both moka pots — moka_2 first; LLM places at
  opposite corners of cook_region with high-altitude carry).
- t9: In solved physically (Pi0 picks mug; rotate_pitch+0.9 → push deep →
  release → rotate_wrist+3.0). Close(microwave) RETRACTED — it relied on
  js_move_to + articulate_to teleport (now removed); redo with a physical
  door-close (pi0_doubled / OSC push) or record as strict_failure.

**The low-pre-pos trick (critical for cluttered scenes like t0, t4):** the
default 95cm pre-pos works for sparse 2-3 object scenes (t1, t3) but on
8-object t0 / 5-object t4, Pi0 ignores the sub-instr and wanders to whatever
object dominates the camera view. Pre-position eef at `z = obj_top + 0.18`
(e.g. z=0.65 for cans, z=1.00 for books on study_table) with `tol=0.006-0.008`
and tight `step_clip`. This spatial constraint forces Pi0 to grasp the nearest
object. t0 went from 0/4 sub-instr picks at z=0.95 to clean 8-chunk picks at
z=0.65.

Cross-ref [[pi0-chunks-egl-crash]] for the Pi0 max_chunks ceiling that forced
t3's drawer-close into a hybrid LLM-push + Pi0-snap approach.
