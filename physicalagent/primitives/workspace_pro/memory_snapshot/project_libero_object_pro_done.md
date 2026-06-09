---
name: libero-object-pro-done
description: "libero_object PRO t0-t9 four-cell hybrid runs solved 2026-05-21 (30/30 strict); new \"object\" scene frame (eef home z≈0.26) characterized; Pi0 baselines running on GPU 1"
metadata: 
  node_type: memory
  type: project
  originSessionId: 796fbb81-ebc3-435f-9308-413943fa0320
---

30/30 libero_object PRO hybrid runs (t0–t9 × {_task, _swap, _lan}, seed 0)
strict-passed and audited at
`physicalagent/primitives/workspace_pro/results_object_pert/` — see
`REPORT_object_t0_t9.md` there. Pi0 fullshot baselines (40 cells incl. base)
launched in background on GPU 1 via `run_baselines_object.sh`.

**Why:** Extending the PRO 4-cell coverage from libero_spatial (already
done) to libero_object. All 10 tasks now done.

**How to apply:** The libero_object scene uses a third frame distinct
from LIVING_ROOM (0.68) and KITCHEN (1.17) in
[[liberopro-driver-patch]]. Calibrated values:

- `eef_home_z ≈ 0.261`, table top z ≈ 0
- Flat-box / can items: `pre_pos_z=0.13`, `carry_z=0.22`,
  `release_z=0.16`
- Tall bottles (salad_dressing, ketchup, milk, bbq_sauce):
  `pre_pos_z=0.16`, `carry_z=0.30`, **release at carry z (no descent)**,
  target eef +y past basket center by ~0.030 (bottle dangles ~3 cm behind
  eef during carry)
- Always `set_gripper(+1, steps=8)` between move stages; cylindrical
  cans slip otherwise (see [[pi0-pick-full-prompt]] for related grip
  finding)
- For tall bottles, `track_obj_lift_thresh = lift_thresh = 0.08` (0.05
  leaves grip too loose for the first lift)

Descent stalls when bottle bottom hits basket rim and OSC can't push
further — symptom is `final_dist_m ≈ 0.06`, `basket xy moves 5+ cm`.
Solution: high-carry-and-drop, never descend with a tall bottle inside
the basket footprint.

**orange_juice y-offset exception**: unlike other tall bottles which
dangle ~3 cm behind eef after pick, OJ is grabbed near-centered
(y-offset ≈ 0). Aim eef at basket *center*, not basket_y + 0.030.
Overshooting by 3 cm makes the bottle land outside the basket rim.
