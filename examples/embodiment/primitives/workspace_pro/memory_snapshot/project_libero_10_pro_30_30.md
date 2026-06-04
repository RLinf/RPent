---
name: libero-10-pro-30-30
description: "libero_10 PRO hybrid 30/30 (2026-05-22) — but the hard cells used TELEPORT (now forbidden); those solves are RETRACTED and being redone physics-only (see [[no-teleport-rule]])"
metadata: 
  node_type: memory
  type: project
  originSessionId: 234721d6-dc80-4ac9-806e-e06977ce7823
---

**⚠ RETRACTION (2026-05-26):** the original "30/30 PASS" claim relied on **teleport
primitives** (`set_object_pose`, `articulate_to`, `js_move_to`/`carry_object`) for the
hard cells — those do not demonstrate physical manipulation and have been **removed from
the codebase** ([[no-teleport-rule]]). The teleport-dependent solves below are no longer
valid and are being **redone with physics-only primitives**.

Audit JSONs in `examples/embodiment/primitives/workspace_pro/results_10_pert/10_<pert>_t<N>_s0.json`.

## Cells that used teleport — redo scope (physics-only)

The 8 source-library `results_10_pert` teleport cells are: lan t2/t3/t8, swap t2/t3/t8,
task t3/t8 (all `articulate_to` — stove-knob TurnOn or drawer open/close). Redone via:
- **Stove TurnOn (t2/t8)**: `pi0_doubled` "turn on the stove" physically rotates the knob
  (burner glows red), then scripted/Pi0 pick + OSC carry + release of the moka.
  ✅ lan t2 verified physics-only 2026-05-26 (libero_term=True, 9 commands, no teleport).
- **Drawer open/close (t3)**: open + close physically via OSC push or `pi0_doubled`; the
  close must be a continuous push that keeps the drawer wall in contact with the object.

The originally-claimed teleport methods for the other hard cells (task t2/t9, swap t9,
lan t5/t9) used `set_object_pose` / `articulate_to overshoot` and are **retracted**; if
those cells matter, redo them physically too.

## Per-perturbation breakdown (PENDING physics-only re-verification)

- **task (P1)** / **swap (P2)** / **lan**: the easy pick-and-place cells stand; the
  articulation / past-IK-reach cells are being re-verified without teleport.

Related: [[no-teleport-rule]] [[swap-perturbs-fixtures]] [[libero-pro-plus-install]] [[stove-turnoff-strict]]
