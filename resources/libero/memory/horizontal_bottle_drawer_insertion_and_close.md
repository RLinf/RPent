---
name: horizontal-bottle-drawer-insertion-and-close
description: Reusable narrow-drawer recipe that rotates a long bottle horizontally, inserts with a tilted pose, and finishes with a hooked close.
aliases:
  - put the bottle in the bottom drawer of the cabinet and close it
  - wine bottle bottom drawer close
metadata:
  node_type: memory
  type: skill
---

# Horizontal bottle drawer insertion and close

## Applicable pattern

Use this pattern when a long object only fits a drawer horizontally and both insertion and final closure require orientation-aware contact motions.

## Reference task alias

`put the bottle in the bottom drawer of the cabinet and close it`

> Note: a prior memory snapshot for this cell described a different technique (vertical drop via wrist-rotation IK overshoot to seat bottle upright). That technique is **NOT what the latest exploration confirmed** — multiple recent attempts showed the bottle bouncing or landing off-centre, and the close phase was unsolved without the tilted hook below. This file reflects the current winning technique.

## Winning technique
Pi0_pick the bottle with high-z prepos and a retry-on-grab-and-drop block (up to 3 tries with re-segmentation, lower z each retry). Lay the bottle **horizontal along Y** (cavity Y dim 13.5 cm > X dim 10 cm > bottle 13 cm) via `rotate_wrist target_yaw=1.57 + rotate_pitch target_pitch=1.4`, then descend `move_pose` to (cavity_center, z=1.00) with `target_pitch=1.4` — the tilt threads the OSC singularity at z=1.025 that traps vertical descents. To close: pi0_doubled gives a partial close, then **tilted hook at `target_pitch=0.6`, z=0.95, slow +y drag (`step_clip=0.012`)** engages the drawer top edge BELOW the cabinet rim and drags the face the remaining ~13 cm.

---

## Magic numbers (RANGES, not single points)

| Knob | Range | Note |
|---|---|---|
| pick prepos z | 1.15–1.20 | longer descent runway = cleaner Pi0 close |
| pi0_pick max_chunks | 10–18 | >18 → Pi0 enters *place mode* and dumps bottle on cabinet top (attempts 2, 5, 11) |
| set_gripper steps after pick | 10–15 | firms grip to qpos≈0.008 |
| lift z after pick | 1.23–1.27 | clear cabinet rim 0.99 |
| rotate_wrist target_yaw | 1.5–1.6 | align bottle long axis with Y |
| rotate_pitch (place) | 1.35–1.45 | horizontal + threads OSC |
| place eef z | 0.98–1.03 | above 1.05 → bottle bounces out (attempt 7) |
| place step_clip | 0.012 | controlled tilted descent |
| pi0_doubled close max_chunks | 20–30 | >30 locks arm in stuck OSC config (attempts 8, 13) |
| drawer-hook pitch | 0.5–0.8 | tilt that escapes OSC vertical wall at z=1.025 |
| drawer-hook eef z | 0.93–0.97 | BELOW cabinet rim 0.99 — needed to hook drawer face |
| drawer-hook step_clip | 0.010–0.018 | slow drag keeps panel contact |
| drawer-hook +y throw | ≥0.18 m | covers full 13 cm close stroke |

---

## Failure modes (mined from 16 failed attempts) → fix

| Failure | Fix |
|---|---|
| Pi0 grab-and-drop on first pick (qpos closes then re-opens) — common, ~70% of the recorded attempts | retry up to 3x: re-segment bottle, re-prepos, lower z each retry (1.18→1.10) |
| `max_chunks ≥ 20` → Pi0 carries the bottle through a full pick-AND-place and dumps it on TOP of cabinet | hard cap max_chunks at 18 for pick, 25 for doubled-close |
| Bottle laid along X (rotate_pitch alone, no wrist) → bridges cavity rim, blocks drawer close | `rotate_wrist 1.57` BEFORE `rotate_pitch 1.4` → lay along Y |
| Vertical descent to place → OSC wall at z=1.025 → bottle drops from too high → bounces out onto cabinet top | tilted `move_pose target_pitch=1.4` threads the singularity down to z≈1.00 |
| Drop eef z > 1.05 → bottle bounces forward onto cabinet front edge | place at eef z = 1.00 (range 0.98–1.03) |
| Vertical move_to "push" to close drawer → OSC wall at z=1.025, drawer moves <5 cm | **tilted-eef low-z drawer hook (`target_pitch=0.6` at z=0.95)** — THE recipe's transfer-critical lever |
| pi0_doubled at max_chunks 40+ leaves arm stuck in OSC config (unrecoverable) | cap at 25; treat as partial-close only, finish with manual hook |
| Wrist yaw direction ambiguity: wrong yaw points bottle in −Y instead of +Y → bottle hangs in front of cavity mouth (attempt 16) | always `rotate_wrist target_yaw=1.57` BEFORE `rotate_pitch` |
| Bottle lands at cavity mouth (y < cy_mid) → drawer slides past it, leaves bottle behind in world frame (attempt 12) | place at y ≥ cy_mid so cavity walls drag bottle along during close |
| `pi0_pick "pick up the wine bottle"` sometimes never closes gripper | alt prompts that worked: `"grasp the wine bottle"` (attempts 6, 14), `"put the bottle in the bottom drawer"` (caution: hard-cap max_chunks ≤ 10) |

---

## Scene re-localization (entities + prompts)

* **Bottle:** `segment({"prompt":"the dark wine bottle","camera":"agentview","min_score":0.15})`. Fallback: `"the wine bottle on the table"`. Use (bx, by) for prepos; bz is unreliable.
* **Cavity / destination:** `segment({"prompt":"the open drawer cavity","camera":"agentview","min_score":0.10})`. Returns (cx, cy_mid, cz_floor). The cabinet was fixed-pose in the reference run, but always re-segment. Expected bounds: cx∈[−0.03,+0.03], cy_mid∈[0.15,0.22], cz_floor≈0.926.
* **Drawer face (for hook target):** `segment({"prompt":"the bottom drawer front panel with handle","camera":"agentview","min_score":0.10})`. The OPEN face y becomes the start point for the hook drag.
* **All XYZ in the recipe are written relative to (bx, by) or (cx, cy_mid)** — never copy recorded literal coordinates.


---

## Difficulty and reliability
**16 failed attempts before the win.** High-difficulty cell — the pick alone is stochastic, the placement is geometrically tight (bottle barely fits cavity), and the close requires escaping an OSC singularity. The verifier should expect the Phase-1 retry block to fire 2–3 times. Do **NOT** treat first pick failure as a recipe break — that's the expected path.

## Related memory

[[feedback_move_pose_covarying_reach]] [[feedback_rotate_pitch_orientation_control]]
