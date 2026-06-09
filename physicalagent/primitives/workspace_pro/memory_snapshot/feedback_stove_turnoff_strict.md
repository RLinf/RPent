---
name: stove-turnoff-strict
description: "FlatStove TurnOff predicate is qpos < 0.0 STRICTLY — use target_qpos=-0.005 (joint range lower bound), not 0.0"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

**Rule**: The `flat_stove_1_button` predicate thresholds (method-agnostic — true no
matter how you physically drive the knob): TurnOn fires at button `qpos >= 0.5`,
TurnOff fires at `qpos < 0.0` STRICTLY. The robot must physically turn the knob to
land in the right band — there is NO teleport (no `articulate_to`); use `pi0_doubled`
("turn on/off the stove") or a scripted OSC contact push on the knob. The qpos=0.0
reset state is in the dead band and satisfies neither predicate.

**Why**: From `liberopro/envs/objects/articulated_objects.py:FlatStove`:
- `default_turnon_ranges = [0.5, 2.1]` → TurnOn fires when `qpos >= 0.5`
- `default_turnoff_ranges = [-0.005, 0.0]` → TurnOff fires when `qpos < max(turnoff_ranges) = 0.0` **strictly less than**

Joint range is `[-0.005, 2.1]` (XML). At reset qpos=0.0 — the stove is "off" visually but the TurnOff *predicate* doesn't fire because 0 is NOT strictly < 0.

**How to apply**:
- `Turnon(stove)` → physically rotate the knob to `qpos >= 0.5`. Verified 2026-05-26:
  `pi0_doubled` with prompt "turn on the stove" (~15 chunks, set `gripper_closed_thresh=0`
  so the pick-success break never fires) physically turns the knob — the burner glows red
  and `(TurnOn stove)` fires once the moka is also placed. Used for libero_10 lan/swap/task
  t2 + t8 physics-only redo.
- `Turnoff(stove)` → physically rotate to `qpos < 0.0` (e.g. pi0_doubled "turn off the stove").
- The reset value 0.0 is the dead band between the two predicate thresholds — neither fires there.

**Related**: [[cook-region-offset]] [[no-teleport-rule]]
