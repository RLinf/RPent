---
name: cook-region-offset
description: "stove cook_region is offset (+0.15 m, 0, 0) from stove fixture origin via burner sub-body — don't aim at the stove_region BDDL center"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

When placing on `flat_stove_1_cook_region` (e.g. "put bowl on stove"):

**Rule**: target eef xy = `(stove_region_center_x + 0.15, stove_region_center_y)`, NOT the stove_region center itself.

**Why**: `flat_stove.xml` structure is `base → burner (pos=0.15 0 0) → cook_region site (pos=0 0 0)`. The cook_region site is centered at the burner sub-body, which is offset +0.15 m in stove-local x from the fixture origin. For LIBERO scenes that place the stove with default yaw=0, this +0.15 carries through to world coords. With yaw=π it flips to -0.15.

**How to apply**:
- libero_goal: stove_region in BDDL at world (-0.41, 0.21) → cook_region center at world **(-0.26, 0.21)** (verified by t1 release at this xy → predicate triggers).
- libero_10 t2: stove_init_region at (-0.21, 0.20) → cook_region at (-0.06, 0.20) (matches the existing recipe target -0.067, 0.20).
- Also add a +0.04 m y compensation for bowl-eef offset: eef y target = cook_region_y + 0.042, so bowl xy lands at cook_region center.
- Cook_region site half-extent is 0.075 (15 cm box). Bowl center must land within ±0.07 m of cook_region center AND `check_contact(stove_body, bowl)` must register — bowl dropped from height (descent stall) lands on burner rim and may evaluate False even within xy bounds.

**Related**: [[libero-10-t0-t5-pro]] [[liberopro-driver-patch]]

A 2-hour t1 failure debug (releasing bowl at x=-0.40..-0.45 → predicate False even with bowl visually on stove) was solved by reading `flat_stove.xml` and seeing the burner offset.
