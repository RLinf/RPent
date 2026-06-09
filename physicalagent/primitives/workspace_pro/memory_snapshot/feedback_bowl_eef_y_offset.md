---
name: bowl-eef-y-offset
description: Pi0 rim-hooks a bowl with ~4.5cm offset in eef-frame -y; compensate eef y target = predicate_y + 0.045 for bowls
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

**Rule**: When placing a bowl into a target predicate site (On/In with ≤3cm xy tolerance), set `eef_y_target = predicate_y + 0.045` (m). The bowl hangs ~4.5cm in eef-frame -y direction after a Pi0 rim-hook grasp.

**Why**: Pi0's grasp on a small round bowl is a rim-hook (one finger pressing on the rim, bowl body offset to the side of the gripper). The bowl center sits ~4.5cm in -y from the eef center. Without compensation, the bowl drops 4cm short of the predicate site even though the eef hits its target exactly.

Discovered 2026-05-21 by comparing libero_goal t3 task PASS (cream_cheese, centered box grasp) vs t3 base FAIL (bowl ended up at xy (0.035, -0.135) when eef targeted (0.027, -0.091); delta y = -0.044 → bowl is 4.4cm in -y of eef).

**How to apply**:
- For `On(bowl, cook_region)` on stove: eef y = cook_region_y + 0.042 → 0.252.
- For `In(bowl, drawer_top_region)`: eef y = drawer_y + 0.045 → -0.046.
- For boxes (cream_cheese), cylinders (alphabet_soup), plates: no offset needed — those grasp centered.
- The offset can be empirically refined per object: subtract `(post_release_bowl_y - eef_target_y)` from your eef y target.

**Related**: [[cook-region-offset]] [[pi0-false-positive-lift]]
