---
name: swap-perturbs-fixtures
description: "libero_goal P2 swap perturbation swaps FIXTURE positions (stove ↔ cabinet ↔ wine_rack), not just object positions — must recompute predicate site world coordinates per swap BDDL"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

**Rule**: For `libero_goal_swap` cells where the goal predicate references a fixture site (`flat_stove_1_cook_region`, `wooden_cabinet_1_top_side`, `wine_rack_1_top_region`), **READ the swap BDDL `:init` block** to find where the fixture was placed. The swap moves entire fixtures across the table, so the target site world coordinates differ completely from the base layout.

**Why**: I spent 6+ iterations on t1/t2/t4/t9 swap assuming the stove/cabinet/rack stayed at their base positions. The Pi0 home-pose pick worked every time, but carry to the "known" cook_region xy (-0.26, 0.21) put the bowl on the CABINET (which the swap had moved to where the stove normally is). Discovered when grep-ing the swap BDDL and seeing:
```
(On wooden_cabinet_1 main_table_stove_region)   ← cabinet at base stove location
(On flat_stove_1 main_table_cabinet_region)     ← stove at base cabinet location
```

**Site world coordinates per t1/t2/t4/t9 swap layout (verified 2026-05-21)**:
- **t1 swap** (`flat_stove_1` → cabinet_region with yaw=π): cook_region world = **(-0.12, -0.24, ~0.92)** (stove origin (0.03, -0.24) + yaw=π × burner offset (+0.15, 0) = -0.15 in world x)
- **t2 swap** (`wooden_cabinet_1` → wine_rack_region with yaw=π): top_side world = **(-0.264, -0.276, ~1.12)** (rack center (-0.26, -0.26) + yaw=π × top_side body-local (0.004, 0.016, 0.22))
- **t4 swap**: same as t2 (cabinet at wine_rack_region) → top_side world = (-0.264, -0.276, ~1.12)
- **t9 swap** (`wine_rack_1` → cabinet_region with yaw=π): top_region world = **(0.03, -0.16, ~1.14)** (cabinet center (0.03, -0.24) + yaw=π × site body-local (0, -0.08, 0.22))

**How to apply**:
1. For any swap-perturbation cell, read the BDDL `:init` block to find which fixture each region maps to.
2. Recompute the predicate site world center using the new fixture position + yaw + body-local site offset.
3. Carry target = site_world - (obj-eef offset read from post-pick state).
4. Carry at z=1.30, descent at z=1.20-1.22 (cabinet top/rack), z=1.04 (stove cook_region).
5. Release at descent, then **retreat with gripper open** — the predicate sometimes fires DURING the retreat as the object settles (not at the release step itself).

**Related**: [[cook-region-offset]] [[bowl-eef-y-offset]] [[pi0-pre-pos-can-hurt]]
