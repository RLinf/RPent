---
name: bddl-ranges-table-frame
description: "BDDL `:ranges (...)` values are in TABLE-relative frame, not world; multiply by table_origin offset (typically -0.2 in x for kitchen/study) to get world coords, OR read from a sister-cell audit's final_state"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 234721d6-dc80-4ac9-806e-e06977ce7823
---

**Rule:** BDDL `(:ranges ((x_min y_min x_max y_max)))` blocks specify positions in the **table-local frame**, not world frame. For most libero scenes, the table is offset from world origin (kitchen_table / study_table at world (-0.20, 0, 0.83) approximately), so the world position of an `init_region` is:

```
world_xy = table_origin_xy + bddl_ranges_center_xy
```

For kitchen_table / study_table, `table_origin_xy ≈ (-0.20, 0)`. So a BDDL region at `:ranges ((-0.21, -0.15, -0.19, -0.13))` has center `(-0.20, -0.14)` in table-frame → world `(-0.40, -0.14)`.

**Why:** lan t5 (book → caddy back compartment, 2026-05-22) — I placed the book at `(-0.20, -0.14, 0.92)` (table-frame BDDL coord) and predicate never fired. Switched to `(-0.407, -0.142, 1.05)` (world coord from swap_t5 audit's final_state) — solved in 1 call.

**How to apply:** Before targeting a predicate site from BDDL `:ranges`:
1. **Best**: read the sister-cell audit (task/swap/lan version of same task) in `workspace_pro/results_*/`'s `final_state.objects` — that's the ground-truth world coord where the predicate fired.
2. **Fallback**: add the table_origin offset (typically -0.20 in x for kitchen/study scenes; 0 for living_room which is at world origin) to the BDDL range center.
3. **Verify by manipulation**: physically place the object at your candidate coord (pick + OSC carry + release) and check if the predicate fires; if not, adjust the target xy and retry. (There is no teleport probe — set_object_pose is removed; see [[no-teleport-rule]].)

This also explains why fixture `(:target kitchen_table)` regions in libero_10 swap BDDLs map to ~+0.20m further from robot in world frame than the bare BDDL coords suggest.

Related: [[swap-perturbs-fixtures]]
