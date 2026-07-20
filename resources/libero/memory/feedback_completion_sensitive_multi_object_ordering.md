---
name: completion-sensitive-multi-object-ordering
description: Choose the first object in a shared-target sequence so partial completion does not hide the remaining object, then carry later objects above the occupied target.
metadata:
  node_type: memory
  type: feedback
---

# Completion-sensitive multi-object ordering

## Rule

For instructions that place multiple similar objects into one shared target region, choose the object order before execution. A closed-loop policy may behave as though the instruction is complete after one matching object reaches the target, leaving the remaining object unattended. Place the object most likely to be ignored or become difficult to reach after partial completion first.

The later carry must also clear the object already at the target. Use separate target footprints and a carry altitude above the occupied-object collision envelope.

## How to apply

1. Identify every required object and the shared target from the current observation.
2. Check which object is more vulnerable to policy ambiguity, reach limits, or obstruction after the first placement.
3. Handle that completion-sensitive or reach-sensitive object first.
4. Allocate separated landing positions inside the target instead of reusing one center point.
5. Carry each later object high enough that the wrist and held object cannot brush the object already placed.

Object identifiers do not define a universal order. Re-evaluate the current layout when one object begins near a reach boundary or along the later carry corridor.

## Reference case

For two moka pots sharing one stove region, the pot labeled `moka_2` was the nearer pot at initial `y ~= +0.04` and had to be handled first in the reference layout. When `moka_1` was placed first, the second closed-loop grasp call hovered without closing around `moka_2`, behaving as though the visible pot on the stove already satisfied the instruction.

Use opposite-corner placements within the stove region. The reference placement used an end-effector drop height of `z = 1.075-1.078` with `step_clip=0.01`. A high carry at `z ~= 1.30` kept the wrist clear of the first pot; at `z ~= 1.20`, the wrist brushed the placed pot and knocked it off the stove.

These object labels and heights are reference-layout values, not universal constants. The reusable rules are to protect the completion-sensitive object with ordering, separate the target footprints, and clear the placed-object collision envelope during later carries.

## Related memories

- [[dual_moka_obstacle_aware_stove_placement]] - a geometry-specific manual placement pattern with its own scene-dependent order and front-arc carry.
- [[feedback_read_image_before_decide]] - inspect the current layout before selecting object order and target footprints.
- [[feedback_pi0_pre_pos_can_hurt]] - choose the policy starting pose from current clutter and target visibility.
