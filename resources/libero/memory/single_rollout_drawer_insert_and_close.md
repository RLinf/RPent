---
name: single-rollout-drawer-insert-and-close
description: Reusable articulated-container pattern that completes opening, insertion, and closure in one closed-loop rollout.
aliases:
  - put the black bowl in the bottom drawer of the cabinet and close it
  - black bowl bottom drawer close
metadata:
  node_type: memory
  type: skill
---

# Single-rollout drawer insert and close

## Applicable pattern

Use this pattern when a compact object must be placed in a drawer and the complete open, grasp, transport, insert, and close sequence can be handled by one closed-loop instruction.

This pattern is not intended for long objects that require explicit reorientation, geometry-aware insertion, or a separate contact strategy. For that case, use [[horizontal_bottle_drawer_insertion_and_close]].

## Canonical execution

| Field | Setting |
|---|---|
| Initial state | Clean task state |
| Instruction | Complete task language from `view_driver_state({"step":0}).task_language` |
| Action | One `pi0_doubled` call |
| `max_chunks` | `600` |
| Manual pre-positioning | None |
| Manual primitives | None |
| Follow-up close call | None |
| Required sequence | Open, grasp, transport, insert, and close in the same rollout |

Issue the complete task language verbatim. Keep the scene unchanged before the call and let the closed-loop policy ground the target object, distractors, and drawer from the current observations.

## Execution rules

- Use `pi0_doubled`, not `pi0_pick` followed by manual carry or closure.
- Keep the full task in one call; do not split placement and closure into separate calls.
- Do not replace the complete task language with a close-only follow-up instruction.
- Do not manually pre-position the arm before the call.
- Do not relocate distractor objects before the call.
- Do not append `set_gripper`, `rotate_pitch`, `move_pose`, or `move_to` as manual recovery actions.
- Do not copy state-dependent coordinates from another scene.
- Do not use short close pulses or an enlarged chunk budget as a separate fallback.

## Initial-state guards

These checks validate the starting state; they are not additional recipe inputs.

| Check | Required condition |
|---|---|
| `task_language` | Matches the active drawer insertion-and-close task |
| `libero_terminated` | `false` |
| Robot and scene state | Clean and not inherited from a previous rollout |

No separate segmentation, pixel target, or world-coordinate offset is required by this pattern.

## Failure patterns to avoid

| Avoid | Canonical alternative |
|---|---|
| `pi0_pick` followed by manual carry and closure | Use one full-task `pi0_doubled` call. |
| One placement call followed by a separate close call | Keep placement and closure in the same rollout. |
| A close-only instruction after the object is placed | Use the complete task language from the start. |
| Manual or tilted drawer closing | Leave closure inside the full closed-loop rollout. |
| Moving a distractor before execution | Preserve the clean scene configuration. |
| Manual pre-positioning or coordinate-based alignment | Let the policy localize from the current observations. |
| Manual primitives after the closed-loop call | Do not chain recovery primitives onto the resulting state. |
| A recipe that assumes the drawer starts open | Use the complete instruction so opening remains part of the sequence. |

## Difficulty

This is a difficult, contact-rich sequence involving opening, grasping, transport, insertion, and closure within one complete rollout.

## Related memories

- [[feedback_move_pose_covarying_reach]] - reach constraints that can limit manual drawer closure.
- [[feedback_no_pi0_end_to_end]] - cases where one closed-loop policy call should own a multi-stage contact sequence.
- [[feedback_pi0_delivery_service]] - using complete task language for bundled manipulation sequences.
- [[feedback_pi0_pre_pos_can_hurt]] - why manual pre-positioning can reduce closed-loop policy reliability.
- [[feedback_failure_forensics]] - separating recorded failure symptoms from unsupported failure explanations.
