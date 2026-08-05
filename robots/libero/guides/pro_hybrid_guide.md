# LIBERO-Pro Hybrid Perception Guide

This guide extends [strict_hybrid_guide.md](./strict_hybrid_guide.md) for the
LIBERO-Pro evaluation tracks. Read the strict guide first; its runtime contract,
localization discipline, and single-attempt rules apply unchanged.

LIBERO-Pro perturbs object placement, fixture placement, spatial relations, and
task language. The agent must solve the observed scene rather than replaying a
seed-specific command sequence.

## Start Of Run

Call:

```json
{"step": 0}
```

through `view_env_state`. From the returned tool result:

- read top-level `task_language` verbatim;
- inspect `state.robot0_eef_pos` to identify the scene frame;
- inspect object names only as a scene inventory, never as coordinates;
- inspect the embedded agentview image for semantic identity and relations;
- inspect the wrist image only when it contains useful close geometry.

Step `-1` means the latest record. Do not inspect or index `states.json`
directly; it is an internal manifest.

## Perturbation Axes

### Object perturbation

Objects move relative to their seed positions. Re-localize every target and
destination from the current images. Seed recipes remain useful for primitive
ordering, prompt wording, safe heights, and known failure modes, but never for
coordinates.

### Spatial perturbation

Relations such as left/right, on-top-of, or next-to may select a different
instance than in the reference scene. Identify the candidate satisfying the
current visual relation, then localize it with `back_project` or `segment`.

### Goal perturbation

The destination or requested interaction changes. Re-read `task_language` and
classify the destination semantically. Do not infer the goal from the suite,
task index, object list, or a sibling result file.

### Task-language perturbation

The authoritative instruction is the current top-level `task_language`. Do not
read BDDL files. They combine language with hidden initialization data and
therefore violate perception isolation.

## Scene Frame Selection

The initial end-effector height distinguishes the principal scene frames:

| Initial EEF z | Frame | Typical fixtures |
|---|---|---|
| approximately 0.26 m | object/low-table | grocery and basket tasks |
| approximately 0.68 m | living-room table | plates, baskets, pudding |
| approximately 1.17 m | kitchen table | stove, cabinet, drawer, microwave |

Use `view_env_state({"step": 0})["state"]["robot0_eef_pos"][2]` as the
measurement. Then use the matching safe-height guidance from
[env_calibration.md](./env_calibration.md).

## Mandatory Perception Table

Before manipulating, create one row for every task-relevant entity:

| field | meaning |
|---|---|
| role | target, destination, support, fixture, or relation landmark |
| semantic evidence | visual properties and relation establishing identity |
| agentview pixels | several interior pixels or a verified segment mask |
| agentview xyz | robust projection from the global camera |
| wrist refinement | accepted, rejected, basket-confirmed, or unnecessary |
| final xyz | coordinate used for planning |
| uncertainty | duplicates, occlusion, rim bias, label ambiguity, etc. |

Do not begin manipulation while a required row lacks a defensible identity or
coordinate.

## Swapped Objects And Fixtures

For swap-style perturbations:

1. Identify both swapped entities in the embedded agentview image.
2. Classify each by appearance and current relation, not expected seed layout.
3. Localize each independently.
4. Verify the chosen target still satisfies `task_language`.
5. Re-localize after any contact that could move either entity.

The runtime withholds privileged coordinates, so there is no coordinate field
to fall back on. The current images and geometry tools are the source of truth.

## Destination Classification

Pro scenes frequently contain look-alike surfaces. Before placement, explicitly
classify all plausible destinations:

- plate versus stove burner;
- cabinet top versus drawer opening;
- basket interior versus rim;
- microwave cavity versus door or surrounding counter;
- movable lid versus fixed fixture surface.

Only after semantic classification should you call `back_project` or `segment`
for coordinates.

## Mid-Carry Re-Localization

Long-horizon Pro tasks often move or occlude objects during earlier steps.
Before each new pick or placement:

1. call `view_env_state({"step": -1})` if the previous primitive result is
   no longer in context;
2. inspect the newest embedded images;
3. re-localize any entity that may have moved;
4. update the working perception table;
5. verify the remaining primitive order still matches `task_language`.

Never assume the initial coordinate remains valid after contact, release, or a
fixture interaction.

## Contact Tasks

Use `pi0_doubled` for learned contact behavior such as turning a knob or
opening/closing a drawer. Its success flag mirrors the benchmark termination
predicate, so an intermediate contact can be useful even when success is false.
Inspect state and image evidence after every contact attempt.

Use short scripted alignment motions around contact skills. Avoid long blind
pushes, which can destabilize MuJoCo or move the end effector into an invalid IK
branch.

## Planning Across Multiple Objects

For multi-object tasks:

- process objects in an order that minimizes collision and occlusion;
- preserve already-correct placements;
- route later carries around placed objects rather than over them;
- re-confirm each source and destination immediately before use;
- verify intermediate relations visually rather than assuming they held.

When order is specified by the task, obey it even if another order appears
easier.

## Reference Results And Memory

Reference results and memory entries provide reusable strategy:

- prompt ladders;
- manipulation ordering;
- safe approach and carry heights;
- fixture-specific contact patterns;
- known failure and recovery modes.

They do not provide valid coordinates for the current run. Re-derive all
positions through perception.

## Outcome And Audit

The current benchmark outcome is top-level `libero_terminated` in the latest
environment tool result. It is not stored inside `state`.

The audit should record:

- exact perturbed `task_language`;
- perturbation type when known;
- semantic identification evidence;
- agentview and accepted wrist localization results;
- primitive sequence and recovery decisions;
- memory files consulted;
- final state and `libero_terminated`.

Recipe export is runtime-managed from recorded primitives and successful
segmentation events. Artifact identifiers returned by tools are for audit and
traceability, not manual file access.

## Quick Checklist

- Read strict guide and relevant memory.
- Call `view_env_state({"step": 0})`.
- Select the scene frame from initial EEF z.
- Read top-level `task_language`.
- Build the complete perception table.
- Re-derive all coordinates for this scene.
- Use agentview for identity and wrist for consistent refinement.
- Re-localize after contacts and placements.
- Check top-level `libero_terminated` after each relevant action.
- Write an honest audit and call `finish` without resetting.
