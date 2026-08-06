# Strict Hybrid LLM + Pi0.5 Perception Guide

This guide describes the perception-isolated LIBERO runtime. The agent receives
robot proprioception, object names, task language, and camera observations. It
does not receive privileged object coordinates.

Pi0.5 performs grasping through `pi0_pick`. The LLM owns semantic perception,
localization, scripted motion, release, verification, and recovery within the
current episode.

## Runtime Contract

Every motion primitive appends a `StepRecord`. The record contains:

- `step_idx`: sequential motion-step number;
- `state`: robot proprioception and coordinate-free scene information;
- `artifacts`: sorted logical base names for all files captured at the step;
- `command`, `result`, and `elapsed_s`;
- `extras`: LIBERO outcome fields such as `task_language`,
  `libero_terminated`, and `episode_truncated`.

Artifact storage is private to `EnvState`. Do not construct filenames or read
observation files manually. Use the structured tools:

- `view_env_state`: state, artifact names, log, and embedded images;
- `view_camera_meta`: calibration data for one camera and step;
- `back_project`: matching image pixel or region to world coordinates;
- `segment`: text- or point-prompted mask plus projected `world_xyz`.

Step `0` is the initial observation. Step `-1` means the latest observation.
When a tool omits `step`, its schema default is `-1`.

`states.json` is an internal versioned manifest. It is not an agent-facing state
API and should not be indexed directly.

## Camera Roles

`view_env_state` embeds the best available images for the selected step:

- **policy image**: the Pi0-oriented agentview input;
- **agentview image**: fixed global view, high resolution when available;
- **wrist image**: eye-in-hand view, high resolution when available.

Use agentview for semantic identity and global relationships. Use wrist for
close-range geometry after the target identity has already been anchored in
agentview. The wrist view must not freely replace the semantic decision when
similar objects are nearby.

Pixels passed to `back_project` must come from the same camera and resolution
specified in the call. The tool selects the matching world map internally.

## Initial Perception Pass

Before the first manipulation primitive:

1. Call `view_env_state({"step": 0})`.
2. Read the returned top-level `task_language` verbatim.
3. Identify every movable target, destination, support, and relation landmark
   named by the task.
4. Inspect the embedded agentview image and classify candidates by color,
   shape, label, container type, and spatial relation.
5. Localize each chosen candidate with several interior pixels and
   `back_project`, or use `segment` when a stable text/point prompt exists.
6. Record a working table with semantic evidence, sampled pixels, projected
   coordinates, uncertainty, and the selected final coordinate.

Do not start manipulation until every required target and destination has a
defensible identity and coordinate estimate.

## Semantic Identity Before Geometry

Depth and world coordinates cannot distinguish visually similar surfaces. A
plate, stove burner, pot lid, and cabinet top can all be flat circular or planar
regions at similar heights.

Classify the destination in RGB before localizing it:

- **plate**: ceramic disc with a clean rim, often white or ringed;
- **stove region**: darker fixture surface, coil, grate, or burner pattern;
- **basket**: open container whose interior center differs from the rim;
- **cabinet or drawer**: fixture geometry associated with the task noun.

When duplicate objects exist, choose by the relation in `task_language`, not by
internal object suffixes.

## Coarse-to-Fine Localization

For each non-basket object:

1. Select the semantic candidate in agentview.
2. Project three to eight interior pixels and take a robust median.
3. Move approximately 15-20 cm above the agentview anchor.
4. Inspect the wrist image and refine the same physical candidate.
5. Accept the wrist estimate only when it remains within roughly 3-5 cm of the
   agentview anchor. Otherwise reject it and retain the global estimate.

For baskets and cavities, use agentview for global identity and wrist for the
interior center. Avoid rim pixels. `back_project` region mode can summarize a
bounded pixel window with optional `z_min` and `z_max` filtering.

## Segmentation

`segment` does not move the robot. Supply exactly one of:

```json
{"prompt": "the black bowl on the stove", "camera": "agentview", "step": -1}
```

```json
{"point": [420, 615], "camera": "agentview", "step": -1}
```

The result includes the mask score, projected `world_xyz`, logical artifact
identifiers for audit, and an embedded overlay image when available. Inspect
the overlay before trusting the projection.

Use plain visual phrases. Remove benchmark or brand names that the segmenter
cannot ground. If segmentation fails, choose pixels manually and call
`back_project`.

## Grasping

Pre-position above the localized object before calling Pi0.5:

```json
{
  "prompt": "pick up the short red-label can",
  "max_chunks": 20,
  "lift_thresh": 0.05,
  "gripper_closed_thresh": 0.06
}
```

Treat `pi0_pick.success` as a hint. Verify the grasp from:

- end-effector lift;
- nonzero gripper opening consistent with holding an object;
- wrist or agentview evidence that the correct object moved with the gripper;
- an empty or changed source location.

If the grasp failed, recover in the same episode by re-localizing,
re-positioning, and improving the visual prompt. Do not reset in single-attempt
evaluation mode.

## Scripted Motion

Use short, staged `move_to` commands. Do not issue a single horizontal move
larger than approximately 0.30 m. Long commands can switch IK branches and move
the end effector into the wrong half-space.

After every motion:

1. Inspect the returned `result` and final distance.
2. Inspect the new state and embedded images.
3. Re-localize anything that may have moved.
4. Continue only when the observed state matches the plan.

Use `rotate_wrist`, `rotate_pitch`, or `move_pose` when orientation matters.
Use `pi0_doubled` for short learned contact interactions such as knobs,
buttons, doors, or drawers. Alternate it with short, capped scripted alignment
motions; never use one long blind push.

## Placement

Carry objects at a collision-safe height. Reconfirm the destination before
release, especially when plate/burner or rim/interior confusion is possible.

For open containers:

- localize the interior rather than the nearest rim;
- place above the interior center;
- lower enough to avoid a high-energy drop;
- inspect the post-release observation and outcome flag.

If the task predicate does not fire, reclassify the destination before assuming
the grasp failed. Wrong-surface placement is a common cause.

## Completion And Audit

The current outcome is the top-level `libero_terminated` value returned by the
latest environment tool. Do not look for it inside `state`.

On completion, write the requested audit with:

- suite, task, seed, and evaluation regime;
- exact memory files consulted;
- semantic identification and localization strategy;
- grasp and placement evidence;
- final state and `libero_terminated`;
- honest failure details when the task remains incomplete.

Then call `finish`. Recipe generation is handled by the runtime from recorded
primitive and successful segmentation events.

## Final Checklist

- Initial state read with `step: 0`.
- Task language copied from the returned tool result.
- All targets and destinations semantically identified in agentview.
- Coordinates obtained through `back_project` or `segment`.
- Wrist refinements checked against the agentview anchor.
- Grasp confirmed visually and proprioceptively.
- Long motion split into safe waypoints.
- Destination reclassified immediately before release.
- Latest result checked for top-level `libero_terminated`.
- Audit written before `finish`.
