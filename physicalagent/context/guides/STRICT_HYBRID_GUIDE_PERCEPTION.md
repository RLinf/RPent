# Strict Hybrid LLM + Pi0.5 — Perception-Isolated Guide

You are taking over a hybrid LIBERO experiment in **perception-isolated** mode.
This is the **sibling** of `STRICT_HYBRID_GUIDE.md` (the oracle-state guide).
If you read only the oracle one you will accidentally use GT object
coordinates — which is not allowed in this mode.

> **Pi0.5 only does the grasp (`pi0_pick`). The LLM (you) handles every motion
> (`move_to`), every release, sequencing, retries — and you do not get GT
> object coordinates. You localize objects yourself from the depth + camera
> calibration the driver dumps each step.**

## What's different from oracle mode (read this first)

| | oracle (`STRICT_HYBRID_GUIDE.md`) | **perception (this guide)** |
|---|---|---|
| driver flags | (none) | **`--hide_object_coords --always_render`** |
| `states.json` entry's state | full `objects:{name:[x,y,z]}` | **`object_names:[…]` only — NO coords** |
| extra obs files | `images/image_NN.png` only | **+ `images_cam/image_cam_NN.png`, `depths/depth_NN.npy`, `camera_meta.json`** |
| how you get an object's xyz | read `state["objects"][name]` | **detect the object pixel in `images_cam/image_cam_NN.png`, read depth at that pixel, back-project with `camera_meta.json`** |
| `pi0_pick` `track_obj` arg | name from state's `objects` keys | name from `state["object_names"]` (the names list is still given; only positions are stripped) |
| cell timeout | 600 (short suites) | **1200** — perceptual localization + manipulation is slower |
| audit `regime` | `strict` | `strict_perception` |

How localization works (the core of this mode):

1. `Read` `images_cam/image_cam_NN.png` — the agentview RGB **in the calibration frame**
   (vertical-flip of the raw buffer). This is the image you pick pixels in.
   `images/image_NN.png` is the Pi0-rotation frame; **do not use it for back-projection**.
2. Find the target object visually → pixel `(row, col)`.
3. Read the metric depth at that pixel from `depths/depth_NN.npy`.
4. Back-project with `camera_meta.json`:
   `P_world = extrinsic_cam2world @ [col·z, row·z, z, 1]; z = depth[row,col]`.
   That gives you the object's surface point in world frame.

**Verified 5/5 vs GT** at protocol design time (plate Δ=6 mm, cookies Δ=14 mm) —
the back-projection is trustworthy. The driver `dump_state` writes `depths/depth_NN.npy`
+ `camera_meta.json` + `images_cam/image_cam_NN.png` every step (always — the perception
interface is unconditional once the driver is launched).

## Before you start: READ THE AUTO-MEMORY

Operating wisdom lives in the in-repo snapshot:

```
physicalagent/context/memory/MEMORY.md
```

Scan the ~40 one-line hooks. For perception cells **always** open:
- `feedback_no_teleport_rule.md` — the deleted primitives.
- `feedback_redo_cell_timeout_1200.md` — why CELL_TIMEOUT_S is 1200 here.
- `project_perception_isolated_protocol.md` — the protocol summary + the
  known oracle leak (`pi0_pick`'s `track_obj` still reads GT z internally for
  grasp early-cut / success validation — *localization* is perception-only,
  the *grasp* is not yet fully oracle-free).

For bowl→plate spatial tasks always also read `feedback_bowl_eef_y_offset.md`
(bowl-eef y-offset 4.5 cm: place at `eef_y = plate_y + 0.045`, not `plate_y`).

## Rule 1 — `pi0_end_to_end` is FORBIDDEN

Same as oracle mode. `pi0_pick` is for the grasp only. You script every
`move_to` and every `release`. Use `track_obj_lift_thresh` to make Pi0 exit
the moment the object lifts.

## Rule 4 — NO TELEPORT primitives (physics-only)

`set_object_pose`, `articulate_to`, `js_move_to`, `carry_object` are **deleted
from the codebase**. They are not callable. If a goal is past OSC reach and
no physical approach works, write an honest `libero_terminated:false` audit —
never warp.

## Rule 0 — Use images for reasoning, not just JSON state

After every command, `Read` the new `images_cam/image_cam_NN.png` (`Read` renders PNG
visually in Claude Code). Even more important here than in oracle mode: this
is *the* signal you use to find objects.

`images_cam/image_cam_NN.png` is the calibration-frame RGB — same scene as
`images/image_NN.png` but vertically flipped so that pixel coordinates align with
the camera matrices in `camera_meta.json`. Pick object pixels from this one.

## Rule 2 — Multi-episode iteration is allowed

Same as oracle mode. A failed pick → re-pre-position, retry Pi0 with the
next rung of the prompt ladder (see Rule 3). Reset between (task, seed)
runs.

## Rule 5 — Assume every task is physically solvable

Same as oracle mode. A localization that "looks right" but moves the gripper
into thin air usually means your pixel was on a wrong surface (e.g. picked
the bowl's reflection on the table). Re-look at `images_cam/image_cam_NN.png`, pick a
different pixel firmly on the target's top, re-back-project. Don't conclude
"unreachable" until you've validated localization.

## Mental model

1. **One Python process runs forever.** Pi0.5 + LIBERO sim, waiting for a command.
2. **You write commands.** Each command is a JSON file at `{workdir}/command.json`.
   The driver consumes it, runs one primitive, appends a step entry to
   `states.json` and writes `images/image_NN.png` + **`images_cam/image_cam_NN.png`** +
   **`depths/depth_NN.npy`** + (once at step 00) **`camera_meta.json`**. Then it
   blocks for the next command.
3. **You read, localize via back-projection, decide next move, write next command.**

## Launch a session

### One-click (recommended)

```bash
bash scripts/libero/run_perception_cell.sh \
     SUITE TASK SEED

# examples:
bash .../run_perception_cell.sh libero_object_swap 2 0    # PRO object swap t2 s0
bash .../run_perception_cell.sh libero_goal_task   5 0    # PRO goal task t5 s0
bash .../run_perception_cell.sh libero_10_task     5 0    # PRO libero_10 task t5 s0 (auto MAX_EPISODE_STEPS=5000)
bash .../run_perception_cell.sh libero_spatial     3 0    # standard libero_spatial t3 s0
```

`run_perception_cell.sh` bakes in the three things this mode needs (perception
prompt, `--hide_object_coords --always_render`, `CELL_TIMEOUT_S=1200`) and
auto-routes `LIBERO_TYPE=pro` for `*_swap/_task/_lan` suites. Override any knob
via env: `CUDA_DEVICE=2 MODEL=claude-opus-4-7 OUTPUT_DIR=/path bash run_perception_cell.sh …`.

### Raw driver launch (if you want to drive the REPL yourself)

```bash
cd ${PHYSICALAGENT_REPO_ROOT:-$(pwd)}
REPL_WORKDIR="${PHYSICALAGENT_WORKDIR_PREFIX:-$(python - <<'PY'
from physicalagent.config import get_default_workdir_prefix
print(get_default_workdir_prefix())
PY
)}"
CUDA_VISIBLE_DEVICES=0 LIBERO_TYPE=pro MUJOCO_GL=egl \
  python \
    physicalagent/backends/rlinf/repl_driver.py \
    --suite libero_object_swap --task 0 --seed 0 \
    --workdir $REPL_WORKDIR \
    --max_episode_steps 600 \
    --hide_object_coords --always_render
```

> Mandatory flags for this mode: **`--hide_object_coords`** strips `objects`
> from each `states.json` entry's state (keeps `object_names` + proprioception);
> **`--always_render`** keeps the depth + camera observables fresh after every
> OSC primitive (the render-skip optimisation in oracle mode leaves depth stale
> after `move_to`).

Set `--max_episode_steps 5000` for libero_10 (long-horizon); 600 is fine
for libero_spatial / object / goal. `LIBERO_TYPE=pro` for PRO perturbation
suites; omit it (or set `standard`) for the base benchmark.

Run in background; wait for readiness:

```bash
until [ -f $REPL_WORKDIR/states.json ] && [ -s $REPL_WORKDIR/states.json ]; do sleep 5; done
```

## The perception files you read each step

| file | what's in it |
|---|---|
| `states.json[NN]` | `step_idx`, `libero_terminated`, `state.{robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, object_names, obj_of_interest}`, `command`, `result`, `elapsed_s`. **No object coordinates.** |
| `images/image_NN.png` | RGB in Pi0 frame (180° rotated). *Do not pick pixels here for back-projection.* |
| `images_cam/image_cam_NN.png` | RGB in **calibration frame** (vertical flip). Pick object pixels HERE. |
| `depths/depth_NN.npy` | `(256, 256) float32` metric depth in meters, calibration frame. Same row/col as `images_cam/image_cam_NN.png`. |
| `camera_meta.json` | `intrinsic_K` (3×3), `extrinsic_cam2world` (4×4), `depth_near`, `depth_far`, plus a `projection` recipe in the file. Dumped once at step 00; cache it on disk read. |

## Localization snippet (run via Bash; substitute pixel + step)

```bash
python - <<'PY'
import json, numpy as np
wd = "$REPL_WORKDIR"; step = "01"; row, col = ROW, COL   # <-- fill in
cm = json.load(open(f"{wd}/camera_meta.json"))
E  = np.array(cm["extrinsic_cam2world"])
depth = np.load(f"{wd}/depths/depth_{step}.npy")
z = float(depth[row, col])
P = E @ np.array([col*z, row*z, z, 1.0])
print("world_xyz =", [round(float(v),3) for v in P[:3]], " depth_m=", round(z,3))
PY
```

Tips:

- Sample 3–5 pixels on the object (centre + a couple of edge pixels) and
  median the back-projected xy — robust to a single mis-picked pixel.
- The returned point is the **visible surface** under your chosen pixel.
  For a flat object (plate, basket) that surface ≈ the place target. For a
  bowl/bottle, the surface is the top of the object; the rim's xy is what
  you want for `release`, not the grasp's eef_y (apply
  `feedback_bowl_eef_y_offset` — bowl eef_y_target = perceived_plate_y + 0.045).
- For the **table z** (when you need a known floor to compare against),
  read a pixel on bare table near the object and back-project.

## The command vocabulary

Write JSON to `{workdir}/command.json`. The full primitive set (this matches
`STRICT_HYBRID_GUIDE.md` — only the *control signals* are listed here):

```jsonc
// === physics-only primitives (the entire allowed set) =====================

// Scripted EEF servo. action_scale=0.05 is the env's units; step_clip caps
// per-step Δxyz (m) BEFORE division by action_scale — smaller = slower.
{"action": "move_to", "xyz": [x, y, z], "gripper": -1|+1,
 "tol": 0.012, "step_clip": 0.02, "max_steps": 80, "action_scale": 0.05,
 "target_yaw": null}

// Pi0.5 closed-loop pick. track_obj is an OBJECT NAME (from state.object_names);
// track_obj_lift_thresh forces Pi0 to exit the moment the object lifts.
{"action": "pi0_pick", "prompt": "pick up the X",
 "max_chunks": 25,
 "track_obj": "akita_black_bowl_1",
 "track_obj_lift_thresh": 0.07,
 "lift_thresh": 0.05, "gripper_closed_thresh": 0.06}

// Pi0.5 for a contact skill (knob turn, drawer open/close — rare here).
{"action": "pi0_doubled", "prompt": "turn off the stove", "max_chunks": 20}

// Open gripper to place. Triggers libero termination if "On"/"In" predicate met.
{"action": "release", "max_steps": 20}

// Hold pose + drive gripper. Use to firm a grip mid-carry.
{"action": "set_gripper", "gripper": -1|+1, "steps": 5}

// Wrist yaw (world-z). Bug-fixed (atan2-based).
{"action": "rotate_wrist", "target_yaw": 0.0, "max_steps": 40, "step_clip": 0.10}

// Tilt eef pitch (axis-angle X). Used for cavity entry / micro-aiming.
{"action": "rotate_pitch", "target_pitch": 0.9, "max_steps": 40, "step_clip": 0.10}

// Co-vary xyz + pitch + yaw — threads cabinet-front IK singularity.
{"action": "move_pose", "xyz": [x,y,z], "target_pitch": 0.0, "target_yaw": 0.0,
 "step_clip": 0.02, "pitch_step": 0.08, "yaw_step": 0.08,
 "tol": 0.012, "ori_tol": 0.05, "max_steps": 150}

// Reset env (between episodes; NOT allowed mid-recipe per Rule 4 in the eval harness).
{"action": "reset"}

// === FORBIDDEN — DELETED FROM THE CODE — DO NOT EMIT ============
//   set_object_pose, articulate_to, js_move_to, carry_object
```

## The strict-hybrid recipe (perception variant)

A typical bowl→plate cell looks like:

1. Read `states.json[0]`, `images_cam/image_cam_00.png`, `camera_meta.json`.
2. Identify the target object name from `task language` (the `obj_of_interest`
   key is often `null`; lean on `state.object_names` + the task instruction).
3. Localize the target object — pixel in `images_cam/image_cam_00.png` → `depth[row,col]`
   → back-project → world xyz.
4. **Pre-position** above it: `move_to [obj_x, obj_y, carry_z]`, gripper open.
5. `pi0_pick` with `track_obj=<obj_name>` and the right prompt (start with
   "pick up the {object}", escalate per Pi0 ladder — see below).
6. `set_gripper +1, 5` to firm the grip.
7. Localize the placement region (basket / plate / drawer slot) the same way.
8. `move_to [place_x, place_y, carry_z]` to traverse at constant height.
9. Optionally descend `move_to [place_x, place_y, place_z]`.
10. `release` — predicate (`On`/`In`) checks → `libero_terminated=True` if hit.
11. Light retreat (`move_to` upward) so the next step's image is clean.

> **Predicate fire timing.** Most LIBERO `On(X, Y)` predicates fire on
> `release` if `X` is above `Y`'s region. `In(X, container)` needs `X` to
> have actually entered the container's volume before release. If a release
> fires `term=False` but the object is on top of the right region, descend
> 1–2 cm more and re-`release`.

## Pi0 prompt ladder (Rule 3 — Pi0 IS the delivery service)

Try in order; each rung uses a slightly more specific prompt or a re-pre-pos.

1. `"pick up the {object}"` — generic sub-instruction.
2. Full BDDL task language (e.g. `"Pick the akita black bowl on the cookies
   box and place it on the plate"`).
3. Add a spatial qualifier (`"…on the wooden cabinet"`, `"…next to the
   basket"`).
4. Re-pre-position 5 cm lower or shifted, then retry rung 2 or 3.

Empirically Pi0 sometimes needs the **full prompt with the spatial
qualifier** for elevated picks (stove, cabinet-top, drawer). See
`feedback_pi0_pick_full_prompt.md` and the prompt-ladder note in MEMORY.

## Key hyperparameters

- Single-step `xyz` within ±0.30 m of current eef or OSC flips IK. Split
  long traversals into 2–3 carry-z waypoints.
- `track_obj_lift_thresh`: 0.05 (flat/stable) / 0.08 (slippery tall bottles).
- `step_clip`: 0.025 (empty / box) / 0.015 (cans) / 0.012 (tall bottles).
- Frame z (from `state.robot0_eef_pos[2]` at step 00):
  ≈ 0.68 → LIVING_ROOM, ≈ 1.17 → KITCHEN, ≈ 0.26 → OBJECT.
- BOWL: `eef_y_target = perceived_plate_y + 0.045` (bowl-eef y-offset).
- TALL BOTTLES: carry at `z=0.30`, release without descending.
- Approach high-then-vertical; recover by re-`pi0_pick`, not by hovering.

## Reading state / log files

After every command:
1. `Read {workdir}/states.json` (jump to entry NN) → check `result.success`,
   `final_dist_m`, `peak_lift_m`, eef pose, gripper width, `libero_terminated`.
2. `Read {workdir}/images_cam/image_cam_NN.png` → visual confirmation;
   pick pixels for any new localization.

You **do not** open `depths/depth_NN.npy` interactively — feed it to the
localization snippet above.

## Common failure modes

- **Pick missed (gripper closed empty).** `log[result].peak_lift_m` < `lift_thresh`,
  `min_gripper_opening` ≈ 0. Re-pre-position 1–2 cm lower or shifted; retry
  Pi0 with next prompt-ladder rung.
- **Object slipped mid-carry.** `release` returns `term=False` and the object
  is no longer where you `release`d it. `release`, re-pre-pos above it,
  `pi0_pick` again, traverse again.
- **OSC stuck.** `move_to` returns `final_dist_m > 0.05` at `max_steps` and
  same xy twice → try `rotate_pitch`, split into more waypoints, or
  `move_pose` (co-varying) for cabinet-front singularity.
- **Placement off because localization was wrong.** The release puts the
  object on bare table instead of on the plate. Re-`Read` the latest
  `images_cam/image_cam_NN.png`, pick a different pixel on the target region
  (sample 3 pixels on the plate's flat top, median the back-projected xy),
  redo. Tip: if depth at your chosen pixel is much closer than the table z,
  you picked the camera's near edge / an object rim — pick again.

## Verifying strict compliance

Before saving the audit, run this check on your recipe-so-far:

```bash
python -c "import json,re; s=json.load(open('$REPL_WORKDIR/states.json')); bad=re.compile(r'set_object_pose|articulate_to|js_move_to|carry_object'); hits=[e['command'].get('action') for e in s if e.get('command') and bad.search(e['command'].get('action',''))]; print('TELEPORT — REJECTED' if hits else 'physics-only ✓')"
```

## Persisting successful runs as audit JSONs

When `state.libero_terminated == true`:

a. Write the working command sequence (copy the `command` field of each
   step entry in `states.json` in order) to
   `{OUTPUT_DIR}/recipe_{TAG}.jsonl` — one JSON per line, no `note` field.
b. Write a minimal audit JSON to `{OUTPUT_DIR}/{TAG}.json` with at least:
   `suite`, `task_id`, `seed`, `regime: "strict_perception"`, `strategy_notes`
   (mention HOW you localized — which pixel, depth, back-projected world xyz),
   `pick_result` (the `result` from your `pi0_pick` step in `states.json`),
   `final_state` (latest `states.json` entry's `state` field),
   `libero_terminated: true`.
c. Stop.

If unrecoverable after honest exploration, write
`{OUTPUT_DIR}/{TAG}.json` with `libero_terminated: false` + `strategy_notes`
describing what you tried, the back-projected xyz you used, and which step
failed. Stop.

> The `regime: "strict_perception"` tag distinguishes these audits from the
> oracle-state `strict` regime in mixed-mode datasets.

## Iteration heuristics

- After 2 failed retries on the same step, **stop tuning numerics** and
  inspect images: open `images_cam/image_cam_NN.png` at pick / pre-release / post-release.
  The visual disagreement is usually the bug (you picked the wrong pixel, or
  Pi0 grabbed the decoy). This is the lesson of
  `feedback_failure_forensics.md` — applies even more strongly here.
- If a release reports `term=False` but the object is visibly on the target
  region, descend 1–2 cm more and `release` again; predicate often needs
  contact, not just hover.
- Trust the back-projection (verified 5/5 vs GT). If a localization "feels
  off", it's because you picked the wrong pixel — not because the depth /
  calibration is wrong.

## What "strict_perception" means concretely

- **No GT object coordinates anywhere in your reasoning.** The `state` you
  read has none; the only legitimate sources of object xyz are
  `images_cam/image_cam_NN.png` + `depths/depth_NN.npy` + `camera_meta.json`.
- **No teleport primitives.** The four are deleted.
- **Pi0 only does the grasp.** You script every motion + release.
- **`track_obj` still receives a NAME from `state.object_names`** — names are
  not coordinates. (Internally `pi0_pick` does still read the GT object z to
  decide when to early-cut on lift — that is a known, documented residual
  oracle dependence in the *grasp* primitive; the *localization* you do is
  perception-only. See `project_perception_isolated_protocol.md`.)
- The expected audit `regime` is `strict_perception`.

## Reference cases from prior sessions

- **libero_object PRO swap+task (200 cells, perception-isolated)** → 198/200
  solved, 100% completed-cell solve, 0 honest failures. The pattern is
  consistent: localize-pre-pos-pi0_pick-set_gripper-move-release in 6–12
  commands per cell. See `multi_seed_exp/percep_object_*` for s0 recipes.
- **goal swap t2 / task t9** (in the 33-cell highlight in
  `result_paper/hybrid_primitive_usage.md`) — solved physics-only with a
  single grasp + carry + release; Pi0 fullshot can't complete either,
  hybrid does.

When you write a new audit, browse `multi_seed_exp/percep_*/recipe_*_s0.jsonl`
for a sibling cell's working sequence as a template — but never paste its
xyz, re-derive every position via back-projection from THIS scene's depth.

Begin by reading MEMORY.md, then `states.json[0] + images_cam/image_cam_00.png + camera_meta`,
localize the target object via back-projection, then plan and execute.
