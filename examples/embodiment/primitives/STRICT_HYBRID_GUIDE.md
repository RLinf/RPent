# Strict Hybrid LLM + Pi0.5 — Handover Guide

You are taking over a hybrid LIBERO experiment from a previous session. This
guide is everything you need to know to continue iterating on **strict** hybrid:

> **Pi0.5 only does `pick` (the gripper grasp). The LLM (you) handles
> everything else — motion planning (move_to), release, sequencing, retries,
> sanity-checking — by writing JSON commands to a REPL driver process and
> reading state/image dumps.**

## Before you start: READ THE AUTO-MEMORY

Past sessions have stored ~20 hard-won lessons in
`/root/.claude/projects/-mnt-public2-zhangyixian/memory/`. The index
file `MEMORY.md` is auto-loaded into your system prompt via CLAUDE.md,
so the one-line hooks are already visible to you. Before launching a
driver, **scan those hooks** for entries whose name/description touches
the cell you're about to attempt (search for the task name, object
type, perturbation suite, or symptom). Then `Read` the matching
`.md` file for the full rule and fix-recipe — it likely captures a
root cause that took 30+ minutes to diagnose, with the working
strategy written down. See **"Memory files to read"** later in this
guide for a curated list of high-signal entries.

Whenever you discover a non-obvious fix that took more than two
iterations to find, write a new `feedback_*.md` and add a hook to
`MEMORY.md` so the next session can skip your debugging cycle.

## Rule 1 — `pi0_end_to_end` is FORBIDDEN

**Using Pi0 to perform the place is not allowed**, regardless of how hard the
task looks. The LLM MUST script every motion (`move_to`) and every release
(`release`). If your scripted placement misses, iterate within LLM scripting
(better coordinates, multi-stage path, different approach angle) — do NOT
hand the place back to Pi0.

Specifically:
- You may call `pi0_pick` to do the grasp (with `track_obj` cut so Pi0 exits
  immediately after lifting the object).
- You may NOT call `pi0_pick` (with high `lift_thresh`/`gripper_closed_thresh`
  and no `track_obj`) to let Pi0 finish the task after a failed LLM place.
- You may NOT call `pi0_pick` with the full task instruction expecting Pi0 to
  also handle the place.

The `pi0_end_to_end` regime that appears in older audit JSONs (e.g. previous
t9 attempt) is **deprecated**. Existing entries can stay as historical
records but new attempts must NOT produce that regime. If you cannot solve a
task without Pi0 doing the place, document it as a strict-regime failure
(`stopped_at_pick=true` or analogous, with notes on which LLM placement
strategies you tried) — the failure is more useful than a Pi0-end-to-end
"success".

`pi0_doubled` remains permitted **only** when Pi0 is used for a non-pick VLA
skill that genuinely cannot be scripted in OSC space — knob turn (t2),
drawer close (t3). In every other case, prefer strict.

## Rule 4 — NO TELEPORT primitives (physics-only)

**The four teleport primitives are DELETED from the codebase and must never be
reintroduced, re-added, or simulated:**

- `set_object_pose` — writes an object's free-joint qpos straight to the goal
- `articulate_to` — writes a door/drawer/knob joint qpos directly
- `js_move_to` — kinematically warps the arm's 7 joint qpos (`mj_forward`, no controller)
- `carry_object` — `js_move_to` variant that rewrites a held object's qpos each waypoint

They bypass contact physics and do not demonstrate physical manipulation, so any
"success" using them is invalid. The driver no longer exposes these actions; issuing
one returns an `unknown action` error. **Every motion must go through the OSC
controller (`move_to`, `rotate_wrist`, `rotate_pitch`, `release`, `set_gripper`) or
through Pi0 (`pi0_pick`).**

`pi0_doubled` — Pi0 driving a non-pick contact skill such as the stove-knob turn or a
drawer open/close — IS allowed and IS the sanctioned replacement for what
`articulate_to` used to do (the robot makes real contact and applies real torque). If a
goal predicate genuinely cannot be satisfied by any physical sequence (e.g. a site past
the Panda reach, or a drawer whose `is_close` needs qpos>0 that no physical push
achieves), record an honest `strict_failure` with the predicate decomposition — a
documented physical failure is worth more than a teleport "success".

## Rule 2 — Multi-episode iteration is allowed and encouraged

A (task, seed) pair gets as many fresh episodes as you need to find a
working strategy. `reset` is cheap; the audit JSON only captures the final
attempt that you persist, so intermediate failed episodes are free.

Concretely:
- If a place misses, `reset` and re-pick from scratch with a different plan
  (different approach vector, different stage order, different release z,
  pre-oriented wrist). You are not required to recover within one episode.
- If Pi0 keeps grabbing the wrong object in a cluttered scene, `reset` and
  try again — a fresh episode often picks the correct object on the next
  try (especially after you tighten the pre-pos), instead of fighting the
  drift in the current one.
- If you discover mid-run that your assumed fixture coordinates are wrong,
  `reset`, re-derive the coordinates (read the relevant XML or post-pick
  image), and run a clean second episode with the corrected target.

This combines with Rule 1: do not escalate to Pi0 end-to-end just because
the first episode failed. Iterate within LLM scripting across multiple
episodes; the only ceiling is "no scripted variant works after honest
exploration," at which point document the strict failure.

## Rule 5 — Assume EVERY task is physically solvable; do NOT mark failure lightly

**Default working assumption: every task in this benchmark is physically
solvable with the physics-only primitives.** The LIBERO tasks were created from
human teleoperation demonstrations on the very same robot and scene, so a
physical solution provably exists. Treat "this cell is blocked" as a *hypothesis
to disprove*, never as a conclusion you reach after a couple of failed episodes.
A `strict_failure` you record is a strong claim ("no physical sequence can satisfy
this predicate") and it is usually WRONG — it almost always means *your strategy or
your applied force was insufficient*, not that the task is infeasible.

This is not a pep talk; it is the empirical track record of this project. Multiple
cells that prior sessions confidently labelled `strict_failure_physical` were later
solved or shown solvable once someone stopped trusting the label:

- **`libero_10 lan_t3` (drawer close to qpos>0)** was declared blocked ("needs qpos>0,
  impossible to push past flush"). An offline probe (set the joint qpos and step with
  zero action) proved qpos>0 is a perfectly **stable** state — the drawer is
  near-frictionless with no collision seat. The close was never impossible; pi0 just
  quits at visual-flush and OSC is rank-deficient at the cabinet-front singularity. It
  is now SOLVED by alternating `pi0_doubled` close with an OSC front-approach push.
- **`libero_10 swap_t8` (mokas → relocated stove)** was declared "cook past Panda reach".
  The cook_region is actually a reachable BOX (the prior session measured the burner
  *body* xpos, not the predicate *site* box). Reach was never the barrier.

**Before you are allowed to write `strict_failure`, you MUST have done ALL of:**

1. **Verify the feasibility claim itself, with numbers.** Read the actual predicate
   threshold from the libero source (e.g. `is_close` / `under` / `in_box`), and the
   actual site/box coordinates and sizes — never trust a coordinate "from the image" or
   a remembered bound. Many "blocked" verdicts are just a wrong number.
2. **Run an offline feasibility probe** when a predicate hinges on a joint/object
   reaching some value: load the env (`OffScreenRenderEnv`), write the target qpos /
   pose, step with zero action, and check whether the predicate region is physically
   attainable and *stable*. If the goal state holds, the task is solvable — keep going.
3. **Exhaust the strategy ladder**, not just parameter tweaks: different force path
   (push vs lift vs momentum-tap on a low-friction joint), different approach pose to
   dodge a singularity, `pi0_doubled` vs scripted OSC and *combinations* of them
   (pi0 reaches joint configs OSC cannot; OSC is precise where pi0 is blind), re-order
   the subtasks, and multi-episode `reset`/relaunch retries (auto-retry scripts that
   relaunch on a sim crash are encouraged — this is the "try 100 times" regime).
4. **Render the failed steps** (Rule 0) and write the diagnosis in words.

Only after all four, and only if you can name the *specific physical reason* (e.g.
"object geometrically too large for the closed cavity", verified) may you record a
failure — and even then, frame it as "not achieved within these limits", note exactly
what you tried, and leave the door open for the next session. A documented honest dead
end is fine; a lazy `strict_failure` after two tries is a bug.

Read [WORK_DONE.md](./WORK_DONE.md) first if you want the full context. This
file is the operating manual.

## Rule 0 — Use images for reasoning, not just JSON state

The driver dumps **both** `state_NN.json` (privileged numerical state) and
`image_NN.png` (agentview render) at every step. The single biggest failure
mode of the previous session was **only reading the JSON**: I had every image
on disk but never called `Read` on a PNG, so I never used the LLM's spatial
reasoning ability. I became a control-parameter tuner instead of a spatial
reasoner — and tuned controllers cannot rescue a bad target layout.

**Mandatory practice:**

1. **Before any non-trivial decision, `Read` the latest `image_NN.png`.**
   Especially: before the first `move_to` of a placement, after every
   failure, and any time you're about to retry the same plan with tweaked
   numbers. JSON tells you where objects are; the image tells you what
   *space remains* and *what could fit there*.

2. **Describe the scene in words first, decide second.** Write 1–2
   sentences: "cook_region center occupied by moka_2; ~6 cm of clear floor
   in the front-right and back-left of the 15×15 cm site." Then decide.
   Numbers are sanity checks for the plan, not the source of it.

3. **If you've tuned controller params (`step_clip`, `max_steps`, drop
   height) twice in a row without success, STOP.** This is the signature
   of being trapped at the wrong abstraction level. Open the latest image,
   redescribe the scene, and ask whether the *target layout itself* is
   feasible. Multi-object placements into a flat region almost always
   need a footprint allocation step (e.g. opposite corners) — single-point
   targeting is the default trap.

The t8 failure that took 4 attempts to fix would have been solved on attempt
2 by a single image inspection: "first moka sits at the center of a 15 cm
square — where else can a 6 cm cylinder go?" Two minutes of vision-language
reasoning, not two hours of step_clip tuning.

### Failure forensics — when a retry fails, render the failed steps

After two retries on the same plan, *stop tuning parameters and start
rendering*. The image is your debugger. Walk the failed run step by
step, `Read` the `image_NN.png` at each critical milestone, and look
for the disagreement between what you expected and what's on screen.

**Render this set after any failed (suite, task, seed) attempt:**

1. `image_00.png` — initial scene (verify object positions match
   `state_00`; spot any unexpected fixture / distractor / configuration
   the BDDL didn't tell you about).
2. The image right after each `pi0_pick` — confirm the object is
   actually held (gripper around object, object lifted) vs hanging
   off a finger or still on the table. JSON's `chunks_used` can lie
   when Pi0 ran out of chunks mid-motion.
3. The image right before every `release` — confirm the object is
   above the target xy zone, not 5 cm off due to a stalled `move_to`.
   This catches "release dropped the object on the basket rim, not
   in it" failures *before* you waste a `release`.
4. The image right after every `release` — confirm the object landed
   where physics predicted, not bounced / rolled / tipped.
5. For articulation / contact tasks: image immediately before AND after each
   push or close. A physical-contact failure (the door swing sweeping the object
   out of the cavity, a drawer that didn't move because contact was lost) leaves
   visually-obvious evidence that JSON only hints at via a z drop.

**Write the diagnosis in words.** For each image, one sentence on
what you see and one sentence on what you expected. The mismatch is
where the bug is. Do this *before* writing the next command.

Example from libero_10 t3 drawer diagnosis (2026-05-21):

- *Expected after the close*: drawer closed with the bowl inside,
  `libero_terminated=True`.
- *Image showed*: drawer slid shut but the bowl left behind on the table
  at the old open-drawer xy.
- *Conclusion*: the close moved the drawer floor without carrying the bowl —
  the fix is a CONTINUOUS push (drawer front wall in contact drags the bowl
  along), never a one-shot motion that loses contact.

That single image inspection turned 3 session-hours of "try a
different drop xy, try a slower carry" into a
10-minute fix. **Whenever a retry of a *strategy* (not just numbers)
fails twice, render. The image will tell you whether the failure is
in your plan or in the physics simulator's behavior.**

## Mental model

1. **One Python process runs forever.** It loaded Pi0.5 (~90s, GPU mem ~6GB),
   built a single-env LIBERO sim, and is now blocked waiting for a command.
2. **You write commands.** Each command is a JSON file at
   `/tmp/hybrid_repl/command.json`. The driver consumes it, runs one
   primitive, dumps `state_NN.json` + `image_NN.png` + `log_NN.json`, then
   touches `done_NN.flag`. Then it blocks for the next command.
3. **You read state, decide next move, write next command. Repeat.**

Each (task, seed) run reloads the model (slow). Within a session you can
freely `reset` to retry the same task.

## Launch a session

```bash
cd /mnt/public2/zhangyixian/RLinf_agentic
CUDA_VISIBLE_DEVICES=0 /opt/venv/openpi/bin/python \
    examples/embodiment/primitives/interactive_driver.py \
    --suite libero_10 --task <N> --seed 0 --max_episode_steps 5000
```

> libero_10 recipes are long-horizon (mean ~940, up to ~1600 env-steps);
> 600 hits robosuite's per-episode cap mid-recipe and raises
> `ValueError("executing action in terminated episode")`. Use 5000.
> (spatial/object stay at 600 — short single pick→place.)
> See `feedback_max_episode_steps_libero.md` + `results_10_pert/PATCH_NOTES.md`.

Run this **in the background** (Bash `run_in_background: true`) so the harness
doesn't block. Then wait for `state_00.json` to exist as the readiness signal
(~90s model load).

```bash
until [ -f /tmp/hybrid_repl/state_00.json ] && [ -s /tmp/hybrid_repl/state_00.json ]; do sleep 5; done
```

Suites currently supported: `libero_spatial`, `libero_10`. Add more via
`make_env(..., suite_name=...)` if needed.

## The command vocabulary

Write JSON to `/tmp/hybrid_repl/command.json`. Brief schemas below; the full
**Extended primitives reference** section (later in this guide) explains
when to use each, gotchas, failure trees, and worked examples (t9 strict
recipe is documented end-to-end).

```jsonc
// === core (always available) =====================================

// Scripted EEF servo. action_scale=0.05 is the env's units; step_clip caps
// per-step Δxyz in metres BEFORE division by action_scale → smaller = slower.
{"action": "move_to", "xyz": [x, y, z], "gripper": -1|+1,
 "tol": 0.012, "step_clip": 0.02, "max_steps": 80, "action_scale": 0.05,
 "target_yaw": null}                          // optional world-z yaw target

// Pi0.5 closed-loop pick. The track_obj hard-cut is the critical knob.
{"action": "pi0_pick", "prompt": "pick up the X",
 "max_chunks": 30,
 "track_obj": "akita_black_bowl_1",           // <object>_pos key in raw obs
 "track_obj_lift_thresh": 0.07,                // metres above init z to break
 "lift_thresh": 0.05, "gripper_closed_thresh": 0.06}

// Hold pose, open gripper. Triggers libero termination if "On" predicate met.
{"action": "release", "max_steps": 25}

// Hold pose, command gripper for N env steps without moving.
{"action": "set_gripper", "gripper": +1|-1, "steps": 5}

// Reset env (same task/seed). State_00.json reappears.
{"action": "reset"}

// Clean shutdown. Use when switching tasks (next session = new --task).
{"action": "exit"}

// === extended (added 2026-05-19, Rule-1 compliant LLM-side motion) ====

// World-z yaw rotation, holds xyz. (Bug-fixed: used to rotate in the
// OPPOSITE direction for gripper-down configs.)
{"action": "rotate_wrist",
 "delta_yaw": 1.57,                           // OR "target_yaw": <abs>
 "gripper": +1|-1,
 "max_steps": 60, "tol": 0.05, "step_clip": 0.15}

// World-x pitch rotation, holds xyz / yaw / gripper.
// Lets the gripper "lean forward" to fit through tight cavity openings
// (e.g. t9 microwave) that 4-DoF xyz+yaw OSC can't thread.
{"action": "rotate_pitch",
 "delta_pitch": 0.9,                          // OR "target_pitch": <abs>
 "gripper": +1|-1,
 "max_steps": 60, "tol": 0.03, "step_clip": 0.15}

// (NO teleport commands. js_move_to / articulate_to / set_object_pose are
//  REMOVED — every motion goes through the OSC controller or Pi0. For
//  Close(articulation) / TurnOn, push physically with move_to or hand the
//  contact skill to pi0_pick (pi0_doubled). See "Rule 4 — NO TELEPORT".)
```

After each command, wait for the flag:

```bash
until [ -f /tmp/hybrid_repl/done_<NN>.flag ]; do sleep 1; done
```

Where `<NN>` is the step number (zero-padded, starts at 01 for first command).

## The strict-hybrid recipe

Per (task, seed):

```
1. Inspect state_00.json + image_00.png. List target objects and their xyz.
2. (Optional) pre-position above target object with gripper open.
3. pi0_pick with track_obj=<target> and lift_thresh=0.05–0.08.
   The track_obj parameter is what enforces "Pi0 only does pick" —
   the loop breaks the moment the object's z lifts by your threshold,
   preventing Pi0 from continuing into a learned place trajectory.
4. Read post-pick state. Compute bowl-eef offset.
5. move_to(plate_xy - offset_xy, plate_z + half + margin - offset_z) keeping
   gripper closed. Often needs 2-3 sub-stages: lift → travel → descend.
6. release(max_steps=25). Watch for libero_terminated=True in state JSON.
7. **Always add a retreat step** — move_to a safe pose with gripper open,
   step_clip=0.02. See "Predicate fire timing" below.
```

Multi-object tasks chain steps 2–6 per object. Pi0 is re-invoked for each
pick; you handle every place.

### Predicate fire timing & mandatory retreat

`libero_terminated=True` can fire at **any env.step**, not only during the
`release` primitive. The libero predicate (`On`, `In`, `Close`, …) is
re-evaluated every physics step the env runs, so the trigger step depends on
*when the object physically lands in the predicate region*, which is not
always the moment you open the gripper.

Empirically observed fire points (libero_goal swap session, 2026-05-22):

| Task | Fires during | Why |
|---|---|---|
| t1 swap (bowl→cook) | **retreat** | OSC stalled at z=1.20, bowl perched on gripper at z=1.17; only after retreat (gripper opens and moves away) did bowl fall the last 17 cm onto cook_region. |
| t9 swap (wine→rack) | **descent** | The descent `move_to` itself put the bottle in contact with rack at z=1.15 → `On` satisfied mid-descent, before any release primitive ran. |
| t2/t4 swap (→cabinet) | **release** | Standard case — release opens gripper, object drops a few cm onto cabinet top, predicate satisfied within `release`'s settle steps. |

Two operational consequences:

1. **Always include a retreat `move_to` after `release`** even when
   `libero_terminated=False` at the release step. Retreat clears the gripper
   away from the placement so the object can fully settle, and lets you
   capture the *true* final state — both for predicate firing and for
   recording the object's resting xyz in the audit JSON. Skipping retreat
   risks (a) missing the predicate that would have fired and (b) saving an
   `final_state` where the object is still half-supported by the gripper.

2. **Don't gate "success" on release's `libero_terminated`.** Check the
   *latest* state file after retreat, not the release log. If your audit
   stitch only inspects `release_result["libero_terminated"]`, you will
   under-count strict passes for any task where settle takes longer than
   release's `max_steps` (e.g. tall objects, cavity placements, OSC-stalled
   descents where the object falls from height after release).

## Extended primitives reference

Four additions / fixes (2026-05-19) cover tasks the basic recipe can't:
narrow cavity openings, OSC-singularity workspace boundaries, and
`Close(articulation)` goal predicates. All four are pure LLM-side motion
primitives — Rule 1 compliant; they do not invoke Pi0.

| Primitive | Purpose | Run cost | Implementation |
|---|---|---|---|
| `rotate_wrist` (bug-fixed) | world-z yaw rotation, holds xyz | ~10–30 env steps | OSC action[5] |
| `rotate_pitch` | world-x pitch rotation, holds xyz/yaw | ~10–30 env steps | OSC action[3] |

(`js_move_to`, `articulate_to`, `set_object_pose`, `carry_object` were removed — see
**Rule 4 — NO TELEPORT**. Only the two OSC reorient primitives above remain in this
"extended" set; everything else is core OSC + `pi0_pick`.)

### 1. `rotate_wrist` — world-z yaw rotation (bug-fix)

**What it does.** Drives the gripper's world-frame yaw to `target_yaw`
(absolute) or by `delta_yaw` (relative). Holds xyz constant. Uses OSC
`action[5]` (axis-angle z).

**Bug fix (2026-05-19).** Earlier versions extracted yaw via
`scipy.Rotation.as_euler('zyx')[0]`, which returns the **negative** of the
world yaw for gripper-down configs (`R[2,2] ≈ -1`). The function rotated
the wrist in the OPPOSITE direction of the commanded yaw. Replaced with
`atan2(R[1,0], R[0,0])` (rotation matrix first column). Also fixed the
same buggy extraction in `move_to(target_yaw=...)`.

**Returns:** `{name, start_yaw, target_yaw, final_yaw, final_err,
steps_used, libero_terminated}`. Pass if `|final_err| < 0.05`.

**Conventions.** `yaw = atan2(R[1,0], R[0,0])`. At gripper-down rest:
yaw = 0. `+yaw` rotates the gripper x-axis from world +x toward world +y.

**When to use.**
- Pre-orienting the gripper before threading into narrow openings whose
  geometry depends on wrist alignment (e.g. libero_10 t9 microwave cavity).
- Twisting the gripper after release to unhook a held object's handle
  from a finger before retreat. **Empirical t9 finding:** `delta_yaw = +3.0`
  after release of a mug also (a) unhooks the handle, (b) retreats the eef
  ~12 cm in -y, AND (c) nudges the held object an extra ~4 cm in +y. Useful
  as a "release + retreat + push deeper" three-in-one combo for cavity
  placements.

### 2. `rotate_pitch` — world-x pitch rotation (NEW)

**What it does.** Tilts the gripper around the world X-axis, so the
gripper z-axis swings in the world yz-plane. Drives via OSC `action[3]`
(axis-angle x). Holds xyz, yaw, and gripper constant.

**Why it exists.** OSC `move_to` only exposes xyz + yaw (4-DoF in pose
space). Some tasks — notably t9 microwave cavity entry — need the gripper
to "lean forward" so the wrist body (~3 cm above the eef site) fits
through a tight opening. Without pitch tilt, the wrist body hits the
cavity ceiling at z=1.088. With `delta_pitch ≈ +0.9` (52°), it threads
the opening and mug placement succeeds.

**Returns:** `{name, start_pitch, target_pitch, final_pitch, final_err,
steps_used, libero_terminated}`. Pass if `|final_err| < 0.05`.

**Conventions.** `pitch = atan2(R[1,2], -R[2,2])`. At gripper-down rest:
pitch = 0 (gripper z-axis points world -z). `+pitch = +π/2`: gripper z
points world +y (gripper "looking forward"). `-pitch = -π/2`: gripper z
points world -y. Sign verified empirically: `action[3]=+1.0` tilts eef z
toward world +y.

**When to use.**
- **t9 mug → microwave**: `rotate_pitch +0.9` lets the gripper thread the
  cavity opening (mandatory for the In predicate).
- Pouring-like motions where the eef needs to tilt forward.
- Any cavity / shelf insertion task where 4-DoF (xyz + yaw) OSC can't fit
  the wrist body geometry.

### 3. (REMOVED) js_move_to / articulate_to / set_object_pose — NO TELEPORT

These four primitives — `js_move_to` (kinematic arm-qpos warp), `carry_object`
(object-qpos warp riding the arm), `articulate_to` (door/drawer/knob joint-qpos
write), and `set_object_pose` (object free-joint qpos write) — have been **deleted
from the codebase** (2026-05-26). They bypass contact physics and do not demonstrate
physical manipulation. See **Rule 4 — NO TELEPORT**. Do not reintroduce or simulate
them. The only motion primitives are the OSC ones (`move_to`, `rotate_wrist`,
`rotate_pitch`, `release`, `set_gripper`) and `pi0_pick`.

**Physics-only replacements for what teleport used to do:**

- **Stove TurnOn / TurnOff** (`flat_stove_1_button`, On at qpos≥0.5, Off at qpos<0):
  turn the knob physically with `pi0_doubled` — `pi0_pick` prompt "turn on the stove",
  `max_chunks≈15`, `gripper_closed_thresh=0` + `track_obj=null` so the pick-success
  break never fires. The burner glows red when on. (Verified libero_10 t2/t8, 2026-05-26.)

- **Moka / object pick**: scripted grasp works for clear-geometry bodies (descend to
  the body, `set_gripper +1` steps≈22, lift), but the moka body grip is laterally
  WEAK — carry it in ~0.08 m hops at `step_clip≈0.01` with a `set_gripper +1` re-clamp
  after each hop. Long fast traverses slip. Use `pi0_pick` for flat/small objects.

- **Drawer open**: `pi0_doubled` — `pi0_pick` prompt "open the bottom drawer"
  (`gripper_closed_thresh=0`, `lift_thresh=0.3`); Pi0 pulls it open (often grabs the
  handle and the nearby object in one go).

- **Close(slide-drawer) with object inside**: the object must be dragged by CONTINUOUS
  contact (a teleport leaves it behind). Push the drawer front continuously — either
  `pi0_doubled` "close the drawer" (repeat; it pushes the front and the object rides
  the floor) or a CAPPED OSC push (`move_to` +slide-direction, `max_steps≤120` to dodge
  the QACC-DOF9 NaN). ⚠ Do NOT scripted-descend the eef to low z right at the cabinet
  front — that collision can crash the MuJoCo worker (EOFError). Note: WhiteCabinet
  `is_close` needs qpos>0 (pushed past flush), which is genuinely hard physically; if no
  physical push achieves it, record an honest strict_failure (In satisfied / Close blocked).

- **Close(hinge-door)** (e.g. microwave): push the door physically with `pi0_doubled`
  "close the door", or reorient (rotate_pitch/rotate_wrist) and OSC-push from a
  non-singular pose. If unreachable, record a strict_failure.

- **Predicate site past Panda reach**: there is no teleport shortcut. Get the eef as far
  as OSC allows and release if the object's xy is over the predicate region; otherwise
  record a strict_failure. Do NOT warp the object there.

### Operational notes for all extended primitives

- **EGL crash budget.** The libero env worker EGL context crashes after
  roughly 9–12 cumulative commands per driver instance, with long
  `move_to` commands (max_steps >= 300) being the heaviest. A long push
  is cheap (warp + few settle_steps), so it doesn't eat the budget the
  way long OSC moves do. Still: restart the driver between (task, seed)
  iterations and keep total commands per attempt <10 when possible.
- **`reset` within a session corrupts pre-pos.** Multiple `{"action":
  "reset"}` calls within a single driver session leave the Panda in
  joint configs where subsequent `move_to` pre-positions don't converge
  (eef stops ~5 cm short of target). Restart the driver fresh per
  iteration.
- **Rule 1 compliance.** None of these primitives invoke Pi0. They are
  all permitted in strict-regime runs. The `pi0_doubled` regime still
  exists for cases where a non-pick VLA skill (e.g. knob turn in t2,
  drawer close in t3) is the sanctioned route now that teleport is gone:
  use `pi0_doubled` (Pi0 pushes the knob/drawer with real contact).
- **Rule 2 (multi-episode iteration).** Still valid. Use it when
  an OSC `move_to` stalls on a target you believe is reachable —
  reset, look at `ik_diagnostics`, adjust knobs, try again.

### File layout (reference)

```
examples/embodiment/primitives/
├── primitives.py              # LiberoPrimitiveDriver — OSC move_to, pick,
│                              # release, set_gripper, rotate_wrist, rotate_pitch.
│                              # (Teleport methods removed — see Rule 4.)
├── interactive_driver.py      # REPL — action handlers for the physics-only primitives.
└── STRICT_HYBRID_GUIDE.md     # THIS FILE — full operating manual + Rules.

rlinf/envs/libero/
└── libero_env.py / venv.py    # env worker + OSC step dispatch. (The teleport
                                 # module js_move_to.py and the js_move_to /
                                 # articulate_to / set_object_pose pipe commands
                                 # were deleted — see Rule 4 — NO TELEPORT.)
```

## Reading state / log files

```bash
cat /tmp/hybrid_repl/state_<NN>.json | python -c "
import json, sys
d = json.load(sys.stdin); s = d['state']
print('libero_term:', d['libero_terminated'])
print('eef:', s['robot0_eef_pos'])
print('grip:', s['robot0_gripper_qpos'])
for k, v in s['objects'].items(): print(f'  {k}: {v}')
"
```

For images, use the Read tool on `/tmp/hybrid_repl/image_<NN>.png` (Claude
Code's Read tool renders PNGs).

## Hyperparameters that actually matter

| Knob | When to use small | When to use large | Reason |
|---|---|---|---|
| `pi0_pick.track_obj_lift_thresh` | 0.05 — interrupt as soon as bowl leaves table | 0.10 — let Pi0 fully secure the grasp | Lower threshold = stricter constraint but risk of slipping. Higher = more secure but Pi0 may start traveling toward place region. |
| `move_to.step_clip` | 0.015–0.02 — cylindrical/heavy objects (cans, moka pots) on libero_10 | 0.04 — bowls, empty gripper | Smaller = slower = less mid-translation slippage. |
| `move_to.tol` | 0.008 — precise pickup re-alignment | 0.02 — general transit | Tight tol can stall on OSC limits; loose tol can miss target. |
| `pi0_pick.max_chunks` | 20–30 — when track_obj will fire | 50–60 — when expecting Pi0 to do extra (knob/drawer) | Each chunk = 5 sim steps. |

## Rule 3 — LLM is a delivery service for Pi0, not a replacement

Pi0.5 is a vision-action model whose **single best skill is grasping
objects it can see in front of it from a stable pose.** The hybrid
pipeline gains its leverage by letting Pi0 do that one thing well and
having the LLM handle everything else — *not* by replacing Pi0's grasp
with LLM-scripted descend+close+lift the moment Pi0 stumbles.

Concretely, the LLM/Pi0 split for any pick is:

- **LLM**: read the BDDL, find the target's xy in `state_00`, drive
  `move_to` to a pre-pos that places the gripper directly above the
  object at a stable z, hand Pi0 the right prompt, *let go*.
- **Pi0**: from that pre-pos + prompt, run the closed-loop grasp.

The Appendix at the end of this guide ("LLM-scripted pick fallback") is
a real escape hatch but should be the **last** thing you reach for. In
practice the failure mode is: `pi0_pick` returns 15 chunks of nothing,
the LLM panics, writes `move_to z=mug_top + set_gripper +1`. That is
premature unless you have first exhausted the escalation ladder below.

### The Pi0 prompt escalation ladder (try in order)

Only after **all four** fail across multiple episodes should you write
an LLM-scripted pick.

1. **Sub-instruction** `"pick up the {object}"`. Works on `libero_spatial`
   (visually unambiguous) and any clean single-target scene.
2. **Full task language** verbatim from BDDL `:language`. Critical for
   `libero_10` cluttered scenes (multiple distractors) and tasks Pi0
   was trained on as a *multi-step* instruction — drawer open+place,
   stove on+place, microwave in+close, etc. Example: instead of
   `"pick up the black bowl"`, use `"put the black bowl in the bottom
   drawer of the cabinet and close it"` — and `track_obj`-cut the
   moment the bowl lifts so Pi0 doesn't run the place too.
3. **Spatial qualifier** `"pick up the X on the cabinet"` /
   `"...next to the basket"`. Use for elevated objects (cabinet top,
   stove, microwave shelf) and edge-of-workspace positions.
4. **Reposition + retry**: if Pi0 wanders or stalls, the gripper is
   often approaching from a bad angle. Adjust the *pre-pos*, not the
   grasp:
   - Lower pre-pos z (drop from z=0.95 to z=0.65 — tight spatial
     constraint forces Pi0 toward the nearest object).
   - Different approach xy (offset by 5 cm toward the handle side for
     mugs; offset toward the body center for boxes).
   - `reset` + fresh episode if the scene has drifted from prior
     attempts (also clears any "wrong-object fixation").

If you reach LLM-scripted pick, document in `strategy_notes` which
prompts and pre-poses you tried first. A "tried only sub-instr once,
then scripted" note is a red flag in audit.

### Empirical evidence: `libero_10_lan t3` retry

First pass used sub-instr `"pick up the black bowl"` → Pi0 ran 15 chunks
with gripper open at end. We bailed to LLM-scripted pick (which also
failed — bowl is thin, vertical grasp slides off), declared strict failure.

Retry on the same (suite, task, seed) used the **full task prompt**
`"put black bowl inside bottom drawer and close it"` from the same
pre-pos. Pi0 picked the bowl in 19 chunks (track_obj cut at +5 cm
lift), gripper qpos sum ≈ 0.004 (firm grasp), no scripting needed.

The retry's failure shifted to the place step (drawer floor friction
releases the object during slide; separate physics problem). The pick
itself was solved by a one-line prompt change.

**Lesson**: when `pi0_pick` returns "nothing happened" on a
`libero_10` (or any cluttered) task, try the full BDDL task language
*before* anything else. Sub-instructions only really shine on
`libero_spatial`.

## Picking the right `pi0_pick` prompt — quick reference

The Pi0.5 checkpoint is `pi05_libero130_fullshot/30000` — SFT on all
130 libero tasks with their original BDDL instructions. The escalation
ladder above is the operating discipline; this table is the quick
lookup once you know which rung to start on.

| Prompt strategy | When to start here | Notes |
|---|---|---|
| `"pick up the {object}"` | `libero_spatial` (visually unambiguous, single canonical object) | Prompt-blind regime — model picks whatever's on the table. Works regardless of prompt details. |
| Full task instruction (e.g. `"put the black bowl in the drawer and close it"`) | `libero_10` *default* — pick this first. Also: any task where Pi0 was trained with a multi-step instruction (drawer, stove, microwave). | Gives Pi0 task context. **Will trigger Pi0's full pick+place if you don't `track_obj`-cut.** |
| Spatial qualifier (`"pick up the X on the cabinet"`) | Elevated objects or edge-of-workspace placements where the default view doesn't isolate the target | Helps with vertical reach on cabinet-top / stove-top picks. Pair with `track_obj_lift_thresh = 0.08` (see [[pi0-pick-full-prompt]] memory). |

## Common failure modes and how to recognize them

Read these signals from `state_NN.json` after each command:

### Pick missed (gripper closed on empty space)
- `pick_result.diagnostics.track_obj_final_z ≈ track_obj_init_z` (object didn't lift)
- `state.objects.<target>_pos[2]` unchanged from init
- Sometimes `gripper_qpos sum ≈ 0` (fully closed = empty) — gripper open near 0.08 is normal for libero 2f85 holding nothing.

**Fix:** re-pre-position more precisely, retry pi0_pick. If still failing,
relax `track_obj_lift_thresh` (Pi0 may have lifted < your threshold).

### Object slipped during move
- Post-move `state.objects.<target>_pos[2]` dropped near table z
- post-move `offset_z` from `bowl-eef` is much smaller (object fell below
  gripper)

**Fix:** reset, retry with smaller `step_clip` (0.015), use multi-stage
move (lift z high first → travel → descend).

### EEF stuck (OSC limits)
- `move_result.final_dist_m > tol AND steps_used == max_steps`
- EEF position doesn't change between two state dumps

**Fix:** check for collision — Panda wrist hits microwave/cabinet top.
Lower target z OR break path into stages that go AROUND obstacles. There's
no collision-aware planner; you must script the detour.

**Special case — OSC stalls at the SAME xy across multiple variants**: if
the EEF stops at the same `(x, y)` for 3+ different staging plans (different
approach z, different x entry, max `step_clip=0.05`), it is **not** a
collision and **not** a tuning issue. It is a Panda IK / OSC singularity
at that workspace location. Scripted `move_to` cannot bypass it.

Permitted responses (Rule 1 forbids Pi0 fallback for place):
1. **Reset and try a completely different task strategy** — push the object
   across the table instead of lifting through workspace; approach the goal
   from a different side; pre-orient the wrist before the move.
2. **Decompose the move differently** — break it into more stages, change
   which axis is varied first, raise the carry altitude to clear the
   singularity region.
3. **Declare strict failure for this task** if no scripted variant works
   and document the dead-end in `strategy_notes`. A negative-result audit
   beats a Pi0-end-to-end "success" that violates Rule 1.

t9 is the canonical example: 6+ scripted variants all stalled at y≈0.26
trying to enter the microwave cavity. The prior session escalated to Pi0
end-to-end (now forbidden by Rule 1) — that audit entry is grandfathered
but must not be reproduced. A fresh t9 attempt should either find a
scripted detour or be documented as a strict failure.

### Pi0 did the place (constraint violation!)
- Right after `pi0_pick`: `pick_result.libero_terminated=True` OR
- `chunks_used` much higher than expected (≥ 25 for a single pick) AND
  `state.objects.<target>_pos` already at the goal location

**Fix:** lower `track_obj_lift_thresh` (e.g. 0.05 → 0.04) to interrupt
Pi0 earlier. Or pre-position EEF directly above target so Pi0 needs
less descent before the trigger fires.

## Verifying strict compliance

After each run, check `log_<pick_step>.json`:

```python
pr = json.load(open("/tmp/hybrid_repl/log_02.json"))["result"]  # the pi0_pick step
assert pr["libero_terminated"] == False, "Pi0 finished the task — violation!"
assert pr["chunks_used"] < 25, "Pi0 ran too long; may have done place"
```

And `log_<release_step>.json`:

```python
rr = json.load(open("/tmp/hybrid_repl/log_06.json"))["result"]  # release step
assert rr["libero_terminated"] == True, "Task didn't terminate during release"
```

These two together are your audit trail: **pick exited without libero_term;
release triggered libero_term ⇒ LLM did the place**.

## Persisting successful runs as audit JSONs

The REPL workflow is great for iteration but **leaves nothing reproducible
behind once the driver exits** — `/tmp/hybrid_repl/state_NN.json` and
`log_NN.json` live only until the next driver run wipes them. The
libero_spatial corpus at `results_all_spatial/tN_sM.json` is the gold
standard: each file is a single self-contained record of one
(task, seed) rollout that an auditor (or a future Claude) can read months
later to verify strict compliance, replay the strategy, or compare runs.

**Goal**: after each successful (task, seed) iteration, produce one such
JSON in `results_all_<suite>/tN_sM.json` matching the libero_spatial
schema, plus an entry in `all_rows.json`.

### Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. REPL iterate (Rule 0 + command vocab + heuristics).          │
│    Find a command sequence that gets libero_terminated=True.    │
│    Write each command as a JSON line in a scratch file as you   │
│    go, e.g. /tmp/hybrid_repl/recipe_t<N>_s<M>.jsonl, so the     │
│    final recipe is captured cleanly (do NOT rely on memory).    │
│                                                                 │
│ 2. Confirm strict compliance via the asserts above.             │
│                                                                 │
│ 3. Stitch the run into a per-rollout audit JSON:                │
│    - the pick log (chunks, lift, gripper closure, diagnostics)  │
│    - the post-pick state (eef + all object positions)           │
│    - the move_to logs (each stage)                              │
│    - the release log (gripper opening progression)              │
│    - the final state and libero_terminated=True                 │
│    - the strategy notes (which prompt, which xy, which staging) │
│                                                                 │
│ 4. Save to results_all_<suite>/tN_sM.json. Append to all_rows.  │
│                                                                 │
│ 5. (Optional) Codify the working sequence as a per-task replay  │
│    script so it can be re-run unattended on a new seed and the  │
│    audit JSON regenerated from scratch.                         │
└─────────────────────────────────────────────────────────────────┘
```

### Where to save

| Suite | Directory | Per-rollout filename |
|---|---|---|
| libero_spatial | `results_all_spatial/` | `tN_sM.json` (gold standard, already populated) |
| libero_10 / libero_long | `results_all_10/` | `tN_sM.json` (to be built — only WORK_DONE.md table exists) |
| libero_goal | `results_all_goal/` | (future) |
| libero_object | `results_all_object/` | (future) |

In each directory also maintain `all_rows.json` — a single list of all the
per-rollout dicts, for easy bulk loading and stats.

### The schema (match libero_spatial exactly)

Top-level keys, in order:

```jsonc
{
  "task_id": 9,
  "seed": 0,
  "elapsed_s": 22.3,
  "regime": "strict",  // or "pi0_doubled" — "pi0_end_to_end" is FORBIDDEN (Rule 1); only kept in historical entries
  "strategy_notes": "lift to z=1.20 before any +y motion; cavity entered at x=-0.05",
  "pick_result":     { /* log_<pick_step>.json["result"] verbatim */ },
  "post_pick_state": { /* state_<pick_step>.json["state"] verbatim */ },
  "object_lifted_during_pick_m": 0.07,         // peak_lift_m, sanity
  "target_xyz":      [x, y, z],                // computed release target
  "target_diag":     { eef, target_obj, offset_obj_minus_eef },
  "move_results":    [ /* list of log_<move_step>.json["result"], one per stage */ ],
  "release_result":  { /* log_<release_step>.json["result"] */ },
  "final_state":     { /* state_<release_step>.json["state"] */ },
  "libero_terminated": true,                   // final
  "<task-specific metric>": 0.0143,            // bowl_to_plate_xy_m,
                                                // mug_in_heating_region, etc.
  "stopped_at_pick": false                     // audit flag
}
```

The two new keys vs. spatial:

- **`regime`** — one of `"strict"` or `"pi0_doubled"` (Pi0 used for a
  non-pick VLA skill like knob/drawer/door). `"pi0_end_to_end"` is
  **forbidden by Rule 1** and must not be produced by new runs — it appears
  only in grandfathered historical entries. Spatial entries are all
  implicitly `"strict"`; libero_10 runs MUST set this explicitly.
- **`strategy_notes`** — a 1-2 sentence human-readable summary of what
  worked, especially the *non-obvious* bit (e.g. for t8 "opposite-corner
  placement, not center"). Future-you will thank present-you. If the run
  was a strict failure (Rule 1 forbids escalation to Pi0 end-to-end), say
  which LLM placement variants you tried and why each failed.

If a run used more than one move_to (typical on libero_10 — lift, traverse,
descend), include them as a **list** under `move_results`, in execution
order. Spatial only needed one move_to so it used a singular `move_result`;
extend the schema rather than collapse.

### Stitch helper

Run this at the end of a successful REPL session, before issuing `exit`:

```bash
/opt/venv/openpi/bin/python - <<'PYEOF'
import json, glob, os, re
WORKDIR = "/tmp/hybrid_repl"
OUTDIR  = "/mnt/public2/zhangyixian/RLinf_agentic/examples/embodiment/primitives/results_all_10"
TASK_ID, SEED = 9, 0                                           # ← fill in
REGIME = "strict"                                               # ← fill in ("strict" or "pi0_doubled" — Rule 1 forbids "pi0_end_to_end")
NOTES  = "OSC IK barrier at cavity entry; Pi0 full task prompt solved in 186 chunks"

logs = {}
for p in sorted(glob.glob(os.path.join(WORKDIR, "log_*.json"))):
    n = int(re.search(r"log_(\d+)\.json", p).group(1))
    logs[n] = json.load(open(p))
states = {}
for p in sorted(glob.glob(os.path.join(WORKDIR, "state_*.json"))):
    n = int(re.search(r"state_(\d+)\.json", p).group(1))
    states[n] = json.load(open(p))

pick_steps    = [n for n, l in logs.items() if l["command"]["action"] == "pi0_pick"]
move_steps    = [n for n, l in logs.items() if l["command"]["action"] == "move_to"]
release_steps = [n for n, l in logs.items() if l["command"]["action"] == "release"]
pick_n    = pick_steps[0]      if pick_steps    else None
release_n = release_steps[-1]  if release_steps else None

record = {
    "task_id": TASK_ID,
    "seed": SEED,
    "regime": REGIME,
    "strategy_notes": NOTES,
    "pick_result":     logs[pick_n]["result"]               if pick_n is not None else None,
    "post_pick_state": states[pick_n]["state"]              if pick_n is not None else None,
    "move_results":    [logs[n]["result"] for n in move_steps],
    "release_result":  logs[release_n]["result"]            if release_n is not None else None,
    "final_state":     states[max(states)]["state"],
    "libero_terminated": states[max(states)]["libero_terminated"],
    "elapsed_s": sum(l.get("elapsed_s", 0.0) for l in logs.values()),
}
os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, f"t{TASK_ID}_s{SEED}.json")
with open(out, "w") as f:
    json.dump(record, f, indent=2)
print(f"saved {out}")

rows_path = os.path.join(OUTDIR, "all_rows.json")
rows = json.load(open(rows_path)) if os.path.exists(rows_path) else []
rows = [r for r in rows if not (r["task_id"] == TASK_ID and r["seed"] == SEED)]
rows.append(record)
rows.sort(key=lambda r: (r["task_id"], r["seed"]))
json.dump(rows, open(rows_path, "w"), indent=2)
print(f"updated {rows_path} (now {len(rows)} rows)")
PYEOF
```

Edit `TASK_ID`, `SEED`, `REGIME`, and `NOTES` for each run. The helper
auto-discovers all command logs in the workdir, picks the first `pi0_pick`
as the canonical pick, the last `release` as the canonical place, and
captures every `move_to` in execution order.

### Codify into a replay script (optional but recommended for benchmark runs)

Once a (task, seed) pair is solved interactively, copy the working command
sequence (from `recipe_t<N>_s<M>.jsonl`) into a small Python module under
`examples/embodiment/primitives/strategies/`:

```python
# strategies/t9_strict_attempt.py — example shell
def commands_for(task_id, seed, state_00):
    """Return a list of REPL command dicts that solve t9. Each dict is
    exactly what you'd write to /tmp/hybrid_repl/command.json."""
    return [
        {"action": "start_recording"},
        {"action": "move_to", "xyz": [-0.020, -0.020, 1.10], "gripper": -1,
         "tol": 0.012, "step_clip": 0.03, "max_steps": 60},
        {"action": "pi0_pick",
         "prompt": "put the yellow and white mug in the microwave and close it",
         "max_chunks": 100, "lift_thresh": 99.0, "gripper_closed_thresh": 99.0},
        # ... etc — paste from recipe_t9_s0.jsonl
        {"action": "save_video",
         "path": "videos/t9_replay.mp4", "fps": 25},
        {"action": "exit"},
    ]
```

Then a single `batch_replay.py` runner walks `strategies/` and produces
`results_all_10/all_rows.json` for the whole suite unattended. This is
the libero_10 equivalent of `test_hybrid_all_spatial.py`. Until that
runner exists, every successful run MUST go through the stitch helper
above so we don't lose the audit trail again.

### Don't skip persistence even on hard runs

If you reach a strict-regime dead end (Rule 1 forbids the Pi0-place
fallback that used to live here), still save the record. Set the regime
to whichever strict-allowed value applies (`"strict"` or `"pi0_doubled"`),
mark `libero_terminated` honestly (false if you couldn't fire the
predicate), and use `strategy_notes` to enumerate the LLM placement
variants you tried and why each failed. Negative-result audits are how
future sessions know not to retry the same dead end.

## Iteration heuristics

You are NOT limited to one shot per (task, seed). `reset` is cheap — keep
running fresh episodes with new strategies until one succeeds. The audit
JSON only captures the final (successful or final-failed) attempt, so
intermediate failed episodes are free.

When something fails, try these in order:

1. **Re-read state** — never assume objects are where you last left them.
2. **Re-pre-position** — gripper open, EEF above object xy at safe z.
3. **Tighten `tol`** for the failing move_to (0.02 → 0.008).
4. **Reduce `step_clip`** if mid-translation slip (0.025 → 0.015).
5. **Break move into stages**: lift in place → travel high → descend.
6. **Adjust `track_obj_lift_thresh`** — lower if Pi0 going too far; higher if
   exiting before secure grasp.
7. **Different prompt — walk the Rule 3 escalation ladder**: sub-instr →
   full BDDL task language → spatial qualifier → reposition+reset. For
   `libero_10` cluttered scenes, **start from full task language**, not
   sub-instr. Lower pre-pos z (e.g. 0.65) constrains Pi0 spatially.
8. **`reset` and try a fresh episode** if scene state has drifted irrecoverably
   (object on floor, knocked off cabinet, etc). Resetting also clears the
   "Pi0 fixated on wrong object" state — a fresh episode often picks the
   correct object on the next try.
8a. **After the second failed retry, render the failed-step images and
   describe what you see.** Rule 0 §"Failure forensics" covers this in
   full — `Read` the PNGs at pick, pre-release, post-release, and
   any articulation step; the visual disagreement with your mental
   model is where the bug is. Numerical tweaks without image
   inspection are how you lose 3 hours on a 10-minute problem
   (libero_10 t3 drawer is the canonical example).
9. **Pi0 fallback for the place is NOT an option** (Rule 1). If after many
   reset-and-retry episodes you still can't script the place, document the
   strict failure and move on.

Don't loop a single strategy more than 3 times within one episode. After 3
retries on the same plan, change the plan (different xy, different prompt,
different staging) — or `reset` and start a fresh episode.

## What "strict" means concretely

A task is **strict** if for EVERY pick-then-place cycle:
- The `pi0_pick` command exited with `libero_terminated=False` (Pi0 didn't
  finish the task).
- The `release` command was the one that triggered `libero_terminated=True`
  (the LLM's release primitive caused libero's `On`/`In` predicate to fire).

A task is **non-strict (Pi0 doubled)** if Pi0 was used for a non-pick VLA
skill (knob turn for stove tasks, drawer push for close-drawer tasks). This
is sometimes unavoidable — note it in your write-up but it's a weaker
demonstration.

A task **failed** if `libero_terminated` never fired before max_steps.

## Reference cases from prior session

Cases that worked strict (use as templates):

| Pattern | Example task | Key trick |
|---|---|---|
| Simple table pick → plate | libero_spatial t0 | Direct offset compensation |
| Bowl in closed drawer | libero_spatial t4 | `pick(..., track_obj=akita_black_bowl_1, lift_thresh=0.05)` with full prompt makes Pi0 open drawer, your track_obj cuts after lift |
| Bowl on cabinet top | libero_spatial t9 | Same as above but track interrupts during ascent above cabinet |
| Two objects → basket | libero_10 t0, t1, t7 | Slow step_clip=0.02, multi-stage move per pot |
| Two objects → two plates | libero_10 t4 | Same; separate pick+place loops per mug |
| Object into narrow cavity (In) | libero_10 t9 | `rotate_pitch +0.9` to thread cavity opening; `rotate_wrist +3.0` after release pushes object deeper + retreats. **Close(door) must be physical** (pi0_doubled "close the door" or OSC push) — no teleport (Rule 4); record strict_failure if unreachable |

Cases that failed strict (next session, try these):

- **libero_10 t8 (both moka → stove)**: ✅ **Solved 2026-05-19** with strict
  hybrid (Pi0 only for both picks, LLM for all moves/places). Key insight:
  the cook_region is a 15×15cm box centered at `(-0.050, -0.200)`. **Place
  the two pots at opposite corners, never one in the middle** — the first
  moka in center leaves no room for the second and they collide on release.
  Working layout: moka_2 → back-left `(-0.091, -0.228)`, moka_1 → front-right
  `(-0.014, -0.155)`, ~11 cm apart. Slow descend to z=1.05 (step_clip=0.01)
  converged in 93 steps without OSC stall. See `videos/t8_v4_SUCCESS.mp4`.
- **libero_10 t9 (mug in microwave + close door)**: In(mug, heating_region)
  is solved physically with `rotate_pitch`/`rotate_wrist`. ⚠ The Close(microwave)
  step previously used `articulate_to` (teleport) — RETRACTED under Rule 4. The
  door must be closed physically (pi0_doubled "close the door" or an OSC push from
  a reoriented pose); if unreachable, record a strict_failure. Key insights:
  1. **In(mug, heating_region)**: `rotate_pitch +0.9` tilts the gripper
     forward so its wrist body (~3 cm above eef) fits through the
     14-cm-tall cavity opening (z ∈ [0.944, 1.088]); without the pitch,
     OSC stalls before the eef can enter and any single-pose-axis variant
     fails for the same reason.
  2. **Close(microwave)**: the Panda struggles to reach the door panel via OSC
     (workspace boundary at x≈-0.08, door at x≈-0.18). Closing it physically is
     hard — hand it to `pi0_doubled` ("close the microwave door") or push from a
     reoriented non-singular pose; a swept push can knock the just-placed mug back
     out along the door's swing arc, so push gently and re-check. No teleport (Rule 4).
  3. The mug must be pushed deeper than y≈0.29 BEFORE closing the door,
     otherwise the door's leading edge sweeps it back out. The
     `rotate_wrist delta_yaw=+3.0` after release does triple duty
     (unhook + retreat + nudge-mug-deeper) and gets the mug to y≈0.33
     reliably.

  The prior session's `pi0_end_to_end` audit entry is grandfathered
  but is forbidden by Rule 1; the current strict recipe replaces it.

## Appendix: LLM-scripted pick fallback (when Pi0 won't cooperate)

**Read Rule 3 ("LLM is a delivery service for Pi0") and walk the Pi0
prompt escalation ladder first.** Specifically — if your only Pi0
attempt was a sub-instruction `"pick up the X"` and it failed once,
try the **full BDDL task language** before reaching this appendix.
That single change has resolved `libero_10` cluttered-scene picks
that looked unrecoverable. See the `libero_10_lan t3` retry case.

Only after Pi0 has failed across (a) sub-instr, (b) full task
language, (c) spatial qualifier, and (d) reposition + reset, may you
script the pick yourself. This violates "Pi0 only for pick" strictly
but is a valid escalation when Pi0's training distribution genuinely
does not cover the current scene state.

The recipe:

```jsonc
// 1. Pre-pos high above the object (gripper open, large step OK)
{"action": "move_to", "xyz": [obj_x, obj_y, obj_z+0.25], "gripper": -1,
 "tol": 0.012, "step_clip": 0.03, "max_steps": 60}

// 2. Descend until EEF auto-stalls on the object (DO NOT force a low target).
//    Aim eef_z = obj_z + 0.05 with small step_clip. Likely stops above target
//    when fingers contact object. Read state — if eef_z > obj_z + 0.08, the
//    EEF stopped early (something else blocking) → re-align xy and retry.
{"action": "move_to", "xyz": [obj_x, obj_y, obj_z+0.05], "gripper": -1,
 "tol": 0.008, "step_clip": 0.015, "max_steps": 60}

// 3. Close gripper firmly (10-15 steps). Critical to wait long enough —
//    the PD takes ~10 steps to fully clamp.
{"action": "set_gripper", "gripper": 1, "steps": 15}

// 4. Read state: gripper qpos sum should be ~0.03-0.06 (NOT ~0.0).
//    If sum ≈ 0 → fingers touched (empty grasp) → re-position xy +/- 5mm and retry.
//    If sum > 0.06 → gripper still partly open (didn't clamp) → repeat set_gripper.

// 5. Lift verify: move EEF up by 15cm; read state. obj_z should track eef_z
//    within ±1cm (the offset_z observed at grasp time).
{"action": "move_to", "xyz": [obj_x, obj_y, eef_z+0.15], "gripper": 1,
 "tol": 0.012, "step_clip": 0.02, "max_steps": 40}
```

**Verification metric (this is your strict-grasp criterion)**:
`(obj_z_after_lift - obj_z_at_grasp) >= 0.07` AND `0.02 < gripper_qpos_sum < 0.07`.

Anything else is a false grasp — release and retry.

## Appendix: When EEF stalls on place (geometry trap)

Symptom: `move_to` runs `max_steps` but `final_dist_m > tol`, and the EEF
position barely moves between successive states.

Causes:
- Robot wrist hitting a housing top (stove, microwave, cabinet top plate).
- Held object's bottom touching the target before EEF reaches its target z.
- Panda IK at a near-singular config can't translate without rotation.

**You CANNOT force the EEF lower with OSC** — there's no command to push past
the limit. Options:

1. **Release here anyway**: if `obj_z - target_z < 0.10`, gravity finishes
   the job. Drift may be 1-3 cm.
2. **Approach from a different angle**: lift back up, translate xy to a
   different approach vector, descend. Sometimes IK frees up.
3. **Aim eef_z higher**: accept release height > optimal. For tall objects
   (moka pot, mug) in stove/microwave context, sometimes `eef_z = obj_z + 0.10`
   is the lowest you can reach.

If after 3 staging variants the EEF still stalls and release induces > 10 cm
drift, **reset and try a completely fresh episode** with a different overall
plan — different approach vector, different stage order, pre-oriented wrist,
or pushing the object across the surface instead of lifting it through the
singular region. Rule 1 forbids using Pi0 to do the place, so iterating
within LLM scripting (across multiple episodes if needed) is the only path.
If after many fresh-episode attempts no scripted variant works, document
the task as a strict failure in `strategy_notes`.

## Memory files to read

**You have access to a persistent auto-memory at**
`/root/.claude/projects/-mnt-public2-zhangyixian/memory/`. The index is
`MEMORY.md` (one-line hooks per memory, auto-injected by CLAUDE.md into
the system prompt of every new session). Individual entries live as
`feedback_*.md` / `project_*.md` / `reference_*.md`.

**Before starting a new (suite, task) attempt**: scan `MEMORY.md` for
entries whose names or descriptions touch the cell you're about to do
(e.g. "swap", "plate", "stove", the task type), then `Read` the matching
`.md` for the full rule. Doing this saves hours: many of the memories
below capture root causes that took 30+ minutes to diagnose, with the
fix-recipe written down so you skip straight to the working approach.

**When to write memory**: any time you fix a failure that took more than
two iterations to diagnose, or find a non-obvious env quirk (predicate
threshold, fixture offset, finger geometry constraint). Save it as a
new `feedback_*.md` with `type: feedback` frontmatter and a one-line
hook in `MEMORY.md`. The next session — yours or someone else's — will
skip your debugging cycle.

### High-signal entries for libero hybrid work

Project-level orientation:
- `project_rlinf_agentic_workdir.md` — default cwd
- `project_liberopro_install.md` — LIBERO-Pro install + benchmark patch
- `project_libero_spatial_t0_t9_pro.md`, `project_libero_10_t0_t9_done.md`,
  `project_libero_object_pro_done.md`, `project_libero_goal_pro_progress.md`
  — per-suite progress / corpus pointers
- `project_libero_hybrid_llm_vla.md` — running record of findings
- `project_pi05_libero_prompt_blind.md` — Pi0 prompt-blind discovery on
  libero_spatial
- `feedback_no_teleport_rule.md` — the four teleport primitives are removed;
  physics-only recipes for stove-knob / drawer (READ THIS — Rule 4)
- `project_rotate_pitch_primitive.md` — `rotate_pitch` (§2)
- `project_libero_t9_pi0_only.md` — superseded by t9 strict, kept as
  historical reference

Pi0 prompt + grasp behavior:
- `feedback_pi0_pick_full_prompt.md` — when to use FULL BDDL prompt
- `feedback_pi0_delivery_service.md` — Pi0 is for grasp, LLM for the
  rest (Rule 3 escalation ladder; see §"Rule 3" above)
- `feedback_pi0_pre_pos_can_hurt.md` — for some objects (plates,
  swap-side picks), SKIP the pre-pos `move_to` and let Pi0 plan from
  the default home pose. Pre-pos can break Pi0's learned approach.
- `feedback_pi0_false_positive_lift.md` — `pi0_pick.success=True` may
  fire on eef-ascent without grasping; gate on `track_obj_final_z`.
- `feedback_pi0_chunks_egl_crash.md` — keep `max_chunks` modest
- `feedback_read_image_before_decide.md` — Rule 0 reinforcement

Env mechanics + predicate quirks:
- `feedback_cook_region_offset.md` — `flat_stove_1_cook_region` is +0.15m
  offset from stove fixture origin via burner sub-body
- `feedback_bowl_eef_y_offset.md` — bowl rim-hook leaves bowl ~4.5cm in
  eef-frame -y; compensate `eef_y_target = predicate_y + 0.045`
- `feedback_stove_turnoff_strict.md` — TurnOn fires at knob qpos≥0.5, TurnOff
  at qpos<0; drive the knob PHYSICALLY (pi0_doubled), no teleport
- `feedback_osc_push_mujoco_nan.md` — for Close(slide-drawer) push continuously
  with a CAPPED OSC push or pi0_doubled; a long push can NaN the sim
- `feedback_swap_perturbs_fixtures.md` — libero_goal P2 swap moves
  ENTIRE FIXTURES (stove/cabinet/rack), not just table objects. Read
  the swap BDDL `:init` block before computing carry targets.

Control + transport tactics:
- `feedback_scripted_pick_limits.md` — fully scripted pick fails for
  small objects (<6cm). Keep Pi0 for picks, scripted only for clear
  geometry.
- `feedback_render_skip_in_env_step.md` — env render toggle to lift the
  EGL command budget
- `feedback_max_episode_steps_libero.md` — use
  `--max_episode_steps 5000` for REPL sessions
- `feedback_failure_forensics.md` — after 2 retries on the same plan,
  STOP tuning and `Read` the images
- `feedback_workspace_bounds_clamp.md` — editing workspace bounds also
  clamps action range

Cross-suite progress + non-obvious past failures:
- `feedback_liberopro_driver_patch.md` — `make_env` benchmark routing
- `feedback_no_pi0_end_to_end.md` — Rule 1 reinforcement
- `feedback_rotate_wrist_yaw_sign.md` — yaw sign bug fix
- `feedback_physics_eval_pipeline.md`, `feedback_checkpoint_frequency.md`
  — (BuilderBench-only, skip for libero work)
- `project_libero_10_t0_t5_pro.md` — libero_10 t0–t5 PRO; drawer-close must
  be a real continuous push (no teleport)

### Workflow

```
1. New (suite, task) → cat MEMORY.md, scan for relevant entries.
2. Read the matching feedback_*.md / project_*.md files BEFORE starting.
3. Apply the documented fix; don't rediscover.
4. If you discover something new (took > 2 iterations to find): write
   a new feedback_*.md and add a one-line hook to MEMORY.md.
```

## TL;DR launch checklist

```
0. cat /root/.claude/projects/-mnt-public2-zhangyixian/memory/MEMORY.md
   → scan one-line hooks; Read matching .md files for relevant fixes.
1. cd /mnt/public2/zhangyixian/RLinf_agentic
2. Bash run_in_background:true
     CUDA_VISIBLE_DEVICES=0 /opt/venv/openpi/bin/python \
         examples/embodiment/primitives/interactive_driver.py \
         --suite libero_10 --task <N> --seed 0 --max_episode_steps 5000
3. Bash run_in_background:true (wait for state_00.json)
     until [ -f /tmp/hybrid_repl/state_00.json ] && \
            [ -s /tmp/hybrid_repl/state_00.json ]; do sleep 5; done
4. Read state_00.json, image_00.png. Identify target objects + goal regions.
5. Iterate: REPL command → done_NN.flag → Read image_NN.png (Rule 0)
            → Read state_NN.json → next command.
   Append each command to a recipe_tN_sM.jsonl scratch file as you go.
6. For each pick:
     Write command.json (pi0_pick + track_obj)
     Confirm pick_result.libero_terminated=False AND target lifted.
7. For each place:
     Compute offset, write command.json (move_to, slow step_clip, multi-stage)
     Then release.
     Confirm libero_terminated=True in release's log.
     If OSC `move_to` stalls at the same xy across 2+ variants, the task
     hits a Panda workspace boundary — reorient (rotate_wrist/rotate_pitch)
     and approach from a non-singular pose, or release if the held object's
     xy is already over the predicate region. There is NO teleport (Rule 4).
     If the cavity opening is too narrow for the wrist body geometry, add
     `rotate_pitch` (§2) BEFORE the push to tilt the gripper forward.
8. For Close(articulation) goals (door / drawer / lever close), PUSH the
   articulation physically AFTER placing the object: `pi0_doubled` (Pi0
   "close the drawer/door") or a CAPPED OSC push that keeps continuous
   contact (drags the object along). No teleport (Rule 4). If no physical
   push satisfies the predicate (e.g. drawer is_close needs qpos>0), record
   a strict_failure with the predicate decomposition.
9. Audit: pick_term=False AND release_term=True (or close fires during the
   physical push) → strict (or pi0_doubled if Pi0 did the contact skill).
   If LLM placement misses, `reset` and run a FRESH episode with a
   different plan (multi-episode iteration is allowed and encouraged — see
   "Iteration heuristics"). Do NOT call Pi0 to finish the place: Rule 1
   forbids it.
10. **Persist**: run the stitch helper above to produce
     results_all_<suite>/tN_sM.json + update all_rows.json,
     set regime + strategy_notes correctly (`"strict"` or `"pi0_doubled"`
     only — never `"pi0_end_to_end"`). Required: REPL files do NOT persist
     past driver exit.
11. Write exit. Move on to next task.
12. **Memory write-back**: if you discovered a non-obvious fix or env
     quirk that took >2 iterations to diagnose, save it as
     `/root/.claude/projects/-mnt-public2-zhangyixian/memory/feedback_<name>.md`
     (YAML frontmatter: `name`, `description`, `type: feedback`; body
     leads with the rule, then `**Why:**` and `**How to apply:**` lines)
     AND append a one-line hook to `MEMORY.md`. Next session reads it
     via the system prompt and skips your debugging cycle.
```

Good luck. Don't let Pi0 do the place — that's the LLM's job (Rule 1).
For OSC singularities, reorient with `rotate_wrist`/`rotate_pitch` and
approach from a non-singular pose; for articulation closes, push physically
(pi0_doubled / capped OSC push). NO teleport — see Rule 4.
**Before debugging from scratch, check memory.**
