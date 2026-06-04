# Hybrid LLM + Pi0.5 VLA — Work Log

This document summarizes what's been built and learned in the hybrid LLM + Pi0.5
VLA experiments (2026-05-18 → 2026-05-19) on the LIBERO benchmark.

## Goal

**Compose Pi0.5 (VLA) with Claude (LLM)** to solve LIBERO tasks, where:
- Pi0.5 handles **VLA-specific skills** (primarily `pick`).
- The LLM handles **symbolic/spatial reasoning** — read scene state, compute
  target poses, sequence scripted primitives (`move_to`, `release`,
  `set_gripper`), inspect rendered images for sanity checking.

The strictest version: **Pi0.5 only does the pick**, LLM scripts all the rest.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  examples/embodiment/primitives/                                    │
│                                                                     │
│  primitives.py            ─→  LiberoPrimitiveDriver class:          │
│                                  pick / release / move_to /         │
│                                  render_agentview /                 │
│                                  get_privileged_state               │
│                                                                     │
│  interactive_driver.py    ─→  REPL: one process loads Pi0.5 once,   │
│                                  polls /tmp/hybrid_repl/command.json│
│                                  dumps state_NN.json + image_NN.png │
│                                  signals via done_NN.flag           │
│                                                                     │
│  hybrid_pick_place.py     ─→  one-shot single-task driver           │
│                                                                     │
│  test_hybrid_all_spatial  ─→  batch runner (formula-based,          │
│                                  no LLM-in-the-loop)                │
│  test_hybrid_failed_retry ─→  retry failures with full prompt       │
└─────────────────────────────────────────────────────────────────────┘
```

### The track_obj mechanism — central trick

`primitives.py:pick(track_obj=..., track_obj_lift_thresh=...)` watches the
named object's z position (privileged sim state) and **breaks the Pi0.5 loop
the moment the object has lifted by the threshold**. This is what enforces
"Pi0 does pick only, not place" even when Pi0 was trained on a full pick+place
trajectory and would otherwise continue.

```python
if track_obj is not None:
    obj_z = float(self.env.current_raw_obs[self.env_idx][f"{track_obj}_pos"][2])
    if (obj_z - track_obj_init_z) >= track_obj_lift_thresh:
        success = True; break
```

### REPL handshake protocol

`interactive_driver.py` polls `/tmp/hybrid_repl/command.json`. Each command is
JSON describing one action:

| action | fields | what it does |
|---|---|---|
| `pi0_pick` | `prompt, max_chunks, track_obj, track_obj_lift_thresh` | Pi0.5 closed-loop chunk rollout |
| `move_to` | `xyz, gripper, tol, step_clip, max_steps, action_scale` | scripted EEF servo |
| `release` | `max_steps` | hold pose, open gripper |
| `set_gripper` | `gripper, steps` | hold pose, command gripper N steps |
| `reset` | — | reset env to initial state |
| `exit` | — | clean shutdown |

After each command the driver writes `state_NN.json` + `image_NN.png` +
`log_NN.json` and creates `done_NN.flag`. The orchestrator (Claude) writes the
next command file.

## Results

### libero_spatial (10 tasks × 2 seeds = 20 rollouts) — 2026-05-18

Two passes:
- **Pass 1 (sub-instr + scripted offset-compensated place):** 14/20 libero_term.
- **Pass 2 (full prompt rescue on 6 failures):** 6/6 → combined **20/20**.

Result by task (strict regime "Pi0 only pick + LLM all place"):

| task | strict success | failure mode if any |
|---|---|---|
| 0–3, 5–8 | ✓ (random misses on t1, t6 individual seeds) | — |
| **4 (drawer)** | ✗ → ✓ via track_obj | bowl inside closed drawer; Pi0 sub-instr reflexively goes to table z. Solved via `pick(track_obj=akita_black_bowl_1, lift_thresh=0.05)` + full prompt → Pi0 opens drawer, picks, my track_obj fires after 5cm lift → LLM places. |
| **9 (cabinet)** | ✗ → ✓ via track_obj | bowl on top of cabinet; same idea. |

Per-rollout audit (`results_all_spatial/t*_s*.json`):
- All 20 had `pick_result.libero_terminated=False` and `release_result.libero_terminated=True`
  → **libero "On" predicate fires inside the LLM's release primitive, not during Pi0's pick**.
- Move converged in 15–80 steps; release in 3–25 steps.

### libero_10 (libero_long) — 2026-05-19

Interactive REPL pass:

| task | description | libero_term | regime |
|---|---|---|---|
| 0 | put both soup + tomato in basket | ✓ | **strict** |
| 1 | put both cream cheese + butter in basket | ✓ | **strict** |
| 2 | turn on stove + put moka pot | ✓ | Pi0 doubled as knob-turn |
| 3 | put bowl in drawer + close | ✓ | Pi0 doubled as drawer-close |
| 4 | 2 mugs → 2 plates | ✓ | **strict** |
| 5 | book in caddy back compartment | ✓ | LLM release stuck on caddy rim → Pi0 re-placed |
| 6 | mug on plate + pudding right-of | ✓ | **strict** |
| 7 | both soup + cream cheese in basket | ✓ | **strict** |
| **8** | both moka pots on stove | ✓ | **strict, v4 opposite-corner** — see below |
| **9** | mug in microwave + close | ✓ | **Pi0 end-to-end (non-strict)** — see below |

**Totals: 10/10 libero_term; 6/10 are strict ("Pi0 only pick + LLM all place").**

#### t9 deep-dive: OSC IK barrier forced Pi0 end-to-end (2026-05-19)

t9 is the only libero_10 task where **strict hybrid was not achievable**. The
LLM scripting could not push the EEF past `y ≈ 0.26` at the cavity-entry
height `z ≈ 1.05-1.10` — across 6+ staging variants (direct push, lift-
then-descend, over-the-top, left/right side, diagonal, max-step_clip), the
EEF stalled with `final_dist > 0.10 m` and `steps_used == max_steps`.
This is not a parameter-tuning problem; it is a Panda IK / OSC singularity
in this specific configuration. The Panda base is at world `(-0.66, 0, 0.912)`
and the cavity centre is at `(-0.01, +0.35, +1.02)` — the EEF must extend
~0.72 m diagonally while threading through a narrow front opening (cavity
`x ∈ [-0.152, +0.055]`, `y ∈ [+0.273, +0.440]`, `z ∈ [+0.944, +1.088]`)
between an open door panel (`x = -0.19`) on the left and a thick right
wall (`x ∈ [+0.055, +0.164]`) on the right. There simply is not enough
free workspace for the Panda's link 6-7 chain at any orientation I could
script.

**Pi0 end-to-end (non-strict) works**: with the full task prompt and no
`track_obj` cut, Pi0 solved t9 in 186 chunks (~930 env steps). It picks,
places into the cavity, and closes the door — all by closed-loop visual
servoing learned from libero_130 SFT. Video: `videos/t9_pi0_SUCCESS.mp4`.

**Geometry notes for future strict attempts (if anyone tries):**
- All microwave geoms are `contype=0 conaffinity=1` (mug/gripper
  `contype=1 conaffinity=1` ⇒ they DO collide; ignore the misleading
  `contype=0`).
- Microwave body: `x ∈ [-0.181, +0.164]`, `y ∈ [+0.271, +0.470]`,
  `z ∈ [+0.920, +1.107]` (left/right/back walls + top + bottom).
- Cavity interior: `x ∈ [-0.152, +0.055]`, `y ∈ [+0.271, +0.440]`,
  `z ∈ [+0.944, +1.088]`. **Sealed on top** — cannot descend from above.
- Open door at `qpos = -1.579 rad`: world bbox `x ∈ [-0.208, -0.182]`,
  `y ∈ [+0.012, +0.272]`, `z ∈ [+0.923, +1.109]`. Door panel hangs
  forward toward robot, fouling much of the left-front workspace.
- Heating-region predicate: `x ∈ [-0.13, +0.11]`, `y ∈ [+0.273, +0.439]`,
  `z ∈ [+0.936, +1.096]` (subset of cavity interior).
- Door close predicate: needs `Close(microwave)` ≈ joint qpos near 0.

#### t8 deep-dive: opposite-corner placement was the unlock (2026-05-19)

After 3 failed strict attempts placing the first moka in cook_region center
(then second moka kept colliding with it), user observed:

> 需要给两个 Moka 都留出位置，而不是一味的给一个 moka 放中间，这样另一个肯定会 collision

- cook_region is a **15×15cm box site**, center `(-0.050, -0.200)`,
  bounds `x ∈ [-0.125, +0.025], y ∈ [-0.275, -0.125]`
- Moka pots are ~6 cm wide; placing one in the middle leaves no clearance
- **Working layout (v4)**: moka_2 at back-left `(-0.091, -0.228)`,
  moka_1 at front-right `(-0.014, -0.155)` — ~11 cm center-to-center
- 93-step descend reached z=1.061 (target 1.05, within 1.1cm tol) — the
  "OSC stall at z=1.13" failure mode from earlier attempts did NOT recur.
  Likely those stalls were caused by wrist mis-alignment or a worse
  starting EEF orientation, not by stove geometry alone.
- Released cleanly → `libero_terminated=True`. See
  `videos/t8_v4_SUCCESS.mp4` (522 frames, ~21s).

## Key Findings

### 1. Pi0.5 on libero_spatial is prompt-blind (vision-driven)

Negative control test: prompt "pick up the alphabet soup" (libero_object item,
absent from libero_spatial scenes) gave **9/9 picks identical to the correct
sub-instruction** (stroke 0.245–0.269 m, gripper closed). Pi0.5 picks whatever
canonical object is on the table regardless of prompt. See
`results_all_spatial/t*_s*_pick_negctrl.json`. This is fine for libero_spatial
(unambiguous scene) but means we can't rely on prompt-following for
multi-object scenes.

On libero_10 (8 distinct objects per scene) prompt-following DOES matter and
Pi0.5 does respond to which object to pick — at least when the scene is in
its training distribution.

### 2. Pick-termination predicate must include post-min ascent

Original `pick()` exited on `peak_z - min_z >= 5 cm`, but that's the descent
depth (peak is the starting z). The model could exit at the BOTTOM of descent
without ever lifting. Fixed to track `post_min_peak_z` separately:

```python
if z < min_z:
    min_z = z
    post_min_peak_z = z   # reset after new deeper min
else:
    post_min_peak_z = max(post_min_peak_z, z)
ascended = (post_min_peak_z - min_z) >= lift_thresh
```

Combined predicate: `descent_done (≥ 10 cm) AND ascended (≥ 5 cm post-min) AND gripper closed`.

### 3. Offset compensation is mandatory

The gripper holds an object with a non-zero `bowl_pos - eef_pos` offset (2–10
cm typical, varies per grasp due to symmetry / approach angle). Naively
moving EEF to `plate_xy` drops the object 3–10 cm off-target. Always read
`get_privileged_state()` immediately after pick to measure the offset:

```python
offset = bowl_pos - eef_pos
target_eef_xy = target_object_xy - offset_xy
target_eef_z  = target_object_z + half_height + margin - offset_z
```

### 4. Release-on-contact > release-from-height

Dropping from ≥ 5 cm above the target induces 3–4 cm lateral drift (asymmetric
gripper opening pushes the object). Aim EEF z so that
`object_bottom = target_top + 1–2 cm` before release.

### 5. Slow step_clip + multi-stage move for cross-table travel

`move_to(target, step_clip=0.04)` works for short moves on libero_spatial
(bowls). But on libero_10 cans/boxes, the same step_clip causes objects to
slip out of the gripper mid-translation. Empirical rule:

- libero_spatial pick-and-place: `step_clip=0.025–0.04` fine.
- libero_10 cylindrical/box objects: `step_clip ≤ 0.02` required.
- Long diagonal moves: break into **lift → translate at high altitude →
  descend → release**. Single-shot diagonal often OSC-stalls partway.

### 6. Object disturbance during approach

Open gripper fingers (~8 cm wide) sweep into objects on the way to the
target. The basket / moka pots get pushed 5–10 cm. After each motion command,
**re-read state JSON** before computing the next target. Don't assume objects
stay put.

### 7. Pi0.5 + scripted hybrid breaks on:
- **Scene state drift**: when the scene is partway through a task (e.g. one
  moka already placed), Pi0 may ignore the remaining object — its training
  distribution had a fixed init.
- **Narrow workspace**: OSC controller can't path-plan around obstacles
  (microwave top plate). EEF stalls when blocked. Would need
  collision-aware planning.

## Files

```
examples/embodiment/primitives/
├── primitives.py                # Driver class
├── interactive_driver.py        # REPL — main tool for LLM-in-the-loop
├── hybrid_pick_place.py         # One-shot single task
├── test_hybrid_all_spatial.py   # Batch (no LLM loop)
├── test_hybrid_failed_retry.py  # Batch retry
├── results_all_spatial/         # 20 rollout JSONs (libero_spatial pass 1)
├── results_all_spatial_retry/   # 6 rescue JSONs (pass 2)
├── WORK_DONE.md                 # this file
└── STRICT_HYBRID_GUIDE.md       # handover for next Claude session
```

## Key checkpoint

`/mnt/public2/data_move/16T_5_slz_zyx/zhangyixian/zhangyixian/pi05_libero130_fullshot/30000`
— Pi0.5 SFT on libero_130 fullshot, 30k steps. Used unchanged throughout.

## Venv

`/opt/venv/openpi/bin/python` has libero + openpi + CUDA installed. No need to
rebuild.
