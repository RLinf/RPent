# Teleport-primitive redo checklist

_Generated 2026-05-26._

Recipes that achieved success using a **teleport** primitive. There are FOUR teleport primitives — all bypass contact physics by writing qpos directly or warping the arm/object kinematically:

- **`set_object_pose`** — warps an object's free-joint qpos straight to the goal coords; the object jumps, bypassing grasp/carry physics.
- **`articulate_to`** — writes a door/drawer/knob joint qpos directly; the articulation snaps to target.
- **`js_move_to`** — kinematically warps the arm's 7 joint qpos (`mj_forward`, no controller) along an RRT path; the arm teleports between waypoints.
- **`carry_object`** — a `js_move_to` variant that ALSO rewrites a held object's qpos every waypoint so the object rides the warped arm (always implies `js_move_to`).

These successes do not demonstrate physical manipulation and must be redone with **physics-only** primitives (OSC `move_to` / `rotate_wrist` / `rotate_pitch` / `release` / `set_gripper`, and `pi0_pick` / `pi0_doubled` for VLA contact skills such as knob turn and drawer open/close).

## Totals

- **Reported 10-seed results (`multi_seed_exp/`): 383 recipes** use a teleport primitive (230 use `set_object_pose`, 198 use `articulate_to`, 57 use `js_move_to`, of which 47 use `carry_object`).
- **Source library (`results_*_pert/`): 22 recipes** use a teleport primitive (0 use `set_object_pose`, 20 use `articulate_to`, 2 use `js_move_to`, of which 2 use `carry_object`).

> **Note:** the previous version of this checklist counted only `set_object_pose` + `articulate_to`. Adding `js_move_to` / `carry_object` (also non-physical arm/object warps) increases the redo scope; the rows below reflect the full four-primitive set.

## Reported results — cells to redo (per env/regime/task: # seeds teleport-affected)

| env | regime | task | seeds w/ teleport | set_object_pose | articulate_to | js_move_to | carry_object |
|---|---|---|---|---|---|---|---|
| libero_10 | lan | t0 | 3/10 | 3 | 0 | 0 | 0 |
| libero_10 | lan | t1 | 1/10 | 1 | 0 | 0 | 0 |
| libero_10 | lan | t2 | 10/10 | 9 | 10 | 0 | 0 |
| libero_10 | lan | t3 | 10/10 | 10 | 10 | 0 | 0 |
| libero_10 | lan | t4 | 2/10 | 2 | 0 | 0 | 0 |
| libero_10 | lan | t5 | 10/10 | 10 | 0 | 0 | 0 |
| libero_10 | lan | t6 | 6/10 | 6 | 0 | 0 | 0 |
| libero_10 | lan | t7 | 7/10 | 7 | 0 | 0 | 0 |
| libero_10 | lan | t8 | 10/10 | 8 | 8 | 0 | 0 |
| libero_10 | lan | t9 | 10/10 | 9 | 10 | 1 | 0 |
| libero_10 | swap | t0 | 8/10 | 8 | 0 | 1 | 1 |
| libero_10 | swap | t1 | 5/10 | 5 | 0 | 0 | 0 |
| libero_10 | swap | t2 | 10/10 | 7 | 10 | 0 | 0 |
| libero_10 | swap | t3 | 10/10 | 5 | 10 | 6 | 6 |
| libero_10 | swap | t5 | 3/10 | 3 | 0 | 0 | 0 |
| libero_10 | swap | t6 | 2/10 | 2 | 0 | 0 | 0 |
| libero_10 | swap | t7 | 8/10 | 8 | 0 | 0 | 0 |
| libero_10 | swap | t8 | 10/10 | 10 | 0 | 0 | 0 |
| libero_10 | swap | t9 | 10/10 | 10 | 10 | 3 | 3 |
| libero_10 | task | t2 | 10/10 | 10 | 10 | 0 | 0 |
| libero_10 | task | t3 | 10/10 | 10 | 10 | 0 | 0 |
| libero_10 | task | t4 | 1/10 | 1 | 0 | 0 | 0 |
| libero_10 | task | t5 | 4/10 | 4 | 0 | 0 | 0 |
| libero_10 | task | t6 | 5/10 | 5 | 0 | 0 | 0 |
| libero_10 | task | t7 | 5/10 | 5 | 0 | 0 | 0 |
| libero_10 | task | t8 | 10/10 | 0 | 10 | 1 | 0 |
| libero_10 | task | t9 | 10/10 | 10 | 10 | 1 | 0 |
| libero_goal | lan | t0 | 10/10 | 0 | 10 | 0 | 0 |
| libero_goal | lan | t3 | 10/10 | 0 | 10 | 0 | 0 |
| libero_goal | lan | t5 | 10/10 | 1 | 0 | 10 | 10 |
| libero_goal | lan | t7 | 10/10 | 0 | 10 | 0 | 0 |
| libero_goal | lan | t9 | 1/10 | 1 | 0 | 0 | 0 |
| libero_goal | swap | t0 | 10/10 | 0 | 10 | 0 | 0 |
| libero_goal | swap | t2 | 3/10 | 3 | 0 | 0 | 0 |
| libero_goal | swap | t3 | 10/10 | 0 | 10 | 0 | 0 |
| libero_goal | swap | t5 | 10/10 | 0 | 0 | 10 | 10 |
| libero_goal | swap | t6 | 1/10 | 1 | 0 | 0 | 0 |
| libero_goal | swap | t7 | 10/10 | 0 | 10 | 0 | 0 |
| libero_goal | task | t0 | 10/10 | 0 | 10 | 0 | 0 |
| libero_goal | task | t1 | 8/10 | 8 | 0 | 0 | 0 |
| libero_goal | task | t3 | 10/10 | 0 | 10 | 0 | 0 |
| libero_goal | task | t4 | 2/10 | 1 | 0 | 1 | 1 |
| libero_goal | task | t6 | 2/10 | 2 | 0 | 0 | 0 |
| libero_goal | task | t7 | 10/10 | 0 | 10 | 0 | 0 |
| libero_goal | task | t8 | 1/10 | 1 | 0 | 0 | 0 |
| libero_object | lan | t2 | 3/10 | 3 | 0 | 1 | 0 |
| libero_object | lan | t7 | 1/10 | 1 | 0 | 0 | 0 |
| libero_object | lan | t9 | 3/10 | 3 | 0 | 1 | 1 |
| libero_object | swap | t2 | 8/10 | 8 | 0 | 0 | 0 |
| libero_object | task | t4 | 3/10 | 3 | 0 | 0 | 0 |
| libero_spatial | lan | t0 | 2/10 | 2 | 0 | 0 | 0 |
| libero_spatial | lan | t7 | 3/10 | 3 | 0 | 0 | 0 |
| libero_spatial | lan | t8 | 1/10 | 1 | 0 | 0 | 0 |
| libero_spatial | swap | t3 | 6/10 | 6 | 0 | 0 | 0 |
| libero_spatial | swap | t4 | 4/10 | 4 | 0 | 0 | 0 |
| libero_spatial | swap | t5 | 3/10 | 3 | 0 | 0 | 0 |
| libero_spatial | swap | t6 | 8/10 | 1 | 0 | 7 | 7 |
| libero_spatial | swap | t7 | 1/10 | 1 | 0 | 0 | 0 |
| libero_spatial | swap | t9 | 8/10 | 0 | 0 | 8 | 8 |
| libero_spatial | task | t1 | 6/10 | 0 | 0 | 6 | 0 |
| libero_spatial | task | t3 | 2/10 | 2 | 0 | 0 | 0 |
| libero_spatial | task | t4 | 1/10 | 1 | 0 | 0 | 0 |
| libero_spatial | task | t7 | 1/10 | 1 | 0 | 0 | 0 |
| libero_spatial | task | t9 | 1/10 | 1 | 0 | 0 | 0 |

## Reported results — by environment

| env | teleport recipes | set_object_pose | articulate_to | js_move_to | carry_object |
|---|---|---|---|---|---|
| libero_spatial | 47 | 26 | 0 | 21 | 15 |
| libero_object | 18 | 18 | 0 | 2 | 1 |
| libero_goal | 128 | 18 | 90 | 21 | 21 |
| libero_10 | 190 | 168 | 108 | 13 | 10 |

## Source library (`results_*_pert/`) — teleport recipes

These are the seed-0 'verified recipes' that multi-seed runs replayed. The `teleport` column lists every teleport primitive each recipe uses.

| folder | env | regime | task | teleport |
|---|---|---|---|---|
| results_10_pert | libero_10 | lan | t2 | articulate_to |
| results_10_pert | libero_10 | lan | t3 | articulate_to |
| results_10_pert | libero_10 | lan | t8 | articulate_to |
| results_10_pert | libero_10 | swap | t2 | articulate_to |
| results_10_pert | libero_10 | swap | t3 | articulate_to |
| results_10_pert | libero_10 | swap | t8 | articulate_to |
| results_10_pert | libero_10 | task | t3 | articulate_to |
| results_10_pert | libero_10 | task | t8 | articulate_to |
| results_goal_pert | libero_goal | base | t0 | articulate_to |
| results_goal_pert | libero_goal | base | t3 | articulate_to |
| results_goal_pert | libero_goal | base | t7 | articulate_to |
| results_goal_pert | libero_goal | lan | t0 | articulate_to |
| results_goal_pert | libero_goal | lan | t3 | articulate_to |
| results_goal_pert | libero_goal | lan | t5 | carry_object |
| results_goal_pert | libero_goal | lan | t7 | articulate_to |
| results_goal_pert | libero_goal | swap | t0 | articulate_to |
| results_goal_pert | libero_goal | swap | t3 | articulate_to |
| results_goal_pert | libero_goal | swap | t5 | carry_object |
| results_goal_pert | libero_goal | swap | t7 | articulate_to |
| results_goal_pert | libero_goal | task | t0 | articulate_to |
| results_goal_pert | libero_goal | task | t3 | articulate_to |
| results_goal_pert | libero_goal | task | t7 | articulate_to |

## Physical-redo status — source library (seed 0), as of 2026-05-27

**Coverage: YES — all 22 source-library teleport recipes have been re-attempted
physics-only and EACH has an audit JSON in its `results_*_pert/` directory (none
missing).** Outcome: **17 SOLVED** (`libero_terminated=True`, no teleport) + **5 honest
`strict_failure_physical`**. The SOLVED audits replace the teleport recipes as the
physical ground truth; the 5 failures are documented dead-ends (the predicates the
teleport originally faked).

### `results_10_pert` (8 cells): 5 SOLVED / 3 blocked

| cell | status | regime | how / why |
|---|---|---|---|
| `10_lan_t2_s0`  | ✅ SOLVED | strict_pi0_doubled_physical | stove: pi0_doubled knob turn + scripted moka pick + OSC carry/release |
| `10_swap_t2_s0` | ✅ SOLVED | strict_pi0_doubled_physical | stove (same pattern) |
| `10_lan_t8_s0`  | ✅ SOLVED | strict_pi0_doubled_physical | both mokas → stove, 2 pick+carry+place cycles |
| `10_task_t8_s0` | ✅ SOLVED | strict_pi0_doubled_physical | left moka → stove, ultra-slow 0.08 m-hop carry |
| `10_lan_t3_s0`  | ✅ **SOLVED 2026-05-27** | pi0_doubled | drawer close to qpos>0: pi0 close → OSC front-approach push → pi0 close (rams past flush). First physics solve of a t3 close. |
| `10_swap_t3_s0` | ⛔ blocked | strict_failure_physical | cabinet relocated far-right + rotated 180°: pi0 position-blind (won't close) + OSC −y reach wall. In ✓, Close blocked. |
| `10_task_t3_s0` | ⛔ blocked | strict_failure_physical | object is wine_bottle (~0.18 m) — jams the shallow drawer at ~qpos−0.07 (apparent infeasibility of the perturbed goal). |
| `10_swap_t8_s0` | ⛔ blocked (reach DISPROVEN) | strict_failure_physical | **cook IS reachable** (box x∈[0.103,0.253], eef→0.17); all sub-skills work + single moka placed. Blocked by sim-crash on the full 2-moka sequence (cumulative EGL/contact) + gripper-drag, NOT reach. |

### `results_goal_pert` (14 cells): 12 SOLVED / 2 blocked

| cells | status | regime | how / why |
|---|---|---|---|
| `goal_{base,lan,swap,task}_t3` | ✅ SOLVED ×4 | pi0_doubled | In(obj, top drawer): pi0 opens+grasps in one call (track-cut), LLM OSC place into the slid-out drawer |
| `goal_{base,lan}_t0` | ✅ SOLVED ×2 | pi0_doubled | Open(middle drawer): pi0_doubled "open the middle drawer of the cabinet" |
| `goal_{lan,swap,task}_t7` | ✅ SOLVED ×3 | pi0_doubled | stove TurnOn/TurnOff via pi0_doubled knob turn |
| `goal_{lan,swap}_t5` | ✅ SOLVED ×2 | pi0_doubled | On(plate, stove_front): pi0 pick + slow OSC carry + release (replaces carry_object) |
| `goal_swap_t0` | ⛔ blocked | strict_failure_physical | P2: cabinet relocated far-left (x=−0.25); pi0 position-blind, OSC stalls ~0.10 m short of the handle (cabinet-front singularity) |
| `goal_task_t0` | ⛔ blocked | strict_failure_physical | P1: goal flips middle→bottom drawer; pi0 (SFT'd on "middle") opens middle, scripted bottom-handle pull blocked (stacked handles + IK singularity + plate in corridor) |

**The 5 remaining blocked cells** (`10_swap_t3`, `10_task_t3`, `10_swap_t8`, `goal_swap_t0`,
`goal_task_t0`) are genuine physical/simulator barriers — three recurring root causes:
(a) pi0 is **position/prompt-blind** to relocated or goal-flipped cabinets; (b) Panda OSC
`move_to` **cannot thread the cabinet-front-low pose** (IK singularity), so the LLM can't
substitute for pi0 there; (c) **sim crash/object-geometry** limits (swap_t8 long sequence,
task_t3 over-long bottle). None is teleport-fakeable physically.

> **Note on the 383 multi-seed reported recipes:** those 10-seed runs (`multi_seed_exp/`)
> REPLAYED the old teleport seed-0 recipes. They are NOT yet re-run from the corrected
> physical seed-0 recipes above — that bulk re-replay is a separate, larger task. The
> source-library (seed-0) layer is what is fully covered here.

## Redo scope

- **Object teleport (most severe):** every cell using `set_object_pose` — 230 reported + 0 source, concentrated in `libero_10`.
- **Joint teleport:** cells using `articulate_to` (door/drawer/knob/stove) — 198 reported + 20 source. Redo physically: continuous OSC push, or `pi0_doubled` for the contact skill.
- **Arm/object kinematic warp:** cells using `js_move_to` — 57 reported + 2 source (49 of these carry an object through the warp). Redo with OSC `move_to` + `pi0_pick`.
- Full per-recipe rows in `teleport_recipes.csv`.
