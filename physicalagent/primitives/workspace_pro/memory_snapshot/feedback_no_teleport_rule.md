---
name: no-teleport-rule
description: NO teleport primitives — set_object_pose / articulate_to / js_move_to / carry_object are REMOVED from the code. Every motion must be real physics (OSC + Pi0). Verified-physical contact recipes for stove-knob and drawer.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bc1d8704-c816-4268-9976-2a28696eeecf
---

**Rule (2026-05-26, user-mandated):** the four teleport primitives are **deleted from the
codebase** and must never be reintroduced or simulated:
- `set_object_pose` — warps an object's free-joint qpos to the goal
- `articulate_to` — writes a door/drawer/knob joint qpos directly
- `js_move_to` — kinematically warps the arm's 7 joint qpos (mj_forward, no controller)
- `carry_object` — js_move_to variant that rewrites a held object's qpos each waypoint

**Why:** teleport "successes" don't demonstrate physical manipulation. Removed from
`interactive_driver.py`, `primitives.py`, `rlinf/envs/libero/venv.py`; `rlinf/envs/libero/js_move_to.py`
deleted. Only physics-true primitives remain: OSC `move_to` / `rotate_wrist` / `rotate_pitch`
/ `release` / `set_gripper`, and `pi0_pick`. `pi0_doubled` (Pi0 driving a non-pick contact
skill — knob turn, drawer open/close) IS allowed and IS the sanctioned replacement for the
articulation teleports.

**How to apply — verified-physical recipes (libero_10 redo, 2026-05-26):**
- **Stove TurnOn**: `pi0_pick` prompt "turn on the stove", `max_chunks≈15`, set
  `gripper_closed_thresh=0` + `track_obj=null` so the pick-success break never fires →
  Pi0 physically rotates the knob (burner glows red). Then scripted/Pi0 pick the moka and
  OSC-carry it onto the cook region. Solved lan/swap t2 + task/lan t8.
- **Moka body grasp + carry is laterally weak** — it slips on fast/long traverses. Grasp
  pristine (descend to body, `set_gripper +1` steps≈22), then carry in 0.08 m hops at
  `step_clip≈0.01` with a `set_gripper +1` re-clamp after each hop. Ultra-slow is mandatory
  for long y-traverses (e.g. task t8 y:0.24→−0.20).
- **Drawer**: open with `pi0_pick` "open the bottom drawer" (`gripper_closed_thresh=0`,
  `lift_thresh=0.3`); place the bowl, release; close with `pi0_doubled` "close the drawer"
  (pushes the front continuously, dragging the bowl) and/or a CAPPED OSC push. Do NOT
  scripted-descend the eef into the cabinet front (z≈0.95 there crashes the MuJoCo worker
  with EOFError). See [[feedback_osc_push_mujoco_nan]].
- The interactive driver's EGL/MuJoCo worker dies (EOFError) after ~15–20 heavy commands or
  on a collision blowup — keep recipes lean; if it crashes, relaunch (env resets to t0).

Supersedes/!related: [[stove-turnoff-strict]] [[feedback_osc_push_mujoco_nan]]
[[scripted-pick-limits]] [[rotate-pitch-primitive]] [[failure-forensics-render-images]].
