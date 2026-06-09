---
name: gripper-ctrl-is-finger-position
description: "Gripper actuator ctrl = target finger POSITION (0=closed, 0.04=open); grip=1.0 clamps to OPEN — the recurring 'closed gripper that's secretly open' bug"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 173e389d-6fb4-4726-99ec-b7c6d73b9da0
---

In the LIBERO/robosuite Panda 2f85, the two gripper actuators are POSITION actuators with
ctrlrange `[0, 0.04]` (finger_joint1) and `[-0.04, 0]` (finger_joint2). So when you torque-drive
the arm and set `d.ctrl[7]=grip; d.ctrl[8]=-grip` directly:
- `grip=0.0`  → fingers to 0   = **CLOSED**
- `grip=0.04` → fingers to ±0.04 = **OPEN**
- `grip=1.0`  → clamped to ±0.04 = **OPEN** (NOT closed!)

**Why:** I spent a whole session with `grip=1.0` meaning "close" — every bottle grab and
"closed-gripper ram" was silently OPEN, so it never grabbed or knocked anything, which looked like
a deep geometric/reachability barrier. It was just an out-of-range ctrl clamped to open.

**How to apply:** In any offline joint-space / direct-ctrl script, CLOSE = `grip=0.0`, OPEN =
`grip=0.04`. After a close, verify it actually grabbed: `width = d.qpos[gf0]-d.qpos[gf1]` ≈ object
thickness (e.g. 0.029 on a drawer bar); width ≈ 0.002 means empty close. (Note: the high-level
interactive_driver `set_gripper`/pi0 path uses its own convention — this is specifically the
raw-actuator path.) Discovered solving goal_swap_t0 — see [[goal-pert-physical-redo]],
[[jointspace-control-bypasses-osc-singularity]], [[handle-bar-grasp-orientation]].
