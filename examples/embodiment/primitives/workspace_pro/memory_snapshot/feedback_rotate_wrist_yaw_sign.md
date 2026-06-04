---
name: rotate-wrist-yaw-sign
description: "`primitives.py:rotate_wrist` was rotating in the OPPOSITE direction from commanded because `as_euler('zyx')[0]` returns the negative of world yaw for gripper-down configs (R[2,2]≈-1). Fixed 2026-05-19 by using `atan2(R[1,0], R[0,0])` directly. For pose-control yaw extraction in libero/robosuite, NEVER use scipy euler decomposition when gripper points down — go through the rotation matrix's first column."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0074d341-bdcd-41b6-b40e-036b75648dc3
---

**The bug**: in `examples/embodiment/primitives/primitives.py:rotate_wrist`, world yaw was being extracted via `scipy.spatial.transform.Rotation.from_quat(quat).as_euler('zyx')[0]`. For the Panda gripper-down home pose where the eef rotation matrix has `R[2,2] ≈ -1`, this returns the **negative** of the true world yaw. Symptom: calling `rotate_wrist(delta_yaw=+1.57)` rotated the wrist by ~−π/2 instead of +π/2.

**Why**: the Z-Y-X intrinsic Euler decomposition picks a chart where γ ≈ π (gripper-down flip around X), and α (the "yaw" component) is α = −φ_world rather than +φ_world. Numerically you can verify with `scipy.spatial.transform.Rotation.from_matrix(np.array([[cos φ, sin φ, 0], [sin φ, −cos φ, 0], [0, 0, −1]])).as_euler('zyx')` — first component comes out as `−φ`.

**The fix** (already in `primitives.py` on `master`):

```python
def _yaw_of(quat_xyzw):
    rot = _R.from_quat([q[0], q[1], q[2], q[3]])
    R = rot.as_matrix()
    return float(np.arctan2(R[1, 0], R[0, 0]))   # world-frame yaw
```

**How to apply (durable lesson):**
- For any future yaw extraction in libero/robosuite (also relevant if you add `rotate_pitch` or a `pose_diff` primitive), use **the rotation matrix's first column** (`atan2(R[1,0], R[0,0])` for yaw, `atan2(-R[2,0], sqrt(R[0,0]²+R[1,0]²))` for pitch). Do NOT rely on `as_euler` — it is chart-dependent and silently flips signs for gripper-down configs.
- `robot0_eef_quat` in this libero version is **xyzw** (scipy convention), so `_R.from_quat([q[0],q[1],q[2],q[3]])` is correct. Verified by `/tmp/probe_yaw.py` which showed `R_xyzw` reproduces the expected eef axis orientation, while `R_wxyz` (treating the quat as `[w,x,y,z]`) gives garbage.

**Operational payoff (t9 retry)**: with the fixed `rotate_wrist`, doing `delta_yaw=+π/2` before pushing the mug toward the microwave cavity moved the eef stall point from y=0.16 to y=0.245 (~+8cm). Full strict completion still blocked because Panda gripper wrist body height (~3-5cm above eef center) interferes with the cavity ceiling at z=1.088 — that requires a new `rotate_pitch` primitive or joint-space planning. See [[libero-10-t0-t9-done]] for the full t9 status.

Cross-ref [[no-pi0-end-to-end]] — rotate_wrist is the kind of LLM-side iteration that Rule 1 explicitly endorses (vs hand-off to Pi0).
