---
name: scripted-pick-limits
description: Fully scripted descent plus gripper close is geometrically unreliable in LIBERO for small and medium objects because the fingers miss by millimeters
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

**Rule**: Do not expect a fully scripted descent plus `set_gripper` close to reliably grasp objects with diameter ≤ 6 cm in LIBERO. The Panda 2f85 gripper geometry combined with the per-object body center/extent makes the alignment band only ~1 cm wide. Pi0's closed-loop visual control finds the band; scripted descent typically does not.

**Why** (measured 2026-05-21 on libero_goal t1 swap akita_black_bowl and t2 swap wine_bottle):

- Gripper finger length (eef site → fingertip) ≈ 8 cm.
- Open gripper width ≈ 7.8 cm; akita_black_bowl outer dia ≈ 6 cm; wine_bottle outer dia ≈ 5 cm. Side clearance ~1 cm per finger — alignment must be ≤ ±5 mm.
- For top-down rim grasp on a bowl, fingertips must sit at bowl mid-height (z = 0.91 m). Eef base z = 0.99 m. OSC stalls at z=0.99 exactly at this xy, so the nominal top-down approach may not reach the narrow grasp band.
- For wine bottle (tall), the body is graspable along ~10 cm of vertical extent but the bottle's narrow radius leaves the open fingers brushing the bottle's outer surface rather than encircling. Closing the gripper from this pose squeezes air next to the bottle, not the bottle itself.

**Effect**: fully scripted pick→carry→place chains fail to grasp ~80% of the time on this geometry. A scripted descent plus close does not recover from misaligned fingers; require a verified grasp before OSC carry.

**How to apply**:
- Keep Pi0 in the pick loop for libero scenes; only use scripted pick when Pi0 fails repeatedly AND the object has clear gripper-friendly features (large flat box, mug handle, squat moka body — verified scripted-graspable on libero_10 t2 moka 2026-05-26).
- If you must script: descend to grasp_z, close with `set_gripper` for at least 12 steps, then lift. Treat EEF ascent and gripper closure as supporting signals only; inspect the latest wrist and agentview images to confirm material between the fingers and displacement from the source location. If the grasp clearly missed, re-localize and make a small lateral correction (about ±5 mm) within the current episode.

**Related**: [[feedback_pi0_false_positive_lift]] [[feedback_pi0_delivery_service]]
