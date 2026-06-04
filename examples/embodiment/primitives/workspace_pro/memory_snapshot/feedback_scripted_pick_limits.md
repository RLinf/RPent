---
name: scripted-pick-limits
description: Fully scripted pick (js_move_to + set_gripper close) is geometrically unreliable in libero for small/medium objects — fingers miss by mm
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

**Rule**: Don't expect a fully scripted pick recipe (`js_move_to to grasp_z` + `set_gripper close`) to reliably grasp objects with diameter ≤ 6 cm in libero. The Panda 2f85 gripper geometry combined with the per-object body center/extent makes the alignment band only ~1 cm wide. Pi0's closed-loop visual control finds the band; scripted descent typically doesn't.

**Why** (measured 2026-05-21 on libero_goal t1 swap akita_black_bowl and t2 swap wine_bottle):

- Gripper finger length (eef site → fingertip) ≈ 8 cm.
- Open gripper width ≈ 7.8 cm; akita_black_bowl outer dia ≈ 6 cm; wine_bottle outer dia ≈ 5 cm. Side clearance ~1 cm per finger — alignment must be ≤ ±5 mm.
- For top-down rim grasp on bowl, fingertips must sit at bowl mid-height (z = 0.91 m). Eef base z = 0.99 m. OSC stalls at z=0.99 exactly (Panda workspace floor at this xy); `js_move_to` can warp past but settle physics during descent gives the bowl a chance to roll slightly out from under the gripper.
- For wine bottle (tall), the body is graspable along ~10 cm of vertical extent but the bottle's narrow radius leaves the open fingers brushing the bottle's outer surface rather than encircling. Closing the gripper from this pose squeezes air next to the bottle, not the bottle itself.

**Effect**: fully scripted pick→carry→place chains fail to grasp ~80% of the time on this geometry — a scripted descent + close doesn't synthesize a grasp from misaligned fingers. (Note: teleport carry is gone — no js_move_to / carry_object — so a held object is now kept only by a real grasp + OSC carry; see [[no-teleport-rule]].)

**How to apply**:
- Keep Pi0 in the pick loop for libero scenes; only use scripted pick when Pi0 fails repeatedly AND the object has clear gripper-friendly features (large flat box, mug handle, squat moka body — verified scripted-graspable on libero_10 t2 moka 2026-05-26).
- If you must script: descend to grasp_z, then SET_GRIPPER CLOSE with `steps≥12`, move_to lift, THEN read `state.objects.<obj>_pos` to verify the obj z rose with the gripper. If not, retry with a different xy offset (±5mm) — small lateral adjustments often fix it.

**Related**: [[pi0-false-positive-lift]] [[pi0-delivery-service]]
