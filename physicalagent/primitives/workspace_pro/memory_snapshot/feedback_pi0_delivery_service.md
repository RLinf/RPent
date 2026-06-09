---
name: pi0-delivery-service
description: "LLM is delivery for Pi0, not replacement. Walk full prompt-escalation ladder before LLM-scripted pick. One sub-instr failure is NOT enough evidence to bail."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 23bda317-5513-4ade-ae34-7d3f3988aeef
---

Pi0 is a vision-action grasper — its strength is closing the loop on a single object from a reasonable pre-pose. The LLM's job is to deliver the gripper to that pre-pose and hand Pi0 the right prompt; LLM-scripted pick (`move_to z=top + set_gripper +1`) is a last resort, not a second-attempt fallback.

**Rule (codified in STRICT_HYBRID_GUIDE.md §"Rule 3 — LLM is a delivery service for Pi0"):** before writing an LLM-scripted pick, exhaust the Pi0 prompt escalation ladder:
1. Sub-instruction `"pick up the {object}"` (libero_spatial default)
2. Full BDDL task language verbatim (libero_10 default, also any drawer/stove/microwave task)
3. Spatial qualifier `"pick up the X on the cabinet"` (elevated objects)
4. Reposition pre-pos and/or `reset` for fresh episode

**Why:** During the libero_10 t1–t5 PRO sweep (2026-05-21), I gave up on Pi0 after a single sub-instr failure on t2_swap, t3_task, t3_lan, t5_task — and went straight to LLM-scripted pick. Some of those are real Pi0 distribution gaps (t3_task bottle = too tall); others are *just the wrong prompt* (t3_lan bowl — base recipe used full task prompt and got 12 chunks; retry with full task prompt picked in 19 chunks). The audit JSON's `regime_history` should show prompts tried before scripted; "tried only sub-instr once" is a red flag.

This is complementary to [[pi0-pick-full-prompt]], which is the *elevated-object* version of the same rule. This memory is the *general* rule: even for table-level libero_10 picks in cluttered scenes (multiple distractors), full task language often outperforms sub-instr because Pi0's training data on libero_10 paired multi-step instructions with the grasp.

**How to apply:** when `pi0_pick` returns "nothing happened" (gripper open, object unmoved, 15+ chunks consumed) on any libero_10 task, my reflex must be:
1. Re-issue `pi0_pick` with `prompt = full BDDL :language` (read from `state_00.task_descriptions[0]` or `benchmark.get_task(i).language`).
2. If still failing, try with `track_obj_lift_thresh = 0.08`.
3. If still failing, reset + retry.
4. *Then and only then*, LLM-scripted pick.

Recipe note in audit JSON's `strategy_notes` MUST document which prompts and pre-poses were tried before scripted, in order. Negative-result audits ("tried sub-instr+full+spatial, all failed, scripted picked") are how future sessions know which Pi0 dead-ends to skip.
