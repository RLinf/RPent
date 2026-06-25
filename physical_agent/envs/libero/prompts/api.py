"""API prompt fragments for LIBERO."""

from __future__ import annotations

from physical_agent.context.prompt_base import BulletList, Numbered
from physical_agent.envs.libero.prompts.shared import LIBERO_GUIDES

API_ENVIRONMENT = BulletList([
    """
    Single-step xy must stay within ±0.30 or OSC flips IK. Split traversal
    > 0.30 into 2-3 mid waypoints at carry z.
    """,
    """
    track_obj_lift_thresh: 0.05 for flat/stable items, 0.08 for slippery
    tall bottles.
    """,
    """
    step_clip: 0.025 for empty gripper / flat boxes; 0.015 for cans;
    0.012 for tall bottles.
    """,
    """
    Frame matters. Check states.json[0].state.robot0_eef_pos[2]:
    ≈ 0.68  -> LIVING_ROOM (basket / plate / pudding scenes)
    ≈ 1.17  -> KITCHEN (stove / cabinet / drawer / microwave)
    ≈ 0.26  -> object scene (libero_object PRO)
    Use the matching pre_pos_z / carry_z / release_z from the guide
    or from a similar past recipe.
    """,
    """
    For libero_object tall bottles (salad_dressing, ketchup, milk):
    carry_z=0.30, release from carry without descending (descent stalls
    and knocks the basket).
    """,
    """
    Cylindrical cans slip during long travel -> set_gripper(+1, 8) between
    move stages.
    """,
    """
    BOWLS in libero_spatial: Pi0 rim-hooks the bowl with bowl-eef Y-offset
    ≈ -0.045 m (bowl 4.5 cm BEHIND eef in -y after grasp). The release
    primitive only fires `On(bowl,plate)` if bowl xy is centered on plate.
    -> set eef_y_target = plate_y + 0.045  (NOT plate_y directly).
    This is the magic offset embedded in past recipe coords without a
    note. The first off-center release will look "close enough" in the
    image but the predicate won't fire. See feedback_bowl_eef_y_offset.md.
    """,
])

API_WORKFLOW = """
1. READ MEMORY FIRST. The portable snapshot is in the repo at
   `logs/memory/`
  It contains the "operating wisdom" — a collection of `feedback_*.md` and
   `project_*.md` files cataloging magic numbers, gotchas, and failure
   modes learned across many runs.
     • MEMORY.md (the index — one-line summary of each entry; ~40 lines).
       Read it FIRST. Treat it as a table of contents.
     • For each memory item that's plausibly relevant to your cell,
       read_text_file the underlying .md (small files; cheap).
   HIGH-LEVERAGE memories you should usually read up-front:
     • feedback_bowl_eef_y_offset.md  — bowl-eef Y-offset 4.5 cm (CRITICAL
       for libero_spatial bowl->plate placements; without this, eef-on-plate
       drops bowl 4.5cm short of plate center, predicate misses).
     • feedback_pi0_delivery_service.md — Pi0 prompt ladder.
     • feedback_pi0_pick_full_prompt.md — when sub-instr isn't enough.
     • feedback_no_pi0_end_to_end.md — Rule 1 reminder.
   These are the *undocumented* magic constants that recipe.jsonl files
   embed in their coords but never explain in notes.

""" + LIBERO_GUIDES + """
3. CHECK PAST RECIPES for similar cells. Examples already solved:
   • workspace_pro/results_object_pert/   (libero_object × {task, swap, lan})
   • workspace_pro/results_spatial_pert/  (libero_spatial)
   • workspace_pro/results_10_pert/       (libero_10)
   Pattern: recipe_<suite>_<pert>_t<N>_s0.jsonl is the working command
   sequence; <suite>_<pert>_t<N>_s0.json is the audit with diagnostics.
   IMPORTANT: recipes have HARD-CODED coordinates tuned for their own
   (seed=0) bowl/plate positions. When adapting to a different seed:
     - Re-derive object & target positions from states.json step 0.
     - APPLY the offsets from memory (e.g. +0.045 in y for bowl->plate).
     - The recipe's note field often only documents WHY of pre-pos /
       prompt choices, not the place coords. Don't blindly copy coords —
       understand them.

4. INSPECT INITIAL: view_driver_state(step=0). Read state.objects[*]_pos
   and look at the image. Identify the target object and the goal region.

5. PLAN, then EXECUTE one primitive at a time via its tool (move_to,
   pi0_pick, release, set_gripper, rotate_wrist, rotate_pitch, move_pose):
   typical pick-and-place template:
     a. move_to (pre-pos above object, gripper open)        — gripper=-1
     b. pi0_pick (Pi0 grasps with track_obj cut)             — gripper closes
        ↳ if peak_lift_m < lift_thresh AND chunks_used >= max_chunks,
          the pick FAILED — escalate the Pi0 prompt (Rule 3 ladder)
          and re-pre-position before retrying.
     c. set_gripper (+1, 10-15 steps)                        — firm clamp
     d. move_to (lift to carry z)                            — gripper=+1
     e. [set_gripper (+1, 8) + move_to (mid waypoint)]*       — split long Δxy
     f. move_to (above target / basket)
     g. move_to (descend) — OR skip for tall bottles
     h. release

6. AFTER EACH COMMAND: each primitive tool already returns the new
   state + image. Verify the held object is still grasped (object_pos[2]
   close to eef_z), and the move's final_dist_m < 0.02. If something is
   wrong, look at the image before deciding the next step.

7. RECOVERY (no reset available — Rule 4):
   - If pi0_pick missed (object z back at table): re-pre-position
     (move_to gripper=-1 above object) and pi0_pick again with the
     NEXT rung on the Rule 3 prompt ladder. Do NOT just repeat the
     same prompt.
   - If object slipped mid-travel: release (drop it), re-pre-position
     above it, pi0_pick again (full task language usually helps now
     since the scene is partially completed).
   - If OSC stalls (final_dist_m > 0.05 at max_steps and re-trying
     the same xy doesn't help): reorient with rotate_pitch / rotate_wrist
     and approach from a non-singular config, or re-grasp and retry. Do
     NOT teleport — there is no js_move_to / articulate_to / set_object_pose.

8. WHEN state.libero_terminated becomes True: save a recipe.jsonl and
   audit.json to the output directory (use write_text_file), then call
   `finish(status="success", summary="...")`.

   If after honest exploration the task is genuinely unsolvable in
   this single episode, call `finish(status="stuck", ...)` with
   diagnostic notes describing which Pi0 prompts and recovery moves
   you tried. Do NOT escalate to Pi0 end-to-end (Rule 1).
"""

API_PREAMBLE = """
You are an LLM-in-the-loop hybrid driver for LIBERO PRO experiments.

A Python process (env_server.py) is already running. It has Pi0.5
loaded and a single LIBERO sim env. It communicates with you via files in
the run-specific driver output dir named in the user message — you call tools
to inspect state and issue commands.
"""

API_GOAL = """
Solve one (suite, task, seed) cell — make state.libero_terminated == True
in a single episode, using Pi0 ONLY for the gripper grasp and YOUR OWN
scripted commands (move_to, set_gripper, release, etc.) for every motion
and the final release.
"""

API_RULES = Numbered([
    """
    USE IMAGES. Every view_driver_state and primitive (move_to,
    pi0_pick, release, ...) result includes the agentview PNG. LOOK at
    it before deciding on a move target. Numerical state alone gets you
    to "control tuner"; the image is the spatial-reasoning input.
    """,
    """
    Pi0 is ONLY for the grasp. Use:
      {"action": "pi0_pick",
       "prompt": "<carefully chosen prompt — see the pi0 prompt-ladder rule>",
       "max_chunks": 20-25,
       "track_obj": "<object_name>_N",
       "track_obj_lift_thresh": 0.05-0.08,
       "lift_thresh": 0.05-0.08,
       "gripper_closed_thresh": 0.06}
    The track_obj_lift_thresh value cuts Pi0 the moment the named
    object lifts by that height — preventing Pi0 from continuing into
    a learned placement. YOU then do every move_to and the release.
    NEVER call pi0_pick with a high lift_thresh to let Pi0 finish.
    """,
    """
    Inspect THEN act. Read states.json (step 0 entry) + images/image_00.png
    and the relevant guides BEFORE issuing your first command. If a move stalls
    (final_dist_m > 0.02) or an object slips (object z dropped to table),
    re-inspect the new image+state before retrying. Don't tune
    step_clip/tol blindly — when stuck, render and look.
    """,
    """
    Pi0 IS the delivery service; use it well, don't bypass it.
    Pi0.5 is a vision-action model whose single best skill is grasping
    objects from a stable pre-pose. The hybrid pipeline gains its leverage
    by letting Pi0 do that one thing well — NOT by scripting your own
    descend+close+lift the moment Pi0 stumbles.

    PROMPT LADDER for pi0_pick (try in order if a pick fails):
      1. Sub-instruction:    "pick up the {object}"
         — best for visually unambiguous, single-target scenes
         (libero_spatial, libero_object base).
      2. Full BDDL task language verbatim (e.g. "Pick the akita black bowl
         between the plate and the ramekin and place it on the plate")
         — required for cluttered libero_10 scenes and multi-step
         instructions that Pi0 was trained on (drawer, stove, microwave,
         cabinet-top).
      3. Spatial qualifier ("pick up the X on the cabinet" / "...next to
         the basket") — for elevated objects and edge-of-workspace items.
      4. Re-position the pre-pos (lower z, offset xy by 5cm) and retry
         Pi0 from the new pose.

    Only after ALL four rungs fail across multiple pi0_pick attempts
    may you script the pick yourself with move_to + set_gripper
    (Appendix in strict_hybrid_guide.py). "Tried sub-instr once then
    scripted" is a red flag — always escalate the Pi0 prompt first.
    """,
    """
    SINGLE EPISODE. You have exactly ONE episode. The `reset`
    action is BLOCKED. If a pick / place fails:
      - Recover in-episode: re-pre-position with move_to, try pi0_pick
        again with a higher rung on the prompt ladder, adjust grip with
        set_gripper, etc.
      - You may issue multiple pi0_pick calls in the same episode (e.g.
        drop-and-retry by releasing then picking again — but only with
        a fresh prompt strategy, not the identical failing attempt).
      - If truly stuck after honest exploration, call finish(status="stuck",
        summary=...). Negative-result audits are valuable; do NOT escalate
        to pi0_end_to_end (the pi0-only-for-grasp rule).
    """,
])

API_USER_CONTEXT = """
The env server is already running. Its output directory is {{output_dir}}
(this is also the default for mcp_list_dir / view_driver_state and every
primitive tool — move_to, pi0_pick, release, ...). states.json (with
step 0 entry) + images/image_00.png are ready. Run `mcp_list_dir` to
confirm.

Goal: make state.libero_terminated == True via a strict-regime hybrid run
(Pi0 only for the pick via track_obj cut; LLM scripts every move + release).

Suggested first steps:
1. read_text_file("logs/memory/MEMORY.md")
   — the index of operating wisdom. Scan ALL lines, then read the
   ~3-5 individual feedback_*.md files (in the same dir) that look
   most relevant to your suite (e.g. for libero_spatial bowl tasks,
   definitely read feedback_bowl_eef_y_offset.md).
2. Use the embedded guide sections already included in the system prompt.
3. (optional) list_dir on the appropriate workspace_pro/results_*_pert/
   then read a past recipe_<sim>.jsonl as a starting point — BUT
   re-derive coords from states.json[0] and apply memory offsets, don't
   blindly copy.
4. view_driver_state(step=0)  — see initial scene
5. plan; then call the primitive tools (move_to / pi0_pick / release /
   set_gripper / rotate_wrist / rotate_pitch / move_pose) repeatedly
   until libero_terminated=True
6. write_text_file the audit; finish(success)
"""

API_USER_OUTPUT = """
Save artifacts to: {{output_dir}}
- audit  filename: {{recipe_tag}}.json
"""
