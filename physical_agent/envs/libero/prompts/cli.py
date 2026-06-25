"""CLI prompt fragments for LIBERO."""

from __future__ import annotations

from physical_agent.context.prompt_base import BulletList, Numbered
from physical_agent.envs.libero.prompts.shared import LIBERO_GUIDES

CLI_WORKFLOW = """
1. READ MEMORY FIRST (the "operating wisdom" — magic numbers + gotchas).
   The snapshot lives IN THE REPO at:
     `logs/memory/MEMORY.md`
   (If you are running on the originating machine and want the LIVE
   feed instead, it's at
   `logs/memory/MEMORY.md`,
   but the in-repo snapshot is the portable source of truth.)
   Scan all ~40 lines, then `Read` the 3-5 most relevant feedback_*.md
   for your cell. For bowl->plate spatial tasks ALWAYS read:
   - feedback_bowl_eef_y_offset.md (CRITICAL: bowl-eef y-offset 4.5cm,
     so place at eef_y = plate_y + 0.045, NOT plate_y directly).
   Other high-leverage memories:
   - feedback_pi0_delivery_service.md (Pi0 prompt ladder)
   - feedback_pi0_pick_full_prompt.md (when sub-instr isn't enough)
   - feedback_no_pi0_end_to_end.md (Rule 1 reminder)

""" + LIBERO_GUIDES + """
3. CHECK PAST RECIPES for similar cells:
   - workspace_pro/results_spatial_pert/   (libero_spatial)
   - workspace_pro/results_object_pert/    (libero_object)
   - workspace_pro/results_10_pert/        (libero_10)
   - workspace_pro/results_goal_pert/      (libero_goal — corrected seed-0
       PHYSICS-ONLY recipes; for goal cells READ the matching
       recipe_goal_<regime>_t<N>_s0.jsonl + audit first. base/lan/swap/task
       share the same task semantics, so a base/lan recipe is a valid template
       when the swap/task variant is missing. See project_goal_pert_physical_redo
       for the per-task method: drawer-open & stove-knob via pi0_doubled,
       top-drawer In-place, plate OSC-carry.)
   Pattern: `recipe_<suite>_<pert>_t<N>_s0.jsonl` is the working command
   sequence; `<suite>_<pert>_t<N>_s0.json` is the audit with diagnostics.
   WARNING: recipes have HARD-CODED coords tuned for seed=0. When
   adapting to a different seed, re-derive object/target positions from
   the step 0 entry in `states.json` and APPLY the offsets from memory.

4. INSPECT INITIAL STATE:
   Call `mcp__physical_agent__view_driver_state` with `{"step": 0}` OR
   `Read {{output_dir}}/states.json` (step 0 entry) AND
   `Read {{output_dir}}/images/image_00.png`.
   Identify target object name (from BDDL) and the goal region.

5. EXECUTE one primitive at a time by calling its MCP tool, e.g.:

       mcp__physical_agent__move_to({"xyz": [x, y, z], "gripper": -1, ...})
       mcp__physical_agent__pi0_pick({"prompt": "...", "track_obj": "...", ...})
       mcp__physical_agent__release({})

   Each MCP tool blocks until the next step is available, and returns the
   new state entry + log + agentview image. Do NOT manually write driver
   command files; use the MCP tool for every primitive. After each tool
   result, inspect the returned state/image (or read the matching
   `images/image_NN.png`) before deciding the next command.

6. ALLOWED PRIMITIVES (see STRICT_HYBRID_GUIDE §"The command vocabulary"
   for full schemas). These are PHYSICS-ONLY — every motion makes real
   contact and applies real torque:
     - move_to        (OSC servo with optional yaw target)
     - pi0_pick       (the ONLY allowed single-grasp Pi0 invocation)
     - pi0_doubled    (Pi0 for a contact skill: knob turn, drawer open/close)
     - release
     - set_gripper
     - rotate_wrist / rotate_pitch    (when needed for cavity entry)
   FORBIDDEN (Rule 4 — NON-NEGOTIABLE):
     - reset, exit
     - set_object_pose, articulate_to, js_move_to, carry_object
       — these four TELEPORT primitives bypass contact physics (they write
       object/arm/joint qpos directly) and have been DELETED from the code.
       They are not callable. NEVER emit them. A door/drawer/knob is closed
       with a short capped OSC push or `pi0_doubled`; a goal region past OSC
       reach is approached physically or honestly reported as unreachable —
       it is NEVER warped to. See STRICT_HYBRID_GUIDE Rule 4.

7. RECOVERY (no reset available):
   - pi0_pick missed (peak_lift < lift_thresh): re-pre-position,
     pi0_pick again with NEXT rung on Pi0 prompt ladder (Rule 3).
   - Object slipped mid-travel: release, re-pre-position above it,
     pi0_pick again (full task language usually helps now since the
     scene is partially completed).
   - OSC stalls (final_dist_m > 0.05 at max_steps, same xy twice):
     try `rotate_pitch` (cavity entry), re-approach high-then-vertical,
     or split the traversal into smaller waypoints. For a door/drawer/knob
     use a SHORT capped OSC push or `pi0_doubled` (never one long push —
     it NaNs MuJoCo). If a goal region is genuinely past OSC reach and no
     physical approach works, write an honest stuck-audit
     (`libero_terminated: false`) — do NOT warp the object/arm there.
     (Teleport primitives are forbidden; see ALLOWED PRIMITIVES above.)

8. WHEN state.libero_terminated becomes True:
   a. Write a minimal audit JSON to `{{output_dir}}/{{recipe_tag}}.json`
      with these keys: `suite`, `task_id`, `seed`, `regime: "strict"`,
      `strategy_notes`, `pick_result` (from the pi0_pick step's entry
      in states.json), `final_state` (from the latest states.json
      entry's `state` field), `libero_terminated: true`.
   b. Stop.

   If unrecoverable after honest exploration, instead write
   `{{recipe_tag}}.json` with `libero_terminated: false` and
   `strategy_notes` describing what you tried. Then stop.
"""

CLI_PREAMBLE = """
You are an LLM-in-the-loop hybrid driver for the LIBERO PRO benchmark.

A server process (`env_server.py`) is already running. It has
Pi0.5 loaded and a single-env LIBERO sim. It communicates with you via
the `physical_agent` MCP tools and writes artifacts in `{{output_dir}}/`:

- Call one of the per-primitive MCP tools (`mcp__physical_agent__move_to`,
  `mcp__physical_agent__pi0_pick`, `mcp__physical_agent__release`,
  `mcp__physical_agent__set_gripper`, `mcp__physical_agent__rotate_wrist`,
  `mcp__physical_agent__rotate_pitch`, `mcp__physical_agent__move_pose`)
  to issue one primitive.
- The driver consumes it and APPENDS a step entry to:
    `{{output_dir}}/states.json`              (top-level JSON array of step blobs;
                                          one entry per step with state +
                                          command + result + elapsed_s)
    `{{output_dir}}/images/image_NN.png`      (agentview camera, ~256x256 PNG)
- NN is zero-padded sequential (`01`, `02`, ...). Initial state is at
  step `00` and is ALREADY ON DISK (you can read it now).
"""

CLI_GOAL = "YOUR GOAL: produce `state.libero_terminated == true` in a single episode."

CLI_ENVIRONMENT = BulletList([
    "Single-step xy must stay within ±0.30 or OSC flips IK. Split traversal > 0.30 into 2-3 mid waypoints at carry z.",
    "track_obj_lift_thresh: 0.05 for flat/stable items, 0.08 for slippery tall bottles.",
    "step_clip: 0.025 (empty gripper / flat boxes), 0.015 (cans), 0.012 (tall bottles).",
    """
    Frame matters. Check states.json[0].state.robot0_eef_pos[2]:
      ≈ 0.68 -> LIVING_ROOM (basket / plate / pudding scenes)
      ≈ 1.17 -> KITCHEN (stove / cabinet / drawer / microwave)
      ≈ 0.26 -> object scene (libero_object PRO)
    """,
    "BOWL: eef_y_target = plate_y + 0.045 (bowl-eef y-offset compensation)",
    "TALL BOTTLES: carry z=0.30, release from carry without descending",
    "CANS: set_gripper(+1, 8) between move stages",
    """
    Approach high-then-vertical; recover by re-pick, not hover. Reach any
    object by moving directly ABOVE it at carry z, then descend straight
    down (the eef sits ~8-10 cm above the fingertips, so a low-z lateral
    move bulldozes the object, and near a table edge knocks it off). If a
    `release` leaves `libero_terminated` still False, re-`pi0_pick` and
    place again: an open gripper cannot move an already-placed object, so
    never hover or repeat a `move_to` that does not change the scene.
    """,
])

CLI_RULES = Numbered([
    """
    USE IMAGES. After every command, also `Read` the new `images/image_NN.png`
    (or inspect the image returned by the MCP result).
    The image is your spatial-reasoning input; numerical state alone is
    insufficient.
    """,
    """
    Pi0 is ONLY for the grasp. Use:
      {"action": "pi0_pick", "prompt": "<carefully chosen prompt>",
       "max_chunks": 20-25, "track_obj": "<object_name>_N",
       "track_obj_lift_thresh": 0.05-0.08,
       "lift_thresh": 0.05-0.08, "gripper_closed_thresh": 0.06}
    The `track_obj_lift_thresh` cuts Pi0 the moment the object lifts —
    prevents Pi0 from continuing into a learned placement. YOU then do
    every `move_to` and the `release`. NEVER let Pi0 finish the place
    (e.g. don't call pi0_pick with full task language and high lift_thresh).
    """,
    """
    Inspect THEN act. Read `states.json` (step 0 entry) +
    `images/image_00.png` and the relevant guides BEFORE issuing your
    first command.
    """,
    """
    Pi0 IS the delivery service; walk the prompt ladder before scripting
    a pick yourself:
      1. sub-instruction:  "pick up the {object}"
      2. full BDDL task language verbatim (from the task instruction)
      3. spatial qualifier ("...on the cabinet" / "...next to the basket")
      4. re-position pre-pos (lower z, offset xy 5cm) and retry Pi0
    See feedback_pi0_delivery_service.md for the worked argument.
    """,
    """
    SINGLE EPISODE. DO NOT issue `{"action": "reset"}` or `{"action": "exit"}`
    mid-run. The experiment requires one attempt to capture honest
    single-shot performance. If unrecoverable, write a stuck-audit and stop.
    """,
])

CLI_NEXT = """
Begin by reading MEMORY.md, then use the embedded guide sections, then
states.json[0] + images/image_00.png. Then plan and execute.
"""
