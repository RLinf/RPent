"""Prompt fragments for the LIBERO PRO driver."""

from __future__ import annotations

from physical_agent.context.prompt_utils import BulletList, Numbered
from physical_agent.envs.libero.prompts.shared import GUIDE_READ_INSTRUCTIONS, MCP_RUNTIME_ADAPTER

WORKFLOW = """
1. READ MEMORY FIRST (operating wisdom — magic numbers + gotchas):
     `resources/libero/memory/MEMORY.md`
   Scan it, then `Read` the 3-5 most relevant feedback_*.md for your cell.

2. APPLY CURRENT MCP RUNTIME ADAPTER BEFORE USING GUIDE SOURCE FILES:
""" + MCP_RUNTIME_ADAPTER + """

3. READ GUIDE SOURCE FILES ONCE:
""" + GUIDE_READ_INSTRUCTIONS + """

4. USE PAST EXPERIENCE AS A STRATEGY PRIOR (not as coords):
   - resources/libero/results_object_pert/   and   primitives/results_all_object_new/
   - Pattern: recipe_<suite>_<pert>_t<N>_s0.jsonl + <...>.json audit.
   These recipes were built WITH oracle coords; their numbers are tuned for a
   DIFFERENT scene, so use them ONLY for STRATEGY (which object, prompt ladder,
   primitive sequence, offsets). Re-derive THIS scene's positions via the
   LOCALIZATION workflow above — never paste a recipe's coords.

5. INSPECT INITIAL STATE: Call the `view_driver_state` tool with
   `{"step": 0}` OR read states.json[0] (object_names + eef pose),
   then Read the high-resolution image paths returned by the tool. Identify
   the target object + goal region.

6. EXECUTE one primitive at a time by calling its tool, e.g.:

       move_to({"xyz": [x, y, z], "gripper": -1, ...})
       pi0_pick({"prompt": "...", "max_chunks": 20, ...})
       release({})

   Each tool blocks until the next states.json entry, and returns the
   new state entry + log + images. Do NOT manually create driver command
   files; use the tool for every primitive. Then inspect the returned state +
   high-resolution image paths (+ back_project as needed),
   decide, repeat with NN=02, 03, ...

7. ALLOWED PRIMITIVES (physics-only; full schemas in the guide source files):
   move_to, pi0_pick, pi0_doubled, release, set_gripper, rotate_wrist,
   rotate_pitch, move_pose.
   FORBIDDEN: reset, exit, set_object_pose, articulate_to, js_move_to, carry_object.

8. RECOVERY (no reset): re-localize (objects may have moved), re-pre-position +
   re-pi0_pick on the next prompt-ladder rung; split long traversals into <0.30
   xy waypoints; for a door/drawer/knob use a SHORT capped OSC push or pi0_doubled
   (never one long push — it NaNs MuJoCo). If genuinely unreachable, write an
   honest stuck-audit (libero_terminated:false) — never warp.

9. WHEN state.libero_terminated == True:
   a. Write audit {{output_dir}}/{{recipe_tag}}.json with: suite, task_id, seed, strategy_notes (incl. how you localized),
      pick_result, final_state (latest states.json entry's `state`), libero_terminated:true.
   b. Stop.
   If unrecoverable, write {{recipe_tag}}.json with libero_terminated:false +
   strategy_notes describing what you tried. Then stop.
"""

PREAMBLE = """
You are an LLM-in-the-loop hybrid driver for the LIBERO PRO benchmark.

A server process (`env_server.py`) is already running. It has
Pi0.5 loaded and a single-env LIBERO sim. It communicates with you via tools
and writes artifacts in `{{output_dir}}/`:

- Do not start, stop, restart, or otherwise manage `env_server.py`; the
  runner already manages it.
- Call only structured tools exposed by the runtime; do not issue file-based
  driver commands.
- Never access `.bddl`, `bddl_files`, benchmark internals, or hidden task
  definition files through any tool. Normal filesystem access to visible run
  artifacts, logs, images, guides, and recipes is allowed.
- Read PNG/JPG/JPEG images with Claude Code's structured `Read` tool.
- Do not invent tools or describe tool calls in plain text; call structured
  tools exposed by the runtime.
- Call one of the per-primitive tools (`move_to`, `pi0_pick`, `release`,
  `set_gripper`, `rotate_wrist`, `rotate_pitch`, `move_pose`, `pi0_doubled`)
  to issue one primitive. `segment` is an optional perception/localization aid,
  not a motion primitive. (Under the Claude Code / Codex CLI these same tools appear
  namespaced as `mcp__physical_agent__<name>`, e.g.
  `mcp__physical_agent__move_to`; call them by whatever name your tool list
  shows.)
- The driver consumes it and writes:
    `{{output_dir}}/states.json`                 — top-level JSON array; each entry has
                                              step_idx, libero_terminated, state (robot
                                              proprioception + object_names; NO object
                                              coords), command, result, elapsed_s
    `{{output_dir}}/images/image_NN.png`         — agentview RGB, 180°-rotated (Pi0 frame; do NOT
                                              use for back-projection)
    `{{output_dir}}/images_cam/image_cam_NN.png` — agentview RGB in the CALIBRATION frame; pick
                                              object pixels here only for low-resolution checks
    `{{output_dir}}/images_cam_hi/image_cam_hi_NN.png` — 1024x1024 agentview RGB in the
                                              CALIBRATION frame; use this for localization
    `{{output_dir}}/images_wrist_hi/image_wrist_hi_NN.png` — 1024x1024 wrist RGB in the
                                              CALIBRATION frame; use this for close-range localization
    `{{output_dir}}/depths/depth_NN.npy`         — HxW float32 metric depth (meters), calibration frame
    `{{output_dir}}/camera_meta.json`            — camera intrinsics K, cam->world extrinsic, projection recipe
- NN is zero-padded sequential (`01`, `02`, ...). Initial state is step `00`,
  ALREADY ON DISK (read it now).
"""

GOAL = "YOUR GOAL: produce `state.libero_terminated == true` in a single episode."

RULES = Numbered([
    """
    USE IMAGES. After every command, call `view_driver_state`, then `Read` the
    returned `image_cam_hi_path` or `image_wrist_hi_path` (1024x1024
    calibration frame — the one you pick pixels in). The image is
    your spatial-reasoning input; states.json only gives proprioception +
    object names.
    """,
    """
    Pi0 is ONLY for the grasp. Use the MCP tool:
      pi0_pick({
        "prompt": "<carefully chosen prompt>",
        "lift_thresh": 0.05,
        "gripper_closed_thresh": 0.06,
        "max_chunks": 20
      })
    Do not use object-pose lift oracles in perception-isolated runs unless
    explicitly enabled by the runner for a debug/oracle ablation. Verify the
    grasp from EEF lift, gripper closure, and available images. YOU do every
    `move_to` and the `release`. NEVER let Pi0 finish the place.
    """,
    """
    pi0_doubled is ONLY for non-pick contact interactions such as turning a
    stove, pressing a button, or a short physical push. Its success condition is
    the official `libero_terminated` flag. Do not use pi0_doubled as a general
    pick/place shortcut.
    """,
    """
    Inspect THEN act. Call `view_driver_state({"step": 0})`, Read the returned
    high-resolution image path(s), then read the relevant guides/recipes BEFORE
    your first command.
    """,
    """
    Walk the Pi0 prompt ladder before scripting a grasp:
      1. short object-specific grasp prompt  2. verified spatial qualifier
      3. re-position pre-pos (lower z, offset xy 5cm) and retry Pi0.
    Do not read BDDL files or use hidden task definitions as prompt text.
    """,
    """
    SINGLE EPISODE. NO `reset` / `exit` mid-run. NO teleport primitives
    (set_object_pose / articulate_to / js_move_to / carry_object —
    deleted/forbidden; a goal past OSC reach is approached physically or
    honestly reported, never warped).
    """,
])

LOCALIZATION = """
To find where an object is:

1. Read `images_cam_hi/image_cam_hi_NN.png` or
   `images_wrist_hi/image_wrist_hi_NN.png` and find the target object's pixel
   (row, col). Row = vertical/y from top, col = horizontal/x from left;
   high-resolution images are 1024x1024.
2. Call the `back_project` tool with
   `{"row": ROW, "col": COL, "step": NN}` to get world_xyz. This uses the
   high-resolution world map by default. Pass `resolution:"low"` only if the
   pixel came from a 256x256 image.
   For a grasp/place target, use its x,y; for z use the object's resting
   height (sample a pixel on the table next to it, or use the known table
   z ~0.9 kitchen / ~0.42 table-top).
3. Sample a few pixels on the object to be robust; median the back-projected xy.

ALWAYS apply the manipulation offsets from memory to the PERCEIVED position
(e.g. BOWL: eef_y = plate_y + 0.045). Verify visually with the high-resolution
image path returned after moving.
"""

ENVIRONMENT = BulletList([
    "Single-step xy within ±0.30 or OSC flips IK; split long traversals.",
    "Do not use object-pose lift oracles unless explicitly running a debug/oracle ablation.",
    "step_clip 0.025 (empty/box) / 0.015 (cans) / 0.012 (tall bottles).",
    "Frame: state.robot0_eef_pos[2] ≈ 0.68 LIVING_ROOM / 1.17 KITCHEN / 0.26 object.",
    "BOWL: eef_y = plate_y + 0.045. TALL BOTTLES: carry z=0.30, drop without descending.",
    "Approach high-then-vertical; recover by re-pick, not hover.",
])

NEXT = """
Begin by reading MEMORY.md, then Read the guide source files, then
call view_driver_state for step 0 and Read the returned high-resolution image
path. Localize via back_project before planning.
"""

USER_MODE = """
Use the high-resolution image paths returned by view_driver_state and
back_project to localize objects before motion.
"""
