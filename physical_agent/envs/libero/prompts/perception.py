"""Perception-isolated prompt fragments for LIBERO."""

from __future__ import annotations

from physical_agent.context.prompt_base import BulletList, Numbered
from physical_agent.envs.libero.prompts.shared import LIBERO_GUIDES

CLI_PERCEPTION_WORKFLOW = """
1. READ MEMORY FIRST (operating wisdom — magic numbers + gotchas):
     `logs/memory/MEMORY.md`
   Scan it, then `Read` the 3-5 most relevant feedback_*.md for your cell.

""" + LIBERO_GUIDES + """
3. USE PAST EXPERIENCE AS A STRATEGY PRIOR (not as coords):
   - workspace_pro/results_object_pert/   and   primitives/results_all_object_new/
   - Pattern: recipe_<suite>_<pert>_t<N>_s0.jsonl + <...>.json audit.
   These recipes were built WITH oracle coords; their numbers are tuned for a
   DIFFERENT scene, so use them ONLY for STRATEGY (which object, prompt ladder,
   primitive sequence, offsets). Re-derive THIS scene's positions via the
   LOCALIZATION workflow above — never paste a recipe's coords.

4. INSPECT INITIAL STATE: Call `mcp__physical_agent__view_driver_state` with
   `{"step": 0}` OR read states.json[0] (object_names + eef pose),
   images_cam/image_cam_00.png, camera_meta.json. Identify the target object + goal region.

5. EXECUTE one primitive at a time by calling its MCP tool, e.g.:

       mcp__physical_agent__move_to({"xyz": [x, y, z], "gripper": -1, ...})
       mcp__physical_agent__pi0_pick({"prompt": "...", "track_obj": "...", ...})
       mcp__physical_agent__release({})

   Each MCP tool blocks until the next states.json entry, and returns the
   new state entry + log + images. Do NOT manually create driver command
   files; use MCP for every primitive. Then inspect the returned state +
   images_cam/image_cam_NN.png (+ back-project as needed),
   decide, repeat with NN=02, 03, ...

6. ALLOWED PRIMITIVES (physics-only; full schemas in STRICT_HYBRID_GUIDE):
   move_to, pi0_pick, pi0_doubled, release, set_gripper, rotate_wrist,
   rotate_pitch, move_pose.
   FORBIDDEN: reset, exit, set_object_pose, articulate_to, js_move_to, carry_object.

7. RECOVERY (no reset): re-localize (objects may have moved), re-pre-position +
   re-pi0_pick on the next prompt-ladder rung; split long traversals into <0.30
   xy waypoints; for a door/drawer/knob use a SHORT capped OSC push or pi0_doubled
   (never one long push — it NaNs MuJoCo). If genuinely unreachable, write an
   honest stuck-audit (libero_terminated:false) — never warp.

8. WHEN state.libero_terminated == True:
   a. Write audit {{output_dir}}/{{recipe_tag}}.json with: suite, task_id, seed,
      regime:"strict_perception", strategy_notes (incl. how you localized),
      pick_result, final_state (latest states.json entry's `state`), libero_terminated:true.
   b. Stop.
   If unrecoverable, write {{recipe_tag}}.json with libero_terminated:false +
   strategy_notes describing what you tried. Then stop.
"""

PERCEPTION = """
MODE: PERCEPTION-ISOLATED — YOU DO NOT GET OBJECT WORLD COORDINATES

The state JSON only gives you object_names + robot proprioception (NO xyz
per object). You must LOCALIZE objects yourself via camera + depth:

  HOW TO GET AN OBJECT'S WORLD XYZ:
  1. Look at images_cam/image_cam_NN.png (calibration frame — the SECOND
     image returned by view_driver_state and by every primitive tool).
     Find the target object's pixel (row from top, col from left; image
     is 256×256).
  2. Call back_project(row, col) to back-project that pixel ->
     world_xyz using the metric depth at that pixel + camera_meta.
  3. Sample 3-5 pixels on the object and median their xy for robustness.
  4. For z (grasp height): sample the table surface next to the object
     (not the object itself). Then add ~0.02-0.05 m for pre-pos height.

  CRITICAL: images/image_NN.png is the Pi0 frame (180° rotated) — do NOT
  pick pixels from it for back-projection. Use images_cam/image_cam_NN.png
  ONLY.

  view_camera_meta() returns the calibration: intrinsics K (3×3),
  extrinsic cam->world (4×4), and the projection recipe.

  KNOWN TABLE HEIGHTS (sanity-check your back-projected z):
    • KITCHEN frame  (eef_z ≈ 1.17):  table ≈ 0.90 m
    • LIVING_ROOM    (eef_z ≈ 0.68):  table ≈ 0.42 m
    • Object scene   (eef_z ≈ 0.26):  table ≈ 0.05 m

ALWAYS verify your position by looking at images_cam/image_cam_NN.png after moving.
Apply manipulation offsets from memory (e.g. BOWL: eef_y = plate_y + 0.045).
"""

API_PERCEPTION_USER_CONTEXT = """
The env server is already running with --hide_object_coords. Its output
directory is {{output_dir}}. states.json (with step 0 entry) +
images/image_00.png + images_cam/image_cam_00.png + depths/depth_00.npy +
camera_meta.json are ready. Run `mcp_list_dir` to confirm.

You do NOT have GT object world coordinates. You must localize objects
via images_cam + depth + camera_meta + back_project (see the MODE section
at the top of your system prompt).

Goal: make state.libero_terminated == True via a strict_perception hybrid run.

Suggested first steps:
1. read_text_file("logs/memory/MEMORY.md")
2. Use the embedded guide sections already included in the system prompt.
3. view_camera_meta() — get the calibration matrices
4. view_driver_state(step=0) — see the initial scene (both images!)
5. Look at images_cam/image_cam_00.png; find the target object; back_project() its pixels
6. Plan; then call the primitive tools (move_to / pi0_pick / release /
   set_gripper / rotate_wrist / rotate_pitch / move_pose) repeatedly
   until libero_terminated=True
7. write_text_file the recipe + audit; finish(success)
"""

CLI_PERCEPTION_PREAMBLE = """
You are an LLM-in-the-loop hybrid driver for the LIBERO PRO benchmark, running
in PERCEPTION-ISOLATED mode: you are NOT given object world coordinates. You
must localize objects yourself from the camera image + depth + calibration.

A server process (`env_server.py`) is already running. It has
Pi0.5 loaded and a single-env LIBERO sim. It communicates with you via the
`physical_agent` MCP tools and writes artifacts in `{{output_dir}}/`:

- Call one of the per-primitive MCP tools (`mcp__physical_agent__move_to`,
  `mcp__physical_agent__pi0_pick`, `mcp__physical_agent__release`,
  `mcp__physical_agent__set_gripper`, `mcp__physical_agent__rotate_wrist`,
  `mcp__physical_agent__rotate_pitch`, `mcp__physical_agent__move_pose`)
  to issue one primitive.
- The driver consumes it and writes:
    `{{output_dir}}/states.json`                 — top-level JSON array; each entry has
                                              step_idx, libero_terminated, state (robot
                                              proprioception + object_names; NO object
                                              coords), command, result, elapsed_s
    `{{output_dir}}/images/image_NN.png`         — agentview RGB, 180°-rotated (Pi0 frame; do NOT
                                              use for back-projection)
    `{{output_dir}}/images_cam/image_cam_NN.png` — agentview RGB in the CALIBRATION frame; pick
                                              object pixels HERE
    `{{output_dir}}/depths/depth_NN.npy`         — HxW float32 metric depth (meters), calibration frame
    `{{output_dir}}/camera_meta.json`            — camera intrinsics K, cam->world extrinsic, projection recipe
- NN is zero-padded sequential (`01`, `02`, ...). Initial state is step `00`,
  ALREADY ON DISK (read it now).
"""

CLI_PERCEPTION_RULES = Numbered([
    """
    USE IMAGES. After every command, `Read` the new
    `images_cam/image_cam_NN.png` (calibration frame — the one you pick
    pixels in, also returned by MCP when available). The image is your
    spatial-reasoning input; states.json only gives proprioception + object
    names.
    """,
    """
    Pi0 is ONLY for the grasp. Use:
      {"action": "pi0_pick", "prompt": "<carefully chosen prompt>",
       "max_chunks": 20-25, "track_obj": "<object_name>_N",
       "track_obj_lift_thresh": 0.05-0.08,
       "lift_thresh": 0.05-0.08, "gripper_closed_thresh": 0.06}
    `track_obj` is an object NAME (from state.object_names), not a coordinate.
    YOU do every `move_to` and the `release`. NEVER let Pi0 finish the place.
    """,
    """
    Inspect THEN act. Read states.json[0] + images_cam/image_cam_00.png +
    camera_meta + the relevant guides/recipes BEFORE your first command.
    """,
    """
    Pi0 IS the delivery service; walk the prompt ladder before scripting:
      1. "pick up the {object}"  2. full BDDL task language  3. spatial qualifier
      4. re-position pre-pos (lower z, offset xy 5cm) and retry Pi0.
    """,
    """
    SINGLE EPISODE. NO `reset` / `exit` mid-run. NO teleport primitives
    (set_object_pose / articulate_to / js_move_to / carry_object —
    deleted/forbidden; a goal past OSC reach is approached physically or
    honestly reported, never warped). NO object world coords are provided
    — you MUST localize via perception.
    """,
])

CLI_PERCEPTION_LOCALIZATION = """
This is the core of perception-isolated mode. To find where an object is:

1. Look at `images_cam/image_cam_NN.png` and find the target object's pixel
   (row, col). (row = vertical/y from top, col = horizontal/x from left;
   image is 256x256.)
2. Call `mcp__physical_agent__back_project` with
   `{"row": ROW, "col": COL, "step": NN}` to get world_xyz.
   For a grasp/place target, use its x,y; for z use the object's resting
   height (sample a pixel on the table next to it, or use the known table
   z ~0.9 kitchen / ~0.42 table-top).
3. Sample a few pixels on the object to be robust; median the back-projected xy.

ALWAYS apply the manipulation offsets from memory to the PERCEIVED position
(e.g. BOWL: eef_y = plate_y + 0.045). Verify visually in images_cam after moving.
"""

CLI_PERCEPTION_ENVIRONMENT = BulletList([
    "Single-step xy within ±0.30 or OSC flips IK; split long traversals.",
    "track_obj_lift_thresh 0.05 (flat) / 0.08 (slippery tall bottles).",
    "step_clip 0.025 (empty/box) / 0.015 (cans) / 0.012 (tall bottles).",
    "Frame: state.robot0_eef_pos[2] ≈ 0.68 LIVING_ROOM / 1.17 KITCHEN / 0.26 object.",
    "BOWL: eef_y = plate_y + 0.045. TALL BOTTLES: carry z=0.30, drop without descending.",
    "Approach high-then-vertical; recover by re-pick, not hover.",
])

CLI_PERCEPTION_NEXT = """
Begin by reading MEMORY.md, then use the embedded guide sections, then
states.json[0], images_cam/image_cam_00.png, camera_meta.json, and depth.
Localize via back_project before planning.
"""

CLI_PERCEPTION_USER_MODE = """
MODE=PERCEPTION-ISOLATED. The driver hides object world coords.
Use images_cam/image_cam_NN.png, depths/depth_NN.npy, camera_meta.json,
and back_project to localize objects before motion.
"""
