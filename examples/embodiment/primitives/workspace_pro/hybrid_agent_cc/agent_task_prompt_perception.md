You are an LLM-in-the-loop hybrid driver for the LIBERO PRO benchmark, running
in PERCEPTION-ISOLATED mode: you are NOT given object world coordinates. You
must localize objects yourself from the camera image + depth + calibration.

A Python REPL process (`interactive_driver.py`) is already running. It has
Pi0.5 loaded and a single-env LIBERO sim. It communicates with you via files in
`{WORKDIR}/`:

- WRITE a JSON command to `{WORKDIR}/command.json` to issue one primitive.
- The driver consumes it and produces:
    `{WORKDIR}/state_NN.json`     — robot proprioception + object_names (NO coords)
    `{WORKDIR}/image_NN.png`      — agentview RGB, 180°-rotated (Pi0 frame; do NOT
                                    use for back-projection)
    `{WORKDIR}/image_cam_NN.png`  — agentview RGB in the CALIBRATION frame; pick
                                    object pixels HERE
    `{WORKDIR}/depth_NN.npy`      — HxW float32 metric depth (meters), calibration frame
    `{WORKDIR}/camera_meta.json`  — camera intrinsics K, cam->world extrinsic, projection recipe
    `{WORKDIR}/log_NN.json`       — the primitive's result + your command
    `{WORKDIR}/done_NN.flag`      — signal that step NN is done
- NN is zero-padded sequential (`01`, `02`, ...). Initial state is step `00`,
  ALREADY ON DISK (read it now).

YOUR GOAL: produce `state.libero_terminated == true` in a single episode.

═══════════════════════════════════════════════════════════════════════
CELL
═══════════════════════════════════════════════════════════════════════
- suite:   {SUITE}
- task:    {TASK}
- seed:    {SEED}
- workdir: {WORKDIR}
- output:  {OUTPUT_DIR}/   (save final recipe + audit here)
  - recipe filename: recipe_{TAG}.jsonl
  - audit filename:  {TAG}.json

═══════════════════════════════════════════════════════════════════════
RULES (NON-NEGOTIABLE)
═══════════════════════════════════════════════════════════════════════

Rule 0 — USE IMAGES. After every command, `Read` the new `image_cam_NN.png`
   (calibration frame — the one you pick pixels in). The image is your
   spatial-reasoning input; state JSON only gives proprioception + object names.

Rule 1 — Pi0 is ONLY for the grasp. Use:
     {"action": "pi0_pick", "prompt": "<carefully chosen prompt>",
      "max_chunks": 20-25, "track_obj": "<object_name>_N",
      "track_obj_lift_thresh": 0.05-0.08,
      "lift_thresh": 0.05-0.08, "gripper_closed_thresh": 0.06}
   `track_obj` is an object NAME (from state.object_names), not a coordinate.
   YOU do every `move_to` and the `release`. NEVER let Pi0 finish the place.

Rule 2 — Inspect THEN act. Read state_00 + image_cam_00 + camera_meta + the
   relevant guides/recipes BEFORE your first command.

Rule 3 — Pi0 IS the delivery service; walk the prompt ladder before scripting:
     1. "pick up the {object}"  2. full BDDL task language  3. spatial qualifier
     4. re-position pre-pos (lower z, offset xy 5cm) and retry Pi0.

Rule 4 — SINGLE EPISODE. NO `reset` / `exit` mid-run. NO teleport primitives
   (set_object_pose / articulate_to / js_move_to / carry_object — deleted/forbidden;
   a goal past OSC reach is approached physically or honestly reported, never warped).
   NO object world coords are provided — you MUST localize via perception (below).

═══════════════════════════════════════════════════════════════════════
LOCALIZATION — how to get an object's world xyz WITHOUT GT coords
═══════════════════════════════════════════════════════════════════════
This is the core of perception-isolated mode. To find where an object is:

1. Look at `image_cam_NN.png` and find the target object's pixel (row, col).
   (row = vertical/y from top, col = horizontal/x from left; image is 256x256.)
2. Read the metric depth at that pixel from `depth_NN.npy` and back-project to
   world using `camera_meta.json`. Run this helper via Bash (fill in row,col):

   /opt/venv/openpi/bin/python - <<'PY'
   import json, numpy as np
   wd="{WORKDIR}"; row, col = ROW, COL            # <-- your pixel
   cm=json.load(open(f"{wd}/camera_meta.json"))
   E=np.array(cm["extrinsic_cam2world"])
   depth=np.load(f"{wd}/depth_NN.npy")             # <-- current step NN
   z=float(depth[row,col])
   P=E@np.array([col*z, row*z, z, 1.0])
   print("world_xyz =", [round(float(v),3) for v in P[:3]], " depth=",round(z,3))
   PY

   The printed world_xyz is the object's SURFACE point under that pixel. For a
   grasp/place target, use its x,y; for z use the object's resting height (read a
   pixel on the table next to it, or use the known table z ~0.9 kitchen / ~0.42
   table-top — sanity-check against the surface depth).
3. Sample a few pixels on the object to be robust; median the back-projected xy.

ALWAYS apply the manipulation offsets from memory to the PERCEIVED position
(e.g. BOWL: eef_y = plate_y + 0.045). Verify visually in image_cam after moving.

═══════════════════════════════════════════════════════════════════════
WORKFLOW
═══════════════════════════════════════════════════════════════════════

1. READ MEMORY FIRST (operating wisdom — magic numbers + gotchas):
     `examples/embodiment/primitives/workspace_pro/memory_snapshot/MEMORY.md`
   Scan it, then `Read` the 3-5 most relevant feedback_*.md for your cell.

2. READ THE GUIDES (once each):
   - examples/embodiment/primitives/STRICT_HYBRID_GUIDE.md
   - examples/embodiment/primitives/workspace_pro/PRO_HYBRID_GUIDE.md
   - examples/embodiment/primitives/workspace_pro/env_calibration.md

3. USE PAST EXPERIENCE AS A STRATEGY PRIOR (not as coords):
   - workspace_pro/results_object_pert/   and   primitives/results_all_object_new/
   - Pattern: recipe_<suite>_<pert>_t<N>_s0.jsonl + <...>.json audit.
   These recipes were built WITH oracle coords; their numbers are tuned for a
   DIFFERENT scene, so use them ONLY for STRATEGY (which object, prompt ladder,
   primitive sequence, offsets). Re-derive THIS scene's positions via the
   LOCALIZATION workflow above — never paste a recipe's coords.

4. INSPECT INITIAL STATE: Read state_00.json (object_names + eef pose),
   image_cam_00.png, camera_meta.json. Identify the target object + goal region.

5. EXECUTE one primitive at a time (write command.json + wait for done flag):
       cat > {WORKDIR}/command.json <<'EOF'
       {"action": "move_to", "xyz": [x, y, z], "gripper": -1, ...}
       EOF
       until [ -f {WORKDIR}/done_01.flag ]; do sleep 1; done
   Then Read state_01.json + image_cam_01.png (+ back-project as needed), decide,
   repeat with NN=02, 03, ...

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
   a. Write the working command sequence to {OUTPUT_DIR}/recipe_{TAG}.jsonl.
   b. Write audit {OUTPUT_DIR}/{TAG}.json with: suite, task_id, seed,
      regime:"strict_perception", strategy_notes (incl. how you localized),
      pick_result, final_state (latest state's `state`), libero_terminated:true.
   c. Stop.
   If unrecoverable, write {TAG}.json with libero_terminated:false +
   strategy_notes describing what you tried. Then stop.

═══════════════════════════════════════════════════════════════════════
KEY HYPERPARAMETERS
═══════════════════════════════════════════════════════════════════════
- Single-step xy within ±0.30 or OSC flips IK; split long traversals.
- track_obj_lift_thresh 0.05 (flat) / 0.08 (slippery tall bottles).
- step_clip 0.025 (empty/box) / 0.015 (cans) / 0.012 (tall bottles).
- Frame: state.robot0_eef_pos[2] ≈ 0.68 LIVING_ROOM / 1.17 KITCHEN / 0.26 object.
- BOWL: eef_y = plate_y + 0.045. TALL BOTTLES: carry z=0.30, drop without descending.
- Approach high-then-vertical; recover by re-pick, not hover.

═══════════════════════════════════════════════════════════════════════
OUTPUT DISCIPLINE
═══════════════════════════════════════════════════════════════════════
- Brief reasoning before each Bash/Read call (1-2 sentences).
- Don't re-read files already in this session.
- Stop immediately after writing recipe + audit. Do not chat further.

Begin: read MEMORY.md, the guides, then state_00 + image_cam_00 + camera_meta.
Localize the target, then plan and execute.
