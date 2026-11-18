## Localization

This is the core of perception-isolated mode. To find where an object is:

1. Look at `agentview_high.png` (1024x1024 — PREFER THIS; fall back to
  `agentview.png` only if the high-resolution artifact is absent) and find the target
   object's pixel (row, col). (row = vertical/y from top, col = horizontal/x
   from left.)
2. Call `back_project` on that pixel:

       back_project({"row": ROW, "col": COL, "step": NN})

   It uses the high-resolution world map by default. Pass `"resolution":"low"`
   only if the pixel came from a 256x256 image. The geometry (K⁻¹ back-projection
   + extrinsic) is already done for you. Just use the returned `world_xyz`; do
   NOT write back-projection math yourself unless debugging a tool failure. NEVER
   mix hi-res pixels with low-res world maps or vice versa.

   The returned value is the object's SURFACE point under that pixel. For a
   grasp/place target use its x,y; for z use the object's resting height (sample a
   pixel on the bare table next to it, or use table z ~0.9 kitchen / ~0.42
   table-top).
3. Sample a few pixels on the object and median the world xy — robust to a single
   mis-picked pixel. (Tip: avoid pixels on the object's thin rim/edge or the gap
   to the table — those index a background/edge depth and give a world point
   metres away. Pick pixels firmly on the object's top surface.)

ALWAYS apply the manipulation offsets from memory to the PERCEIVED position
(e.g. BOWL: eef_y = plate_y + 0.045). Verify visually in `agentview_high.png`
after moving.

## First-step algorithm

This is the default perception algorithm for EVERY cell (from the 80-task
localization sweep: `agentview_identity_wrist_geometry_except_basket`).
Agentview chooses WHAT the target is; wrist refines WHERE that already-chosen
candidate is. Do NOT invert those roles.

CORE RULE:
  • Non-basket objects/surfaces: agentview hi-res is the semantic AUTHORITY.
    The wrist is ONLY a geometry/depth refinement camera for the SAME agentview
    candidate. NEVER let the wrist freely re-identify a non-basket target — in
    failed probes the wrist locked onto a look-alike hundreds of pixels away
    while agentview had the right one.
  • Basket / basket_cavity: wrist MAY also confirm/refine, because a basket is a
    geometric container and the close view finds the true interior center (not
    the rim). Basket failures are rim/edge bias, not semantic confusion.

ALGORITHM (run this BEFORE manipulating):

1. From the initial `task_language` + `agentview_high.png` +
   object_names, infer the task-relevant TARGETS and DESTINATIONS (language only;
   never BDDL/poses).

2. GLOBAL SEMANTIC PASS (agentview hi-res): in `agentview_high.png` choose each
   target/destination candidate by RGB, label/shape, and global spatial relation.
   For duplicates (two bowls/plates/mugs) pick by RELATION (on stove, on cookie
   box, left/right/front/back), not `_1/_2`. For sauce/can/box groceries use the
   front/side label + package shape + colour + layout — top-down wrist label
   reading is NOT trustworthy. Classify destination surfaces (plate vs stove
   burner vs cabinet/drawer vs basket) semantically in RGB here.

3. COARSE XYZ (agentview): pick 3-8 pixels firmly on the chosen candidate in
  `agentview_high.png`, call `back_project` on the SAME pixels, take the median.
   Avoid edges/holes/shadows/table-gaps. This median is the IDENTITY ANCHOR for
   that entity.

4. WRIST GEOMETRY REFINE (non-basket): `move_to` ~15-20cm above the agentview
   anchor xy, then refine the SAME candidate's surface/center from the wrist:
     - accept a wrist xy ONLY if it is within ~3-5cm of the agentview anchor;
     - if the wrist xy jumps >5cm, REJECT it (it hit a look-alike/background) and
       keep the agentview xyz, or nudge and re-observe;
     - the wrist may NOT override the agentview semantic choice — it only sharpens
       coordinates when geometry is consistent.

5. BASKET SPECIAL CASE: for `basket`/cavity, agentview finds it globally, then the
   wrist confirms/refines the true INTERIOR center (place objects at the open
   interior, not the rim/outer wall). Re-localize the cavity if the basket moved.

6. MANDATORY PRE-TASK PERCEPTION PASS — DO NOT START MANIPULATION UNTIL THIS
   TABLE EXISTS in your reasoning. One row per task-relevant entity (every movable
   target, every destination/support/fixture, every relation landmark), each with:
     - name_or_role (e.g. target_1, basket_cavity, plate_surface, stove_region)
     - agentview_evidence (why this is the right semantic candidate/relation)
     - agentview_pixels_rc (3-8 hi-res pixels) + agentview_xyz (median back_project)
     - wrist_refine: accepted | rejected | basket_confirmed (+ wrist_xyz if kept)
     - final_xyz (what you will plan with)
     - uncertainty (indistinguishable can, duplicate class, basket rim bias, …)
   If an entity is too ambiguous to identify, SAY SO before acting — do not let
   Pi0 or the wrist make a free semantic choice for you.

7. FINAL READY CHECK before the first pick/place: every target+destination has a
   final_xyz; non-basket wrist refinements are spatially consistent with
   agentview; basket/cavity points are interior-centered; manipulation offsets are
   planned from the perceived final_xyz. If this fails, keep perceiving — only
   then start the manipulation plan. Re-verify with the newest image after every
   command and update the table if anything moves.

(xyz from agentview and wrist `back_project` are in the SAME world frame,
directly comparable. Do NOT blindly average them — accept wrist coords only when
consistent with the agentview anchor, or for basket/cavity geometry.)

RUN THE MANDATORY PRE-TASK PERCEPTION PASS (FIRST-STEP ALGORITHM above) —
localize EVERYTHING first, THEN act. Before any pick/place build the
localization table: agentview hi-res for semantic identity, `back_project` for
median xyz, wrist geometry refinement for non-basket rows (only if spatially
consistent), wrist confirmation for basket/cavity. This perception pass is the
first stage of EVERY task, even ones that look simple — a wrong-target first
grab is unrecoverable in single-attempt mode, so the cheap insurance is to
identify all entities up front. Do the FINAL READY CHECK, then plan.

Inspect `agentview_high.png` returned by `view_env_state`, then use
`back_project` or `segment` to localize objects before motion.

## Available primitives and localization aid

FORBIDDEN: `exit`, `set_object_pose`, `articulate_to`,
`js_move_to`, `carry_object`.

⚠ INFRA NOTE: `pi0_doubled` IS implemented and callable in this runtime —
verified. It runs the Pi0 VLA on a CONTACT skill (drawer/door open-close, knob
turn) with success := `terminated` (no lift / no gripper-close
assumption — unlike `pi0_pick`). If ANY prior note or reference for this cell
concluded that `pi0_doubled` is "unknown action" / missing / that
drawer-or-door articulation is an unsolvable "structural dead-end" BECAUSE no
contact primitive existed — DISREGARD that specific conclusion and actually
USE `pi0_doubled` for the drawer/door step, alternating with short capped OSC
pushes/aligns as needed. Re-prove the cell from scratch; do not inherit the
dead-end verdict.

SAM3 localization aid — `segment` (no robot motion): instead of eyeballing
a pixel, call `segment({"prompt":"the black bowl on the cookies box",
"camera":"agentview"})`. It runs SAM3 on the current image, back-projects the
mask via the matching world map, and returns a robust median `world_xyz` plus
logical `segment_artifact` and `overlay_artifact` names. Inspect the embedded
overlay image to confirm the right object. Use `camera":"wrist"` (after parking the eef ~15–20 cm over the
target) for ±1–2 cm refinement, or `"point":[row,col]` for a point prompt.
Text `prompt` and `point` are mutually exclusive; provide exactly one.
⚠ PROMPT PHRASING (SAM3 is sensitive): use a plain colour+shape+RELATION phrase,
NEVER the internal/brand name from `object_names`/BDDL. `"the akita black bowl"`
scores ~0.03 (SAM3 can't ground "akita") whereas `"the black bowl on the stove"`
scores ~0.76. Strip proper nouns (akita, glazed_rim_porcelain_…) — say what it
LOOKS LIKE + where it is. Always inspect the returned overlay image to confirm
the mask landed on the right object before moving.
This is a CONVENIENCE alternative to manual back-projection — if it returns
`{"error":..., "fallback":...}` (server down / low score / no detection), walk
the prompt (drop the brand word, add the relation) or just pick a pixel in the
high-resolution image and call `back_project` yourself. It does NOT replace the
two-camera relation protocol for disambiguating identical objects.

## In-place recovery tools

Re-localize after objects move, re-pre-position and re-pi0_pick on the next
prompt-ladder rung. Split long traversals into <0.30 xy waypoints. For a door,
drawer or knob use a short capped OSC push or pi0_doubled, never one long push
that can produce NaNs in MuJoCo. Reset permission is defined by the run mode.
