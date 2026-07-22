---
name: feedback-read-image-before-decide
description: "Inspect the latest view_driver_state images before every non-trivial decision; states.json alone turns the agent into a control tuner instead of a spatial reasoner."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 9df802c0-b380-4d01-8d48-706e324854e2
---

When operating RPent, call `view_driver_state` after each primitive and inspect
the returned `image_cam_hi_path` or `image_wrist_hi_path` before deciding the
next command; do not reason from `states.json` alone.
Describe the scene in 1–2 sentences first; then choose the action.

**Why:** On libero_10 t8 (2026-05-19), I spent 3 failed strict-hybrid
attempts placing two moka pots at the same cook_region center, then
debugging release dynamics, OSC stall, drop height, wrist rotation — all
controller-level tuning. The actual problem was that the 15×15 cm
cook_region had plenty of room for two 6 cm pots placed at opposite
corners. **I had a scene image saved at every step but never opened one
during planning** — I only read JSON state. The user pointed out the layout
fix in one sentence; I should have arrived there in attempt 2 by
inspecting the scene image. Repeating controller param tweaks (step_clip,
max_steps, drop height) without ever opening the image is the signature
of this failure mode. See
`robots/libero/guides/strict_hybrid_guide.md`
"Rule 0" section. Linked: [[feedback_staged_held_object_transport]].

**How to apply:**

- Any time a driver dumps an image alongside state, **default to Reading
  the image before each decision**, especially the first command of a new
  primitive and every retry after a failure.
- After tuning controller parameters twice without success, **stop and
  open the image** — this is the abstraction-level switch. Ask whether the
  target layout itself is feasible, not whether the controller can be
  tuned harder.
- For multi-object placements into a flat region (vs. a container), do a
  **footprint allocation pass first**: N × object_diameter vs region size,
  then choose layout (corners / line / single center) — single-point
  targeting is the wrong default for flat regions.
- Numbers (coordinates, distances) are sanity checks; images are the
  reasoning input. Don't reverse this.
