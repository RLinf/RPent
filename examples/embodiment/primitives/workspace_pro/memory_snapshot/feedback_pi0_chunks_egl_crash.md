---
name: pi0-chunks-egl-crash
description: "In the libero hybrid REPL driver, Pi0 invocations with max_chunks >= ~50 reliably crash the libero env subprocess with EOFError / EGL_NOT_INITIALIZED; keep pi0_pick calls <= ~25 chunks per invocation."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0074d341-bdcd-41b6-b40e-036b75648dc3
---

When iterating in `examples/embodiment/primitives/interactive_driver.py`, calling
`pi0_pick` with `max_chunks >= 50` causes the libero env worker subprocess to die
with `EOFError` (parent_remote.recv inside `venv.py`) and the renderer with
`EGLError(err=EGL_NOT_INITIALIZED, baseOperation=eglMakeCurrent)`. The driver
process exits and you lose all in-progress scene state.

**Why:** likely a cumulative leak in the MuJoCo/EGL context inside the venv
worker over the long Pi0 inference loop. The crash happens mid-chunk, not at
exit, so once the driver process is gone you cannot resume — only restart from
seed 0, redo all picks/places.

**How to apply:**
- Set `max_chunks <= 25` on every `pi0_pick`. If Pi0 needs more steps for a
  complex skill (e.g. drawer close, knob turn), split into multiple shorter
  invocations rather than one long one.
- When [[libero-t3-drawer-close]] or similar tasks need Pi0 to complete a
  multi-step VLA skill, prefer 2x `pi0_pick(max_chunks=20)` calls over one
  `pi0_pick(max_chunks=50)`.
- If the driver dies, restart with `--max_episode_steps 3000 --max_steps 100`
  and execute the recipe from the start — there is no resume.

Observed in 2026-05-19 t3 session at chunks=80 and chunks=50.
