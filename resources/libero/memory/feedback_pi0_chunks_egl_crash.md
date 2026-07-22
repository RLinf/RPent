---
name: pi0-chunks-egl-crash
description: "In the libero hybrid REPL driver, Pi0 invocations with max_chunks >= ~50 reliably crash the libero env subprocess with EOFError / EGL_NOT_INITIALIZED; keep pi0_pick calls <= ~25 chunks per invocation."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0074d341-bdcd-41b6-b40e-036b75648dc3
---

When iterating in `robots/libero/tools.py`, calling
`pi0_pick` with `max_chunks >= 50` causes the libero env worker subprocess to die
with `EOFError` (parent_remote.recv inside `venv.py`) and the renderer with
`EGLError(err=EGL_NOT_INITIALIZED, baseOperation=eglMakeCurrent)`. The driver
process exits and you lose all in-progress scene state.

**Why:** likely a cumulative leak in the MuJoCo/EGL context inside the venv
worker over the long Pi0 inference loop. The crash happens mid-chunk, not at
exit, so once the driver process is gone the current episode cannot resume.

**How to apply:**
- Set `max_chunks <= 25` on every `pi0_pick`. If Pi0 needs more steps for a
  complex skill (e.g. drawer close, knob turn), split into multiple shorter
  invocations rather than one long one.
- When [[feedback_thin_handle_drawer_closure]] or a similar pattern needs Pi0 to complete a
  multi-step VLA skill, prefer 2x `pi0_pick(max_chunks=20)` calls over one
  `pi0_pick(max_chunks=50)`.
- If the worker dies, stop the current attempt.

Observed in a 2026-05-19 session at `max_chunks=80` and `max_chunks=50`.
