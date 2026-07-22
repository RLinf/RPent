---
name: render-skip-env-step
description: "Historical RLinF render-skip patch avoided EGL accumulation by disabling camera rendering during OSC primitives. Current RPent retains a cached-frame fallback in robots/libero/env_server.py; revalidate render toggling before porting the old optimization."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 23bda317-5513-4ade-ae34-7d3f3988aeef
---

The driver crash at ~20 commands wasn't Pi0 chunks — it was robosuite's per-env.step camera render piling EGL contexts. With render-skip, a 300-step move_to + 20 OSC commands ran clean (~600 env.steps, ~30× the previous budget).

**Historical implementation (the old RLinF files are not included in this checkout):**
1. Add an `env_call` dispatch path to both generic and LIBERO-specific environment workers.
2. Add `set_image_render_enabled(enabled)` to broadcast observable toggles.
3. Cache the last good image and substitute it when an observation lacks image keys.
4. Disable rendering before OSC primitives and enable it before `pi0_pick`.
5. Fall back to `LiberoEnv._cached_full_image` when a rendered frame has invalid shape or dtype.

Current RPent keeps the cached-frame fallback in `robots/libero/env_server.py`, but the old automatic pre-primitive render toggle is not exposed in `robots/libero/tools.py`.

**Current image behavior:** after every primitive, `dump_state` attempts a fresh
agentview and wrist render before appending `states.json`. It uses a cached frame
only when active rendering fails or returns an invalid image. Therefore an
unchanged frame after OSC motion is a render-fallback symptom to investigate,
not the expected behavior of current RPent.

**Validated 2026-05-21** on libero_10 t6 base layout: 20 OSC `move_to` commands + 1 × 300-step move_to back-to-back → no crash, all images valid. Previous max was ~20 commands before EGL_NOT_INITIALIZED killed the env worker.

**How to apply:** treat this as a porting pattern, not as a claim that current RPent auto-toggles rendering. Monitor EGL behavior first; if the failure recurs, reintroduce the toggle through `robots/libero/env_server.py` and verify that Pi0 receives fresh frames after re-enable.

## Related memory

[[feedback_pi0_chunks_egl_crash]] [[feedback_read_image_before_decide]]
