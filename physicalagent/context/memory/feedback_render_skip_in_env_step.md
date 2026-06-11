---
name: render-skip-env-step
description: "env.step renders agentview every step → EGL accumulation crashes ~20 cmds. Patch added LiberoEnv.set_image_render_enabled() to toggle robosuite observables; interactive_driver disables for OSC primitives, enables for pi0_pick. Stale image fallback uses LiberoEnv._cached_full_image."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 23bda317-5513-4ade-ae34-7d3f3988aeef
---

The driver crash at ~20 commands wasn't Pi0 chunks — it was robosuite's per-env.step camera render piling EGL contexts. With render-skip, a 300-step move_to + 20 OSC commands ran clean (~600 env.steps, ~30× the previous budget).

**Changes (already merged)**:
1. `rlinf/envs/venv/venv.py` worker: handle `env_call` cmd → dispatch to `env.env.env...method(*args, **kwargs)`.
2. `rlinf/envs/libero/venv.py` worker: mirror env_call handler (libero-specific worker, used by ReconfigureSubprocEnv).
3. `rlinf/envs/venv/venv.py` SubprocEnvWorker: new `env_call(method, args, kwargs, target='robosuite')`.
4. `rlinf/envs/libero/libero_env.py` LiberoEnv: new `set_image_render_enabled(enabled)` → broadcasts `modify_observable('agentview_image'/'robot0_eye_in_hand_image', 'enabled', flag)` to all workers.
5. `rlinf/envs/libero/libero_env.py` `_extract_image_and_state`: cache last good image; substitute when obs lacks image keys.
6. `physicalagent/primitives/interactive_driver.py` `execute()`: pre-primitive toggle (OSC primitives → disable, pi0_pick → enable). No post-primitive refresh step (robosuite's first sample after re-enable returns degenerate (1,1,3) float64).
7. `interactive_driver.py` `dump_state()`: fallback to LiberoEnv._cached_full_image when render_agentview returns bad shape/dtype.

**Trade-off — stale image**: when in OSC-only sequence, `images/image_NN.png` stays at last pi0_pick frame (or initial reset). LLM-visual debugging sees pre-OSC scene, not post-OSC. For most cases (move-then-release-then-look) this is fine; for "tweak then look" debugging it's confusing. Document in audit notes when reviewing.

**Validated 2026-05-21** on libero_10 t6 base layout: 20 OSC `move_to` commands + 1 × 300-step move_to back-to-back → no crash, all images valid. Previous max was ~20 commands before EGL_NOT_INITIALIZED killed the env worker.

**How to apply**: nothing to do — driver auto-toggles. Recipe count limit was ~18 due to EGL; new effective limit unclear but at least 30-50× higher. Long multi-stage tasks (drawer push close, dual-moka placement, microwave + close) now fit in a single session.
