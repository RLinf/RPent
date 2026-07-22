---
name: max-episode-steps-libero
description: LiberoEnv inherits robosuite's per-episode step counter; when the --max-episode-steps cap is hit (e.g. 600), the env enters terminated state and any further env.step throws 'executing action in terminated episode' ValueError that kills the worker. For long hybrid sessions use 5000+.
metadata:
  type: feedback
---

When `robots/libero/env_server.py --max-episode-steps N` is too small (the current default is 600), long hybrid recipes can exhaust it. **Each primitive performs many internal env.steps** (`move_to` up to 80-300 steps and `pi0_pick` roughly chunks×5). Twenty primitives can accumulate 1000+ env.steps.

Once the libero/robosuite episode is terminated (success OR truncation), `robosuite/environments/base.py:379` raises:
```python
raise ValueError("executing action in terminated episode")
```
The worker crashes, parent gets EOFError, driver dies. Looks identical to EGL accumulation but is unrelated.

**Diagnosis:** the historical RLinF worker exposed the underlying `ValueError`; in current RPent, inspect the environment-server log. If the error appears after roughly 600 cumulative env.steps and `--max-episode-steps` is 600, this is the likely cause.

**Fix**: bump `--max-episode-steps` to 5000 (or higher). 1 OSC env.step ≈ 5ms with render skip, so 5000 steps ≈ 25s of sim time — plenty for any single (suite, task, scene) attempt.

Verified on 2026-05-21 retry session: with `--max-episode-steps 5000` + render-skip ([[feedback_render_skip_in_env_step]]), a t3_swap session ran 7 primitives without crash; t8_lan ran 10 primitives + 30-chunk pi0_pick without crash. Driver finally stays alive as long as recipes need.
