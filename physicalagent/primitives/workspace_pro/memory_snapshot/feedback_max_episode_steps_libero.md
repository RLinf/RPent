---
name: max-episode-steps-libero
description: LiberoEnv inherits robosuite's per-episode step counter; when --max_episode_steps cap is hit (e.g. 600), the env enters terminated state and any further env.step throws 'executing action in terminated episode' ValueError that kills the worker. For long hybrid sessions use 5000+.
metadata:
  type: feedback
---

When `interactive_driver --max_episode_steps N` is small (default 240, common 600), long hybrid recipes blow through it. **Each primitive does many internal env.steps** (move_to up to max_steps=80-300, pi0_pick chunks×5=75-150, articulate settle_steps=40). 20 primitives can accumulate 1000+ env.steps.

Once the libero/robosuite episode is terminated (success OR truncation), `robosuite/environments/base.py:379` raises:
```python
raise ValueError("executing action in terminated episode")
```
The worker crashes, parent gets EOFError, driver dies. Looks identical to EGL accumulation but is unrelated.

**Diagnosis**: I added try/except around env.step in `rlinf/envs/libero/venv.py:_worker` that prints the actual exception to stderr. The ValueError signature is the giveaway — if you see it after ~600 cumulative env.steps and your --max_episode_steps was 600, this is the cause.

**Fix**: bump `--max_episode_steps` to 5000 (or higher). 1 OSC env.step ≈ 5ms with render skip, so 5000 steps ≈ 25s of sim time — plenty for any single (suite, task, seed) attempt.

Verified on 2026-05-21 retry session: with `--max_episode_steps 5000` + render-skip ([[render-skip-env-step]]), a t3_swap session ran 7 primitives without crash; t8_lan ran 10 primitives + 30-chunk pi0_pick without crash. Driver finally stays alive as long as recipes need.
