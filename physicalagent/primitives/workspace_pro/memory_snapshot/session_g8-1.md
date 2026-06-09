---
name: Session label G8-1
description: User-assigned label for the long-running v13/v14 LLM-prior + 32-emb + xy experiment session — UUID 9519e98c-3886-44d5-b9e9-a8f0fa0bb0d4
type: project
originSessionId: 9519e98c-3886-44d5-b9e9-a8f0fa0bb0d4
---
User labeled the session with UUID `9519e98c-3886-44d5-b9e9-a8f0fa0bb0d4` (jsonl at `/root/.claude/projects/-mnt-public2-zhangyixian/9519e98c-3886-44d5-b9e9-a8f0fa0bb0d4.jsonl`) as **G8-1**.

**Why:** Claude Code has no built-in session rename — sessions are tracked by UUID only. The user wanted a human-readable handle to refer back to this work later.

**How to apply:** When the user mentions "G8-1" in future conversations, point them to that session jsonl file. The session covers:
- Initial CUDA13 / vllm pool / Qwen3-Embedding-0.6B + Qwen3.5-9B setup on `/mnt/public2/zhangyixian`
- v13 / v14 batch launchers (`scripts/run_v14_4envs_llm_prior_32xy.sh`) for 4 envs (arm_push_hard, arm_binpick_hard, humanoid_u_maze, humanoid_big_maze) with `--use-llm-prior` + `--adaptive-prior` + `--llm-analyze` against Qwen3.5-9B chat pool on :9002-9004
- Patch sequence: `evaluate_frontier` rewrite for `use_lang_state=1` (state_emb + waypoint embedding), system-prompt anti-leap rule, system-prompt dense weight allocation table (mastered 0.05–0.10 / frontier 0.30–0.50 / distant 0.05–0.10 / eval target ≥0.05)
- Logs in `/mnt/public2/zhangyixian/Agentic-CRL/logs/exp_v14_4envs_*`
