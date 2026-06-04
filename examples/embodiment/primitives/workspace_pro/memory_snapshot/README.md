# memory_snapshot — frozen copy of operating wisdom

This directory is a **versioned snapshot** of the user's live Claude
Code memory at
`/root/.claude/projects/-mnt-public2-zhangyixian/memory/`. Each
`feedback_*.md` and `project_*.md` is a single-paragraph note capturing
a magic number, gotcha, or failure mode learned across many runs.
`MEMORY.md` is the index (one line per entry).

## Why a snapshot

The hybrid agent (both `hybrid_agent/` API variant and `hybrid_agent_cc/`
claude -p variant) tells the worker to read these notes BEFORE the
first command — they contain "the +0.045 bowl-eef y-offset" type
magic constants that recipe JSONLs embed without explanation.

The live copy lives outside the repo (`/root/.claude/...`) so a fresh
clone on another machine wouldn't have access. This dir solves that —
clone the repo and the wisdom comes with it.

## Updating

The live copy on this machine accumulates new entries over time. To
re-sync the snapshot from the live source:

```bash
bash workspace_pro/sync_memory.sh
git add memory_snapshot/
git commit -m "memory_snapshot: sync from live <date>"
```

Do this any time the live memory has gained an entry that's relevant
to the experiments captured in this repo.

## Where the prompts point

- `hybrid_agent_cc/agent_task_prompt.md` — uses the absolute path
  rooted at the repo (your clone's checkout).
- `hybrid_agent/prompts.py` — same.

If you fork the repo and put it at a different absolute path, both
prompts use relative-to-repo-root form (`examples/.../memory_snapshot/`)
so the absolute layout doesn't matter, as long as `Read` /
`read_text_file` is called from the repo root (which the agent runners
already do).
