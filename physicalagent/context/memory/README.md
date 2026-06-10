# memory_snapshot — frozen copy of operating wisdom

This directory is a **versioned snapshot** of operating memory captured
from prior experiments. Each
`feedback_*.md` and `project_*.md` is a single-paragraph note capturing
a magic number, gotcha, or failure mode learned across many runs.
`MEMORY.md` is the index (one line per entry).

## Why a snapshot

The hybrid agent (both Anthropic API and `claude -p` cerebrum variants)
tells the worker to read these notes BEFORE the
first command — they contain "the +0.045 bowl-eef y-offset" type
magic constants that recipe JSONLs embed without explanation.

The source copy may live outside the repo on a developer machine, so a fresh
clone should not depend on that private location. This directory solves that:
clone the repo and the wisdom comes with it.

## Updating

If an external memory source accumulates new entries over time, set
`PHYSICALAGENT_MEMORY_SOURCE` and re-sync the in-repo snapshot:

```bash
cp -f "${PHYSICALAGENT_MEMORY_SOURCE:-/path/to/source/memory}"/*.md physicalagent/context/memory/
git add physicalagent/context/memory/
git commit -m "memory_snapshot: sync from live <date>"
```

Do this any time the live memory has gained an entry that's relevant
to the experiments captured in this repo.

## Where the prompts point

- `physicalagent/context/libero_prompts.py` contains the full Claude Code
  prompts and points agents at this directory relative to the repo root.
- Runners use `physicalagent.config.get_memory_dir()` and grant this directory
  to `claude -p` with `--add-dir`.

If you fork the repo and put it at a different absolute path, the
relative-to-repo-root references keep working as long as `Read` /
`read_text_file` is called from the repo root (which the agent runners
already do).
