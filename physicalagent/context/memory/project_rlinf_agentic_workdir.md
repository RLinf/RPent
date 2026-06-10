---
name: project-rlinf-agentic-workdir
description: "Default working directory for PhysicalAgent/RLinf work is the configured repo root; external RLinf is selected with PHYSICALAGENT_RLINF_ROOT or RLINF_REPO_PATH."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9df802c0-b380-4d01-8d48-706e324854e2
---

Default working directory for PhysicalAgent/RLinf orchestration is the active PhysicalAgent checkout: `PHYSICALAGENT_REPO_ROOT` when set, otherwise the current repo root. The external RLinf checkout should be selected with `PHYSICALAGENT_RLINF_ROOT` or `RLINF_REPO_PATH`.

**Why:** User stated on 2026-05-18 to keep subsequent work in the then-current PhysicalAgent/RLinf checkout; this should be interpreted as "use the configured repo root", not as a portable absolute path.

**How to apply:** Default `cwd` for PhysicalAgent/RLinf tasks to `PHYSICALAGENT_REPO_ROOT` or the current checkout unless the user names another directory. Only consult older internal forks when the user explicitly configures their path or references those experiments.
