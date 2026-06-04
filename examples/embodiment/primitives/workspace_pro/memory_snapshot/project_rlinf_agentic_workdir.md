---
name: project-rlinf-agentic-workdir
description: "Default working directory for RLinf work — /mnt/public2/zhangyixian/RLinf_agentic (clone of github.com/RLinf/RLinf.git, branch main)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9df802c0-b380-4d01-8d48-706e324854e2
---

Default working directory for RLinf work is `/mnt/public2/zhangyixian/RLinf_agentic`. It is a clone of `https://github.com/RLinf/RLinf.git` on branch `main` (as of 2026-05-18, HEAD = 87840a8 "feat(dreamzero): change default video backend to torchcodec (#1159)").

**Why:** User stated on 2026-05-18 "我们之后都在/mnt/public2/zhangyixian/RLinf_agentic工作".

**How to apply:** Default `cwd` for RLinf tasks to this path unless the user names another directory. The related fork `/mnt/public2/zhangyixian/rlinf_0309` is the user's older internal branch (Pi0.5 DSRL / Expert SAC work) — only consult it when the user references "0309" or older experiments.
