---
name: liberopro-install
description: "How LIBERO-Pro was installed in this env — tarball via ghfast.top mirror, editable install, patch applied; libero_spatial_swap missing"
metadata: 
  node_type: memory
  type: project
  originSessionId: 23bda317-5513-4ade-ae34-7d3f3988aeef
---

LIBERO-Pro installed 2026-05-20 at the configured `LIBERO_PRO_PATH` (commit `0bcf736`).

Why tarball, not git clone: direct `git clone https://github.com/RLinf/LIBERO-PRO.git` repeatedly fails mid-pack with GnuTLS RPC errors in this network. `curl -fL https://ghfast.top/https://github.com/RLinf/LIBERO-PRO/archive/0bcf736.tar.gz` (HTTP/1.1, ~300 KB/s) succeeded — 358 MB tarball, integrity OK.

Install steps that worked:
1. `tar -xzf libero_pro.tar.gz -C "$LIBERO_PRO_PATH" --strip-components=1`
2. `cd "$LIBERO_PRO_PATH" && git init && git add -A && git commit -m snapshot` (needed so install.sh's `clone_or_reuse_repo` later finds an intact repo)
3. `git apply <workspace_pro>/liberopro_register_perturbations.patch`
4. `pip install --no-deps --no-build-isolation -e .` — `--no-build-isolation` is mandatory because USTC pip mirror SSL is broken here, but setuptools 75.8.2 + wheel 0.46.3 are already in the venv.

Sanity passed: all 15 perturbation suites instantiate, README §"Sanity check" exact-string asserts (`libero_10_{swap,task,lan}` t8 + `libero_spatial_{task,lan}` t0) all match BDDL `:language`. Frame eef-z probe matches [[env_calibration]] — KITCHEN 1.173, LIVING_ROOM 0.681.

Missing: `libero_spatial_swap` BDDL/init not in git repo; HF dataset `zhouxueyang/LIBERO-Pro` is unreachable (direct HF, hf-mirror.com, ghfast.top all blocked or no-route). Skip this suite or fetch manually on a network with HF access.

Why: needed to start the LIBERO-Pro track per [[liberopro-pro-hybrid-guide]] (physicalagent/context/guides/PRO_HYBRID_GUIDE.md). Hybrid LLM+Pi0.5 evaluation on P1/P2 perturbations.

How to apply: every fresh PRO session must
- `cd "$PHYSICALAGENT_REPO_ROOT"` (NOT an unrelated checkout; set `PHYSICALAGENT_RLINF_ROOT` / `RLINF_REPO_PATH` explicitly so an older RLinf tree cannot shadow-import first)
- `export LIBERO_TYPE=pro MUJOCO_GL=egl`
- launch via `physicalagent/backends/rlinf/repl_driver.py` (now patched, see [[liberopro-driver-patch]]).
