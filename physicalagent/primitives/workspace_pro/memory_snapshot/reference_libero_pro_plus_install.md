---
name: libero-pro-plus-install
description: Reproduce LIBERO-Pro + LIBERO-Plus install on the existing openpi venv via workspace_pro/install_libero_pro_plus.sh; HF snapshot at /mnt/public2/zhangyixian/datasets/liberopro_hf/
metadata: 
  node_type: memory
  type: reference
  originSessionId: 234721d6-dc80-4ac9-806e-e06977ce7823
---

**Install script**: `physicalagent/primitives/workspace_pro/install_libero_pro_plus.sh`

Re-runs LIBERO-Pro + LIBERO-Plus install on top of the **existing** openpi venv (no fresh venv, no torch reinstall). Idempotent:

```bash
bash physicalagent/primitives/workspace_pro/install_libero_pro_plus.sh           # both
bash physicalagent/primitives/workspace_pro/install_libero_pro_plus.sh --only pro
bash physicalagent/primitives/workspace_pro/install_libero_pro_plus.sh --only plus
```

What it does, on top of existing `libero` install at `/opt/venv/openpi/libero/`:

1. **LIBERO-Pro**:
   - Clone/reuse `LIBERO-PRO` repo at `/opt/venv/openpi/libero_pro/`
   - `pip install -e --no-build-isolation` (falls back to `.pth` file on network failure)
   - Apply `liberopro_register_perturbations.patch` (registers 16 perturbation suites + overrides Task.language from BDDL); idempotent via `git apply --check` / `--reverse --check`
   - **Sync HF snapshot** from `/mnt/public2/zhangyixian/datasets/liberopro_hf/` (BDDLs + init_files, 1.2MB, 16 suites × ~10 files each). Upstream ships several `.pruned_init` at 0/364 bytes — sync is REQUIRED for swap suites to have nonzero trials.
   - Verify all 12 perturbation suites resolve with nonzero trials (skip known-empty `libero_spatial_swap`)

2. **LIBERO-Plus**:
   - Optional apt deps (libexpat1, libfontconfig1-dev, libpython3-stdlib, libmagickwand-dev) — best-effort, warns if no sudo / no apt
   - Clone/reuse at `/mnt/public2/zhangyixian/LIBERO-plus/`
   - `pip install -r extra_requirements.txt && pip install -e --no-build-isolation`
   - Verify import + assets dir present (assets bundled in repo, 8.5GB at `liberoplus/liberoplus/assets/`)

**Env vars** to override defaults:
- `VENV_PY` — python in target venv (default `/opt/venv/openpi/bin/python`)
- `LIBERO_PRO_PATH` / `LIBERO_PLUS_PATH` — repo locations
- `LIBERO_PRO_HF_DIR` — HF snapshot path
- `USE_MIRROR=1` — use `ghfast.top` mirror for unstable github TLS

**HF snapshot re-download** (if `LIBERO_PRO_HF_DIR` missing):
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id='zhouxueyang/LIBERO-Pro', repo_type='dataset',
                  local_dir='/mnt/public2/zhangyixian/datasets/liberopro_hf',
                  allow_patterns=['bddl_files/**','init_files/**'])
```

Related: [[liberopro-install]] [[liberopro-driver-patch]]
