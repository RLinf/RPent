---
name: liberopro-driver-patch
description: "interactive_driver.py make_env must import get_benchmark via rlinf.envs.libero.utils (LIBERO_TYPE-aware), not base libero"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 23bda317-5513-4ade-ae34-7d3f3988aeef
---

In `physicalagent/primitives/interactive_driver.py`, `make_env` originally did `from libero.libero.benchmark import get_benchmark`. That registry has no PRO suites — `get_benchmark('libero_spatial_task')` raises `KeyError` even with `LIBERO_TYPE=pro`.

Fix (applied 2026-05-20):
```python
from rlinf.envs.libero.utils import benchmark as _bench_mod
suite = _bench_mod.get_benchmark(suite_name)()
```

**Why:** `utils.py` reads `LIBERO_TYPE` at module load and binds `benchmark` to `liberopro.liberopro.benchmark` (or `liberoplus...`) for `pro`/`plus`. Using it routes correctly without touching the rest of the driver. `LiberoEnv.get_env_fns` separately remaps `sys.modules["libero.libero.benchmark"]` per-worker — but that happens AFTER make_env runs in the main process, so the main-process import has to be variant-aware on its own.

**How to apply:** check this is still patched on any fresh RLinf_agentic checkout. If `interactive_driver.py` line ~58 reverts to `from libero.libero.benchmark`, re-apply the change before running any PRO suite. Related: [[liberopro-install]].
