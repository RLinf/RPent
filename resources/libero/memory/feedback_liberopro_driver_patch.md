---
name: liberopro-driver-patch
description: "RPent LIBERO environment setup must resolve get_benchmark through the LIBERO_TYPE-aware RLinF benchmark module, not base libero"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 23bda317-5513-4ade-ae34-7d3f3988aeef
---

In `robots/libero/tools.py`, `make_env` originally did `from libero.libero.benchmark import get_benchmark`. That registry has no PRO suites — `get_benchmark('libero_spatial_task')` raises `KeyError` even with `LIBERO_TYPE=pro`.

Fix (applied 2026-05-20):
```python
from rlinf.envs.libero.utils import benchmark as _bench_mod
suite = _bench_mod.get_benchmark(suite_name)()
```

**Why:** `utils.py` reads `LIBERO_TYPE` at module load and binds `benchmark` to `liberopro.liberopro.benchmark` (or `liberoplus...`) for `pro`/`plus`. Using it routes correctly without touching the rest of the driver. `LiberoEnv.get_env_fns` separately remaps `sys.modules["libero.libero.benchmark"]` per-worker — but that happens AFTER make_env runs in the main process, so the main-process import has to be variant-aware on its own.

**How to apply:** check `robots/libero/env_server.py` on any fresh RPent checkout. It should resolve the suite through `rlinf.envs.libero.utils.benchmark`; if it reverts to base `libero.libero.benchmark`, restore the variant-aware lookup before running any PRO suite.
