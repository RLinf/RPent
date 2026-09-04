# Test suite structure

RPent keeps its fast, offline unit tests under `unit_tests` and its
checkpoint-backed embodied GPU tests under `e2e_tests`. Unit tests mirror the
production module they exercise. GPU E2E tests are grouped by robot stack and
exercise the real simulator and model runtime.

## Directory layout

```text
tests/
├── README.md
├── e2e_tests/
│   ├── common.py         # shared assertions and runtime lifecycle helpers
│   ├── run_gpu_suite.sh  # clean-environment entry point for one robot stack
│   ├── libero/
│   ├── robocasa/
│   └── robotwin/
└── unit_tests/
    ├── rpent/             # tests for the core rpent package
    │   ├── cli/
    │   ├── dashboard/
    │   ├── memory/
    │   ├── planner/
    │   ├── robots/
    │   ├── session/
    │   ├── tools/
    │   └── utils/
    └── robots/           # tests for the top-level robot extensions
        ├── libero/
        ├── robocasa/
        └── robotwin/
```

Directories that do not yet have tests do not need empty placeholders. Add
them when the first test for that module lands.

## Placement rules

- Mirror core modules under `tests/unit_tests/rpent/`. For example, tests for
  `rpent.utils.rpc` belong in `tests/unit_tests/rpent/utils/rpc/`.
- Mirror top-level robot extensions under `tests/unit_tests/robots/`. Put
  robot-specific coverage in that directory's `<robot>/` child.
- Place cross-layer tests with the primary contract owner. Registry and config
  contracts belong to `rpent/robots/`; extension toolkit and schema contracts
  belong to `robots/`.
- Keep one-off fakes in the test module that uses them. Put shared fixtures in
  the nearest `conftest.py`: use `tests/conftest.py` only for suite-wide
  fixtures and a module directory's `conftest.py` for local fixtures.
- Name files after the behavior they verify: `*_contracts.py` for stable API
  contracts, `*_loopback.py` for real local transports, `*_lifecycle.py` for
  resource ownership, and `*_smoke.py` for installation or startup checks.
- Put checkpoint-backed simulator and policy-chain coverage under the matching
  `tests/e2e_tests/<robot>/` directory. Keep reusable validation and lifecycle
  code in `tests/e2e_tests/common.py`.

Unit tests must run offline on an ordinary CPU machine. They may cross a real
local boundary, such as a loopback TCP connection or child process, but must
not contact external services.

Run the complete suite with:

```bash
pytest tests/unit_tests -v
```

## Embodied GPU E2E tests

The runner must expose one GPU and the checkpoint and simulator assets required
by the selected target. Run one GPU suite with two new output paths:

```bash
export CUDA_VISIBLE_DEVICES=<one-physical-GPU-ordinal>
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

bash tests/e2e_tests/run_gpu_suite.sh \
  <libero-pro|robocasa|robotwin> \
  /path/to/new-output-dir \
  /path/to/new-venv-root
```
