# Test suite structure

RPent keeps fast, offline unit tests under `unit_tests`. Real-simulator smoke
tests, model component and policy-chain tests, and explicit real-robot
diagnostics belong under `e2e_tests`. Unit tests mirror the production module
they exercise; GPU tests are grouped by robot stack. Environment smoke tests
can run without model checkpoints.

## Directory layout

```text
tests/
├── README.md
├── e2e_tests/
│   ├── common.py         # shared assertions and runtime lifecycle helpers
│   ├── run_gpu_suite.sh  # clean-environment entry point for one robot stack
│   ├── dual_franka/      # opt-in real-robot VLA diagnostic console
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
- Put real-simulator smoke tests, model component tests and policy-chain tests
  under `tests/e2e_tests/<robot>/`. Keep environment smoke tests independent of
  fixtures that require model checkpoints. Reusable validation and lifecycle
  code belongs in `tests/e2e_tests/common.py`.

Unit tests must run offline on an ordinary CPU machine. They may cross a real
local boundary, such as a loopback TCP connection or child process, but must
not contact external services.

Run the complete suite with:

```bash
pytest tests/unit_tests -v
```

## RoboCasa environment smoke tests

RoboCasa separates simulator checks, model component checks and the full call chain:

| Tests | Coverage | Requirements |
| --- | --- | --- |
| `test_environment_smoke.py` | Three scenes, navigation RGB-D/world map and mobile-camera tracking | RoboCasa extra, assets and GPU/EGL; no model weights or planner credentials |
| `test_components.py` | Environment observations over RPC and RLDX inference | Model checkpoint, assets and GPU/EGL |
| `test_policy_chain.py` | CLI → tools → RLDX → simulator action → `finish` | Model checkpoint, assets and GPU/EGL; local scripted planner |

To check simulator installation and interfaces independently, install the
RoboCasa extra and assets, then run:

```bash
RPENT_RUN_ROBOCASA_INTEGRATION=1 \
  pytest tests/e2e_tests/robocasa/test_environment_smoke.py -v
```

This command needs neither `RLDX_MODEL_PATH` nor `RPENT_E2E_OUTPUT_DIR`. Without
`RPENT_RUN_ROBOCASA_INTEGRATION=1`, the four smoke cases are skipped.

The four cases cover `OpenDrawer`, `NavigateKitchen`, and
`PickPlaceCounterToCabinet` at seed 1, plus mobile-camera movement. Task checks
verify construction/reset, 12D actions, operation cameras, navigation RGB-D/world
map, the success predicate, and clean close. The camera check verifies pose and
image changes after eight base steps.

See the RoboCasa usage guide in
[English](../docs/source-en/rst_source/simulators/robocasa.rst#environment-smoke-tests)
or [Chinese](../docs/source-zh/rst_source/simulators/robocasa.rst#environment-smoke-tests)
for setup and test prerequisites.

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

For RoboCasa, the runner enables the four environment smoke cases alongside
the two component tests and one policy-chain test. The existing
[GPU CI](../.github/workflows/embodied-gpu-e2e-run.yml) uses this same runner for
all seven checks. The scripted planner needs no external planner credentials.

## Dual-Franka VLA diagnostic console

Run the real-robot diagnostic explicitly from a source checkout after installing
its robot dependencies and configuring the checkpoint, dataset normalization
statistics, cameras, reachable robot environment, and easy_handeye YAML paths under
`perception.calibration` in the robot config:

```bash
python -m tests.e2e_tests.dual_franka.dual_franka_vla --task-id 1 \
  --robot-config /path/to/robot.yaml \
  --vla-model-path /path/to/checkpoint --vla-repo-id org/dataset
```

This console is not collected by pytest or included in the automated GPU suite.
Initialization may reset the robot. Only `step` and `run N` execute predictions;
`infer` records a prediction without execution. Action validation expects 20
steps per chunk; use `--expected-action-steps` for a different checkpoint chunk
length. The console starts only env and VLA components, with no SAM3 dependency.
Offline regression tests remain in `unit_tests/robots/dual_franka/`.

See the deployment prerequisites in the
[English](../docs/source-en/rst_source/real_world_robots/dual_franka.rst) or
[Chinese](../docs/source-zh/rst_source/real_world_robots/dual_franka.rst) guide.
