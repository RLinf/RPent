# Test suite structure

RPent keeps its fast, offline unit tests under `unit_tests` and its
checkpoint-backed embodied GPU tests and explicit real-robot diagnostics under
`e2e_tests`. Unit tests mirror the
production module they exercise. GPU E2E tests are grouped by robot stack and
exercise the real simulator and model runtime.

## Directory layout

```text
tests/
├── README.md
├── e2e_tests/
│   ├── common.py         # shared assertions and runtime lifecycle helpers
│   ├── run_gpu_suite.sh  # clean-environment entry point for one robot stack
│   ├── dual_franka/     # opt-in real-robot VLA diagnostic console
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

## RoboCasa environment smoke tests

The opt-in RoboCasa smoke suite verifies simulator installation and environment
interfaces without loading a planner or VLA checkpoint. Install the RoboCasa
extra and assets, then run:

```bash
RPENT_RUN_ROBOCASA_INTEGRATION=1 \
  pytest tests/integration_tests/robots/robocasa/test_target50_runtime_smoke.py -v
```

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

## External WAM GPU services

The opt-in backend tests use separately provisioned checkpoint services. They
are not included in `run_gpu_suite.sh`, which provisions the standard robot
stacks. Start the workers described in the
[WAM backend guide](../docs/source-en/rst_source/development/wam_backends.rst),
then run from a LIBERO-enabled test environment:

```bash
RPENT_FAST_WAM_ENDPOINT=http://127.0.0.1:8117 \
  pytest tests/e2e_tests/libero/test_components.py \
    tests/e2e_tests/libero/test_policy_chain.py -k 'wam or horizon' -v \
    --junitxml=fast-wam.xml
```

The component test uses real LIBERO observations, predicts and executes one
action chunk, and records inference latency and checkpoint identity. The CLI
chain executes up to four chunks within a 32-action episode horizon. A separate
environment check verifies that reset settling does not consume that horizon.
These tests do not require native task success. Configure simulator assets
independently of the model environment. Set `RPENT_COSMOS_ENDPOINT` instead to
run the same cases against Cosmos Policy; missing endpoints skip their cases.
Set `RPENT_WAM_SUITE=libero_spatial_task` and configure LIBERO-Pro assets to
exercise Pro instead of the default `libero_spatial`, task 0, seed 0.

RoboTwin uses its own Fast-WAM checkpoint and three-camera configuration. From
the RoboTwin simulator environment, with `ROBOTWIN_ASSETS_PATH` configured:

```bash
RPENT_FAST_WAM_ROBOTWIN_ENDPOINT=http://127.0.0.1:8117 \
  pytest tests/e2e_tests/robotwin/test_components.py \
    tests/e2e_tests/robotwin/test_policy_chain.py -k wam -v \
    --junitxml=robotwin-fast-wam.xml
```

This test uses `RoboTwinWAMClient`, reads real camera and joint-target state,
and executes a component chunk through `env.chunk_step(action_type="qpos")`.
The separate CLI chain executes up to two chunks within a 64-action horizon.
The worker must advertise `robotwin.joint_action.qpos14.v1`. It records latency,
executed action counts and episode status; no LingBot checkpoint is needed.
The RoboTwin CLI selects this backend with `--wam-backend fast-wam`; its toolkit
uses `wam_act`. Without a WAM backend, it retains LingBot's EEF16 interface.

The policy-chain tests exercise the public CLI and `wam_act` toolkit path
with a scripted planner, recording observations and `finish`. These are bounded
integration checks, not task-success benchmarks. Component checks close their
clients, and CLI checks stop their owned daemons. Externally started model
services remain the operator's responsibility.

Offline WAM checks follow the VLA/SAM layout: `test_wam.py` covers public
boundaries and startup configuration, `test_wam_adapters.py` checks physical
input and native output transforms, and `test_wam_loopback.py` exercises real
local HTTP/socket sessions. Platform tool checks stay in platform unit tests.
Full benchmark experiments are separate from these bounded integration checks.

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
