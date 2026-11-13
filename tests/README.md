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

## LeKiwi camera contract check

The unit suite checks RPC recovery after a missing camera frame with a local
camera stub. To also check the real RLinf queue reader, install the companion
RLinf source used by LeKiwi (including its dependencies and the
`get_frame(return_timestamp=True)` API), then run:

```bash
RPENT_RUN_LEKIWI_INTEGRATION=1 \
  pytest tests/e2e_tests/lekiwi/test_camera_contracts.py -v
```

This CPU-only check uses an in-memory frame queue and loopback HTTP. It does not
open a camera, start a robot or require a GPU. It checks that a missing frame
reopens the camera and the next timestamped frame reaches the RPC client.
Without the opt-in flag it is skipped; when enabled, missing or incompatible
RLinf fails the test.

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

### RoboDojo primitive smoke test

RoboDojo has a separate opt-in environment check; it is not a target of
`run_gpu_suite.sh` or the GPU CI workflow. Use the RoboDojo environment and
asset setup from the simulator guide, reserve one GPU, then run:

```bash
CUDA_VISIBLE_DEVICES=<reserved-GPU-ordinal> \
  OMNI_KIT_ACCEPT_EULA=YES \
  RPENT_RUN_ROBODOJO_INTEGRATION=1 \
  RPENT_E2E_OUTPUT_DIR=/path/to/new-output-dir \
  python -m pytest tests/e2e_tests/robodojo/test_environment_smoke.py -v -s
```

This checks saved RGB-D/world maps, both grippers at 0/0.5/1, held controller
targets, small Cartesian moves and wrist rotations, native EE hold execution,
action-aligned VPP2 history and reset, and `finish`.
It requires Isaac Sim, cuRobo, RoboDojo assets and a GPU, but no SAM3,
VLA or external planner. The 900-second test timeout includes simulator
startup. It is not a SAM3 test, a planner-policy-chain test, or a benchmark
success claim. Without the opt-in flag it is skipped.

With the same opt-in variables, SAM3 installed in the runtime environment, and
`SAM3_CHECKPOINT_PATH` pointing to its local checkpoint, run:

```bash
python -m pytest tests/e2e_tests/robodojo/test_components.py::test_sam3_component -v -s
```

The SAM3 component test segments a real saved frame and back-projects a foreground
pixel without advancing the environment. It needs no VLA or external planner.

With VPP2 dependencies and `VPP2_MODEL_PATH` configured as in the simulator guide,
run `test_components.py::test_vpp2_component` and `test_policy_chain.py`.
To reuse an already running VPP2 service, set
`RPENT_ROBODOJO_VLA_ENDPOINT=socket://host:port` instead of `VPP2_MODEL_PATH`.
The tests leave this borrowed service running.
The VPP2 component checks real prediction after an analytic action and executes
one native EE target. The two policy-chain cases use the public CLI and a
scripted local planner to request images, segmentation, a depth query, either
a gripper action or one VPP2 chunk, fresh image feedback and `finish`.
Normal CLI startup requires all three services, even in the analytic case.
The model-backed tests have a 2400-second timeout and are skipped without
either `VPP2_MODEL_PATH` or `RPENT_ROBODOJO_VLA_ENDPOINT`; the Env and SAM3
cases retain their 900-second timeout.
For Awomo05, set `RPENT_ROBODOJO_VLA_BACKEND=awomo05` and configure
`AWOMO05_MODEL_PATH` or `RPENT_ROBODOJO_VLA_ENDPOINT`. Run
`test_components.py::test_awomo05_component` and `test_policy_chain.py`.
The component check updates native history after an analytic action, predicts
one eight-action chunk, and executes one joint target.

This checks integration, not autonomous planning quality or benchmark task success.
RoboDojo is not yet a target of
`run_gpu_suite.sh`; run these opt-in tests directly with a fresh output directory.

### Supported full suites

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

## External WAM GPU services

The opt-in backend tests use separately provisioned checkpoint services. They
are not included in `run_gpu_suite.sh`, which provisions the standard robot
stacks. Start the workers described in the
[action-model guide](../docs/source-en/rst_source/development/add_action_model.rst),
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
run the same cases against Cosmos Policy, or `RPENT_COSMOS3_ENDPOINT` for
Cosmos3's RPent RPC worker. Set `RPENT_WAM_SUITE=libero_10` when using a
LIBERO-10 checkpoint. Missing endpoints skip their cases.
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

The policy-chain tests exercise the public CLI and `wam_act` toolkit path
with a scripted planner, recording observations and `finish`. These are bounded
integration checks, not task-success benchmarks. Component checks close their
clients, and CLI checks stop their owned daemons. Externally started model
services remain the operator's responsibility.

Offline AM checks follow the VLA/SAM layout: `test_am.py` covers startup,
configuration and public boundaries; `test_wam_adapters.py` checks physical
input and native output transforms. `test_wam_loopback.py` covers HTTP/socket
session isolation, reset and cleanup; `test_cosmos_policy_loopback.py` covers
Cosmos Policy conversion through those transports. Platform tool checks stay
in platform unit tests.

## Dual-Franka VLA diagnostic console

Run the real-robot diagnostic explicitly from a source checkout after installing
its robot dependencies and configuring the checkpoint, dataset normalization
statistics, cameras, reachable robot environment, and easy_handeye YAML paths under
`perception.calibration` in the robot config:

```bash
python -m tests.e2e_tests.dual_franka.dual_franka_vla \
  --robot-config /path/to/robot.yaml \
  --am-checkpoint /path/to/checkpoint --pi05-repo-id org/dataset
```

This console is not collected by pytest or included in the automated GPU suite.
Startup does not reset the robot. Enter `run N` for N prediction chunks, `run`
for continuous rollout, `reset` for an operator-confirmed reset, or `quit` to exit.
The instruction comes from `vla.instruction` in the robot config or `--instruction`,
not a registered task. Ctrl-C, SIGTERM, or the printed `STOP` file stops rollout;
stop checks occur between individual actions, but an in-flight RPC may still finish.
Inputs and predictions are saved as JSON/NPZ. The prediction chunk length comes
from `vla.status`; `--expected-action-steps` overrides it. The console starts only
env and VLA components, with no SAM3 dependency.
Offline regression tests remain in `unit_tests/robots/dual_franka/`.

See the deployment prerequisites in the
[English](../docs/source-en/rst_source/real_world_robots/dual_franka.rst) or
[Chinese](../docs/source-zh/rst_source/real_world_robots/dual_franka.rst) guide.
