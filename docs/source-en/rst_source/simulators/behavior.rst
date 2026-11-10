BEHAVIOR
========

Run the BEHAVIOR-1K microwave popcorn task on a public-test instance with RPent. The Planner selects tools
using images and task instructions. The official evaluator receives the resulting
native actions and owns simulation, reset, termination and scoring.

.. _behavior-overview:

Overview
--------

Use one Env service and evaluator connection for each episode.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      RLinf Pi0.5 BEHAVIOR; optional gripper, arm and base primitives.

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``

   .. grid-item-card:: Tasks

      100 challenge task IDs, ``0..99``.

   .. grid-item-card:: Hardware

      Linux and NVIDIA GPU; compatible OmniGibson graphics and model CUDA runtimes.

Tasks
~~~~~

Task recognition describes the integration's scope, not measured task success.

.. list-table::
   :header-rows: 1

   * - Scope
     - Selection
     - Runtime
   * - 100 official challenge tasks
     - Matching task ID, official name and complete main-task text.
     - One R1Pro environment and one episode per Env service.
   * - Public-test episode example
     - Task ``40``, ``make_microwave_popcorn``; split index ``0`` is instance ``301``.
     - Keep the official task step budget and use ``--num-envs 1``.

.. _behavior-observation-action:

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The Planner's tool calls and the simulator's native actions are separate interfaces.

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - Head, left-wrist and right-wrist RGB in that order. The Pi0.5 adapter accepts a 61-channel public robot state or an already encoded 256-channel state. It maps the former to the model layout and rejects other state lengths.
   * - Action
     - The Planner calls VLA or motion tools. The environment receives 23-channel native actions. The BEHAVIOR Pi0.5 preset predicts 32 actions per call using BF16 and five denoising steps.
   * - Reward / success
     - Use the official evaluator JSON success and QScore. Tool completion and Planner ``finish`` are not official task scores.
   * - Task prompt
     - The Agent sends the task instruction with each prediction. Main-task mode enforces the full configured instruction.

Installation and Resources
--------------------------

Install OmniGibson, the official evaluator and the challenge assets using the
`BEHAVIOR-1K instructions <https://github.com/StanfordVL/BEHAVIOR-1K>`_.
RPent supports Python 3.10–3.12. Keep the simulator, model and Planner
environments separate to satisfy their respective Python, graphics and CUDA
requirements. The Env bridge can use the lightweight Planner dependencies
unless you enable CuRobo.

Choose a fresh output directory outside the source checkout. Replace the paths
and Planner model below, then run these exports in every terminal, or save them
in a local shell file and source it in each terminal. Keep the three task
variables together when selecting another task from ``robots/behavior/tasks.py``.

.. code-block:: bash

   export RPENT_REPO=/path/to/RPent
   export RPENT_RUN=/path/to/behavior-runs/run-name
   export VLA_PYTHON=/path/to/vla-env/bin/python
   export ENV_PYTHON=/path/to/env-service-env/bin/python
   export SIM_PYTHON=/path/to/simulator-env/bin/python
   export PLANNER_PYTHON=/path/to/planner-env/bin/python
   export VLA_CHECKPOINT=/path/to/checkpoint
   export PLANNER_MODEL=provider:vision-model-name
   export TASK_ID=40
   export TASK_NAME=make_microwave_popcorn
   export TASK_LANGUAGE='In the kitchen, take the popcorn bag from the countertop, put it into the microwave, and heat it until the popcorn is cooked so the cooked popcorn ends up inside the bag.'
   cd "$RPENT_REPO"

Install RPent from the checkout into each environment that will import it:

.. code-block:: bash

   "$PLANNER_PYTHON" -m pip install -e '.[behavior]'
   "$ENV_PYTHON" -m pip install -e '.[behavior]'
   "$VLA_PYTHON" -m pip install -e '.[behavior-vla]'

The model environment also needs an RLinf/OpenPI build with BEHAVIOR support.
A compatible public checkpoint is
`RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_.
Use its matching normalization statistics and tokenizer. Set ``OPENPI_DATA_HOME``
if your RLinf build uses it to locate a local tokenizer. Configure the
image-capable Planner model and its credentials following
:doc:`../guides/configure_planner`. The VLA checkpoint and the Planner model
are selected independently.

Run a Task
----------

Use a separate terminal for each of the following four commands. The example
uses the ``vla_only.yaml`` tool configuration, which requires just the VLA
worker and Env bridge. An image-capable Planner chooses when to observe, call
the VLA, or finish.

**1. Start the shared Pi0.5 worker.** The worker loads the selected checkpoint
and predicts native actions from three camera views and the public robot state.
Wait for the model to finish loading and the service to listen on port 28011.

.. code-block:: bash

   "$VLA_PYTHON" -m rpent.robots.components.pi05_vla_server \
     --embodiment behavior --model-path "$VLA_CHECKPOINT" \
     --host 127.0.0.1 --port 28011 --transport socket

The example selects the default ``comet`` state layout, with both grippers
following the arms. For a compatible checkpoint trained with each gripper
directly after its arm, append ``--behavior-state-layout abs_joint`` and use
an RLinf environment that provides the ``openpi_pytorch`` loader.

**2. Start the Env bridge.** It connects to the worker and listens on two
ports. RPent sends internal Env RPC requests to port 28010. The official
evaluator connects to the policy WebSocket on port 28012.

.. code-block:: bash

   "$ENV_PYTHON" -m robots.behavior.env_server \
     --task-id "$TASK_ID" --task-language "$TASK_LANGUAGE" \
     --am-endpoint socket://127.0.0.1:28011 \
     --host 127.0.0.1 --port 28010 --transport socket \
     --policy-host 127.0.0.1 --policy-port 28012

**3. Start the official evaluator.** For task 40, public-test split index 0 is
instance 301. Run a single environment and preserve the evaluator's default
step budget. The bridge needs one observation/action exchange per simulator
step.

.. code-block:: bash

   "$SIM_PYTHON" -m omnigibson.eval.eval \
     --task-name "$TASK_NAME" --mode public_test --instance-indices 0 \
     --num-rollouts 1 --num-envs 1 --host 127.0.0.1 --port 28012 \
     --write-video --video-fps 120 --output-dir "$RPENT_RUN/official"

The default RGB observations suffice for VLA inference and 2D segmentation.
For 3D tools such as ``pixel_to_robot_base``, append
``--env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper`` to this evaluator
command. Projection requires a depth image and camera calibration matching the
selected RGB frame.

**4. Start the Planner when the first evaluator observation is available.**
The task ID and text must match the Env bridge. The Planner's VLA endpoint
must point to the same worker the bridge resets at episode start.

.. code-block:: bash

   "$PLANNER_PYTHON" -m rpent.cli.main --robot behavior --planner api \
     --model "$PLANNER_MODEL" \
     --behavior-config robots/behavior/configs/vla_only.yaml \
     --task-id "$TASK_ID" --task-language "$TASK_LANGUAGE" \
     --env-endpoint socket://127.0.0.1:28010 \
     --am-backend pi05 --am-endpoint socket://127.0.0.1:28011 \
     --output-dir "$RPENT_RUN/agent"

To enable the complete ``agent.yaml`` tool set, prepare CuRobo v0.7.8 with
compatible PyTorch, Warp and CUDA in the Env bridge environment. Install SAM3
with its model dependencies in a separate environment, then start its service
in another terminal before starting the Planner:

.. code-block:: bash

   export SAM3_PYTHON=/path/to/sam3-env/bin/python
   export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
   "$SAM3_PYTHON" -m pip install -e '.[sam3]'
   "$SAM3_PYTHON" -m rpent.robots.components.sam3_server \
     --host 127.0.0.1 --port 28013 --transport socket

Start the Env bridge with ``--motion-backend curobo``, select
``--behavior-config robots/behavior/configs/agent.yaml`` on the Planner and
add ``--sam3-endpoint socket://127.0.0.1:28013``. After an episode, stop the
separately started services yourself. Use a fresh Env bridge and output
directory for the next episode. This integration supports one evaluator
connection per bridge and has no Dashboard launcher.

View Results
------------

Under ``RPENT_RUN``, ``agent/`` contains the Planner transcript, observations
and tool results. The evaluator writes JSON scores and videos to ``official/``.
Read the evaluator JSON for task success and QScore. A Planner's ``finish``
records its assessment. The example encodes videos at 120 FPS; playback rate
is independent of simulation timestep and action frequency. The Pi0.5 worker logs
its selected embodiment and checkpoint path at startup.

We evaluated the same VLA trained by our team both as a standalone policy and
as the action model in RPent. For each task and method, the table reports
the mean QScore across 10 instances, as scored by the official BEHAVIOR evaluator.

.. list-table::
   :header-rows: 1

   * - Task
     - VLA only (baseline)
     - RPent + VLA
   * - Turn on the radio
     - 0.20
     - 0.80
   * - Make microwave popcorn
     - 0.70
     - 0.90
   * - Pick up trash
     - 0.33
     - 0.73

Common Options
--------------

CLI options override the corresponding fields in the Planner YAML. The values
below describe the two shipped templates wherever a template supplies them.

.. list-table:: Planner options
   :header-rows: 1

   * - Option
     - Behavior
   * - ``--task-id``, ``--task-language``
     - Select an ID from 0–99. When the text is omitted, the Planner uses the official instruction for that ID. The Env bridge requires the text explicitly.
   * - ``--behavior-config``, ``--tool-allowlist``
     - Select the YAML template or override its robot tools. An explicit allowlist must include ``view_env_state`` and ``finish``. Shared memory/file tools remain available.
   * - ``--env-endpoint``, ``--am-backend``, ``--am-endpoint``, ``--sam3-endpoint``
     - Use internal ``socket://`` or ``http://`` addresses. BEHAVIOR uses the ``pi05`` action backend. The ``--vla-endpoint`` alias for ``--am-endpoint`` is also accepted. Supply SAM3 when enabling ``segment``.
   * - ``--control-mode``, ``--instruction-mode``
     - Both templates use ``agent`` control and return to the Planner after each bounded VLA call. In ``main`` mode, the Planner checks the complete task instruction against the Env bridge on connection. The ``override`` mode permits a tool call to supply its own instruction. The parser also accepts ``scripted`` control.
   * - ``--execution-horizon``, ``--max-chunks-per-call``
     - Both templates request up to 32 native actions per chunk and at most one chunk per tool call. Reduce the horizon to submit a shorter prefix of each prediction.
   * - ``--env-rpc-timeout-s``
     - The templates allow 1,230 seconds for an Env call, including overhead on top of the bridge's 1,200-second chunk deadline.
   * - ``--model``, ``--memory-dir``, ``--output-dir``
     - Select an image-capable Planner, a local memory corpus and a dedicated output directory. The Planner requires an output directory for run artifacts.

.. list-table:: Model and bridge options
   :header-rows: 1

   * - Process and option
     - Behavior
   * - Pi0.5 ``--embodiment behavior``, ``--model-path``
     - Select the ``pi05_behavior`` model configuration and compatible weights. The model path can also come from ``PI05_CHECKPOINT_PATH``.
   * - Pi0.5 ``--behavior-state-layout``, ``--norm-stats-path``, ``--repo-id``
     - Match the training layout (default ``comet``). An explicit ``--norm-stats-path`` selects ``norm_stats.json`` or its directory and takes precedence over ``--repo-id``. With a dataset ID, the worker looks for that ID's statistics under the checkpoint root or its ``assets/`` directory. The public PT50-CS32 checkpoint uses ``assets/behavior-1k/2025-challenge-demos/norm_stats.json`` by default. Missing statistics stop worker startup.
   * - Pi0.5 ``--seed``, ``--cuda-device``
     - BEHAVIOR defaults to seed 0. Use a fresh worker and the same seed to repeat the first prediction. Select the GPU visible to the worker if necessary.
   * - Env ``--task-id``, ``--task-language``, ``--am-endpoint``
     - Supply the evaluator task identity and the same worker the Planner uses.
   * - Env ``--host``, ``--port``, ``--transport``, ``--policy-host``, ``--policy-port``
     - Set the internal RPent RPC listener and the separate official policy WebSocket address. The evaluator uses the latter.
   * - Env ``--simulation-timeout-s``, ``--action-timeout-s``, ``--chunk-timeout-s``
     - Defaults are 600 seconds waiting for an observation, 180 seconds waiting for an action, and 1,200 seconds for the entire chunk.
   * - Env ``--motion-backend``, ``--motion-device``
     - The default backend is ``none``. Select ``curobo`` for IK-based arm tools. Its default device is ``cuda:0``.

Task Memory
-----------

Pass ``--memory-dir`` to select a local corpus. The default is
``memory/behavior/`` under the repository root. The Planner uses RPent's
shared ``list_dir`` and ``read_text_file`` tools to read ``MEMORY.md`` after
the first observation, then up to three relevant entries before acting. The
layout includes ``global/`` and ``task-family/`` as described in
:doc:`../development/memory`. Evaluation opens the corpus read-only. Use the
current observations to localize targets and confirm grasps. Evaluation uses a
local corpus without remote synchronization.

.. _runtime-contracts-and-troubleshooting:

Tools and Troubleshooting
-------------------------

The complete Agent exposes twelve robot tools. The three in ``vla_only.yaml``
are ``view_env_state``, ``vla_act`` and ``finish``.

.. list-table::
   :header-rows: 1

   * - Tool
     - Result
     - Advances simulation
   * - ``view_env_state``
     - Saved RGB views, state and execution feedback.
     - No
   * - ``read_robot_state``
     - Measured joints and velocities for a saved observation.
     - No
   * - ``segment``
     - SAM3 mask and overlay from text or an original-image pixel in ``[row, column]`` order.
     - No
   * - ``pixel_to_robot_base``
     - A saved-frame pixel and depth projected into that frame's robot base.
     - No
   * - ``vla_act``
     - A bounded action chunk predicted from the current observation and instruction.
     - Yes
   * - ``set_gripper``
     - A left or right gripper command, from ``val=0`` (closed) to ``val=1`` (open).
     - Yes
   * - ``move_to``, ``move_eef_delta``
     - A nearby arm pose or a relative XYZ offset solved by CuRobo IK.
     - Yes
   * - ``rotate_wrist``
     - A relative end-effector rotation about robot-base Z, from −180° to 180°, using IK.
     - Yes
   * - ``rotate_wrist_joint``
     - A relative roll, pitch or yaw wrist-joint turn from −90° to 90°, clipped to joint limits.
     - Yes
   * - ``move_base``
     - Bounded base velocity followed by a zero command.
     - Yes
   * - ``finish``
     - End the Planner session and save its assessment.
     - No

An additional ``release`` tool opens the selected gripper when explicitly
included in the allowlist. Tools return RPent's ``ToolResult(data, images)``.
Action tools record post-action observations, including three views, ``state``
and ``log.result``. An observation's ``step`` is a record index. Read-only
tools use that saved frame. Re-localize geometric targets after motion.

IK-based arm tools plan from a stationary start and interpolate seven joints
under their velocity limits. They hold the trunk, the other arm and both
gripper commands. Left-arm targets use action channels 7–13 and right-arm
targets 15–21. The base channels 0–2 stay at zero. Check measured pose error
and action confirmations for completion. Zero terminal velocity is not required. A permitted position-only fallback
reports ``orientation_relaxed`` and orientation error. A failed IK attempt can
report a nearest reachable pose without issuing the planned arm trajectory.

The simulator resolves contact. The arm tools plan short adjustments without
path-level scene collision checking. Even when the wrist's target XYZ is
fixed, the fingers can sweep through nearby objects. Use this tool for
free-space orientation adjustments. A direct wrist-joint turn
uses a joint ramp and hold, keeping the remaining joints and gripper commands fixed while
the hand follows an arc. Check its measured
change and clipping flag. Gripper command completion alone cannot confirm a
grasp, and ``move_base`` does not measure reliable odometry or guarantee arrival.

For a connection failure, check the addresses and logs of all processes.
Passing the official ``ws://`` endpoint as ``--env-endpoint`` will fail.
After a timeout, inspect issued, confirmed and unconfirmed counts before
restarting. Only a subsequent observation confirms an issued action. Preserve
unknown counts and avoid replaying a timed-out chunk.

Development and Tests
---------------------

The following paths are relative to ``robots/behavior/`` unless otherwise
specified.

.. list-table:: Code map
   :header-rows: 1

   * - Area
     - Files and responsibilities
   * - Task catalog and registration
     - ``tasks.py`` maps official task IDs, names and instructions. ``__init__.py``, ``robot_spec.py`` and ``configs/`` connect the robot, templates and borrowed services to RPent.
   * - Planner prompts
     - Edit general decision guidance in ``prompts/evaluate.py``. ``prompt_bundle.py`` assembles it for the selected tools. The task goal is supplied at runtime.
   * - Robot tools
     - ``primitives.py`` implements actions and execution accounting. ``tools.py`` decodes public observations (``snapshot_observation``, ``extract_public_proprio``) and handles segmentation, projection and ``finish``. ``toolkit.py`` registers tool groups, binds resources and captures post-action observations. Keep units, limits and result semantics in the typed ``@tool`` definitions.
   * - Motion
     - ``motion.py`` and ``assets/r1pro/`` provide the Env-side CuRobo implementation and robot model. ``R1ProMotionModel`` reads the bundled URDF and CuRobo robot configuration for joint limits and self-collision geometry. ``R1ProMotionPlanner`` solves IK and provides waypoints for the action tools.
   * - Pi0.5
     - ``pi05.py`` adapts R1Pro observations, state layouts and actions. The shared ``pi05_vla_client.py``, ``pi05_vla_server.py`` and ``am_runtime.py`` live under ``rpent/robots/components/`` in the repository root.
   * - Env and evaluator
     - ``env_client.py`` and ``env_server.py`` implement internal RPC and service startup. In ``eval/evaluator.py``, ``WebsocketPolicyServer`` handles the wire protocol, ``OfficialEvaluatorPolicy`` provides ``reset`` and ``act``, and ``EvaluatorStepBridge`` matches actions to subsequent observations. Both service interfaces share one bridge.

**Offline tests.** From the checkout in a Linux CPU environment, set
``PLANNER_PYTHON`` to its interpreter. Install the test dependencies and CPU
PyTorch, then run the robot tests and shared Pi0.5 contracts:

.. code-block:: bash

   "$PLANNER_PYTHON" -m pip install -e '.[test,behavior]'
   "$PLANNER_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
   "$PLANNER_PYTHON" -m pytest tests/unit_tests/robots/behavior \
     tests/unit_tests/rpent/robots/components/test_pi05_vla_client.py \
     tests/unit_tests/rpent/robots/components/test_pi05_vla_server_contracts.py -q

**Real-service tests.** For an exclusive test episode, follow steps 1–3 above.
Once the evaluator supplies its first observation, run the following command
in place of the Planner. Use a new output directory and run serially with no
other action client or ``pytest-xdist``. The three tests borrow the services
and need no VLM credentials.

.. code-block:: bash

   export RPENT_RUN_BEHAVIOR_INTEGRATION=1
   export RPENT_BEHAVIOR_TASK_ID="$TASK_ID"
   export RPENT_BEHAVIOR_TASK_LANGUAGE="$TASK_LANGUAGE"
   export RPENT_BEHAVIOR_ENV_ENDPOINT=socket://127.0.0.1:28010
   export RPENT_BEHAVIOR_VLA_ENDPOINT=socket://127.0.0.1:28011
   export RPENT_E2E_OUTPUT_DIR="$RPENT_RUN/integration"
   "$PLANNER_PYTHON" -m pytest tests/e2e_tests/behavior -v -s

The tests skip before connecting unless ``RPENT_RUN_BEHAVIOR_INTEGRATION=1``.
Select ``tests/e2e_tests/behavior/test_components.py`` to check three RGB
views, the public 61-channel state and a real ``[T, 23]`` VLA prediction without
submitting actions. The component checks issue one inference request.
``test_policy_chain.py`` then uses ``BehaviorToolkit`` to predict and submit
one chunk of up to 32 native actions, receive three new images and call
``finish(blocked)``. It saves the prediction, submitted actions, execution
counts and Toolkit state/images under ``RPENT_E2E_OUTPUT_DIR``.

A passing policy-chain check confirms every submitted action from an
observation. The final ``blocked`` status ends this bounded check. It exercises the Toolkit directly. Use a full episode with the
Planner and evaluator to assess autonomous decisions and official task
success. The tests release only their client connections. Stop the borrowed
services yourself, including the evaluator if it is still waiting for actions.
These checks run separately from ``run_gpu_suite.sh``.
