BEHAVIOR
========

Use RPent's vision-language planner to perform household tasks in the
BEHAVIOR-1K simulator. The planner uses camera observations to call shared
Pi0.5 and robot action tools. The simulated R1Pro dual-arm mobile robot executes
the actions. A dedicated RPent Env process uses RLinf’s BEHAVIOR 3.9.3
adapter to advance the simulation and read official task metrics.

This guide selects “make microwave popcorn” (task ID 40) from the 100
BEHAVIOR-1K challenge tasks to show how to prepare the environment, run a task,
and find its observations and scores.

.. _behavior-overview:

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      RPent's shared Pi0.5 BEHAVIOR configuration; optional gripper, arm, and base tools.

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``

   .. grid-item-card:: Tasks

      100 challenge task IDs; comparison results reported for three tasks.

   .. grid-item-card:: Hardware

      Linux, NVIDIA GPU, and compatible OmniGibson graphics dependencies and model CUDA environment.

Each Env process owns one simulator instance and one episode. The planner
borrows its RPC endpoint; shared Pi0.5 inference runs in a separate process.

Tasks
~~~~~

Task IDs, official names, and full instructions are listed in
``robots/behavior/tasks.py``. All three must identify the same task.

.. list-table::
   :header-rows: 1

   * - Scope
     - Selection
   * - 100 challenge tasks
     - IDs ``0..99``. When given only ``--task-id``, the planner uses the official task instruction. The Env service uses the same default; any explicit override must match.
   * - Public-test example in this guide
     - ID ``40``, name ``make_microwave_popcorn``; ``--instance-split public_test --instance-id 301`` selects the actual instance ID, not its position in a list. The first-run example caps execution at 320 steps.

.. _behavior-observation-action:

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - Head, left-wrist, and right-wrist camera images, together with measured robot state such as joint and gripper positions, inform the planner and action model.
   * - Action
     - The planner calls the VLA or robot control tools, then uses new observations to choose the next action. Both templates execute up to 32 steps per VLA call; parameters can shorten this prefix.
   * - Reward / success
     - The Env service reports native termination/truncation and official QScore metrics from RLinf’s simulator adapter. A termination flag alone is not a task-success verdict. See “View Results” for the score file locations.
   * - Task prompt
     - By default, every VLA call uses the selected task's full instruction. The planner can choose different tools, but does not replace the VLA instruction with a newly generated subtask instruction.

Installation and Resources
--------------------------

Use Linux, Python 3.11, and an NVIDIA GPU supported by the
`Isaac Sim 5.1 requirements <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_.
Choose the Env API for the tools you need.

* For observations and native actions, the public
  `RLinf BehaviorEnv checkout <https://github.com/Forging-ahead/RLinf/tree/b4a7bfabaa54e9acde9aad86369eeb9ad0d6196a>`_
  exposes the official 61-value R1Pro state, three camera views, rewards and
  episode progress. Use its BEHAVIOR installation instructions and install RPent
  with the ``behavior`` extra in the Env interpreter. Start one local service::

     python -m robots.behavior.env_server --env-api behavior-env \
       --env-config /path/to/radio.yaml --task-id 0 --instance-id 301 \
       --camera-size 128 --output-dir /path/to/run --port 25000

  Env RPC returns RLinf's batched ``main_images``, ``wrist_images`` (left, right)
  and ``states`` (1 × 61). Pi0.5 receives these arrays and the configured
  instruction. Available metric depth stays with the tools.

  Base ``radio.yaml`` on RLinf's ``examples/embodiment/config/env/behavior_r1pro.yaml``.
  For the public test instance, set ``omni_config.task.activity_instance_dir`` to the
  directory containing the task's cached public instances and use the matching
  scene configuration. The service selects RLinf's 2026 state layout and runs a
  single Ray environment. Each acknowledged action corresponds to a returned
  frame. OmniGibson can report episode index zero for both reset and the first action.
* For the CuRobo, calibrated camera, legacy 256-value model input and official
  QScore examples below, use the in-process ``BehaviorSimulator`` interface,
  available in `this RLinf revision <https://github.com/Forging-ahead/RLinf/tree/1edcc27dc0dde23d520da1112545146b4902e846>`_.
  Select it with ``--env-api simulator``. When the flag is omitted, RPent uses
  this interface if the installed package exports it and otherwise selects
  ``BehaviorEnv``.

Use three separate Python environments. The planner and SAM3 can share one
interpreter while running as separate processes; Env and Pi0.5 use their own
interpreters. This keeps simulator, action-model, and segmentation dependencies
from replacing each other.

.. list-table::
   :header-rows: 1

   * - Interpreter
     - Purpose and installation profile
   * - ``ENV_PYTHON``
     - OmniGibson 3.9.3 / Isaac Sim 5.1.0; RLinf ``behavior`` env-only profile uses PyTorch 2.7.0 with CUDA 12.8 wheels.
   * - ``VLA_PYTHON``
     - Shared Pi0.5; RLinf's model-only ``--model openpi`` profile uses PyTorch 2.7.1 with CUDA 12.8 wheels and does not install the simulator SDK.
   * - ``PLANNER_PYTHON`` / ``SAM3_PYTHON``
     - RPent planner and optional SAM3 service; install separately from the Env and OpenPI environments.

Replace the paths below with local paths. Use new environment directories and
a fresh ``RPENT_RUN`` outside the source checkout. Load the variables in every
terminal, or save them in a local shell file and source it in each terminal.

.. code-block:: bash

   export RPENT_REPO=/path/to/RPent
   export RLINF_REPO=/path/to/RLinf
   export ENV_VENV=/path/to/envs/behavior-393
   export VLA_VENV=/path/to/envs/behavior-pi05
   export PLANNER_VENV=/path/to/envs/behavior-planner
   export ENV_PYTHON="$ENV_VENV/bin/python"
   export VLA_PYTHON="$VLA_VENV/bin/python"
   export PLANNER_PYTHON="$PLANNER_VENV/bin/python"
   export SAM3_PYTHON="$PLANNER_PYTHON"
   export RPENT_RUN=/path/to/behavior-runs/run-name
   export ENV_GPU=0
   export VLA_GPU=1
   export SAM3_GPU=2
   export OMNIGIBSON_DATA_PATH=/path/to/2026/datasets
   export OMNIGIBSON_DATASET_PATH="$OMNIGIBSON_DATA_PATH/behavior-1k-assets"
   export OMNIGIBSON_ASSET_PATH="$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets"
   export OMNIGIBSON_KEY_PATH="$OMNIGIBSON_DATA_PATH/omnigibson.key"
   export VLA_CHECKPOINT=/path/to/checkpoints/PT50-CS32
   export OPENPI_DATA_HOME=/path/to/cache/openpi
   export SAM3_CHECKPOINT_PATH=/path/to/checkpoints/sam3/sam3.pt
   export PLANNER_MODEL=provider:vision-model-name
   export TASK_ID=40

Install uv following :doc:`../get_started/quickstart`. From the matching RLinf
checkout, use the existing BEHAVIOR profile for the simulator and the model-only
OpenPI profile for Pi0.5. RPent launches these services with separate interpreters.

.. code-block:: bash

   cd "$RLINF_REPO"
   bash requirements/install.sh embodied --env behavior \
     --python 3.11 --venv "$ENV_VENV" --install-rlinf --no-flash-attn
   bash requirements/install.sh embodied --model openpi \
     --python 3.11 --venv "$VLA_VENV" --install-rlinf --no-flash-attn

The Env installer uses official BEHAVIOR ``v3.9.3-post2`` source revision
``a8247a8cc1633fe1ca0cc66aa07243d46c64f155`` and pins its LeRobot dependency to
``436812bd8ee39b768c645c248c91f1330834e687`` for Python 3.11 compatibility.
It applies this dependency pin in a separate SDK installation copy, leaving
the supplied SDK source checkout intact. Keep the installation copy after
setup. Use ``--no-root`` only when the required system libraries and
build tools are already installed. Model setup invokes RLinf's existing OpenPI
asset helper. The download below explicitly places the tokenizer in the cache
selected by ``OPENPI_DATA_HOME``; the runtime reads its
``big_vision/paligemma_tokenizer.model`` file.

Install RPent into all three environments. The ``behavior`` extra supplies
RPent's integration dependencies; ``behavior-vla`` uses the separately installed
RLinf/OpenPI runtime. Neither extra downloads simulator assets or model weights.

.. code-block:: bash

   uv venv --python 3.11 "$PLANNER_VENV"
   cd "$RPENT_REPO"
   uv pip install --python "$PLANNER_PYTHON" -e '.[behavior]'
   uv pip install --python "$ENV_PYTHON" \
     --override "$RLINF_REPO/requirements/embodied/envs/behavior-overrides.txt" \
     -e '.[behavior]'
   uv pip install --python "$VLA_PYTHON" -e '.[behavior-vla]'
   uv pip check --python "$PLANNER_PYTHON"
   "$ENV_PYTHON" -m pip check
   uv pip check --python "$VLA_PYTHON"

The Env install uses RLinf's BEHAVIOR dependency overrides. When reusing the
official SDK, the Env interpreter's pip check includes inherited packages. The
validated runtime reported four metadata conflicts involving packaging,
websockets, wrapt, and Pillow. See the BEHAVIOR installation guide in the
matching RLinf checkout for the constraints and verify the service starts.

Verify the adapter import before loading a scene:

.. code-block:: bash

   "$ENV_PYTHON" -c 'import inspect; from rlinf.envs.sim.behavior import BehaviorSimulator; print(inspect.getfile(BehaviorSimulator))'

The path should be inside ``RLINF_REPO``. Resolve an unexpected checkout or
an installation conflict before starting services. This checks import resolution,
not simulator assets or successful task execution.

Download the 2026 challenge assets and the selected task's public-test
instances using the
`official BEHAVIOR-1K instructions <https://github.com/StanfordVL/BEHAVIOR-1K>`_.
``OMNIGIBSON_DATA_PATH`` must contain ``omnigibson-robot-assets``,
``behavior-1k-assets``, and ``2026-challenge-task-instances``, with the dataset
key at ``OMNIGIBSON_KEY_PATH``. A package installation does not provide these
data. Motion tools use the loaded R1Pro robot's joint limits, USD model, and
CuRobo configuration from these assets.

Download the `public PT50-CS32 checkpoint
<https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_ using the
Hugging Face CLI installed in the model environment:

.. code-block:: bash

   "$VLA_VENV/bin/hf" download RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 \
     --local-dir "$VLA_CHECKPOINT"
   test -s "$VLA_CHECKPOINT/model.safetensors"
   test -s "$VLA_CHECKPOINT/config.json"
   test -s "$VLA_CHECKPOINT/assets/behavior-1k/2025-challenge-demos/norm_stats.json"
   "$VLA_VENV/bin/hf" download RLinf/openpi_tokenizer --local-dir "$OPENPI_DATA_HOME"
   test -s "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"

Keep the checkpoint's normalization statistics with its weights. Configure the
vision-language planner's model, service URL, and credentials following
:doc:`../guides/configure_planner`; choose this model independently of the VLA.
Before starting GPU services, check the chosen planner's authentication:

.. code-block:: bash

   "$PLANNER_VENV/bin/rpent-check-llm" --planner api --model "$PLANNER_MODEL" --json

This check does not exercise images or tools. For a run without prior task
experience, create an empty local memory directory; it need not contain a
``MEMORY.md`` file. The planner command below selects this read-only corpus.

.. code-block:: bash

   mkdir -p "$RPENT_RUN/memory-empty"
   cd "$RPENT_REPO"

The examples assign GPU 0 to Env, GPU 1 to Pi0.5, and GPU 2 to optional SAM3;
adjust the IDs to your allocated devices. Env uses ``OMNIGIBSON_GPU_ID`` and
removes ``CUDA_VISIBLE_DEVICES`` so CUDA masking does not change numbering
independently of Vulkan. Keep ``omni_config.env.device: null`` in the YAML;
model services use their own ``--cuda-device`` argument.

Run a Task
----------

First choose which tools the planner can call. Both templates let a
vision-language planner choose the next action, and both retain the VLA. The
difference is whether additional perception and robot control tools are
available. Therefore, ``vla_only.yaml`` is not the standalone VLA baseline in
the experiments below.

.. list-table::
   :header-rows: 1

   * - Configuration
     - Robot tools available to the planner
     - Additional dependencies
   * - ``vla_only.yaml``
     - View observations, call the VLA, and end the planner session.
     - No SAM3 or CuRobo dependency; suitable for a first run.
   * - ``agent.yaml``
     - Retains observation, VLA, and finish tools, and adds detailed robot state, segmentation, 3D localization, and gripper, arm, and base control: 12 robot tools in total.
     - SAM3 and CuRobo; 3D localization requires RGB-D observations.

The following example uses ``vla_only.yaml`` with three processes in separate
terminals. Its 320-step limit is a bounded integration check, not a complete
benchmark rollout. Match the task budget and protocol when comparing results.

**1. Start the shared Pi0.5 service.** It loads the checkpoint and matching
normalization statistics. Wait for the service to listen on port ``28011``.
The statistics path below belongs to the public PT50-CS32 checkpoint; select
the corresponding statistics when changing checkpoints.

.. code-block:: bash

   "$VLA_PYTHON" -m rpent.robots.components.pi05_vla_server \
     --embodiment behavior --model-path "$VLA_CHECKPOINT" \
     --cuda-device "$VLA_GPU" \
     --norm-stats-path "$VLA_CHECKPOINT/assets/behavior-1k/2025-challenge-demos/norm_stats.json" \
     --host 127.0.0.1 --port 28011 --transport socket

**2. Start the simulator Env service.** It composes the RLinf YAML, applies
the explicit overrides below, constructs one simulator, and resets it before
accepting actions on RPC port ``28010``. No separate evaluator process or
policy WebSocket is used. Allow scene initialization to finish before
starting the planner.

.. code-block:: bash

   env -u CUDA_VISIBLE_DEVICES OMNIGIBSON_GPU_ID="$ENV_GPU" \
     "$ENV_PYTHON" -m robots.behavior.env_server \
     --env-config "$RLINF_REPO/examples/embodiment/config/env/behavior_r1pro.yaml" \
     --task-id "$TASK_ID" --instance-split public_test --instance-id 301 \
     --max-steps 320 --seed 0 \
     --action-frequency 30 --rendering-frequency 30 --physics-frequency 120 \
     --camera-size 128 --env-wrapper rgbd \
     --host 127.0.0.1 --port 28010 --transport socket \
     --output-dir "$RPENT_RUN/env"

The bounded example explicitly uses 30 Hz control and rendering with 120 Hz
physics. Without these flags, the composed YAML retains its configured
frequencies. Keep frequencies consistent with the selected checkpoint and
evaluation protocol. ``--task-id`` resolves the official
task name and instruction for both Env and Planner.

**3. Start the planner once Env is ready.** The task ID must match the Env
service. To override the official instruction, pass the same
``--task-language`` to both commands. Point the action-model endpoint to the
service started in step 1.

.. code-block:: bash

   "$PLANNER_PYTHON" -m rpent.cli.main --robot behavior --planner api \
     --model "$PLANNER_MODEL" \
     --behavior-config robots/behavior/configs/vla_only.yaml \
     --task-id "$TASK_ID" \
     --memory-profile local --memory-dir "$RPENT_RUN/memory-empty" \
     --env-endpoint socket://127.0.0.1:28010 \
     --am-backend pi05 --am-endpoint socket://127.0.0.1:28011 \
     --output-dir "$RPENT_RUN/agent"

Exiting the planner closes its client connections and leaves the borrowed
services running. After the planner finishes, press Ctrl+C in the Env terminal
to collect its final result and stop it, then stop the Pi0.5 terminal and the
SAM3 terminal if you started it. Stop only services owned by this run. The
official SDK shutdown may terminate the dedicated Env process immediately;
Env writes its results before that call. Verify ``env-result.json`` reports
``server_status: stopped``, inspect its metrics, and check that all owned
processes have exited and released their GPUs. Use a new Env process and
output directory for the next episode.

Connecting a client does not reset Env. An explicit ``BehaviorEnvClient.reset()``
resets the simulator and that client's cached observation and feedback only.
Each Planner/Toolkit session belongs to one episode: after resetting Env,
create a new session because an existing Toolkit retains action and termination
state from the previous episode.

View Results
------------

A run produces two sets of results: the planner saves tool calls and
observations; Env saves its resolved configuration and current official
metrics on shutdown. While running, ``env-result.json`` records service readiness;
stop Env cleanly to collect its final metrics. Depth files are generated only
when observations provide depth.

.. code-block:: text

   $RPENT_RUN/
   ├── agent/
   │   ├── states.json
   │   ├── head_rgb.png/00.png
   │   ├── left_wrist_rgb.png/00.png
   │   ├── right_wrist_rgb.png/00.png
   │   ├── proprio.npy/00.npy
   │   └── head_depth.npy/00.npy
   └── env/
       ├── env_config.yaml
       └── env-result.json

``00`` is the initial observation, followed by ``01``, ``02``, and so on.
Names such as ``head_rgb.png`` are directories. Left- and right-wrist depth
are saved in the same way. This number indexes saved records; one tool call
can execute several simulation actions.

.. list-table:: Fields to Inspect
   :header-rows: 1

   * - File and fields
     - Meaning
   * - ``env-result.json``: ``server_status``, ``session``
     - Service state and the episode's native termination/truncation information. A normally stopped service records ``stopped``; ``error`` or ``cleanup_error`` describes failure. Check this file together with process exit, since SDK shutdown can exit with code 0 even after a recorded failure.
   * - ``env-result.json``: ``metrics[0]``
     - Official metric data in ``official_metrics``, cached ``q_score``, ``instance_id``, and ``episode_length``. Before any completed action, QScore is ``null``, not zero. A stopped service does not imply the task succeeded.
   * - ``env_config.yaml``
     - The effective task instance, seed, frequencies, step limit, and simulator configuration after applying CLI overrides.
   * - ``states.json``: ``steps[i].command``, ``steps[i].result``
     - Tool calls and execution feedback, including requested, issued, confirmed, and unconfirmed action counts. Unknown values remain ``null``.
   * - ``states.json``: ``steps[i].state.robot_state``, ``steps[i].state.artifacts.views``
     - Measured robot state and image paths for the record, useful for inspecting changes around an action.

The Env launcher does not automatically encode a video. Saved tool
observations are sparse records, not a frame for every simulation step. A
separate recording harness must capture intermediate frames when a full
trajectory video is needed. Assess performance using the official task
metrics, including QScore; neither ``terminated`` alone, planner ``finish``,
nor a clean process exit establishes a successful task.

Enable More Tools
-----------------

Before using ``agent.yaml``, install the motion dependencies in the Env
environment. The env-only installer above omits OmniGibson's optional
``primitives`` dependencies. Use its pinned public CuRobo revision and
``warp-lang==1.12.0``. Build against the Env interpreter's existing PyTorch
2.7.0 and CUDA 12.8 stack. The NVIDIA driver alone is insufficient: install
the CUDA Toolkit 12.8, including ``nvcc``, and the host C++ build tools.
Set ``CUDA_HOME`` to your Toolkit installation; ``nvcc --version`` should
report release 12.8.

.. code-block:: bash

   export CUDA_HOME=/usr/local/cuda-12.8
   export PATH="$CUDA_HOME/bin:$PATH"
   nvcc --version
   uv pip install --python "$ENV_PYTHON" setuptools-scm wheel ninja pybind11
   MAX_JOBS=2 uv pip install --python "$ENV_PYTHON" --no-build-isolation \
     'warp-lang==1.12.0' \
     'nvidia-curobo @ git+https://github.com/StanfordVL/curobo@78612f45cef52c3fa0298de243a54cd7ca614414'
   "$ENV_PYTHON" -m pip check
   "$ENV_PYTHON" -c 'import importlib.metadata as m; import curobo; print({p: m.version(p) for p in ("torch", "warp-lang", "nvidia-curobo")}); print(curobo.__file__)'

Keep build isolation disabled so the CUDA extensions compile against the
installed Torch rather than another build-time version. The final command
checks installed versions and import location; native motion behavior is
checked separately under “Development and Tests”.

For a new episode, follow the startup order above with three changes:

1. Add ``--motion-backend curobo --motion-device "cuda:$ENV_GPU"`` to the Env command.
2. Keep ``--env-wrapper rgbd`` on Env for RGB, depth, robot state, and live camera calibration used by 3D localization; ``rgb`` omits depth.
3. Prepare and start SAM3 as shown below. Change the planner configuration to ``robots/behavior/configs/agent.yaml`` and add ``--sam3-endpoint socket://127.0.0.1:28013``. Retain the task, Env/VLA endpoints, local-memory, and output options from the first-run command.

Use ``--action-frequency 30`` for the motion tools. Their measured-velocity
checks and joint interpolation currently use a 30 Hz control period; other
control rates require separate motion validation.

SAM3 runs in the planner environment, separate from Env and OpenPI. Install
the matching PyTorch 2.7.0 / torchvision 0.22.0 CUDA 12.8 wheels with RPent's
existing SAM3 extra:

.. code-block:: bash

   cd "$RPENT_REPO"
   uv pip install --python "$SAM3_PYTHON" --torch-backend cu128 \
     'torch==2.7.0' 'torchvision==0.22.0' -e '.[behavior,sam3]'
   uv pip check --python "$SAM3_PYTHON"

Request access on the `facebook/sam3 model page
<https://huggingface.co/facebook/sam3>`_, then download ``sam3.pt``. The service
loads this local file; setting its path does not download it.

.. code-block:: bash

   "$PLANNER_VENV/bin/hf" auth login
   "$PLANNER_VENV/bin/hf" download facebook/sam3 sam3.pt \
     --local-dir "$(dirname "$SAM3_CHECKPOINT_PATH")"
   test -s "$SAM3_CHECKPOINT_PATH"
   "$SAM3_PYTHON" -m rpent.robots.components.sam3_server \
     --cuda-device "$SAM3_GPU" \
     --host 127.0.0.1 --port 28013 --transport socket

Wait for SAM3 to finish loading and report its RPC service ready before starting
the planner. Endpoint readiness does not establish that segmentation will find
the requested object; inspect the returned score and ``found`` status. SAM3's
request timeout is 120 seconds, independent of the Env timeout. Planner exit
leaves this borrowed service running; stop it along with the other services
owned by the run.

RGB images suffice for the VLA and 2D segmentation. For 3D
coordinates, RGB, depth, and camera calibration must come from the same frame.

``agent.yaml`` exposes the following 12 robot tools, alongside RPent's shared
file and memory tools.

.. list-table:: Tool Selection
   :header-rows: 1

   * - Tool
     - Purpose and main limitations
   * - ``view_env_state``, ``read_robot_state``
     - Inspect images and measured robot state without advancing physics. The SDK may render while initializing camera calibration.
   * - ``segment``, ``pixel_to_robot_base``
     - Segment a target and find a pixel's coordinates in the robot base frame. Pixels use original-image ``[row, column]`` order. Localize again after the robot or target moves.
   * - ``vla_act``
     - Predict and execute an action prefix, obtaining a new observation after each chunk.
   * - ``set_gripper``
     - ``val=0`` closes the gripper; ``val=1`` opens it. Interpolation and final holds total at most 5000 steps. Add ``release`` to the allowlist for a gripper-opening shortcut.
   * - ``move_to``, ``move_eef_delta``
     - Move one arm in the current robot base frame, with positions in meters. The former specifies a target; the latter requests a small displacement. Both require CuRobo.
   * - ``rotate_wrist``
     - Use CuRobo to rotate the end effector about the base Z axis, from −180° to 180°, while keeping its target position. Fingers still sweep through surrounding space during motion.
   * - ``rotate_wrist_joint``
     - Turn a selected roll, pitch, or yaw joint from −90° to 90° without inverse kinematics. Inspect the measured rotation and joint-limit clipping feedback.
   * - ``move_base``
     - Execute a local base-velocity command followed by zero velocity; arrival at a specific position is not guaranteed.
   * - ``finish``
     - Stop the planner's decision loop and record a reason; it does not shut down the borrowed Env process or mark the native task successful.

The Env service uses native termination and the configured step limit to stop
actions and retain final metrics. ``finish`` stops only the RPent planner. If
called before the environment ends, the saved run remains partial; explicitly
stop the borrowed Env service after inspecting its result.

Arm planning starts from a stable measured pose and uses OmniGibson's
``CuRoboMotionGenerator`` with the loaded R1Pro model. It optimizes only the selected arm's seven joints, locking the
base, trunk, other arm, and fingers to their measured positions for IK;
execution preserves their hold commands. The IK solver checks self-collision.
It solves a local target and interpolates a joint trajectory, without scene
collision checking or a collision guarantee along that path. Use images to
assess available space. Judge tool completion using action confirmations and
measured pose error; gripper closure alone does not establish a successful grasp.

When no ``quat`` is specified, ``move_to`` and ``move_eef_delta`` may retry
with position constraints alone if they cannot maintain the current
orientation, reporting ``orientation_relaxed``. ``move_to`` with an explicit
``quat`` and ``rotate_wrist`` retain their orientation constraints. A failed
IK attempt does not issue the planned arm trajectory and may report the
nearest reachable position to help adjust the target.

Common Options
--------------

CLI arguments override the corresponding settings in ``--behavior-config``.
See :doc:`../guides/cli` for shared planner options. Common BEHAVIOR settings
are listed below.

.. list-table::
   :header-rows: 1

   * - Setting
     - Purpose and defaults
   * - ``--behavior-config``, ``--tool-allowlist``
     - Select a template or override the robot tool list. The list must retain ``view_env_state`` and ``finish``. Shared file and memory tools remain available.
   * - ``--control-mode``, ``--instruction-mode``
     - This guide uses ``agent`` control, where the planner selects tools. ``--instruction-mode main`` uses the full task instruction on every VLA call and checks it when connecting to the Env. ``override`` allows a different instruction for an individual call if the selected model supports it.
   * - ``--execution-horizon``, ``--max-chunks-per-call``
     - Both templates execute at most 32 steps per chunk and predict at most one chunk per tool call. Reduce the horizon to shorten the executed prefix.
   * - ``--env-endpoint``, ``--am-endpoint``, ``--sam3-endpoint``
     - Accept internal ``socket://`` or ``http://`` service addresses. Select ``--am-backend pi05`` for the action model; ``--vla-endpoint`` remains an alias for its endpoint.
   * - Pi0.5 ``--model-path``, ``--norm-stats-path``, ``--repo-id``
     - Select the checkpoint, matching normalization statistics, and dataset identifier for RLinf's data configuration. If no model path is supplied, read ``PI05_CHECKPOINT_PATH``. ``--cuda-device`` selects the model GPU.
   * - Env ``--env-config``, ``--instance-split``, ``--instance-id``
     - Compose the RLinf simulator YAML and optionally override its instance selection. IDs are actual asset IDs: public-test instances start at 301.
   * - Env ``--max-steps``, ``--seed``, frequency flags
     - Override the episode limit, seed, and action/rendering/physics frequencies; omitted values retain the composed YAML settings.
   * - Env ``--camera-size``, ``--env-wrapper``, ``--torch-num-threads``
     - Optionally set square camera resolution. The default ``rgbd`` profile provides RGB, robot state, and linear depth; ``rgb`` omits depth, while ``default`` retains the RLinf base observation profile. The default 4 PyTorch CPU threads do not control all simulator or PhysX threads.
   * - Planner ``--env-rpc-timeout-s``
     - Timeout for a complete Env request; both tool templates set 1230 seconds.
   * - Env ``--motion-backend``, ``--motion-device``
     - The backend defaults to ``none``. With ``curobo``, the default device is ``cuda:0``.

Task Memory
-----------

BEHAVIOR defaults to ``--memory-profile local``. ``--memory-dir`` selects a local experience corpus, defaulting to
``memory/behavior/`` under the repository root. The planner inspects the initial
observation before reading ``MEMORY.md`` and up to three relevant entries.
Historical object positions and grasp outcomes must be confirmed from current
observations. During evaluation, the corpus is read-only and is not synchronized
from a remote source. See :doc:`../development/memory` for the index and the
``global/`` and ``task-family/`` layout.

An empty directory is valid for a run without prior experience, as in the
first-run command. Leave it empty instead of creating a synthetic strategy;
no separate read-only flag or operating-system permission change is required.
The read-only restriction applies to the memory tools, while run artifacts
are written to ``--output-dir``.

Experiment Results
------------------

The comparison below uses the same VLA trained by our team. The baseline
executes tasks directly with that VLA; RPent uses its planner to call the same
VLA and tools. Each task and method is evaluated on 10 instances, and the table
reports mean QScore from the official evaluator. These reported results were
collected with the earlier evaluator-based integration; they are not a new
evaluation of this RLinf adapter or the public checkpoint used above.

.. list-table::
   :header-rows: 1

   * - Task
     - VLA only (baseline)
     - RPent + the Same VLA
   * - Turn on the radio
     - 0.20
     - 0.80
   * - Make microwave popcorn
     - 0.70
     - 0.90
   * - Pick up trash
     - 0.33
     - 0.73

With the same VLA trained by our team, RPent achieves higher mean QScores
than the standalone VLA baseline on all three evaluated tasks.

Development and Tests
---------------------

Configuration and registration start in ``robots/behavior/robot_spec.py``.
``env_server.py`` loads the RLinf configuration and owns the simulator process;
``rlinf_env.py`` adapts its local API to the existing RPent Env contract. Tool
implementations remain in ``primitives.py`` and ``tools.py`` in the same directory. The action model reuses
``rpent/robots/components/pi05_vla_client.py`` and ``pi05_vla_server.py``.

``R1ProMotionModel`` in ``motion.py`` receives ``simulator.get_robot``
from the Env process. It reads the native joint limits
and ``robot.curobo_path`` configuration, using the loaded USD robot for IK
instead of maintaining a separate kinematic model.

The native Env validation configuration uses 128×128 cameras. Other
resolutions require separate runtime checks.

Use a separate Linux CPU test environment, keeping the SAM3 CUDA dependencies
intact. Run the BEHAVIOR and shared Pi0.5 offline tests:

.. code-block:: bash

   uv venv --python 3.11 /path/to/envs/behavior-tests
   export TEST_PYTHON=/path/to/envs/behavior-tests/bin/python
   uv pip install --python "$TEST_PYTHON" --torch-backend cpu -e '.[test,behavior]' torch
   "$TEST_PYTHON" -m pytest tests/unit_tests/robots/behavior \
     tests/unit_tests/rpent/robots/components/test_pi05_vla*.py -q

Real-service checks use the same test interpreter and borrow running GPU
services. For a dedicated episode, complete
startup steps 1–2 and wait for Env to be ready, then replace step 3 with
the commands below. No planner model or credentials are needed. Run tests
serially, with no other action clients or ``pytest-xdist`` running concurrently.

.. code-block:: bash

   export RPENT_RUN_BEHAVIOR_INTEGRATION=1
   export RPENT_BEHAVIOR_TASK_ID="$TASK_ID"
   export RPENT_BEHAVIOR_TASK_LANGUAGE="$("$TEST_PYTHON" -c "from robots.behavior.tasks import TASK_LANGUAGE; print(TASK_LANGUAGE[$TASK_ID])")"
   export RPENT_BEHAVIOR_ENV_ENDPOINT=socket://127.0.0.1:28010
   export RPENT_BEHAVIOR_VLA_ENDPOINT=socket://127.0.0.1:28011
   export RPENT_E2E_OUTPUT_DIR="$RPENT_RUN/integration"
   "$TEST_PYTHON" -m pytest tests/e2e_tests/behavior -v -s

Without ``RPENT_RUN_BEHAVIOR_INTEGRATION``, the tests skip without connecting
to services. The environment and VLA checks in ``test_components.py`` do not
issue actions. The optional native-motion check requires an Env started
with ``--motion-backend curobo`` and
``RPENT_RUN_BEHAVIOR_MOTION_INTEGRATION=1``. It requests IK for both arms'
current end-effector poses through Env RPC after executing a bounded arm-hold
motion and 13 explicit settling actions. It checks native joint limits and
observation identity, and saves ``motion-component.json``; the returned IK
paths are not executed. ``test_policy_chain.py`` executes and confirms at most
32 steps through the Toolkit, saves new observations, and calls ``finish(blocked)``.
Here, ``blocked`` marks the end of a bounded check, not a failed task.

Inspect predictions, submitted actions, confirmation counts, and saved images
in ``RPENT_E2E_OUTPUT_DIR``. These tests verify components and the action chain,
not autonomous planner decisions or complete task performance. They do not
start, reset, or stop services and are not launched through
``run_gpu_suite.sh``. Stop the borrowed services after inspecting their saved results.
