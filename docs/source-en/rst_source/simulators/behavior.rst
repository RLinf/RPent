BEHAVIOR
========

Use RPent's vision-language planner to perform household tasks in the
BEHAVIOR-1K simulator. The planner uses camera observations to call shared
Pi0.5 and robot action tools. The simulated R1Pro dual-arm mobile robot executes
the actions. The official evaluator advances the simulation, ends episodes,
and scores task outcomes.

This guide selects “make microwave popcorn” (task ID 40) from the 100
BEHAVIOR-1K challenge tasks to show how to prepare the environment, run a task,
and find its video and scores.

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

Each Env bridge connects to one evaluator and serves one episode.

Tasks
~~~~~

Task IDs, official names, and full instructions are listed in
``robots/behavior/tasks.py``. All three must identify the same task.

.. list-table::
   :header-rows: 1

   * - Scope
     - Selection
   * - 100 challenge tasks
     - IDs ``0..99``. When given only ``--task-id``, the planner uses the official task instruction. The Env startup command still requires that instruction explicitly.
   * - Public-test example in this guide
     - ID ``40``, name ``make_microwave_popcorn``; index ``0`` in ``public_test`` selects instance ``301``. Use a single environment and retain the official step budget.

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
     - The official evaluator reports success and QScore from the simulated task state. See “View Results” for the score file locations.
   * - Task prompt
     - By default, every VLA call uses the selected task's full instruction. The planner can choose different tools, but does not replace the VLA instruction with a newly generated subtask instruction.

Installation and Resources
--------------------------

Install the simulator and evaluator following the
`official BEHAVIOR-1K instructions <https://github.com/StanfordVL/BEHAVIOR-1K>`_,
and prepare the challenge assets and public-test instances for the selected
task. RPent supports Python 3.10–3.12. Use separate environments for the
simulator, Pi0.5, and planner to satisfy their respective Python, CUDA, and
graphics requirements. Without CuRobo, the Env bridge can use the planner
environment.

Before using CuRobo arm planning or ``rotate_wrist_joint``, point the official
environment variable to your existing datasets directory in the terminal that
starts the Env service. Keep its current value if it is already configured:

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH="/path/to/your/existing/datasets"

Use an absolute path containing ``omnigibson-robot-assets/models/r1pro/``.
RPent reads the robot structure, joint limits, and collision configuration from
these assets. The motion adapter supports robot asset version ``3.8.2`` (see
``omnigibson-robot-assets/VERSION``). The Env service and official evaluator can
share this installation; mount it where Env can read it when using separate
containers. The planner queries joint limits through Env and needs neither a
simulator installation nor local robot assets. With ``vla_only.yaml``, RPent Env
does not load motion assets; the official evaluator still requires its own
simulation resources.

The model environment needs RLinf/OpenPI with ``pi05_behavior`` support,
normalization statistics matching the checkpoint, and a tokenizer. This guide
uses the `public PT50-CS32 checkpoint <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_.
Set ``OPENPI_DATA_HOME`` to the local tokenizer resources as required by your
RLinf version. Configure the vision-language planner's model, service URL,
and credentials following :doc:`../guides/configure_planner`. Select the
planner model independently of the VLA checkpoint.

Replace the placeholder paths with local paths and use a fresh ``RPENT_RUN``
directory outside the source checkout. Load these variables and enter the
checkout in every terminal, or save them in a local shell file and source it
in each terminal.

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

Install RPent in the planner, Env, and model environments. The simulator keeps
its own interpreter; ``.[behavior-vla]`` does not replace the RLinf/OpenPI
environment described above.

.. code-block:: bash

   "$PLANNER_PYTHON" -m pip install -e '.[behavior]'
   "$ENV_PYTHON" -m pip install -e '.[behavior]'
   "$VLA_PYTHON" -m pip install -e '.[behavior-vla]'

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

The following example uses ``vla_only.yaml`` with four processes in separate
terminals.

**1. Start the shared Pi0.5 service.** It loads the checkpoint and matching
normalization statistics. Wait for the service to listen on port ``28011``.
The statistics path below belongs to the public PT50-CS32 checkpoint; select
the corresponding statistics when changing checkpoints.

.. code-block:: bash

   "$VLA_PYTHON" -m rpent.robots.components.pi05_vla_server \
     --embodiment behavior --model-path "$VLA_CHECKPOINT" \
     --norm-stats-path "$VLA_CHECKPOINT/assets/behavior-1k/2025-challenge-demos/norm_stats.json" \
     --host 127.0.0.1 --port 28011 --transport socket

**2. Start the Env bridge.** Port ``28010`` receives RPent's internal RPC
requests; port ``28012`` waits for the official evaluator's WebSocket
connection.

.. code-block:: bash

   "$ENV_PYTHON" -m robots.behavior.env_server \
     --task-id "$TASK_ID" --task-language "$TASK_LANGUAGE" \
     --host 127.0.0.1 --port 28010 --transport socket \
     --policy-host 127.0.0.1 --policy-port 28012

**3. Start the official evaluator.** It loads the scene, connects to ``28012``,
and waits for actions after sending the first observation. Keep
``--num-envs 1`` and exchange one observation and action per simulation step;
the client must not replay cached action chunks.

.. code-block:: bash

   "$SIM_PYTHON" -m omnigibson.eval.eval \
     --task-name "$TASK_NAME" --mode public_test --instance-indices 0 \
     --num-rollouts 1 --num-envs 1 --host 127.0.0.1 --port 28012 \
     --write-video --video-fps 120 --output-dir "$RPENT_RUN/official"

**4. Start the planner once the first observation is available.** The task ID
and instruction must match the Env bridge. Point the action-model endpoint to
the service started in step 1.

.. code-block:: bash

   "$PLANNER_PYTHON" -m rpent.cli.main --robot behavior --planner api \
     --model "$PLANNER_MODEL" \
     --behavior-config robots/behavior/configs/vla_only.yaml \
     --task-id "$TASK_ID" --task-language "$TASK_LANGUAGE" \
     --env-endpoint socket://127.0.0.1:28010 \
     --am-backend pi05 --am-endpoint socket://127.0.0.1:28011 \
     --output-dir "$RPENT_RUN/agent"

Exiting the planner only closes its client connections. Stop the independently
started services manually after the run. Use a new Env service and output
directory for the next episode.

View Results
------------

A run produces two sets of results: RPent saves tool calls and observations;
the official evaluator saves scores and video. The key files are shown below.
Depth files are generated only when observations provide depth.

.. code-block:: text

   $RPENT_RUN/
   ├── agent/
   │   ├── states.json
   │   ├── head_rgb.png/00.png
   │   ├── left_wrist_rgb.png/00.png
   │   ├── right_wrist_rgb.png/00.png
   │   ├── proprio.npy/00.npy
   │   └── head_depth.npy/00.npy
   └── official/
       ├── json/<task_name>_<instance_id>_<rollout_id>.json
       └── videos/<task_name>_<instance_id>_<rollout_id>.mp4

``00`` is the initial observation, followed by ``01``, ``02``, and so on.
Names such as ``head_rgb.png`` are directories. Left- and right-wrist depth
are saved in the same way. This number indexes saved records; one tool call
can execute several simulation actions.

.. list-table:: Fields to Inspect
   :header-rows: 1

   * - File and fields
     - Meaning
   * - Official JSON: ``success``, ``q_score.final``
     - Whether the task completed and its final QScore. Written when the evaluator completes a rollout; an interrupted run may not have complete scores.
   * - Official JSON: ``instance_id``, ``rollout_id``, ``steps``
     - Scene instance, repetition index, and actual evaluation step count.
   * - ``states.json``: ``steps[i].command``, ``steps[i].result``
     - Tool calls and execution feedback, including requested, issued, confirmed, and unconfirmed action counts. Unknown values remain ``null``.
   * - ``states.json``: ``steps[i].state.robot_state``, ``steps[i].state.artifacts.views``
     - Measured robot state and image paths for the record, useful for inspecting changes around an action.

Enable video with ``--write-video``. The example's ``--video-fps 120`` sets
encoded playback speed without changing the simulation control frequency.
Use the official scores and video to assess task performance; saved tool
calls alone do not establish task success.

Enable More Tools
-----------------

Before using ``agent.yaml``, prepare CuRobo v0.7.8 and compatible PyTorch,
Warp, and CUDA in the Env environment, and SAM3 weights in a separate
environment. For a new episode, follow the startup order above with three
changes:

1. Add ``--motion-backend curobo`` to the Env command.
2. Add ``--env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper`` to the official evaluator command to supply depth and calibration for 3D localization.
3. Start the SAM3 service below, then change the planner configuration to ``robots/behavior/configs/agent.yaml`` and add ``--sam3-endpoint socket://127.0.0.1:28013``.

.. code-block:: bash

   export SAM3_PYTHON=/path/to/sam3-env/bin/python
   export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
   "$SAM3_PYTHON" -m pip install -e '.[sam3]'
   "$SAM3_PYTHON" -m rpent.robots.components.sam3_server \
     --host 127.0.0.1 --port 28013 --transport socket

Default RGB observations suffice for the VLA and 2D segmentation. For 3D
coordinates, RGB, depth, and camera calibration must come from the same frame.

``agent.yaml`` exposes the following 12 robot tools, alongside RPent's shared
file and memory tools.

.. list-table:: Tool Selection
   :header-rows: 1

   * - Tool
     - Purpose and main limitations
   * - ``view_env_state``, ``read_robot_state``
     - Inspect saved images and robot state without refreshing cameras or advancing simulation.
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
     - Stop the planner's decision loop and record a reason; it does not end the official evaluation episode.

The official evaluator ends an episode and saves its scores based on
termination or truncation signals from the environment. ``finish`` only stops
the RPent planner: it neither tells the evaluator to end the episode nor
closes the simulator. If called early, the evaluator may still be waiting
for actions. This is not a normally completed evaluation; remaining services
require manual cleanup.

Arm planning starts from a stable pose and holds the trunk, other arm, and
gripper commands. It solves a local target and interpolates a joint trajectory
without checking scene collisions along the entire path. Use images to assess
available space. Judge tool completion using action confirmations and measured
pose error; gripper closure alone does not establish a successful grasp.

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
   * - Env ``--simulation-timeout-s``, ``--action-timeout-s``, ``--chunk-timeout-s``
     - Default deadlines are 600 seconds waiting for observations, 180 seconds waiting for actions, and 1200 seconds for a complete chunk. Both planner templates set ``--env-rpc-timeout-s`` to 1230 seconds.
   * - Env ``--motion-backend``, ``--motion-device``
     - The backend defaults to ``none``. With ``curobo``, the default device is ``cuda:0``.

Task Memory
-----------

``--memory-dir`` selects a local experience corpus, defaulting to
``memory/behavior/`` under the repository root. The planner inspects the initial
observation before reading ``MEMORY.md`` and up to three relevant entries.
Historical object positions and grasp outcomes must be confirmed from current
observations. During evaluation, the corpus is read-only and is not synchronized
from a remote source. See :doc:`../development/memory` for the index and the
``global/`` and ``task-family/`` layout.

Experiment Results
------------------

The comparison below uses the same VLA trained by our team. The baseline
executes tasks directly with that VLA; RPent uses its planner to call the same
VLA and tools. Each task and method is evaluated on 10 instances, and the table
reports mean QScore from the official evaluator.

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
Tool implementations are in ``primitives.py`` and ``tools.py`` in the same
directory. The action model reuses
``rpent/robots/components/pi05_vla_client.py`` and ``pi05_vla_server.py``.

Install test dependencies in a Linux CPU environment and run the BEHAVIOR and
shared Pi0.5 offline tests:

.. code-block:: bash

   "$PLANNER_PYTHON" -m pip install -e '.[test,behavior]'
   "$PLANNER_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
   "$PLANNER_PYTHON" -m pytest tests/unit_tests/robots/behavior \
     tests/unit_tests/rpent/robots/components/test_pi05_vla*.py -q

Real-service checks borrow running services. For a dedicated episode, complete
startup steps 1–3 and wait for the first observation, then replace step 4 with
the commands below. No planner model or credentials are needed. Run tests
serially, with no other action clients or ``pytest-xdist`` running concurrently.

.. code-block:: bash

   export RPENT_RUN_BEHAVIOR_INTEGRATION=1
   export RPENT_BEHAVIOR_TASK_ID="$TASK_ID"
   export RPENT_BEHAVIOR_TASK_LANGUAGE="$TASK_LANGUAGE"
   export RPENT_BEHAVIOR_ENV_ENDPOINT=socket://127.0.0.1:28010
   export RPENT_BEHAVIOR_VLA_ENDPOINT=socket://127.0.0.1:28011
   export RPENT_E2E_OUTPUT_DIR="$RPENT_RUN/integration"
   "$PLANNER_PYTHON" -m pytest tests/e2e_tests/behavior -v -s

Without the opt-in flag, the three tests skip without connecting to services.
``test_components.py`` checks observations and inference without issuing
actions. ``test_policy_chain.py`` executes and confirms at most 32 steps
through the Toolkit, saves new observations, and calls ``finish(blocked)``.
Here, ``blocked`` marks the end of a bounded check, not a failed task.

Inspect predictions, submitted actions, confirmation counts, and saved images
in ``RPENT_E2E_OUTPUT_DIR``. These tests verify components and the action chain,
not autonomous planner decisions or complete task performance. They do not
start, reset, or stop services and are not launched through
``run_gpu_suite.sh``. Manually clean up services still waiting for actions
after the check.
