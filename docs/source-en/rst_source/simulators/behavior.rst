BEHAVIOR
========

Run an official BEHAVIOR-1K episode with RPent. The Planner selects tools;
the official evaluator executes native robot actions and owns simulation,
termination and scoring. The
`source guide <https://github.com/RLinf/RPent/blob/main/robots/behavior/guides/README.md>`_
contains the complete installation and startup commands.

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
     - Head, left-wrist and right-wrist RGB in that order, with public robot state. Geometric tools require matching depth and calibration.
   * - Action
     - The Planner calls VLA or motion tools. The environment receives 23-channel native actions. The RLinf Pi0.5 preset uses ``abs_joint``, BF16, five denoising steps and a 32-action chunk.
   * - Reward / success
     - Use the official evaluator JSON success and QScore. Tool completion and Planner ``finish`` are not official task scores.
   * - Task prompt
     - Supply the full main-task instruction consistently to the VLA, Env service and Planner. The main-task VLA service rejects a different instruction.

Installation and Resources
--------------------------

Install the simulator, evaluator and challenge assets using the official
`BEHAVIOR-1K instructions <https://github.com/StanfordVL/BEHAVIOR-1K>`_. Keep the
simulator, model and Planner environments separate. RPent supports Python
3.10–3.12; the simulator and model environments must also satisfy their own
Python, CUDA and graphics requirements.

In the source checkout, install the lightweight Planner/Env dependencies:

.. code-block:: bash

   python -m pip install -e '.[behavior]'

The VLA environment needs ``.[behavior-vla]`` and a separately installed
BEHAVIOR-capable RLinf build. It must provide
``rlinf.models.embodiment.openpi_pytorch.get_model``, ``pi05_behavior`` and
``predict_action_batch(..., mode="eval")``. Prepare a compatible checkpoint,
normalization statistics and tokenizer assets; set ``OPENPI_DATA_HOME`` if
the RLinf build requires it. Installing the RPent extra does not install a
particular usable RLinf fork or download these model resources.

The complete Agent additionally requires the shared SAM3 service, its checkpoint
and ``.[sam3]`` environment, plus CuRobo v0.7.8 with compatible PyTorch/Warp/CUDA
for ``move_to``. Start with ``vla_only.yaml`` if these optional tools are not
needed. Configure an image-capable Planner through
:doc:`../guides/configure_planner`; its model is separate from the VLA checkpoint.

Run a Task
----------

Follow the source guide's complete example for task ``40``. Set installation
paths, a fresh output directory outside the checkout and the same task variables
in each terminal. Start the services in this order:

1. **VLA:** ``python -m robots.behavior.vla_server`` loads the checkpoint and
   serves predictions on internal RPC port ``28011``. It never advances simulation.
2. **Env bridge:** ``python -m robots.behavior.env_server`` connects to the ready
   VLA, exposes internal Env RPC on ``28010`` and the official policy WebSocket
   on ``28012``.
3. **Official evaluator:** ``python -m omnigibson.eval.eval`` connects to policy
   port ``28012`` and runs one public-test episode. Use ``--num-envs 1`` and
   ``--write-video --video-fps 120``; retain the official step budget.
4. **Planner:** after the first observation arrives, run
   ``python -m rpent.cli.main --robot behavior`` with the selected Planner model,
   task ID/text, service endpoints and
   ``--behavior-config robots/behavior/configs/vla_only.yaml``.

Each command uses its own prepared Python environment and terminal. Full flags
and variable definitions are in the
`one-episode example <https://github.com/RLinf/RPent/blob/main/robots/behavior/guides/README.md#run-one-episode>`_.
``vla_only.yaml`` still uses a VLM to choose observation, VLA and finish tools;
it is not a fixed-loop pure-VLA benchmark baseline.

To use ``agent.yaml``, also start SAM3, enable ``--motion-backend curobo`` on
the Env service and pass ``--sam3-endpoint`` to the Planner. ``agent.yaml``
enables twelve robot tools; RPent's shared file/memory tools remain.
Keep credentials and provider URLs in Planner configuration, not robot templates.

View Results
------------

The example separates outputs into three directories under the external run root:

- ``agent/``: Planner transcript, saved observations and tool results.
- ``vla/``: model settings, inference timings and predicted action chunks.
- ``official/``: evaluator JSON scores and videos; task success comes from this JSON.

Video encoding is set to 120 FPS in the example. Playback rate does not set
simulation timestep or action frequency. A clean service exit or a successful
tool call does not establish task success.

Task Memory
-----------

BEHAVIOR reads a local memory corpus through RPent's shared file tools, as other
simulators do. Pass ``--memory-dir`` to the Planner to select it; without it the
default ``memory/behavior/`` directory is used. The Planner reads
``MEMORY.md`` after the initial observation and up to three relevant entries
before acting. The corpus uses the layout in :doc:`../development/memory`;
BEHAVIOR evaluation reads it only and does not synchronize a remote corpus.

Runtime Contracts and Troubleshooting
-------------------------------------

RPent borrows the services: closing the Planner closes client connections, not
the service processes. Stop them explicitly. A new episode requires a fresh
Env service and output directory; Dashboard startup and multiple evaluator
sessions are not supported by this integration.

Saved observation ``step`` values are record indices, not action counts.
Read-only perception tools neither refresh cameras nor step simulation.
Geometric points belong to their saved frame; re-localize after motion.
``move_to`` covers short single-arm adjustments solved by IK, not general grasp
planning or path-level collision avoidance. ``move_base`` does not provide
reliable odometry or an arrival guarantee.

Keep issued, confirmed and unconfirmed action counts separate. Unknown execution
remains unknown; a timed-out chunk is not automatically replayed. For connection
failures, inspect the service logs. The ``ws://`` official policy endpoint is not
an ``--env-endpoint`` RPC address.

Development and Tests
---------------------

Robot action tools live in ``primitives.py``. Observation, perception, geometry
and session tools live in ``tools.py``. ``toolkit.py`` registers both groups and
binds their runtime resources. ``motion.py`` supplies the environment-side CuRobo
planner and robot model; it does not register tools.

The `source guide's code map <https://github.com/RLinf/RPent/blob/main/robots/behavior/guides/README.md#code-map-and-development>`_
locates configuration, prompts, tools, motion and service adapters. Edit general
decision guidance in ``prompts/evaluate.py`` and tool parameter/result descriptions
at their typed ``@tool`` definitions. Keep task goals dynamic.

Use the `test guide <https://github.com/RLinf/RPent/blob/main/tests/README.md>`_
for offline `BEHAVIOR unit tests <https://github.com/RLinf/RPent/tree/main/tests/unit_tests/robots/behavior>`_
and explicit `real-service checks <https://github.com/RLinf/RPent/tree/main/tests/e2e_tests/behavior>`_.
Integration checks establish interface behavior; task performance requires
official evaluation under recorded, comparable conditions.
