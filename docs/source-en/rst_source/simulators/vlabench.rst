VLABench
============

Run language-conditioned tabletop tasks with RPent in `VLABench <https://github.com/OpenMOSS/VLABench>`_. A planner controls a Franka arm through pose and gripper tools and localizes targets from calibrated RGB-D cameras; no action model is used.

.. _vlabench-overview:

Overview
------------

Check the runtime requirements before following the installation and run steps.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      None. The planner issues ``move_pose``, ``move_delta`` and ``set_gripper``.

   .. grid-item-card:: Planners

      See :doc:`../guides/configure_planner`.

   .. grid-item-card:: Tasks

      Tasks listed in a VLABench evaluation track.

   .. grid-item-card:: Hardware

      Linux; Python 3.10; EGL rendering.

Tasks
~~~~~~~~~~~~

A run selects one episode by task name, track and episode index. The track file ``configs/evaluation/tracks/<track>.json`` in the VLABench checkout lists each task's episodes. ``--track`` defaults to ``track_1_in_distribution``; add ``--eval-unseen`` for the unseen-object split.

.. _vlabench-observation-action:

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The table distinguishes planner tools, observations, and the environment's success criterion.

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - RGB, metric depth and calibration of the ``agentview``, ``left``, ``right`` and ``wrist`` cameras (512 x 512), the Franka joint and finger state, and the task instruction.
   * - Action
     - ``move_pose`` moves the end effector to an absolute world pose and ``move_delta`` moves relative to the measured pose, both through bounded Franka IK. ``set_gripper`` drives the fingers. ``back_project`` reads world XYZ at ``[row, col]`` pixels.
   * - Reward / success
     - Evaluate success using the native task success flag. Planner completion and the step budget are separate outcomes.
   * - Task prompt
     - The instruction comes from the environment after reset.

Installation
------------

VLABench requires Python 3.10. Create a virtual environment from the RPent repository root, install RPent in editable mode, and then install VLABench into the same environment (or a dedicated Python 3.10 environment passed via ``--vlabench-python``) by following its upstream instructions:

.. code-block:: bash

   uv venv --python 3.10 .venv-vlabench
   source .venv-vlabench/bin/activate
   uv pip install -e .

RPent does not bundle VLABench or its scene assets; the environment server imports VLABench from the checkout you point it to.

Prepare Assets
--------------

Keep VLABench's large ``obj/`` and ``scenes/`` assets in a directory outside the checkout and link them into its ``assets/`` directory:

.. code-block:: bash

   export VLABENCH_ROOT=/path/to/VLABench/VLABench
   export VLABENCH_ASSET_ROOT=/path/to/vlabench-assets
   ln -s "$VLABENCH_ASSET_ROOT/obj" "$VLABENCH_ROOT/assets/obj"
   ln -s "$VLABENCH_ASSET_ROOT/scenes" "$VLABENCH_ROOT/assets/scenes"

``VLABENCH_ROOT`` is the inner package directory that contains ``configs/``. The server refuses to start when either link is missing or points elsewhere.

Run a Task
----------

Run one episode from the activated environment:

.. code-block:: bash

   rpent --robot vlabench \
      --task-name select_painting \
      --track track_1_in_distribution \
      --episode-index 0 \
      --seed 42 \
      --planner codex \
      --model <planner-model>

RPent starts the VLABench environment server, waits for it, and stops it when the run ends. See ``rpent --robot vlabench --help`` for the complete option list.

View Results
------------

By default, the run is saved under ``logs/<timestamp>_vlabench_<task-name>_e<episode>_s<seed>/``. Start with these files:

- ``run.log`` contains the RPent process log.
- ``vlabench_env_server.log`` contains simulator startup errors.
- ``transcript_<recipe_tag>.json`` contains the planner conversation and final response.
- ``result.json`` records the task identity, native ``environment_success``, episode budget status, native action count, and planner stats.

The native task success flag in the latest ``task_status`` is the task-success source. Calling ``finish`` ends the planner loop; a success claim is reported as ``failure`` unless the environment confirms it.

Common Options
--------------

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning
   * - ``--vlabench-root``, ``--vlabench-asset-root``
     - Override ``VLABENCH_ROOT`` and ``VLABENCH_ASSET_ROOT``.
   * - ``--vlabench-python``
     - Interpreter that runs the environment server; defaults to ``VLABENCH_PYTHON``, then the interpreter running RPent.
   * - ``--max-episode-steps``
     - Native control-step budget; defaults to 10000. This is a custom run budget, and the task's official limit is reported for provenance.
   * - ``--env-endpoint``
     - Connect to a running environment server instead of launching one.
   * - ``--env-cuda-device``
     - ``CUDA_VISIBLE_DEVICES`` for the local server; defaults to ``--cuda-device``.

VLABench has no published memory corpus, so RPent creates an empty local memory directory and runs in evaluation mode only.
