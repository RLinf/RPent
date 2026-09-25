LIBERO
============

.. figure:: https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/images/fig1.png
   :alt: LIBERO environment overview
   :width: 90%
   :align: center

   LIBERO benchmark overview. Source: `LIBERO project <https://libero-project.github.io/>`_.

Run tabletop manipulation tasks with RPent in `LIBERO <https://libero-project.github.io/>`_, then reproduce LIBERO-PRO experiments. The simulator uses MuJoCo/robosuite. RPent supports the ``standard``, ``pro``, and ``plus`` variants and uses Pi0.5 as the default VLA.

.. _libero-overview:

Overview
------------

Check the model, task, and runtime requirements before following the installation and run steps.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      Pi0.5

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``; also see :doc:`flash`.

   .. grid-item-card:: Tasks

      Object, Goal, Spatial, LIBERO-10

   .. grid-item-card:: Hardware

      Linux, NVIDIA GPU; Python 3.11; CUDA and EGL.

.. _core-libero-pro-suites:

.. _libero-pro-core-suites:

Tasks
~~~~~~~~~~~~

This table covers RPent's four core LIBERO-PRO task families and all of
their perturbation suites.

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Family
     - Base suite
     - Perturbation suites
   * - Object
     - ``libero_object``
     - ``libero_object_task``, ``libero_object_swap``,
       ``libero_object_lan``, ``libero_object_object``
   * - Goal
     - ``libero_goal``
     - ``libero_goal_task``, ``libero_goal_swap``,
       ``libero_goal_lan``, ``libero_goal_object``
   * - Spatial
     - ``libero_spatial``
     - ``libero_spatial_task``, ``libero_spatial_swap``,
       ``libero_spatial_lan``, ``libero_spatial_object``
   * - LIBERO-10
     - ``libero_10``
     - ``libero_10_task``, ``libero_10_swap``, ``libero_10_lan``,
       ``libero_10_object``

.. _libero-observation-action:

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The table distinguishes planner tools, model inputs, and the environment’s success criterion.

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - RGB/depth camera views and end-effector/gripper state. Pi0.5 uses the scene and wrist images with robot state.
   * - Action
     - The planner calls Pi0.5 tools or motion primitives such as ``move_to`` and ``set_gripper``; these execute environment actions.
   * - Reward / success
     - Evaluate success using the final state’s top-level ``terminated`` value. A step limit or planner ``finish`` is not a success label.
   * - Task prompt
     - The task language comes from the selected suite/task and the current environment.

.. _installation-and-variants:

Installation and Resources
--------------------------

For a first LIBERO-PRO run, complete the installation, asset downloads, and model setup in :doc:`../quickstart`. Continue here to select tasks and reproduce experiments.

For another variant, install the matching extra in an isolated environment and download its assets. Install only one LIBERO variant per Python environment:

.. list-table::
   :header-rows: 1

   * - Variant
     - Install
     - Assets
   * - ``standard``
     - ``uv pip install -e ".[libero]"``
     - ``libero-download-assets --skip-existing``
   * - ``pro``
     - ``uv pip install -e ".[libero-pro]"``
     - ``liberopro-download-assets --skip-existing``
   * - ``plus``
     - ``uv pip install -e ".[libero-plus]"``
     - ``liberoplus-download-assets --skip-existing``

At runtime, set ``--libero-type`` to match the installed variant.

VLA Configuration
-----------------

Download the recommended SFT checkpoint
`RLinf-Pi05-LIBERO-130-fullshot-SFT
<https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT>`_,
then point at it via ``PI05_CHECKPOINT_PATH``:

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
     --local-dir /path/to/rlinf-pi05-libero-130-fullshot-sft

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

SAM3 Configuration
------------------

SAM 3.0 segmentation is enabled for every LIBERO run. Download ``sam3.pt``
from `Hugging Face: facebook/sam3 <https://huggingface.co/facebook/sam3>`_
or `ModelScope: facebook/sam3 <https://modelscope.cn/models/facebook/sam3>`_,
then point at it via ``SAM3_CHECKPOINT_PATH``:

.. code-block:: bash

   # Hugging Face (request access on the model page first)
   hf auth login
   hf download facebook/sam3 sam3.pt --local-dir /path/to/sam3

   # ModelScope (use this instead of the Hugging Face commands above)
   modelscope download --model facebook/sam3 sam3.pt --local_dir /path/to/sam3

   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt

Task Selection
--------------

A LIBERO run uses the following task settings:

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning
   * - ``--suite``
     - Select a suite from :ref:`Core LIBERO-PRO suites <libero-pro-core-suites>`.
   * - ``--task``
     - Task index within the suite.
   * - ``--seed``
     - Environment seed; defaults to ``0``.
   * - ``--libero-type``
     - Installed LIBERO variant: ``standard`` | ``pro`` | ``plus``.

.. _minimal-command:

Run a Task
----------

Complete the model setup above and configure a planner with :doc:`configure_planner`. This command runs task 2 of ``libero_object_swap`` at seed 0.

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

   rpent --robot libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

To switch planners, see :doc:`configure_planner`.

View Results
------------

The terminal shows server startup, planner conversation, and tool calls. After the run, inspect ``episode.mp4``, ``transcript_*.json``, and ``run.log`` in the output directory; see :ref:`run-output-files` for default paths and step artifacts.

Inspect the final state with ``view_env_state(step=-1)``. Its top-level ``terminated`` value is the native success result; ``finish`` ends planning. See :doc:`dashboard` for live monitoring.

.. _libero-exploration:

.. _exploration-and-memory:

Task Memory and Exploration Mode
--------------------------------

Use ``--explore`` to build local memory or ``--memory-profile local`` to read a prepared corpus. See :doc:`memory` for the complete LIBERO workflow and commands. LIBERO’s native success result determines whether successful task records are published.

.. raw:: html

   <span id="reproducing-results"></span>

Experiment Reproduction
-----------------------

See :doc:`../leaderboard` for the unified RPent model comparison on LIBERO-PRO
Task/Swap and the corresponding model configurations.

The :doc:`GPT-6 Astra suite results <../leaderboard>`
cover all eight complete suites and 800 verified episodes: 741 successes,
59 failures, and 92.63% Overall, with Codex / GPT-6 Astra / low / reasoning.

The following historical reproduction records use the `reproduce/libero
<https://github.com/RLinf/RPent/tree/reproduce/libero>`_ branch with
``gpt-5.5`` and ``xhigh`` reasoning effort:

- ``libero_10_task``: 70% (70/100)
- ``libero_10_swap``: 55% (55/100)

Use the reproduction branch above and complete this page’s resource setup. The example selects task 0, seed 0; reproducing the reported score requires the full task and seed coverage. Start services at the example ports using :doc:`advanced_deployment` first.

Reproduction command:

.. code-block:: bash

   rpent --robot libero \
     --suite libero_10_task --task 0 --seed 0 \
     --planner codex \
     --model gpt-5.5 --reasoning-effort xhigh \
     --max-turns 100 \
     --planner-timeout-s 5000 \
     --max-episode-steps 10000 \
     --libero-type pro \
     --vla-endpoint http://127.0.0.1:8220 \
     --sam3-endpoint http://127.0.0.1:8114

What Runs Where
---------------

LIBERO separates simulation and model inference into services. The toolkit
uses their clients to execute the planner's requests.

- **env_server** (``robots/libero/env_server.py``) — owns the LIBERO
  MuJoCo env and EGL rendering. Exposes ``reset``, ``step``,
  ``chunk_step``, ``render_camera``, ``get_camera_meta``, … over an RPC
  transport (HTTP by default; socket via ``--transport socket``).
- **vla_server** (``rpent/robots/components/pi05_vla_server.py``) — owns the Pi0.5
  weights. Exposes ``predict`` over the same RPC transport (HTTP or
  socket).
- **sam3_server** (``rpent/robots/components/sam3_server.py``) — loads SAM 3.0 and
  segments a target region from a text description or one pixel on the target
  object. If the model produces multiple candidates, it selects the one with
  the highest model score and returns its mask as a compressed PNG through the
  same RPC transports (HTTP or socket).
- **toolkit** (``robots/libero/toolkit.py``) — defines the tools the
  LLM can call: ``pi0_pick`` (fed to Pi0.5), ``move_to``,
  ``rotate_wrist``, ``back_project``, ``view_env_state``,
  ``finish``, …

Tools the Planner Can Call
--------------------------

LIBERO tools fall into two groups: physical action tools and read-only tools.

**Physical action tools:**

- ``pi0_pick(prompt, ...)`` — use Pi0.5 to execute a closed-loop grasp.
- ``pi0_doubled(prompt, ...)`` — use Pi0.5 for a non-pick contact action.
- ``move_to(xyz, ...)`` — move the end effector to a world-frame position.
- ``move_pose(xyz, target_pitch=..., target_yaw=..., ...)`` — move position
  and orientation together.
- ``rotate_wrist(target_yaw=... / delta_yaw=..., ...)`` — rotate wrist yaw
  to an absolute target or by a relative amount.
- ``rotate_pitch(target_pitch=... / delta_pitch=..., ...)`` — tilt the
  gripper to an absolute pitch or by a relative amount.
- ``set_gripper(gripper=..., steps=...)`` — hold the pose and drive the
  gripper for a fixed number of steps.
- ``release(...)`` — open the gripper.

Physical action tools advance the environment and record new state and images.

**Read-only tools:**

- ``back_project(row, col, ...)`` — back-project an image pixel to world
  coordinates.
- ``segment(prompt=... / point=..., ...)`` — use SAM3 to segment an existing
  image with a text or point prompt.
- ``view_env_state(step=-1)`` — read a recorded state and its embedded
  observation images. Step ``0`` is initial; ``-1`` is latest.
- ``view_camera_meta(camera=..., step=-1)`` — read camera metadata for a
  recorded step. Step ``-1`` is latest.
- ``finish(status, summary)`` — end the current run.

These tools do not advance the environment.

Dashboard
---------

See :doc:`dashboard` to monitor runs, submit tasks, and stop a session.

Bringing Your Own VLA
---------------------

If you have a LIBERO-compatible VLA that is not Pi0.5, swap the model
client without touching the robot by:

1. Writing a new ``vla_server.py`` that exposes the same ``predict``
   RPC contract (over HTTP or socket).
2. Pointing at it with ``--vla-endpoint [protocol://]host:port``.
3. Optionally updating ``robots/libero/toolkit.py`` if the tool
   surface (e.g. ``pi0_pick`` → ``mymodel_pick``) needs to change.

See :doc:`../development/add_primitive` for the full walkthrough.
