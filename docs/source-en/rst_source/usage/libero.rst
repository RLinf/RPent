LIBERO
======

`LIBERO <https://libero-project.github.io/>`_ is RPent's primary simulation
benchmark for MuJoCo/robosuite-based tabletop manipulation.
RPent focuses on four core base task families (``libero_object``,
``libero_goal``, ``libero_spatial``, ``libero_10``) and three variants
(``standard``, ``pro``, ``plus``).
The default VLA is **Pi0.5**, served over HTTP by
``robots/libero/vla_server.py``.

VLA configuration
-----------------

Download the recommended SFT checkpoint
`RLinf-Pi05-LIBERO-130-fullshot-SFT
<https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT>`_,
then point at it via ``PI05_CHECKPOINT_PATH``:

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
     --local-dir /path/to/rlinf-pi05-libero-130-fullshot-sft

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

SAM3 configuration
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

Task selection
--------------

A LIBERO run uses the following task settings:

- ``--suite`` — selects the task suite to run. See
  :ref:`libero-pro-core-suites` for the complete core-suite list.
- ``--task`` — the task index within the suite.
- ``--seed`` — the environment seed.
- ``--libero-type`` — the LIBERO variant: ``standard`` | ``pro`` |
  ``plus``.

.. _libero-pro-core-suites:

Core LIBERO-PRO suites
~~~~~~~~~~~~~~~~~~~~~~

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

Minimal command
---------------

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

   rpent --env libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

To switch planners, see :doc:`configure_planner`.

Exploration and local-memory evaluation
---------------------------------------

RPent supports two LIBERO run modes:

- **Exploration** uses multiple resettable attempts and independent planner
  sessions to discover successful strategies and distil them into a local
  global/suite/task memory corpus. It is a memory-generation workflow, not the
  benchmark success-rate measurement.
- **Evaluation** is the default, single-attempt mode. It does not reset the
  episode or update memory. Local-memory evaluation consumes the validated
  audit, recipe, and lessons produced by exploration. The HarnessVLA success
  rate is reproduced in evaluation mode.

Evaluation remains the default mode.  Omitting ``--memory-profile`` preserves
the original Hugging Face resource sync and prompt:

Both profiles run the same single-attempt evaluation workflow; they differ only
in where the evaluation memory comes from and which memory prompt is used.

.. code-block:: bash

   rpent --env libero --suite libero_10_task --task 0 --seed 1 \
     --planner claude_code --memory-profile hf

Use ``local`` only after a local global/suite/task corpus exists, for example
after running the exploration workflow below. This option does not enable
exploration and does not download memory from Hugging Face; it runs the normal
single-attempt evaluation against ``--memory-dir`` (default:
``resources/libero/memory``) without overwriting that directory. If you want to
evaluate with the prebuilt Hugging Face corpus, keep ``--memory-profile hf``:

.. code-block:: bash

   rpent --env libero --suite libero_10_task --task 0 --seed 1 \
     --planner codex --memory-profile local

Exploration uses the same CLI, runtime, tools, and planner implementations.  It
adds resettable attempts and fresh planner sessions, then distils drafts into
``<memory-dir>/_inbox/<cell>/``.  On normal completion the Python runner
validates and merges those drafts, publishes a task audit/recipe pair only when
LIBERO reported success, and rebuilds ``MEMORY.md``. Exploration can start with
an empty ``--memory-dir`` and always uses the local profile; ``--explore`` is
the flag that enables this workflow:

.. code-block:: bash

   rpent --env libero --suite libero_10_task --task 0 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir /path/to/local/libero-memory

Each planner session owns a fresh toolkit. Its state trace and observation
artifacts are retained under ``<output-dir>/sessions/session_NNN/`` for final
memory distillation, while reset-based attempts within that session reuse the
same toolkit.

Add ``--dashboard`` to the exploration command to watch its reasoning, camera
frames, and continuous action timeline across planner sessions.

Pass ``--no-auto-merge-memory`` to retain inbox drafts for manual review.
Maintainers can validate the corpus, rebuild its index, or merge one reviewed
inbox cell explicitly with ``rpent-memory``:

.. code-block:: bash

   rpent-memory --memory-dir /path/to/local/libero-memory validate
   rpent-memory --memory-dir /path/to/local/libero-memory build-index
   rpent-memory --memory-dir /path/to/local/libero-memory merge \
     --cell 10_task_t0_s0 --output-dir logs/explore_10_task_t0_s0

Generated memory is runtime data and is not committed to this repository.

What runs where
---------------

- **env_server** (``robots/libero/env_server.py``) — owns the LIBERO
  MuJoCo env and EGL rendering. Exposes ``reset``, ``step``,
  ``chunk_step``, ``render_camera``, ``get_camera_meta``,
  ``cached_image``, … over an RPC transport (HTTP by default; socket
  via ``--transport socket``).
- **vla_server** (``robots/libero/vla_server.py``) — owns the Pi0.5
  weights. Exposes ``predict`` over the same RPC transport (HTTP or
  socket).
- **sam3_server** (``robots/libero/sam3_server.py``) — owns SAM 3.0 and
  exposes text or single-positive-point segmentation through the same RPC
  transports (HTTP or socket). It returns only the top compressed PNG mask.
- **toolkit** (``robots/libero/toolkit.py``) — defines the tools the
  LLM can call: ``pi0_pick`` (fed to Pi0.5), ``move_to``,
  ``rotate_wrist``, ``back_project``, ``view_env_state``,
  ``finish``, …

Tools the planner can call
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

Live dashboard
--------------

Add ``--dashboard`` to start a long-lived local Dashboard Session. It
selects an available port and prints the URL in the terminal:

.. code-block:: bash

   rpent --env libero --dashboard \
     --planner claude_code --model claude-opus-4-8

Open the URL, confirm the Session configuration, and click **Start Session**.
After the shared services are ready, start a TaskRun from the page with:

.. code-block:: text

   /rpent-task libero_object_swap 2 0

The Dashboard launcher supports the ``api``, ``claude_code``, and ``codex``
planners. Configure ``--planner`` and ``--model`` as for a normal run; see
:doc:`configure_planner`.

Each TaskRun gets a fresh environment while the VLA and SAM3 services are
reused by the Session. Submit a new ``/rpent-task`` to start or switch tasks;
during a run, normal messages steer the agent and Esc requests an interruption.
Press Ctrl+C in the terminal to stop the Session.

``--dashboard`` cannot be combined with ``--interactive`` or
``--env-endpoint``. External ``--vla-endpoint`` and ``--sam3-endpoint``
services remain supported. Use ``--dashboard-language zh-cn`` for the
Chinese UI.

Bringing your own VLA
---------------------

If you have a LIBERO-compatible VLA that is not Pi0.5, swap the model
client without touching the env by:

1. Writing a new ``vla_server.py`` that exposes the same ``predict``
   RPC contract (over HTTP or socket).
2. Pointing at it with ``--vla-endpoint [protocol://]host:port``.
3. Optionally updating ``robots/libero/toolkit.py`` if the tool
   surface (e.g. ``pi0_pick`` → ``mymodel_pick``) needs to change.

See :doc:`../development/add_primitive` for the full walkthrough.

Reproducing results
-------------------

The following results reproduce
:doc:`Harness VLA <../awesome_works/harnessvla>` on two LIBERO-PRO suites.
On the `reproduce/libero
<https://github.com/RLinf/RPent/tree/reproduce/libero>`_ branch, use
``gpt-5.5`` to reproduce these results:

- ``libero_10_task``: 70% (70/100)
- ``libero_10_swap``: 55% (55/100)

Reproduction command:

.. code-block:: bash

   rpent --env libero \
     --suite libero_10_task --task "task" --seed "seed" \
     --planner codex \
     --model gpt-5.5 \
     --max-turns 100 \
     --planner-timeout-s 5000 \
     --max-episode-steps 10000 \
     --libero-type pro \
     --vla-endpoint http://127.0.0.1:8220 \
     --sam3-endpoint http://127.0.0.1:8114
