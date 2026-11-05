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

      Pi0.5; Cosmos Policy, Cosmos3 and Fast-WAM for standard/Pro evaluation.

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``; also see :doc:`../guides/flash`.

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

For a first LIBERO-PRO run, complete the installation, asset downloads, and model setup in :doc:`../get_started/quickstart`. Continue here to select tasks and reproduce experiments.

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

Each extra includes Pi0.5 (``rpent-openpi``), SAM3, and the RLinf runtime
(``rpent-rlinf``). Download model weights separately as described below.

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
     - Zero-based task index within the suite. Out-of-range indices and tasks with no stored initial states are rejected; invalid selections never substitute another task.
   * - ``--seed``
     - Environment seed; defaults to ``0``.
   * - ``--libero-type``
     - Installed LIBERO variant: ``standard`` | ``pro`` | ``plus``.

.. _minimal-command:

Run a Task
----------

Complete the model setup above and configure a planner with :doc:`../guides/configure_planner`. This command runs task 2 of ``libero_object_swap`` at seed 0.

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

   rpent --robot libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

To switch planners, see :doc:`../guides/configure_planner`.

WAM Evaluation
~~~~~~~~~~~~~~

Prepare a worker with :doc:`../development/add_action_model` and keep the simulator
and SAM3 setup above. Pi0.5 weights are not needed for WAM. With Cosmos on port 8116:

.. code-block:: bash

   rpent --robot libero --suite libero_spatial --task 0 --seed 0 \
     --libero-type standard --am-backend cosmos-policy \
     --am-endpoint http://127.0.0.1:8116 --memory-profile local \
     --planner api --model <planner-model>

Fast-WAM uses ``--am-backend fast-wam`` and its worker endpoint. For LIBERO-Pro,
select a Pro suite and ``--libero-type pro`` in a Pro environment. WAM excludes
LIBERO-Plus, Flash and exploration. WAM uses a required local memory manager
and file tools, with a backend-specific directory that may be empty.
Use ``--memory-dir`` to select a compatible local corpus; HF memory is unsupported.
``wam_act(prompt=None, max_chunks=1)`` replaces the Pi0.5 tools and executes one
to four chunks with fresh observations. The worker's ``chunk_size`` metadata
declares the number of actions per chunk. A prompt override applies to
that call only. Chunk budgets do not specify where to stop inside a chunk.

View Results
------------

The terminal shows server startup, planner conversation, and tool calls. After the run, inspect ``episode.mp4``, ``transcript_*.json``, and ``run.log`` in the output directory; see :ref:`run-output-files` for default paths and step artifacts.

Inspect the final state with ``view_env_state(step=-1)``. Its top-level ``terminated`` value is the native success result; ``finish`` ends planning. See :doc:`../guides/dashboard` for live monitoring.

.. _libero-exploration:

.. _exploration-and-memory:

Task Memory and Exploration Mode
--------------------------------

The HF memory and exploration workflows in this section apply to VLA.
WAM uses local memory as described above.

Evaluation is single-attempt and reads memory without updating it. The default
``hf`` profile uses ``--memory-version auto`` to select memory by model. See
:doc:`../guides/memory` for explicit versions, offline preparation, and release provenance.

Use ``--explore`` to build local memory or ``--memory-profile local`` to read a prepared corpus. See :doc:`../guides/memory` for the complete LIBERO workflow and commands. LIBERO’s native success result determines whether successful task records are published.

.. raw:: html

   <span id="reproducing-results"></span>

Experiment Reproduction
-----------------------

See :doc:`../leaderboard/index` for the unified RPent model comparison on LIBERO-PRO
Task/Swap and the corresponding model configurations.

The :doc:`GPT-6 Astra suite results <../leaderboard/index>`
cover all eight complete suites and 800 verified episodes: 741 successes,
59 failures, and 92.63% Overall, with Codex / GPT-6 Astra / low / reasoning.

The Long results use the `reproduce/libero
<https://github.com/RLinf/RPent/tree/reproduce/libero>`_ branch with
``gpt-5.5`` and ``xhigh`` reasoning effort:

- ``libero_10_task``: 70% (70/100)
- ``libero_10_swap``: 55% (55/100)

Together with the six unchanged Spatial/Object/Goal scores (81%, 69%, 94%,
91%, 75%, and 66%), these Long results give **75.13% Overall** across the eight
suites. The :doc:`Leaderboard <../leaderboard/performance>` uses this updated
aggregate, combining the six paper results with the two Long reproduction results.

Use the reproduction branch above and complete this page’s resource setup. The example selects task 0, seed 0; reproducing the reported score requires the full task and seed coverage. Start services at the example ports using :doc:`../guides/advanced_deployment` first.

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
     --am-endpoint http://127.0.0.1:8220 \
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

Localization Demo
-----------------

The shared ``rpent.tools.localization.localization`` module turns RGB, depth
and calibration into a ``LocalizationFrame``. Each pixel in ``frame.rgb``
has the same array position in ``depth_m``, ``world_xyz`` and ``valid_mask``.
Depth and world coordinates use metres. Row 0 is the top of the image;
OpenCV camera axes are x right, y down and z forward.

LIBERO is the first integration of this capture API. Its env server captures
all requested views and their calibration under one exclusive RPC dispatch,
without resetting or stepping the simulation. The client computes the world
tables with NumPy. RoboCasa and RoboTwin retain their existing integrations.

After installing LIBERO and downloading SAM3 weights as above, start the env
and SAM3 services using :doc:`../guides/advanced_deployment`. Both the demo
client and env server must use this revision's capture API. With those services
running, use an environment that has already been initialized for its task.
For a newly launched env server, explicitly reset it once before the demo:

.. code-block:: python

   from contextlib import closing
   from rpent.utils.rpc import make_rpc_client

   with closing(make_rpc_client("http://ENV_HOST:ENV_PORT")) as rpc:
       rpc.call("env.reset", timeout_s=120.0)

Then execute this from the source checkout, replacing the endpoints:

.. code-block:: bash

   python -m robots.libero.localization_demo \
     --env-endpoint http://ENV_HOST:ENV_PORT \
     --sam3-endpoint http://SAM3_HOST:SAM3_PORT \
     --prompt "black bowl" \
     --output-dir /tmp/libero-localization-demo

Use a new output directory for each run. The demo captures one 1024x1024
``agentview`` image, asks the existing ``Sam3Tools`` helper to segment it,
and reports the median world XYZ of valid pixels in the object mask.
Use ``--camera robot0_eye_in_hand`` for the wrist or ``--height 256 --width 256``
for a smaller capture. This demo uses the existing services without a planner
or action-model service and leaves their lifetime with the operator.

Inspect ``camera.png/00.png``, ``overlay.png/00.png`` and
``result.json/00.json`` in the output directory. ``result.json`` contains
``world_xyz`` in metres and valid-pixel counts; it reports ``world_error``
when no object, insufficient valid depth or a mask/image grid mismatch prevents
localization. The directory also contains metric depth, world coordinates,
geometry and object masks, calibration metadata and the EnvState manifest.
An overlay is saved only when a detected mask matches the image grid.

To use capture independently of SAM3, attach to a running server without
constructing an environment client that would reset it by default:

.. code-block:: python

   from contextlib import closing
   from rpent.tools.localization.localization import RpcFrameSource, capture
   from rpent.utils.rpc import make_rpc_client

   with closing(make_rpc_client("http://ENV_HOST:ENV_PORT")) as rpc:
       frame = capture(RpcFrameSource(rpc), "agentview", height=256, width=256)
   point = frame.world_xyz[row, col]  # meaningful only if valid_mask[row, col]
   object_points = frame.points(object_mask)  # mask must match frame.rgb exactly

``row``, ``col`` and ``object_mask`` must come from this captured RGB grid.
SAM3 point prompts use ``[row, col]``; projecting world points with
``project_to_image`` returns ``(u, v) = (col, row)``. A resized or flipped image
needs an explicit coordinate conversion before its pixels or mask can be used.

Recorded-data Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~

LIBERO keeps the existing image, depth, world-map and metadata artifact names.
State recording now uses canonical frames for both cameras at the observation
resolution and at 1024x1024. If the high-resolution batch fails, it retries
the low-resolution pair; ``segment`` selects a complete high-resolution
image/world pair when present and otherwise uses the low-resolution pair.
Recording now re-renders the two low-resolution views as part of the batch;
this adds rendering work relative to reusing raw observations. It introduces
no additional dependencies.

Readers of existing LIBERO artifacts should account for these changes:

- Invalid depth and world points are ``NaN``. Filter with ``valid_mask`` or
  ``np.isfinite(world).all(axis=-1)`` before computing statistics. Do not use
  nonzero coordinates as the validity test for new captures.
- High-resolution world maps use ``float32`` instead of ``float16``. Their dense
  arrays require twice as much memory before compression; remove fixed dtype
  assumptions and preserve ``float32`` when processing new files.
- New ``agentview_valid_mask.npz``, ``wrist_valid_mask.npz``,
  ``agentview_valid_mask_high.npz`` and ``wrist_valid_mask_high.npz`` artifacts
  store geometric validity. These masks describe usable geometry, not the
  segmented object.
- Metadata's top-level ``intrinsic_K`` now belongs to the saved top-down RGB
  grid, with integer pixel centres, and ``T_world_from_cv`` is the canonical
  camera-to-world transform. Raw calibration is under ``native``. Read
  ``height``, ``width`` and ``pixel_convention`` rather than assuming a fixed
  size or applying another flip or principal-point correction. Existing
  ``get_camera_meta`` and ``render_camera`` RPCs retain their native formats.

For a new low-resolution state artifact, the NPZ key remains ``array``:

.. code-block:: python

   import numpy as np

   with np.load(run_dir / "agentview_world.npz" / "00.npz") as archive:
       world = archive["array"]
   with np.load(run_dir / "agentview_valid_mask.npz" / "00.npz") as archive:
       valid = archive["array"]
   points = world[object_mask & valid]
   centre = np.median(points, axis=0) if len(points) else None

For high-resolution artifacts use the matching ``_high`` world map, image
and validity mask. Old runs have no validity-mask artifact; their reader must
also retain the old nonzero-coordinate check when excluding the old zero
sentinel. ``agentview_policy.png`` is a separate model input and must not
supply coordinates for these world maps.

Dashboard
---------

See :doc:`../guides/dashboard` to monitor runs, submit tasks, and stop a session.

Bringing Your Own VLA
---------------------

If you have a LIBERO-compatible VLA that is not Pi0.5, swap the model
client without touching the robot by:

1. Writing a new ``vla_server.py`` that exposes the same ``predict``
   RPC contract (over HTTP or socket).
2. Pointing at it with ``--am-endpoint [protocol://]host:port``.
3. Optionally updating ``robots/libero/toolkit.py`` if the tool
   surface (e.g. ``pi0_pick`` → ``mymodel_pick``) needs to change.

See :doc:`../development/add_primitive` for the full walkthrough.
