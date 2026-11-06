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

SAM 3.0 is an internal GeometryAgent dependency. Download ``sam3.pt``
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
  ``rotate_wrist``, ``localize``, ``view_env_state``,
  ``finish``, …
- **GeometryAgent** (``tools/libero/geometry_agent.py``) — receives target/part
  descriptions through ``localize``. Its visual grounding model selects a
  material pixel; internal segmentation and coordinate tools compute the result.
  The parent planner has no direct ``segment`` or ``back_project`` tool.

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

- ``localize(query, camera="agentview", step=-1, measurement="surface_point")`` —
  delegate localization of a described object or part to GeometryAgent.
  ``surface_point`` returns a visible material point; ``visible_region`` returns
  the coordinate-wise median and bounds of visible segmented samples.
  Add ``required_geometry=["rim_center", "body_center", "bottom_center", "axis",
  "radius", "height"]`` to request part-specific cleaning and fitting instead.
  The parent supplies target text and step/camera, with no calibration parameters.
- ``view_env_state(step=-1)`` — read a recorded state and its embedded
  observation images. Step ``0`` is initial; ``-1`` is latest.
- ``view_camera_meta(camera=..., step=-1)`` — read camera metadata for a
  recorded step. Step ``-1`` is latest.
- ``finish(status, summary)`` — end the current run.

These tools do not advance the environment.

GeometryAgent pins one recorded frame. Each part uses one visual-model call and
at most two SAM calls. Cup requests use at most three parts; adding a plane uses
at most four. The agent saves ``geometry_XXX.json`` and mask, candidate, cleaned
point-cloud and fit-inlier artifacts. No arrays are sent as tool-result text.

Rim and bottom boundaries use robust 3-D circle fitting; the curved body compares
cylinder and frustum models; visible planar patches use RANSAC and SVD. Cleanup
uses voxel sampling, a target-connected component and neighbor outlier rejection.
Body/rim requests may remove a horizontal background plane found outside the
mask; bottom/plane requests preserve their target plane. A body fit treats
handle/interior samples as outliers and is rejected when support is insufficient.

Exact reference requests use ``inner_rim_center`` for the inner upper opening,
``plate_center`` for the central supporting face, ``lid_center`` for a sealed
circular lid, ``rectangle_center`` for the main rectangular top face, and
``body_segment_center`` for the main bottle barrel excluding its neck and mouth.
The inner opening uses resolved inner contours instead of center-pixel depth.
Its candidates retain their largest connected component without requiring
proximity to the visual material point. That point still identifies the SAM
region and supports background-plane separation and circle consistency checks;
outlier rejection and circle acceptance gates remain active.
Circular face fits take their height from the selected plane. Rectangles require
four physical edges and corner support; foreground covers, coplanar label crops
and image clipping cannot establish a full-face center. Bottle fitting needs
both barrel limits from observed radius transitions or physical terminal edges,
and supports a horizontal axis. Visible sample extents alone are insufficient.

Fitted results are in ``geometry[field]``, with definition, assumptions, evidence
references and quality. ``radius`` means rim radius; ``body_radii`` contains only
visible cross-section radii. Body center is the midpoint of observed coaxial rim
and bottom circles. Full height requires both end boundaries; a visible axial
extent alone is insufficient. Axis sign is conventional unless both ends are
observed. Low residuals alone do not establish reliability: coverage, degeneracy
and consistency gates may reject a fit. Algorithm thresholds and residuals are
not calibrated positional error; quantized world maps retain their precision.

``partial`` means some requested fields are available and others remain unknown.
When geometry is unavailable, the agent requests another view; the parent
executes any motion. Multi-view fusion, full box dimensions and cavity geometry
are not implemented by these fitting models.

API and Codex planners inherit their model and provider URL for visual grounding.
Use ``--geometry-model provider:model`` or ``--geometry-model codex:model`` to
select another model. Bare ``codex`` uses the configured Codex default.
``--geometry-base-url`` overrides the inherited ``--base-url``. Claude Code
planners require an explicit ``--geometry-model``. The visual model receives
only the selected RGB image and target query, with no robot tools or object poses.

Fixed Reference Checklist Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implementation is in ``tools/libero/``. The previous
``python -m robots.libero.localization_experiment`` command delegates to the
same entry point for existing experiment scripts.

The source checkout includes ``选点清单.md`` and its expanded ``test-cases.json``:
16 Task/Swap rows, seeds 1 through 5, 80 cases and 160 fixed reference requests.
Run the following from the repository root in the LIBERO-Pro environment with
the simulator assets, EGL/CUDA and SAM3 checkpoint configured above. The
experiment needs neither Pi0.5/WAM weights nor task memory. Collection starts
only ENV; inference starts only SAM3 and the explicitly selected visual model.

.. code-block:: bash

   python -m tools.libero.localization_experiment validate-manifest
   python -m tools.libero.localization_experiment collect \
     --output experiments/localization/observations
   python -m tools.libero.localization_experiment infer \
     --inputs experiments/localization/observations \
     --geometry-model codex \
     --output experiments/localization/predictions

``codex`` uses its configured default; ``codex:model`` or ``provider:model``
also work, with optional ``--geometry-base-url``. Each case uses only step 0 /
agentview, with fixed A/B targets. Collection records native paired 1024×1024
RGB-D, internal calibration and float32 world maps. Inference copies only allowed
sensor artifacts, checks their hashes and never modifies the collected source.
The protocol records refusals without executing actions or additional views.
Each reference has a budget of one grounding call and at most two SAM calls.

Use matching ``--limit N`` or repeated ``--case <case_id>`` options in collection
and inference for a pilot subset. Add ``--permissive`` to ``infer`` when measuring
errors for rejected checklist centers. It retains strict-fit diagnostics, then
tries relaxed shape fitting or a labelled visible-extent center without asking
for another view. ``strict_accepted`` and ``estimate_method`` distinguish strict
fits from approximate candidates; candidate fields have ``evaluation_only=true``.
In this mode, ``usable`` means a coordinate was returned, not that strict quality
gates passed. Invalid/missing observations still produce errors. Truth remains
evaluator-only. The default is strict; the linked experiment guide specifies the
relaxed gates and methods. Output directories must be new or empty.
``--cuda-device``, ``--env-cuda-device`` and ``--sam3-cuda-device`` select GPUs.
Inference can borrow ``--sam3-endpoint``; ``--env-endpoint`` requires exactly one
selected case and matching server metadata. Only owned processes are stopped.

``results.jsonl`` records every fixed request, including errors and refusals.
``summary.json`` reports status counts, returned centers, calls and latency.
Each case has full ``result_A.json`` / ``result_B.json`` and an ``episode`` with
masks, overlays, candidate/cleaned clouds and fit evidence. Read centers only
from the requested ``geometry[field].value``; surface points or medians are never
fallback centers. ``fit_points_artifact`` is indexed by the fit-inlier array;
``observed_support_artifact`` preserves raw boundary samples separately when a
circular-face fit uses plane-projected points.

For numerical error, generate ``truth-template`` and independently annotate all
160 references before evaluation. Fill ``evaluable``, ``annotation_source`` and
either ``world_xyz`` or an explicit ``not_evaluable_reason``. Preserve each ID,
definition and frame. Truth may instead specify ``model_local_xyz`` with a rigid
``model_to_world`` matrix; bottle references may use a pre-labelled local
``barrel_profile`` (unit ``axis``, ``axis_origin``, ordered ``bounds_m``) with that
matrix. The evaluator alone reads these inputs. Null template values mean unlabelled, not zero
coordinates or automatic exclusion.

.. code-block:: bash

   python -m tools.libero.localization_experiment truth-template \
     --output experiments/localization/truth.json
   # Complete the independent annotations before the next command.
   python -m tools.libero.localization_experiment evaluate \
     --predictions experiments/localization/predictions/results.jsonl \
     --truth experiments/localization/truth.json --tolerance-m 0.01 \
     --output experiments/localization/evaluation

For references defined from complete LIBERO models, add ``--with-references``
to collection. Each case then has a separate ``evaluator_snapshot.json`` with
compiled visual meshes and body poses from that same initial state. The ENV
server must have ``--evaluation-references`` enabled when borrowing an endpoint.
Fix each part's model-local reference before inference, then transform it with
the matching body pose; the object origin is not a default part center. These
snapshots contain no part annotations. They remain outside the episode and are
excluded by inference's sensor whitelist, so GeometryAgent never receives them.

``metrics.json`` and ``errors.csv`` include per-reference errors and object/suite
groups. Success within tolerance uses all independently evaluable references,
including refusals, failures and unrun items. Mean, median, P95 and RMSE are
conditional on returned centers and must be read with coverage. The default
1 cm tolerance is a configurable criterion. Missing labels, duplicate IDs,
wrong units, frames or definitions reject evaluation. For a pilot evaluation,
use the matching ``--case`` / ``--limit`` selection; annotations still cover the
complete checklist. CPU analytic/fake-perception tests establish execution and
geometric contracts; real-scene accuracy and manipulation success require
separate measurements.

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
