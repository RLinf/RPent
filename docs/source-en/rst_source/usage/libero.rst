LIBERO
======

`LIBERO <https://libero-project.github.io/>`_ is RPent's primary simulation
benchmark for MuJoCo/robosuite-based tabletop manipulation.
RPent focuses on four core base task families (``libero_object``,
``libero_goal``, ``libero_spatial``, ``libero_10``) and three variants
(``standard``, ``pro``, ``plus``).
The default VLA is **Pi0.5**, served over HTTP by
``rpent/robots/components/pi05_vla_server.py``.

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

SAM 3.0 segmentation is enabled for LIBERO runs except ``--planner onejev``. Download ``sam3.pt``
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

   rpent --robot libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

To switch planners, see :doc:`configure_planner`.

.. _libero-exploration:

Exploration and local-memory evaluation
---------------------------------------

RPent supports two LIBERO run modes:

- **Exploration** uses multiple resettable attempts and independent planner
  sessions to discover successful strategies and distil them into a local
  global/task-family/task-specific memory corpus. It is a memory-generation workflow, not the
  benchmark success-rate measurement.
- **Evaluation** is the default, single-attempt mode. It does not reset the
  episode or update memory. Local-memory evaluation consumes the validated
  audit, recipe, and lessons produced by exploration. The HarnessVLA success
  rate is reproduced in evaluation mode.

Evaluation remains the default mode. Omitting ``--memory-profile`` selects
Hugging Face memory; ``--memory-version auto`` chooses its model-specific
version. See :ref:`Memory Management <memory-management>` for overrides,
offline downloads and release provenance.

Both profiles run the same single-attempt evaluation workflow; they differ only
in where the evaluation memory comes from and which memory prompt is used.

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner claude_code --memory-profile hf

Use ``local`` after a local memory corpus has been prepared, for example
after running the exploration workflow below. This option does not enable
exploration and does not download memory from Hugging Face; it runs the normal
single-attempt evaluation against ``--memory-dir`` (default:
``memory/libero``) without overwriting that directory. If you want to
evaluate with the prebuilt Hugging Face corpus, keep ``--memory-profile hf``:

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner codex --memory-profile local

Exploration uses the same CLI, runtime, tools, and planner implementations.  It
adds resettable attempts and fresh planner sessions, then distils drafts into
``<memory-dir>/_internal/inbox/<cell>/``.  On normal completion the Python runner
validates and merges those drafts, publishes a task audit/recipe pair only when
LIBERO reported success, and refreshes ``MEMORY.md``. Exploration can start with
an empty ``--memory-dir`` and always uses the local profile; ``--explore`` is
the flag that enables this workflow:

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 0 \
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
  ``chunk_step``, ``render_camera``, ``get_camera_meta``, … over an RPC
  transport (HTTP by default; socket via ``--transport socket``).
- **vla_server** (``rpent/robots/components/pi05_vla_server.py``) — owns the Pi0.5
  weights. Exposes ``predict`` over the same RPC transport (HTTP or
  socket).
- **sam3_server** (``rpent/robots/components/sam3_server.py``) — owns SAM 3.0 and
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

   rpent --robot libero --dashboard \
     --planner claude_code --model claude-opus-4-8

Session configuration comes from the command line, and the URL opens directly
in the live monitor. After the shared services are ready, start a TaskRun with:

.. code-block:: text

   /rpent-task libero_object_swap 2 0

The Dashboard supports the ``api``, ``claude_code``, and ``codex`` planners.
Configure ``--planner`` and ``--model`` on the command line as for a normal
run; see :doc:`configure_planner`.

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
client without touching the robot by:

1. Writing a new ``vla_server.py`` that exposes the same ``predict``
   RPC contract (over HTTP or socket).
2. Pointing at it with ``--vla-endpoint [protocol://]host:port``.
3. Optionally updating ``robots/libero/toolkit.py`` if the tool
   surface (e.g. ``pi0_pick`` → ``mymodel_pick``) needs to change.

See :doc:`../development/add_primitive` for the full walkthrough.

Reproducing results
-------------------

See :doc:`../leaderboard` for the unified RPent model comparison on LIBERO-PRO
Task/Swap and the corresponding model configurations.

The :doc:`GPT-6 Astra suite results <../leaderboard>`
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

Reproduction command:

.. code-block:: bash

   rpent --robot libero \
     --suite libero_10_task --task "task" --seed "seed" \
     --planner codex \
     --model gpt-5.5 \
     --max-turns 100 \
     --planner-timeout-s 5000 \
     --max-episode-steps 10000 \
     --libero-type pro \
     --vla-endpoint http://127.0.0.1:8220 \
     --sam3-endpoint http://127.0.0.1:8114

OneJev decision backend
-----------------------

``--planner onejev`` uses an external OneJev-9B System One service to score
fully parameterized actions computed by the LIBERO adapter. RPent executes
the highest-scoring candidate through its existing toolkit. Pi0.5 performs
grasps and drawer contact; scripted primitives perform transport, lowering
and release.

Install OneJev in a working directory separate from RPent. All paths in the
following commands are relative to that directory:

.. code-block:: bash

   git clone https://github.com/OmniJev/OneJev.git
   uv venv .venv-onejev --python 3.11
   uv pip install --python .venv-onejev/bin/python --torch-backend auto \
     ./OneJev \
     'transformers>=5.17.0,<6' pillow torchvision
   hf download OmniJev/OneJev-9B --local-dir ./OneJev-9B

Start the OneJev service from the same directory and keep it running:

.. code-block:: bash

   ./.venv-onejev/bin/qev serve \
     --model ./OneJev-9B --name OneJev-9B \
     --host 127.0.0.1 --port 8008 \
     --device cuda:0 --dtype bfloat16 --multimodal \
     --no-cuda-graphs --no-gpu-preprocess

Once the service is ready, activate the RPent environment in another terminal.
From the RPent repository root, run:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH="data/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT/"
   export LIBERO_TYPE=pro
   rpent --robot libero --suite libero_object_swap --task 2 --seed 0 \
     --planner onejev --model OneJev-9B --base-url http://127.0.0.1:8008

This mode starts only Env and Pi0.5. It uses run-local artifacts without
loading a memory corpus, and does not start SAM3 or Molmo. OneJev dependencies
remain in their separate environment; the RPent client uses HTTP only.
The Env server is started with ``--public-observations-only``. It exposes
RGB-D, robot state, camera calibration and public episode termination flags;
it omits private object state, segmentation, contact data and reward/info
payloads. A borrowed ``--env-endpoint`` must have the same public mode and
suite/task/seed settings. It does not expose the full ``env.raw_obs`` route.

The adapter in ``robots/libero/onejev/`` supports opening or closing a named
drawer and placing one or more named objects in the same basket, bowl, plate
or tray. It preserves spatial descriptions such as ``between ... and ...``
as one object, and rejects unsupported verbs and separate task stages.
Drawer tasks use bounded Pi0.5 contact actions. Placement tasks use RGB-D
rules to propose destinations; those rules do not identify objects. OneJev
compares the marked RGB regions against the task. Missing geometry stops a
placement run before a grasp. Native LIBERO termination alone establishes
task success.

Optionally provide ``--onejev-config /path/to/config.json``. ``target_roi`` is an operator-supplied
RGB region in normalized ``[row_min, col_min, row_max, col_max]`` coordinates.
Other settings include workspace bounds, grasp and drawer-contact budgets,
carry clearance and
EEF/object offset. These are explicit control priors, not simulator-derived
object geometry. For example, a configuration may contain:

.. code-block:: json

   {
     "target_roi": [0.2, 0.2, 0.8, 0.8],
     "pick_max_chunks": 24,
     "eef_object_offset": 0.09
   }

The ROI above is an illustrative image region, not a calibrated task preset.
Inspect the saved public RGB before choosing one. Each run stores three
decision trace files as JSON arrays with 4-space indentation:

- ``onejev_state.json``: the exact state supplied on each OneJev call.
- ``onejev_question.json``: the question, instructions and fully parameterized
  options supplied on that call.
- ``onejev_decision.json``: the original service output, HTTP status, latency,
  selected candidate ID, ``selected_tool_name``, execution result and any error.

The ``turn`` field joins these records within a rollout. Requests are saved
before HTTP; responses are saved before validation and motion, including
invalid responses. PNG images remain in the step artifacts; base64 data is
not duplicated in the traces. These files are standard JSON arrays, so
``json.load`` can read them and ``grep`` can find fields on their own lines.
Older ``onejev_decisions.jsonl`` logs remain untouched. Placement steps save
``onejev_geometry.json`` and ``onejev_regions.png`` for the measurements and
marked regions; drawer steps use the unmarked public RGB views.
``onejev_outcome.json`` records termination and usage.
Native environment completion determines success. Clean process exit and a
grasp heuristic are separate from task success.

The first version supports non-interactive CLI evaluation. Exploration,
Dashboard control, interactive task changes and ``--no-images`` are rejected.
A planner deadline requests toolkit cancellation at existing safe boundaries;
an in-flight simulator or VLA RPC may finish before cancellation takes effect.
