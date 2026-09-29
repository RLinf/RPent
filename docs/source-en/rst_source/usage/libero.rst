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

Cosmos Policy (experimental)
----------------------------------------

``--wam-backend cosmos-policy`` uses NVIDIA's
`Cosmos Policy <https://github.com/NVlabs/cosmos-policy>`_ LIBERO checkpoint
through an external or RPent-owned RPC service. Suite selection uses the shared
LIBERO catalog, including standard suites and Pro ``_task``, ``_swap``, ``_lan``
and ``_object`` variants. The first 120-episode evaluation covered the four
core base families and their ``_task`` / ``_swap`` variants only; catalog
availability does not establish performance on the remaining suites.
Exploration, Flash Mode, plus variants and best-of-N world-model planning
are not supported by this adapter.

Set up the official Cosmos Policy environment using its
`setup guide <https://github.com/NVlabs/cosmos-policy/blob/main/SETUP.md>`_.
The adapter follows upstream revision
``18a2accadf4e7a3531e56754102af5a24d2316da``. Keep that environment separate:
Cosmos pins its own Torch and CUDA extension versions, which differ from
RPent's OpenPI stack.
From the Cosmos Policy checkout, with RPent available at ``/path/to/RPent``,
start the worker using the official CUDA 12.8 / Python 3.10 environment:

.. code-block:: bash

   cd /path/to/cosmos-policy
   uv run --extra cu128 --group libero --python 3.10 \
     --with-editable /path/to/RPent \
     python -m rpent.robots.components.cosmos_policy_server \
     --cuda-device 0 --host 127.0.0.1 --port 8116

The defaults download ``nvidia/Cosmos-Policy-LIBERO-Predict2-2B``, its
dataset statistics and T5 instruction embeddings. For local files, provide
``--checkpoint``; statistics and embeddings default to files beside the weights.
Use ``--dataset-stats`` and ``--text-embeddings`` to override those paths.
Run from the Cosmos checkout so its relative configuration and tokenizer
paths resolve. Before starting, obtain Hugging Face access to
``nvidia/Cosmos-Predict2-2B-Video2World`` and authenticate the worker environment;
its video tokenizer is required even with a local policy checkpoint. The
referenced upstream revision also downloads base Video2World and ALOHA policy
weights when importing experiment configurations. Reserve disk space and network
access for these additional files.

In the RPent environment, install ``.[libero]``, download standard LIBERO
assets with ``libero-download-assets --skip-existing``, and configure SAM3
as below. A Pi0.5 checkpoint is not needed for Cosmos runs:

.. code-block:: bash

   rpent --robot libero --suite libero_spatial --task 0 --seed 0 \
     --wam-backend cosmos-policy --wam-endpoint http://127.0.0.1:8116 \
     --memory-profile local \
     --cuda-device 1 --planner api --model anthropic:claude-opus-4-8

The worker's GPU and RPent's ``--cuda-device`` are independent; choose GPUs
with enough free memory for the policy, simulator and SAM3. They may share a
GPU when memory permits. The external worker remains running after RPent exits.
For a worker in a container or on another host, bind to a reachable interface
and use its reachable address in ``--wam-endpoint``.

Alternatively, let RPent start and stop the worker in an already provisioned
Cosmos environment. Replace ``--wam-endpoint`` with:

.. code-block:: bash

   --wam-checkpoint /path/to/cosmos-checkpoint \
   --wam-python /path/to/cosmos-policy/.venv/bin/python \
   --wam-root /path/to/cosmos-policy

The two connection modes are mutually exclusive. ``COSMOS_POLICY_PYTHON`` and
``COSMOS_POLICY_ROOT`` may supply the latter two defaults. The owned worker
inherits CUDA/cache/library settings and uses RPent's ``--cuda-device`` when
specified; RPent does not install its dependencies or assume a CUDA directory.
Startup failures and session shutdown stop owned workers; external endpoints
remain operator-owned. ``--wam-backend cosmos`` is an alias for ``cosmos-policy``.

``cosmos_act(max_chunks=1)`` replaces ``pi0_pick`` and ``pi0_doubled``. It
uses the environment's full task language and executes 16 native LIBERO
actions per prediction, re-reading observations between chunks. Raw camera
images are vertically flipped once; proprioception preserves the upstream
``[gripper_qpos, eef_pos, eef_quat_xyzw]`` layout. NVIDIA's code owns image
preprocessing, normalization and action unnormalization. RPent records the
executed state/images using the existing LIBERO toolkit. Future video/value
prediction is disabled by default. An owned worker accepts
``--wam-predict-future``; an external worker accepts ``--predict-future``.
``CosmosPolicyClient.predict_result()`` returns those optional outputs, while
``predict()`` supplies actions to the existing control loop. Predicted images
are not environment observations, and these outputs do not enable planning.

Pass a concrete instruction such as
``cosmos_act(prompt="pick up the black bowl", max_chunks=1)`` to execute a
planner-selected subtask. The override applies only to that call; omitting
``prompt`` or passing ``null`` restores the native task instruction. It does not
change the environment task or its success predicate. Inspect the resulting
images for subtask completion; the returned ``success`` still describes the
full task. Subtask effectiveness depends on the checkpoint and requires separate
evaluation.

``max_chunks`` limits complete action chunks, not individual actions. It accepts
1-4 (default 1): one chunk contains 16 actions, so a call executes at most
16-64 actions. Each started chunk runs in full. The tool checks termination and
truncation between chunks and starts no further chunk after either is reported;
it does not stop at a particular action within a chunk. For example, with five
actions left in the episode budget, a started chunk still executes all 16 actions.
The budget can therefore be exceeded by up to 15 actions.

Start with the full task and preserve that instruction while progress is visible;
use a subtask to address an observed failure. Unlike ``pi0_pick``, this tool has
no grasp-completion stopping rule. ``terminated`` and ``success`` record whether
native task success occurred during execution; they do not guarantee that the
final pose still satisfies the goal after the remaining actions in that chunk.
``truncated`` records whether the episode budget was reached. These flags are
accumulated across the executed actions, so both may be true. The tool result
does not establish whether success preceded the budget limit; use the benchmark
runner below for success measurement with an exact action budget.
Calls to ``finish(status="success", ...)`` are refused unless native success
has been recorded. Once an ended episode is reported, further motion tool calls
are refused and the tool result directs the planner to finish with that outcome.

For instructions absent from the supplied embeddings cache, the official worker
loads ``google-t5/t5-11b`` on demand and caches the computed embeddings. Provision
its tokenizer and weights in the worker's Hugging Face cache before offline use,
and allow additional GPU memory and first-request latency for text encoding.
Use a writable local copy of ``--text-embeddings`` because upstream updates it
when new instructions are encoded.
For an owned worker, select that copy with ``--wam-text-embeddings``.
To forbid online T5 loading, use ``--cached-instructions-only`` on an external
worker or ``--wam-cached-instructions-only`` for an owned worker. Cache misses
then fail before inference; this can exclude Pro instructions and subtasks.

Cosmos runs use current observations without Memory. Pass the global
``--memory-profile local`` option to skip automatic HF synchronization in the
CLI and Dashboard; it does not enable a local Cosmos corpus. ``--memory-dir``
and ``--memory-profile hf`` are rejected. The Cosmos toolkit does not expose
``read_text_file``, ``write_text_file`` or ``list_dir``; observations and run
artifacts are still recorded by RPent.

The model client and worker live in ``rpent/robots/components/``. Their
observation format and checkpoint currently support LIBERO only; environment
wiring and ``cosmos_act`` remain in ``robots/libero/``. Start the worker with
``python -m rpent.robots.components.cosmos_policy_server``; the former
``robots/libero/cosmos_policy_server.py`` entry point has been removed.
The CLI and Dashboard distinguish ``--vla-backend pi05`` from
``--wam-backend cosmos-policy``; the Dashboard labels the component VLA or WAM.
Both reuse shared policy prediction and runtime lifecycle code. Model selection
lives in ``robots/libero/policy.py``; task names and automatic environment
routing live in ``robots/libero/suites.py``.
See :ref:`action-model-layers` for the ownership map and the distinction between
shared policy infrastructure, robot tools and benchmark code.

When updating an earlier Cosmos deployment, replace ``--vla-backend`` and
``--vla-endpoint`` with ``--wam-backend`` and ``--wam-endpoint``. Restart the
worker from the same revision as the client: Cosmos uses
``action_model.capabilities`` and ``action_model.predict``. The earlier
``wam.predict`` service is not wire-compatible. The script
``scripts/wam/cosmos_policy_rpc_bridge.py`` delegates to the same component
server; there is only one Cosmos implementation. ``cosmos_act`` is the
planner-facing tool; ``wam_act`` and simultaneous Pi0.5/WAM tools are not exposed.
Pi0.5 continues to use ``vla.predict`` and its existing flags.

For Pro, install ``.[libero-pro]`` and prepare its assets with
``liberopro-download-assets --skip-existing``. Select a full suite name such as
``--suite libero_spatial_task`` or ``--suite libero_goal_swap``. Shared LIBERO
routing selects ``pro`` for perturbation suites and ``standard`` for base suites
when neither ``--libero-type`` nor ``LIBERO_TYPE`` is set. The CLI flag overrides
the environment variable. Perturbation suites require ``pro``; base suites
may explicitly select another installed variant (Cosmos supports standard/pro).
Use a separate ``LIBERO_CONFIG_PATH`` for Pro if the standard
configuration points to a different package. Check that every selected task has
nonempty initial states and that its instruction and goal come from the Pro
BDDL. Some source distributions contain empty initial-state files; obtain the
corresponding official ``zhouxueyang/LIBERO-Pro`` HF dataset files before running.
Do not substitute standard-task initial states for missing Pro states.

To check a running real worker and the bounded policy chain with an offline
planner, install ``.[test,libero]`` and run:

.. code-block:: bash

   RPENT_COSMOS_ENDPOINT=http://127.0.0.1:8116 CUDA_VISIBLE_DEVICES=1 \
     pytest tests/e2e_tests/libero/test_cosmos_policy.py -v

These checks require real LIBERO assets and, for the complete chain, SAM3 and
the T5 encoder weights for the subtask instruction case.
A passing chain verifies action execution and artifacts, not task success.
Set ``RPENT_COSMOS_SUITE=libero_spatial_task`` (or a supported swap suite) and
the Pro resource configuration to run the same checks on Pro.

To measure policy performance separately, run from the RPent checkout with
a running worker and standard LIBERO assets:

.. code-block:: bash

   RPENT_COSMOS_ENDPOINT=http://127.0.0.1:8116 CUDA_VISIBLE_DEVICES=1 \
     python -m tests.e2e_tests.libero.benchmark_cosmos_policy \
     --output-dir /path/to/new-cosmos-results

The runner measures 100 sequential RPC calls after five warm-up calls on a
fixed real observation, then evaluates all ten Spatial tasks with initial
states 0, 1 and 2, capped at 220 policy actions per episode. ``results.json``
contains raw timings, percentiles and every episode outcome, including errors.
RPC timing includes transport and inference, but excludes simulator steps;
each prediction produces 16 actions. Success comes from native simulator
termination. This evaluates the policy without an LLM planner or SAM3.
These 30 episodes are a small integration evaluation, not a reproduction of
the published benchmark. RPent uses RLinf's reset behavior and its installed
LIBERO/robosuite versions; record those versions and the worker's checkpoint,
denoising steps and seed alongside results.

The runner also accepts ``--suite``, ``--tasks`` and ``--horizon``. It uses the
shared suite catalog except ``libero_90``: this runner covers ten-task suites.
Default
action budgets are Spatial 220, Object 280, Goal 300 and Long 520, including
their Pro variants. For example, ``--suite libero_spatial_task --seeds 0
--warmup 0 --samples 0`` evaluates its ten tasks from initial state 0 without
a separate latency probe. It saves initial/final camera images, per-call RPC
times, control-loop time excluding startup, and total episode time including
startup. There is no LLM output, so ``total_output_tokens`` is zero.
Unlike ``cosmos_act``, this runner executes predictions one action at a time,
stopping immediately on native success or the action limit, including within a
16-action prediction. Its success rates and action counts use that stricter
evaluation protocol.

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

For Pi0.5, evaluation remains the default mode. Omitting ``--memory-profile`` selects
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
