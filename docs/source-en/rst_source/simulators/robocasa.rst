RoboCasa365
============

.. figure:: https://raw.githubusercontent.com/robocasa/robocasa/main/docs/images/readme.webp
   :alt: RoboCasa365 environment overview
   :width: 90%
   :align: center

   Kitchen scenes, objects, and tasks in RoboCasa365. Source: `RoboCasa365 project <https://github.com/robocasa/robocasa>`_.

Run kitchen manipulation tasks with RPent in RoboCasa365, then reproduce Target50 experiments. This integration uses the PandaOmron mobile manipulator and the RLDX-1 action model; its CLI name is ``robocasa``.

.. _robocasa-overview:

Overview
------------

Check the model, task, and runtime requirements before following the installation and run steps.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      RLDX-1

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``

   .. grid-item-card:: Tasks

      Target50 kitchen tasks

   .. grid-item-card:: Hardware

      Linux, NVIDIA GPU; Python 3.10; CUDA and EGL.

Tasks
~~~~~~~~~~~~

Target50 covers the following categories. The reproduction section lists the complete task/seed matrix and run limits.

.. list-table::
   :header-rows: 1

   * - Category
     - Tasks
     - Scope
   * - Atomic
     - 18
     - Individual kitchen operations.
   * - Composite-Seen
     - 16
     - Composite tasks in seen categories.
   * - Composite-Unseen
     - 16
     - Composite tasks in unseen categories.

.. _robocasa-observation-action:

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The table distinguishes planner tools, model inputs, and the environment’s success criterion.

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - Camera views, depth/world coordinates, and mobile-base, end-effector, and gripper state. RLDX-1 receives three camera histories plus state and task text.
   * - Action
     - The planner calls ``rldx_skill`` and motion primitives. RLDX-1 predicts end-effector, gripper, base-motion, and control-mode commands.
   * - Reward / success
     - Evaluate success using the environment’s ``_check_success()`` result exposed as ``state.success``.
   * - Task prompt
     - Use the current environment’s complete ``task_language``; historical memory does not replace it.

Installation and Resources
--------------------------

Use Linux, an NVIDIA GPU, a working CUDA/EGL setup, ``git``, and `uv <https://docs.astral.sh/uv/getting-started/installation/>`_. If you already have the repository, enter it and start at environment creation.

RLDX-1 requires Python ``3.10``. Create a dedicated environment and install
the complete RoboCasa365 stack with ``.[robocasa]``:

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   uv venv --python 3.10 .venv-robocasa
   source .venv-robocasa/bin/activate

First install a matching CUDA-enabled PyTorch and torchvision pair using the
`PyTorch installation selector <https://pytorch.org/get-started/locally/>`_
for your GPU, driver and Python version. Run the selected command in this
environment (use ``uv pip`` in place of ``pip``). The RLDX dependency requires
Torch >= 2.7 and torchvision >= 0.22; choose a mutually compatible pair, not
two independent versions. Then install RPent:

.. code-block:: bash

   uv pip install -e ".[robocasa]"
   uv pip check

Current installation requirements come from ``pyproject.toml``.
The ``robocasa`` extra installs the ``rpent`` branches of RoboCasa, RLDX, and
Robosuite. Do not also install ``rlinf-robocasa365``, which provides the same
import package.

RLDX must come from the extra's ``rpent`` source branch. Older PyPI wheels can
have the same package version but lack interfaces required by this runtime.
When reusing an environment with an older RLDX wheel, add
``--reinstall-package rlinf-rldx`` to the installation command above.

Choose Torch, torchvision, and CUDA for your machine. The reference run used Torch 2.7.0, torchvision 0.22.0, and CUDA 12.6. Record resolved dependencies and Git revisions for every reproduction because branches can advance:

.. code-block:: bash

   uv pip freeze > installed-requirements.txt
   git rev-parse HEAD > rpent-revision.txt

``flash-attn`` is optional; RLDX-1 uses PyTorch SDPA when it is absent. If needed, follow the `FlashAttention instructions <https://github.com/Dao-AILab/flash-attention#installation-and-features>`_ to select a compatible build.

**Post-install setup**

Download the kitchen assets (~10 GB) outside ``site-packages`` so they survive
reinstalls. Target50 does not use RoboCasa dataset or teleop macros, so skip
the optional private-macros setup:

.. code-block:: bash

   robocasa-download-assets --assets-path ~/.robocasa/assets --no-macros -y

It prints the environment variable to export afterwards; add it to the shell
that launches ``rpent``:

.. code-block:: bash

   export ROBOCASA_ASSETS_PATH=~/.robocasa/assets

The installer supplies downloaded collections and bundled scene files. Add ``--skip-existing`` on subsequent runs to check existing downloads. See troubleshooting below for resource conflicts, disk space, and camera errors.

**RLDX-1 checkpoint**

The ``--vla-model-path`` flag on the run commands below expects a
local path to the ``RLDX-1-FT-RC365`` checkpoint (the RoboCasa365
fine-tune). Download it from HuggingFace:

.. code-block:: bash

   hf download RLWRLD/RLDX-1-FT-RC365 \
      --revision 587e9ecdcc5e7184fcc17f58713908edff5af041 \
      --local-dir ./checkpoints/rldx-1-ft-rc365

If the download is slow, use the HF mirror:

.. code-block:: bash

   HF_ENDPOINT=https://hf-mirror.com hf download RLWRLD/RLDX-1-FT-RC365 \
      --revision 587e9ecdcc5e7184fcc17f58713908edff5af041 \
      --local-dir ./checkpoints/rldx-1-ft-rc365

**RLDX-1 backbone support files**

The FT checkpoint contains the weights but also references
``RLWRLD/RLDX-1-VLM`` for architecture, processor and tokenizer metadata.
Target50 freezes revision ``4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7``:
15 non-weight files, about 16.4 MB including documentation and images.
Download these into the same cache used when launching RPent:

.. code-block:: bash

   export HF_HOME="$PWD/.cache/huggingface"
   export HF_HUB_CACHE="$HF_HOME/hub"
   hf download RLWRLD/RLDX-1-VLM \
      --revision 4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7 \
      --include "*.json" "*.txt" "*.jinja" "*.md" "*.png" ".gitattributes" \
      --exclude "*.safetensors.index.json"

No additional base weights are required. Keep these cache variables in the
launch shell; do not shadow them with an empty ``TRANSFORMERS_CACHE``.
The RoboCasa VLA worker automatically uses this same support revision for
both ordinary and Target50 runs, including separately started RPent VLA
servers. There is no extra revision flag or manual cache-ref edit. The pin
applies only to backbone metadata, not the weights selected by
``--vla-model-path``. Model and asset licenses apply separately from RPent's
code license.

Run a Task
----------

Configure your model service with :doc:`../guides/configure_planner` and check it using ``rpent-check-llm``. Run ``OpenDrawer`` with seed 1:

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer \
         --split target \
         --seed 1 \
         --vla-model-path ./checkpoints/rldx-1-ft-rc365 \
         --planner claude_code \
         --model claude-opus-4-8

RoboCasa does not select a planner implementation; the ``api``, ``claude_code``, and ``codex`` planners can run this robot. See :doc:`../guides/configure_planner` for configuration.

.. _inspect-the-result:

View Results
------------

Task success comes from the environment’s ``_check_success()`` result, exposed as ``state.success``. The planner’s ``finish`` status ends its conversation and is not an evaluation label. Inspect ``result.json``, ``transcript_*.json``, and ``run.log`` in the output directory; service startup errors are in ``env_server.log`` and ``vla_server.log``.

Use ``--dashboard`` to watch cameras and planner output; see :doc:`../guides/dashboard` for the shared workflow.

Task Memory
-----------

By default, CLI and Dashboard download memory from
`HF main <https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/robocasa>`_
and provide the current task's memory (task-specific) and general guidance
(global). The planner reads as needed, giving priority to live observations
and task instructions.

.. code-block:: text

   memory/robocasa/
   ├── task-specific/
   └── global/

Global memory is required. Task JSON and recipe must both exist or both be
absent; task Markdown is optional. RPent file tools expose only the memory
allowed for the current task.

.. _custom-memory-sources:

To evaluate with a fixed local copy, use
``--memory-profile local --memory-dir <directory>``; download and run commands
are in :ref:`reproduce-target50`. Expand the details below for filenames,
local exploration output and custom sources.

.. dropdown:: Memory File Selection and Custom Sources

   .. _robocasa-memory-selection:

   HF mode uses ``robocasa/`` from the dataset's current main branch. It provides
   ``<Task>_s0.json``, ``<Task>_s0_recipe.jsonl``, optional ``<Task>.md`` and
   ``global/GLOBAL_MEMORY.md``. The evaluation ``--seed`` changes the scene;
   reference memory still uses ``_s0``.

   Local evaluation uses ``--memory-profile local --memory-dir <directory>`` and
   supports two task file conventions:

   - Published corpus: ``task-specific/<Task>_s0.json`` and ``<Task>_s0_recipe.jsonl``.
   - Exploration output: ``task-specific/<Task>_<split>_s0.json`` and ``<Task>_<split>_s0_recipe.jsonl``.

   JSON and recipe must both exist or both be absent. If absent, the planner uses
   global guidance and live observations. If both conventions exist for the same
   task, separate them with different ``--memory-dir`` directories. ``<Task>.md``
   is optional; its absence is logged.

   Local evaluation also exposes ``global/*.md`` and ``task-family/*.md`` whose
   YAML frontmatter matches all of ``suite: robocasa``, ``regime: <split>`` and
   ``task_id: <Task>``. At least one readable global file is required.

   Prompts and RPent file tools share the selected file list. The tools deny
   other tasks, other splits, the root ``MEMORY.md`` index and ``_internal/``;
   this restriction applies only to RPent tools. CLI checks memory before starting
   services. Dashboard checks the directory and global layer before starting its
   shared VLA, then task files before starting each environment. A task-file error
   leaves the shared VLA running.

   The planner reads memory as needed. Live task language, RGB-D, progress and
   tool results take precedence. Continue VLA calls while contact, a held object
   or visible progress persists; after two consecutive calls without contact or
   progress, re-ground and make a bounded pose adjustment. Each call uses the
   full live task language. Historical ``vla_act`` describes strategy; historical
   coordinates must not be replayed.

   Each run starts a fresh record of selected files, missing layers and actual
   reads, even when reusing an output directory. Zero or partial reads can produce
   valid environment results; missing or corrupt audits are reported separately.
   Results record the profile and task-family identity for boundary checks.

   .. rubric:: Custom Memory Sources

   For another HF dataset with the same layout, set
   ``RPENT_MEMORY_HF_REPO=<owner>/<dataset>`` and use ``--memory-profile hf``.
   The variable takes a repository ID.

   For a custom subtree or branch, download into a fresh directory and use local
   mode. Add ``--revision <branch>`` to select a branch:

   .. code-block:: bash

      hf download <owner>/<dataset> --repo-type dataset \
         --include '<subpath>/**' --local-dir ./custom-memory

      # Add to the RPent run command:
      # --memory-profile local --memory-dir ./custom-memory/<subpath>

   The selected directory should contain ``task-specific/`` and ``global/`` and
   meet the file-selection rules above.

.. _exploration:

Exploration Mode
----------------

Add ``--explore`` to let the planner retry a task across fresh episodes and
write local memory. The directory may start empty; exploration can read its
index and write to its own inbox. As with LIBERO, one run allows up to three planner sessions
with at most five attempts per session by default:

.. code-block:: bash

   rpent --robot robocasa --task-name OpenDrawer --split target --seed 0 \
     --vla-model-path /path/to/rldx \
     --planner codex --reasoning-effort high --planner-timeout-s 7200 \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir /path/to/robocasa-memory

``reset`` uses the environment's ordinary episode reset. The runner exports
only the winning commands after the final reset. Exploration memory is written
to the current local inbox. Drafts are merged when the run completes normally
without an agent execution error. Pass ``--no-auto-merge-memory`` to disable
automatic merging. The exploration prompt is in
``robots/robocasa/prompts/explore.py`` and covers mobile-base use,
``task_progress``, RLDX continuity, and failed-attempt notes.

Shared memory merge publishes accepted proposals under ``task-family/`` and
``global/``, copies the successful audit/recipe pair into ``task-specific/``,
and refreshes ``MEMORY.md``. To evaluate these native files, use seed-0
exploration output and ``--memory-profile local`` with the same directory.
Evaluation requires global memory to have been published first and uses its
own task/split access boundary; exploration keeps its retry and inbox workflow.

.. _reproduce-target50:

Experiment Reproduction (Target50)
----------------------------------

Target50 covers 50 kitchen tasks and 340 runs. The example below uses Codex,
GPT-5.5 and xhigh with task-specific and global memory.

.. _run-and-check-results:

1. Prepare Memory and Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete the installation, resource downloads and Codex login above. Download
memory into a fresh directory and keep it unchanged throughout the evaluation:

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
      --include 'robocasa/**' --local-dir ./target50-memory

Set the action parameters and clear any legacy environment override for ``--seed``:

.. code-block:: bash

   export RLDX_MAX_CHUNKS=40
   export RLDX_SETTLE_PATIENCE=999
   export RLDX_ACTION_STEPS_PER_CHUNK=8
   unset RLDX_RESET_SEED

2. Run Each Task and Seed
~~~~~~~~~~~~~~~~~~~~~~~~~

Start with ``OpenDrawer`` at seed 1:

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer --split target --seed 1 \
         --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 \
         --planner codex --model gpt-5.5 --reasoning-effort xhigh \
         --max-turns 100 --planner-timeout-s 1800 \
         --memory-profile local \
         --memory-dir ./target50-memory/robocasa \
         --output-dir ./runs/target50/atomic/OpenDrawer_s1

Repeat for every task and seed in the matrix below. Task names are listed in
:ref:`Full Task List <robocasa-task-list>`; the manifest is ``robots/robocasa/eval/target50.json``.

.. list-table:: RoboCasa Target50 matrix
   :header-rows: 1
   :widths: 30 15 20 20 15

   * - Split
     - Tasks
     - Seed range per task
     - Run timeout
     - Runs
   * - Atomic
     - 18
     - 1--10
     - 1800 s
     - 180
   * - Composite-Seen
     - 16
     - 1--5
     - 3600 s
     - 80
   * - Composite-Unseen
     - 16
     - 1--5
     - 3600 s
     - 80
   * - **Total**
     - **50**
     -
     -
     - **340**

Run Atomic, then Composite-Seen, then Composite-Unseen. For both composite
groups, use ``--planner-timeout-s 3600`` and output directories
``composite_seen/<Task>_s<seed>`` or ``composite_unseen/<Task>_s<seed>`` under
``./runs/target50/``.

Do not reset the environment during evaluation. Keep task failures and planner
timeouts as results; retry infrastructure failures only when they produced no
valid environment result.

3. Check Results
~~~~~~~~~~~~~~~~~

Each run saves ``result.json`` in its output directory. Success comes from the
environment's ``state.success``. Check and summarize the results with:

.. code-block:: bash

   python -m robots.robocasa.eval.validate_target50 ./runs/target50

A complete evaluation reports ``valid_cells=340`` and ``expected_cells=340``,
with no validation errors and exit code 0. Partial evaluations list missing
results and return a nonzero exit code. Overall success weights all 50 tasks
equally.

.. _task-matrix:

.. _evaluation-protocol:

.. _leaderboard-and-historical-compatibility:

.. _reported-and-historical-target50-results:

.. _historical-codex-reproduction:

Published scores are on the :doc:`leaderboard <../leaderboard/performance>`.
Expand the details below for task names, evaluation settings and published scores.

.. dropdown:: Full Task List

   .. _robocasa-task-list:

   Seen/unseen describes whether tasks occur in the pretraining data; target kitchens
   form a separate held-out scene split. See the `RoboCasa dataset definitions
   <https://robocasa.ai/docs/build/html/datasets/datasets_overview.html>`_.
   The tasks split into three groups:

   - **Atomic (18)** — single-primitive articulation and pick-place
     tasks: ``CloseBlenderLid``, ``CloseFridge``,
     ``CloseToasterOvenDoor``, ``CoffeeSetupMug``, ``NavigateKitchen``,
     ``OpenCabinet``, ``OpenDrawer``, ``OpenStandMixerHead``,
     ``PickPlaceCounterToCabinet``, ``PickPlaceCounterToStove``,
     ``PickPlaceDrawerToCounter``, ``PickPlaceSinkToCounter``,
     ``PickPlaceToasterToCounter``, ``SlideDishwasherRack``,
     ``TurnOffStove``, ``TurnOnElectricKettle``, ``TurnOnMicrowave``,
     ``TurnOnSinkFaucet``.
   - **Composite seen (16)** — multi-step tasks represented in the pretraining data: ``ScrubCuttingBoard``, ``StackBowlsCabinet``,
     ``WashLettuce``, ``RinseSinkBasin``, ``PreSoakPan``,
     ``StirVegetables``, ``LoadDishwasher``, ``SteamInMicrowave``,
     ``SetUpCuttingStation``, ``GetToastedBread``, ``DeliverStraw``,
     ``KettleBoiling``, ``PrepareCoffee``, ``StoreLeftoversInBowl``,
     ``SearingMeat``, ``PackIdenticalLunches``.
   - **Composite unseen (16)** — multi-step tasks absent from the pretraining data: ``ArrangeBreadBasket``,
     ``ArrangeTea``, ``BreadSelection``, ``CategorizeCondiments``,
     ``CuttingToolSelection``, ``GarnishPancake``, ``GatherTableware``,
     ``HeatKebabSandwich``, ``MakeIceLemonade``, ``PanTransfer``,
     ``PortionHotDogs``, ``RecycleBottlesByType``,
     ``SeparateFreezerRack``, ``WaffleReheat``, ``WashFruitColander``,
     ``WeighIngredients``.

   Pass any of these to ``--task-name``. The full RoboCasa catalog is
   larger; see the `RoboCasa <https://robocasa.ai>`_ upstream.

.. dropdown:: Action Settings

   .. _robocasa-action-settings:

   The three Target50 settings apply to each RLDX tool call:

   .. list-table:: Target50 RLDX settings
      :header-rows: 1
      :widths: 42 12 46

      * - Environment variable
        - Value
        - Meaning
      * - ``RLDX_MAX_CHUNKS``
        - 40
        - Maximum prediction chunks per call; ordinary RoboCasa uses 70.
      * - ``RLDX_SETTLE_PATIENCE``
        - 999
        - Consecutive chunks with little end-effector and gripper motion before stopping as settled. This exceeds the 40-chunk cap.
      * - ``RLDX_ACTION_STEPS_PER_CHUNK``
        - 8
        - Actions executed from each prediction chunk.

.. dropdown:: Evaluation Protocol

   .. _robocasa-protocol-history:
   .. _robocasa-evaluation-protocol:

   The current evaluation uses task-specific and global memory together.
   ``target50.json`` uses protocol ``robocasa-harness-vla-v2`` and
   result schema ``1.1``, with GPT-5.5 as the reference configuration. Installation
   dependencies come from ``pyproject.toml``. For custom tasks, seeds or planner
   settings, pass a manifest using the current protocol:

   .. code-block:: bash

      python -m robots.robocasa.eval.validate_target50 /path/to/results \
         --manifest /path/to/manifest.json

   For comparisons, keep a copy of the memory and record its HF commit or file
   hashes locally, alongside the resolved commits of source dependencies on their
   ``rpent`` branches. Memory is not version-pinned; the validator does not compare
   its contents across runs.

   Leaderboard scores come from independent experiment reports. Completing 340
   runs with the current protocol does not establish reproduction of a leaderboard
   entry; compare the model, memory and code configuration too.

.. dropdown:: Published Target50 Scores

   .. _robocasa-reported-results:

   The :doc:`leaderboard <../leaderboard/performance>` is the source for the
   reported rates below. The RPent configurations are Codex / GPT-5.5 / xhigh /
   reasoning, Codex / GPT-6 Astra / low / reasoning, and Claude Code /
   Opus-4.7 / max.reasoning. The Harness VLA reference column reports GPT-5.5
   results from `paper Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_.
   Overall weights all 50 tasks equally; it is not the fraction of successful
   cells among 340.

   .. list-table:: Reported Target50 success rates
      :header-rows: 1
      :widths: 24 18 18 18 22

      * - Split
        - RPent / GPT-5.5
        - RPent / GPT-6 Astra
        - RPent / Opus-4.7
        - Harness VLA / GPT-5.5 reference
      * - Atomic-Seen
        - 92.0%
        - 87.78%
        - 79.4%
        - 92.0%
      * - Composite-Seen
        - 61.0%
        - 43.75%
        - 47.5%
        - 61.0%
      * - Composite-Unseen
        - 13.8%
        - 42.50%
        - 15.0%
        - 13.8%
      * - Overall (task-weighted)
        - 57.1%
        - 59.20%
        - 48.6%
        - 57.1%

   The Astra entry reports **59.20% Overall**, with **87.78% / 43.75% / 42.50%**
   for the three splits. Its episode count is **340 (180/80/80)** following the
   `contributor-confirmed correction
   <https://github.com/RLinf/RPent/pull/205#issuecomment-5749514622>`_. The earlier
   250-cell information was an unsynchronized historical record. The correction
   preserves reported rates; it does not infer success counts from rounded rates
   or claim a new audit of all 340 original results.

.. _environment-smoke-tests:

Environment Checks
------------------

After installing RoboCasa and its assets, run the opt-in environment smoke suite
to check simulator installation and interfaces. It requires no planner credentials
or VLA checkpoint:

.. code-block:: bash

   uv pip install pytest pytest-timeout
   RPENT_RUN_ROBOCASA_INTEGRATION=1 \
      pytest tests/integration_tests/robots/robocasa/test_target50_runtime_smoke.py -v

The four cases cover ``OpenDrawer``, ``NavigateKitchen``, and
``PickPlaceCounterToCabinet`` at seed 1, plus mobile-camera movement. Task checks
verify construction/reset, 12D actions, operation cameras, navigation RGB-D/world
map, the success predicate, and clean close. The camera check verifies pose and
image changes after eight base steps. These real-simulator tests require a working
GPU/EGL setup and are separate from offline CPU CI; skipped tests are not passes.

Troubleshooting
---------------

The asset root must contain downloaded collections and bundled scene, arena, and fixture files. Preserve attribution files. ``--skip-existing`` checks download inventories; for conflicting files, confirm that replacement is intended before using:

.. code-block:: bash

   robocasa-download-assets --assets-path ~/.robocasa/assets --no-macros --overwrite -y

``--overwrite`` takes precedence over ``--skip-existing`` and replaces only files in the installation scope. By default, conflicting files are preserved and the destination must support hard links. Allow space for ZIP archives and unpacked data, plus existing data during replacement. Rerun the download after an interruption.

Run the :ref:`environment smoke tests <environment-smoke-tests>` first. After
downloading all four resources, use the existing RoboCasa E2E component test
to verify VLA worker startup, HTTP RPC and first inference. Install ``.[test]``
if needed, select one available GPU and use a fresh output directory:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 \
   RLDX_MODEL_PATH="$PWD/checkpoints/rldx-1-ft-rc365" \
   RPENT_E2E_OUTPUT_DIR="$PWD/e2e-robocasa" \
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
   MUJOCO_GL=egl python -m pytest -q \
      tests/e2e_tests/robocasa/test_components.py::test_rldx_component --timeout=300

These checks do not run a planner or create benchmark results. Skipped tests
are not passes. Offline variables apply only to the check; ordinary HF memory
sync needs network access. Keep remote planner proxies unchanged.
The lightweight protocol tests still validate all 50 tasks and the fixed
340-cell denominator; full benchmark execution is a separate procedure.

- For slow package downloads, use ``UV_HTTP_TIMEOUT=600`` and put caches and
  temporary files on a sufficiently large filesystem. Retry the pinned HF
  download; apparent shard size is not a completeness check. Do not disable TLS.
- A read-only asset failure needs the corrected RoboCasa dependency, not
  writable canonical assets. Transformed XML uses the temporary directory.
- For an RLDX offline cache miss, check the support snapshot and cache variables
  above. ``NO_ALBUMENTATIONS_UPDATE=1`` disables only an import-time version
  check, not image processing. Keep the existing image-geometry fallback.
- Test the selected Torch/CUDA build with a GPU operation and EGL render,
  not just the driver's version display. Use a build compatible with the host.
- For shared read-only installations, set ``NUMBA_CACHE_DIR`` to a writable
  per-user directory instead of making package code writable.

- If navigation RGB-D or world-map rendering reports a missing
  ``mobilebase0_navview``, reinstall ``.[robocasa]`` to refresh the
  ``RLinf/robosuite`` ``rpent`` branch. Do not patch installed XML files
  manually.
- If ``read_text_file`` reports a missing current-task result, check the
  ``memory/robocasa/task-specific/`` corpus or the selected local directory.
  RPent does not fall back to another task's memory.
  Markdown is optional; Atomic tasks have no published ``<Task>.md``.
- Environment and VLA startup failures are recorded in
  ``<output_dir>/env_server.log`` and ``<output_dir>/vla_server.log``; also
  inspect ``<output_dir>/run.log`` for the run-level error.
- Only the exact ``127.0.0.1`` and ``localhost`` hostnames bypass HTTP proxies
  automatically. Other hostnames and IPs use the standard proxy environment;
  add the exact host to ``NO_PROXY`` and ``no_proxy`` only when it should be
  reached directly.

Implementation Notes
--------------------

The RoboCasa toolkit exposes the same *shape* of tools as LIBERO (a
primitive call, a state view, a ``finish``), with two RoboCasa-specific
aspects:

- **Env-side helpers.** Grasp checks and action assembly need the live
  simulator env, so they live in ``env_server`` as RPCs. The agent-side
  skill holds **both** clients: the env client for render/step, the
  model client for RLDX-1 inference. See
  :doc:`../development/add_robot` for the rationale.
- **Observation shape.** RLDX-1 sees 3 camera video tensors
  ``(1, T, H, W, 3)`` stacked over history ``T``, plus ``state.*``
  and ``annotation.*`` fields. The session id is **not** part of the
  observation — it is managed automatically by the RPC framework:
  ``RpcClient`` generates a private ``rpc_`` + uuid hex session id,
  ``wait_for_ready`` registers it with the server on connect; the
  server tracks each session's idle time and a background sweep thread
  reaps sessions idle longer than the timeout (default 3600s), and the
  client sends ``session.close`` via atexit on process exit. Business
  code (``rldx_skill`` / ``vla_client``) never sees the session id
  directly; the server injects it into ``predict`` / ``reset_session``
  to isolate per-client RLDX memory/RTC policy state.
