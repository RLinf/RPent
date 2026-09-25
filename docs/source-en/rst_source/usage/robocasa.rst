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

   uv pip install -e ".[robocasa]" \
      --constraint robots/robocasa/eval/target50-constraints.txt
   uv pip check

The constraints file pins compatibility-sensitive Target50 dependencies. The ``robocasa`` extra installs the ``rpent`` branches of RoboCasa, RLDX, and Robosuite. Do not also install ``rlinf-robocasa365``, which provides the same import package.

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

Configure your model service with :doc:`configure_planner` and check it using ``rpent-check-llm``. Run ``OpenDrawer`` with seed 1:

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer \
         --split target \
         --seed 1 \
         --vla-model-path ./checkpoints/rldx-1-ft-rc365 \
         --planner claude_code \
         --model claude-opus-4-8

RoboCasa does not select a planner implementation; the ``api``, ``claude_code``, and ``codex`` planners can run this robot. See :doc:`configure_planner` for configuration.

.. _inspect-the-result:

View Results
------------

Task success comes from the environment’s ``_check_success()`` result, exposed as ``state.success``. The planner’s ``finish`` status ends its conversation and is not an evaluation label. Inspect ``result.json``, ``transcript_*.json``, and ``run.log`` in the output directory; service startup errors are in ``env_server.log`` and ``vla_server.log``.

Use ``--dashboard`` to watch cameras and planner output; see :doc:`dashboard` for the shared workflow.

Task Memory
-----------

Select automatic synchronization with ``--memory-profile hf`` (the evaluation default).
Before a run using this profile, RPent's shared memory manager synchronizes the
``robocasa/**`` subtree from the
`RLinf/RPent-memory dataset
<https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/robocasa/results>`_
into ``memory/robocasa``. An online ordinary run therefore requires no separate
memory download. The current task may use only these task-matched files under
``results/``:

.. code-block:: text

   memory/robocasa/results/<Task>_s0.json
   memory/robocasa/results/recipe_<Task>_s0.jsonl
   memory/robocasa/results/<Task>.md  # optional

The final published corpus contains 43 audit JSON files, 43 recipe JSONL files,
and 25 task Markdown files, for 111 files in total and no global memory. The
JSON/JSONL pair contains reviewed seed-0 evidence. The optional Markdown file
contains task-specific exploration memory and may summarize multiple attempts;
all 16 Composite-Seen and 9 Composite-Unseen tasks provide one. The prompt
requires the planner to read every current-task file that exists before acting.
RPent makes those files available through ``read_text_file`` but does not inject
their contents into the prompt.

When using the published HF memory above, the planner uses neither global
memory nor another task's memory.
Seven Composite-Unseen tasks have no task memory and remain in the evaluation:
``HeatKebabSandwich``, ``PanTransfer``, ``PortionHotDogs``,
``SeparateFreezerRack``, ``WaffleReheat``, ``WashFruitColander``, and
``WeighIngredients``. They continue from live observations. Memory is strategy
evidence; historical coordinates, poses, pixels, and subtask prompts must not
replace current localization or the live task language.

.. _exploration:

Exploration Mode
----------------

Add ``--explore`` to let the planner retry a task across fresh episodes and
write local memory. As with LIBERO, one run allows up to three planner sessions
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

.. _reproduce-target50:

Experiment Reproduction (Target50)
----------------------------------

``robots/robocasa/eval/target50.json`` is the canonical manifest for reproducing
Harness VLA on RoboCasa Target50. It freezes the ``target`` environment split,
HF resource revisions, memory scope, task and seed matrix, cell time limits,
success source, and retry policy. Each cell is one task/seed combination. Its protocol ID is
``robocasa-harness-vla-v1``. Source dependencies follow the recorded ``rpent``
branches and must be recorded at their resolved revisions for each run:

.. list-table:: RoboCasa Target50 matrix
   :header-rows: 1
   :widths: 30 15 20 20 15

   * - Split
     - Tasks
     - Seed range per task
     - Cell timeout
     - Cells
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
- **Composite seen (16)** — multi-step tasks on kitchen layouts seen
  during training: ``ScrubCuttingBoard``, ``StackBowlsCabinet``,
  ``WashLettuce``, ``RinseSinkBasin``, ``PreSoakPan``,
  ``StirVegetables``, ``LoadDishwasher``, ``SteamInMicrowave``,
  ``SetUpCuttingStation``, ``GetToastedBread``, ``DeliverStraw``,
  ``KettleBoiling``, ``PrepareCoffee``, ``StoreLeftoversInBowl``,
  ``SearingMeat``, ``PackIdenticalLunches``.
- **Composite unseen (16)** — multi-step tasks on layouts *not* seen
  during training (generalization eval): ``ArrangeBreadBasket``,
  ``ArrangeTea``, ``BreadSelection``, ``CategorizeCondiments``,
  ``CuttingToolSelection``, ``GarnishPancake``, ``GatherTableware``,
  ``HeatKebabSandwich``, ``MakeIceLemonade``, ``PanTransfer``,
  ``PortionHotDogs``, ``RecycleBottlesByType``,
  ``SeparateFreezerRack``, ``WaffleReheat``, ``WashFruitColander``,
  ``WeighIngredients``.

Pass any of these to ``--task-name``. The full RoboCasa catalog is
larger; see the `RoboCasa <https://robocasa.ai>`_ upstream.

Ordinary runs with the HF profile synchronize Hugging Face ``main``. Formal Target50 runs use the
immutable memory snapshot
``551fc3157b3e56b40a3d3a3b4c7ff81721ebe89b``:

.. code-block:: bash

   hf download RLinf/RPent-memory \
      --repo-type dataset \
      --revision 551fc3157b3e56b40a3d3a3b4c7ff81721ebe89b \
      --include "robocasa/**" \
      --local-dir ./target50-memory

For Target50, first download the fixed resources above, then invoke one ordinary
command for each manifest cell. The Codex reference profile is ``gpt-5.5``,
``xhigh``, and ``max_turns=100``; RoboCasa itself remains planner-agnostic. For
the scene identity, use the ordinary ``--seed`` argument and do not set
``RLDX_RESET_SEED``. Ordinary RoboCasa uses ``max_chunks=70``; Target50 alone
overrides it to 40. Freeze the Target50 RLDX execution values first:

.. code-block:: bash

   export RLDX_MAX_CHUNKS=40
   export RLDX_SETTLE_PATIENCE=999
   export RLDX_ACTION_STEPS_PER_CHUNK=8
   unset RLDX_RESET_SEED

The first ``OpenDrawer`` Atomic cell is:

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer --split target --seed 1 \
         --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 \
         --planner codex --model gpt-5.5 --reasoning-effort xhigh \
         --max-turns 100 --planner-timeout-s 1800 \
         --memory-profile local \
         --memory-dir ./target50-memory/robocasa \
         --output-dir ./runs/target50/atomic/OpenDrawer_s1

Use ``--planner-timeout-s 3600`` for either composite split. Execute Atomic,
Composite-Seen, and Composite-Unseen in that order. A cell succeeds only when
the final recorded environment state has ``state.success=true``; the planner's
``finish(status=...)`` argument is not an evaluation label. Valid task failures
and planner timeouts are not retried. Retry an infrastructure failure only when
no valid environment result was produced for that cell.

Every completed command atomically writes ``<output-dir>/result.json`` using
the final environment ``state.success``. The record includes the effective
protocol values but omits provider errors and credentials. Once all cells are
present under ``<results-root>/<manifest-split>/<Task>_s<seed>/result.json``,
validate the fixed denominator and print the task-weighted score with:

.. code-block:: bash

   python -m robots.robocasa.eval.validate_target50 ./runs/target50


Published Target50 Results
--------------------------

The published Codex reproduction contains all 340 cells and reports the
following task-level aggregates:

.. list-table:: Codex Target50 reproduction
   :header-rows: 1
   :widths: 30 20 20 30

   * - Split
     - Successful cells
     - Success rate
     - Harness VLA reference
   * - Atomic
     - 163/180
     - 90.56%
     - 165/180 (91.67%)
   * - Composite-Seen
     - 49/80
     - 61.25%
     - 45/80 (56.25%)
   * - Composite-Unseen
     - 12/80
     - 15.00%
     - 11/80 (13.75%)
   * - Overall (task-weighted)
     - N/A
     - 57.00%
     - 55.40%

The `complete per-task table
<https://github.com/RLinf/RPent/blob/main/robots/robocasa/eval/target50_codex_results.md>`_
contains the success count and accuracy for every task. The published record is
task-level aggregate data; it does not include per-seed traces, raw trajectories,
or failure classifications and therefore is not a per-cell audit artifact.

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
  ``memory/robocasa/results/`` corpus or the selected local directory.
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
