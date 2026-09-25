RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ is the kitchen-scale, long-horizon
manipulation environment. In RPent it is driven by the **RLDX-1** VLA
policy, served over HTTP RPC by default (matching LIBERO); a
pickle-framed socket transport is also supported. See
``robots/robocasa/vla_server.py`` and ``robots/robocasa/robot_spec.py``
for the wire/transport selection.

.. note::

   The task/global Target50 protocol is in
   ``robots/robocasa/eval/target50_v2.json``; ``target50.json`` retains v1. RPent uses ordinary single-task
   ``rpent --robot robocasa`` commands for its 340 cells.

Runtime flow
------------

RoboCasa365 uses a PandaOmron mobile manipulator and the frozen RLDX-1 policy.
The integration is planner-agnostic: API planners, Claude Code and Codex use
the same RoboCasa toolkit. See :doc:`configure_planner` for credentials and
backend configuration; users supply credentials outside the repository.

.. code-block:: text

   rpent CLI -> task-memory sync -> environment and VLA servers
             -> planner toolkit -> final environment state.success

Unless external endpoints are supplied, RPent starts an environment server
and a VLA server for each run. The planner selects primitives using the live
task language and observations; RLDX-1 executes manipulation skills. Only the
environment's own ``_check_success()`` result, surfaced as ``state.success``,
determines evaluation success. The public protocol uses ordinary single-cell
commands, not an included batch launcher. See :doc:`../awesome_works/harnessvla`
for the Harness VLA overview.

Installation
------------

RLDX-1 requires Python ``3.10``. Create a dedicated environment and install
the complete RoboCasa365 stack with ``.[robocasa]``:

.. code-block:: bash

   uv venv --python 3.10
   source .venv/bin/activate

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

The RoboCasa-specific constraints file pins the compatibility-sensitive
package versions validated for Target50 reproduction without narrowing
RPent's shared LIBERO or RoboTwin dependencies. The ``robocasa`` extra tracks
the maintained ``rpent`` branches of RoboCasa, RLDX and Robosuite for both
ordinary runs and Target50. The manifest records these branches, not frozen
source commits. RoboCasa's ``rpent`` branch declares the distribution name
``rpent-robocasa365``; do not co-install
the ``rlinf-robocasa365`` distribution, which provides the same import package.
No second source-install step is required. Branches can advance, so record
the resolved Git commits and installed versions with each evaluation:

.. code-block:: bash

   uv pip freeze > installed-requirements.txt

Keep this environment record with the experiment artifacts. Installing the
same branch later is not a guarantee of identical source code. The checkpoint,
backbone support resources remain fixed below; task/global memory follows the selected branch.
The constraints do not pin Torch, torchvision or a CUDA backend. Installation
retains a compatible installed pair; dependency conflicts must be
resolved before running. The manifest's ``reference_accelerator`` records the
previously used Torch 2.7.0 / torchvision 0.22.0 / CUDA 12.6 combination as
provenance only, not an installation requirement. Record your actual versions
with your results and run the component checks below; other combinations are
not presumed to have identical numerical results. Package mirrors are optional
user configuration, not part of the evaluation protocol.

.. note::

   flash-attn is optional; RLDX-1 uses PyTorch SDPA when it is absent.
   If installing it, follow the `FlashAttention installation guidance
   <https://github.com/Dao-AILab/flash-attention#installation-and-features>`_
   and select a build compatible with your Python, Torch, CUDA and GPU.
   This guide does not prescribe a machine-specific wheel.

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

The external root requires the six downloaded collections and the bundled
static scene, arena and fixture files. The corrected installer supplements
the latter without replacing different existing content. Re-run with
``--skip-existing`` to verify successful download inventories; a nonempty
directory alone is not a complete installation. Keep official attribution
files and finish interrupted downloads before starting experiments.

Resource publication is atomic: interrupted copies do not leave half-written
final files. To repair conflicting resources left by an earlier installer,
rerun the same command with explicit overwrite permission:

.. code-block:: bash

   robocasa-download-assets --assets-path ~/.robocasa/assets --no-macros --overwrite -y

``--overwrite`` takes precedence over ``--skip-existing`` and replaces only
resource files in the installation scope, not unrelated files or whole
directories. Without it, different existing content is preserved. Atomic
no-overwrite publication requires hard-link support on the destination
filesystem. Temporary files left by a killed process do not block retries.

New collections require space for the ZIP and one unpacked copy: staging is
published without copying the payload again. Existing installations need
additional space during replacement. ``--skip-existing`` avoids downloading
and comparing completed collections; bundled static files are checked separately.

**Navigation camera**

The ``robocasa`` extra installs the ``rpent`` branch of ``RLinf/robosuite``,
which provides the Omron base's fixed ``navview`` camera. Its composed MuJoCo
name is ``mobilebase0_navview``. Navigation RGB-D and world-map rendering
validate the camera when they first request it, and report an error if it is
missing. No manual ``site-packages`` XML patch is required.
Target50 uses this same maintained branch; record the resolved revision with
the environment information above.

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

**Task and global memory**

With ``--memory-profile hf`` (the default), the CLI and Dashboard synchronize
``robocasa/**`` from the current ``main`` branch of the
`RLinf/RPent-memory dataset
<https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/robocasa>`_.
Memory is not pinned to a commit. The layout is:

.. code-block:: text

   memory/robocasa/
   ├── task-specific/
   │   ├── <Task>_s0.json
   │   ├── <Task>_s0_recipe.jsonl
   │   └── <Task>.md              # optional
   └── global/
       └── GLOBAL_MEMORY.md

In HF evaluation, RoboCasa provides the current task's available JSON, recipe and Markdown
alongside ``global/GLOBAL_MEMORY.md``. The planner uses ``read_text_file`` to
consult relevant task-specific and global guidance as needed. It chooses when
and how much to read; actions and completion do not require every file to be
read first. There is no option to disable the global layer.

The prompt and file tools share the same selection. RPent file tools deny
other tasks' memory; this is a tool restriction, not an operating-system sandbox.
A missing JSON/JSONL pair is allowed: the planner continues with live
observations and global guidance. A half-present pair is an error. Missing
optional Markdown is logged, and the global file must exist. Files are
discovered by task name and directory; no extra index is needed.
The CLI validates memory before starting robot services. Dashboard validates
the memory root and global layer before starting its shared VLA, then checks
each selected task's files before starting that task's environment.
A task memory error leaves the existing shared VLA available for other tasks.

Live ``task_language``, RGB-D observations, task progress and tool results take
precedence over memory. Apply a global strategy only when its visible
preconditions hold. Continue VLA calls while contact, a held object, fixture
progress or a counter increase shows progress. After two consecutive calls
without contact or visible progress, re-ground and make a bounded pose
adjustment. Every VLA call uses the full, verbatim live task language.
Historical ``vla_act`` entries describe strategies; use current tools and
never replay historical coordinates. Reset remains unavailable during evaluation.

To use local memory, download into a fresh directory and select the local
profile. This also avoids retaining deleted files in an older download directory:

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
      --include 'robocasa/**' --local-dir ./target50-memory

   rpent --robot robocasa \
         --task-name OpenDrawer --split target --seed 1 \
         --vla-model-path /path/to/rldx \
         --planner codex --model gpt-5.5 --reasoning-effort xhigh \
         --memory-profile local --memory-dir ./target50-memory/robocasa

For the maintained reproduction memory branch, add
``--revision reproduce/memory`` to the download command. This selects a mutable
branch, not a fixed data version. Use a fresh directory when switching branches.
Both branches use the same RoboCasa task/global layout. Future memory updates
require only editing the appropriate task or global file; code changes are not
needed.

Local exploration output is also supported directly, without conversion. For
``--task-name <Task> --split <split>``, local evaluation selects either the
published ``<Task>_s0.json`` / ``<Task>_s0_recipe.jsonl`` pair or the native
``<Task>_<split>_s0.json`` / ``<Task>_<split>_s0_recipe.jsonl`` pair in
``task-specific/``. Either half-pair is an error. If both pairs exist, use
separate ``--memory-dir`` directories; RPent does not choose between them.
When neither pair exists, evaluation can use global guidance alone.

Local evaluation also exposes ``global/*.md`` and the ``task-family/*.md``
leaves whose YAML frontmatter matches ``suite: robocasa``, ``regime: <split>``
and ``task_id: <Task>``. Other tasks and splits are excluded. An optional
``task-specific/<Task>.md`` remains available. Evaluation neither requires nor
exposes the corpus-wide ``MEMORY.md`` index or ``_internal/``; it lists the
selected files directly. Missing global memory prevents evaluation startup,
including with the local profile. The same selection drives prompts, file
permissions, and read audits; results record the actual profile and matching
family identity for validation.

Custom memory sources
~~~~~~~~~~~~~~~~~~~~~

To use another HF dataset with the same ``robocasa/`` layout, set
``RPENT_MEMORY_HF_REPO=<owner>/<dataset>`` when launching RPent with
``--memory-profile hf``. This accepts a dataset repository ID, not a browser URL.

For a custom subdirectory or a maintained branch, download that subtree into a
fresh directory and use the local profile:

.. code-block:: bash

   hf download <owner>/<dataset> --repo-type dataset \
      --include '<subpath>/**' --local-dir ./custom-memory

   # Add to the RPent run command:
   # --memory-profile local --memory-dir ./custom-memory/<subpath>

The selected directory uses ``task-specific/`` and must provide at least one
readable ``global/*.md`` file. Add ``--revision <branch>`` to the HF download
command when selecting a branch. No delivery package or migration script is
required.

Harness VLA Target50 reproduction protocol
-------------------------------------------

The current ``robots/robocasa/eval/target50_v2.json`` protocol
(``robocasa-harness-vla-v2``) uses task/global memory without pinning its
data version. It preserves the target task/seed matrix, cell time limits,
no-reset rule, environment success predicate, and 40/999/8 RLDX settings.
The protocol ID identifies the result format and evaluation rules; it lets the
validator distinguish v1 from v2 and does not select a memory data version.

Results record the fixed task/global selection, missing files and actual reads.
The validator accepts zero or partial reads, while checking task boundaries and
the audit structure. Missing or corrupt audit files are reported separately;
read completeness does not determine the environment result's validity or
success. Each run starts a fresh audit, even when reusing an output directory.

Memory contents are not compared across runs. Both ``main`` and
``reproduce/memory`` are maintained branches. For a repeatable comparison,
download memory once and use the same unchanged directory with
``--memory-profile local --memory-dir`` for every cell. Retain the files and
record the HF commit or hashes in local experiment notes. RPent does not pin
memory or add data revision identifiers to result metadata.

The manifests describe the evaluation matrix and validation rules; the
:doc:`leaderboard <../leaderboard/performance>` displays independently reported
scores. A 340-cell result alone does not establish which memory, model or code
configuration produced it. The current v2 manifest includes a GPT-5.5 reference
profile; it is not a universal validator for every model on the leaderboard.

- ``target50.json`` retains the historical v1 task-specific protocol. Validate
  compatible historical records with
  ``--manifest robots/robocasa/eval/target50.json``.
- ``target50_v2.json`` describes current runs with task-specific and global
  memory. It is the default for new results and validation.
- Published leaderboard scores retain their original reported sources; they
  are not reclassified as v2 results without matching run evidence.

Source dependencies follow the recorded ``rpent`` branches.

.. list-table:: RoboCasa Target50 matrix
   :header-rows: 1
   :widths: 30 15 20 20 15

   * - Split
     - Tasks
     - Seeds per task
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

Running a task
--------------

HTTP RPC endpoints whose hostname is ``127.0.0.1`` or ``localhost`` are reached
directly, whether RPent starts the worker or the user supplies the endpoint.
Every other hostname and IP uses the standard proxy environment. Codex applies
the same two-host exception only to its child process for the local MCP
connection. Leave ``HTTP_PROXY`` and ``HTTPS_PROXY`` unchanged when Hugging
Face, a remote planner, or another remote service requires them; the default
runtime does not require a shell-wide ``NO_PROXY`` setup.

If a user-supplied local service uses another hostname or IP and should be
reached directly, add that exact value to the user's existing ``NO_PROXY`` and
``no_proxy`` configuration.

The RoboCasa CLI flags are registered by ``robots/robocasa/__init__`` and
are visible under ``rpent --robot robocasa --help``:

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer \
         --split target \
         --seed 1 \
         --vla-model-path /path/to/rldx \
         --planner claude_code \
         --model claude-opus-4-8

RoboCasa does not select a planner implementation; any planner supported by
RPent can run this robot. See :doc:`configure_planner` for configuration.

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

.. note::

   Use ``--env-endpoint`` / ``--vla-endpoint`` to point at already-running
   servers (``[protocol://]host:port``); when omitted, RPent spawns the env
   and VLA daemons in-process and writes their logs to
   ``<output_dir>/env_server.log`` and ``<output_dir>/vla_server.log``.

Reported results and historical records
---------------------------------------

The :doc:`leaderboard <../leaderboard/performance>` is the source for the
currently reported GPT-5.5 scores: **57.1% Overall**, **92.0% Atomic-Seen**,
**61.0% Composite-Seen** and **13.8% Composite-Unseen**. Overall weights all
50 tasks equally; it is not the fraction of successful cells among 340.

The Astra entry reports **59.20% Overall**, with **87.78% / 43.75% / 42.50%**
for the three splits. Its episode count is **340 (180/80/80)** following the
`contributor-confirmed correction
<https://github.com/RLinf/RPent/pull/205#issuecomment-5749514622>`_. The earlier
250-cell information was an unsynchronized historical record. The correction
preserves reported rates; it does not infer success counts from rounded rates
or claim a new audit of all 340 original results.

The `archived per-task reproduction table
<https://github.com/RLinf/RPent/blob/57088f6df30b227f2229ead985aa75403c0ce291/robots/robocasa/eval/target50_codex_results.md>`_
remains available at its original commit as a separate historical record. It
is not used to derive the current leaderboard values or establish v2 results.

.. _environment-smoke-tests:

Environment smoke tests
-----------------------

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
  ``memory/robocasa/task-specific/`` directory or the selected local directory.
  RPent does not fall back to another task's memory.
  Markdown is optional; Atomic tasks have no published ``<Task>.md``.
- Environment and VLA startup failures are recorded in
  ``<output_dir>/env_server.log`` and ``<output_dir>/vla_server.log``; also
  inspect ``<output_dir>/run.log`` for the run-level error.
- Only the exact ``127.0.0.1`` and ``localhost`` hostnames bypass HTTP proxies
  automatically. Other hostnames and IPs use the standard proxy environment;
  add the exact host to ``NO_PROXY`` and ``no_proxy`` only when it should be
  reached directly.

Toolkit design vs. LIBERO
-------------------------

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

Exploration mode
----------------

Add ``--explore`` to let the planner retry a task across fresh episodes and
write local memory. Exploration may start with an empty memory directory and
can read its index and write to its own inbox. As with LIBERO, one run uses three planner sessions with
five attempts per session by default:

.. code-block:: bash

   rpent --robot robocasa --task-name OpenDrawer --split target --seed 0 \
     --vla-model-path /path/to/rldx \
     --planner codex --reasoning-effort high --planner-timeout-s 7200 \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir /path/to/robocasa-memory

``reset`` uses the environment's ordinary episode reset. The runner exports
only the winning commands after the final reset. Exploration memory is written
to the current local inbox and merged after the run unless
``--no-auto-merge-memory`` is passed. The exploration prompt is in
``robots/robocasa/prompts/explore.py`` and covers mobile-base use,
``task_progress``, RLDX continuity, and failed-attempt notes.

Shared memory merge publishes accepted proposals under ``task-family/`` and
``global/``, copies the successful audit/recipe pair into ``task-specific/``,
and refreshes ``MEMORY.md``. To evaluate these native files, use seed-0
exploration output and ``--memory-profile local`` with the same directory.
Evaluation requires global memory to have been published first and uses its
own task/split access boundary; exploration keeps its retry and inbox workflow.
