RoboCasa Evaluation Reference
=============================

Use this page to check evaluation settings, diagnose memory files or validate
historical results. For the run procedure, see :ref:`reproduce-target50`.

.. _robocasa-task-list:

Full Task List
--------------

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

Action Settings
---------------

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

Memory File Selection
---------------------

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

Custom Memory Sources
~~~~~~~~~~~~~~~~~~~~~~

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

Protocol and Historical Compatibility
-------------------------------------

The current ``target50.json`` uses protocol ``robocasa-harness-vla-v2`` and
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

Historical v1 results used task-specific memory alone. Validate them in the
`historical code checkout <https://github.com/RLinf/RPent/tree/ec4e18fc2f6a73a00c6a5c035a8a3fdb17950b61>`_
with its matching `v1 manifest <https://github.com/RLinf/RPent/blob/ec4e18fc2f6a73a00c6a5c035a8a3fdb17950b61/robots/robocasa/eval/target50.json>`_:

.. code-block:: bash

   python -m robots.robocasa.eval.validate_target50 /path/to/historical-results \
      --manifest robots/robocasa/eval/target50.json

The HF `reproduce/memory archive <https://huggingface.co/datasets/RLinf/RPent-memory/tree/reproduce/memory>`_
retains GPT-5.5 Harness-VLA resources at ``d8c25a7f``; future updates go only to
main. Its RoboCasa ``task_only/`` layout is incompatible with the current
``task-specific/`` layout, and its README describes historical usage. This
guide has not established a matching RoboCasa code/data snapshot for that
archive; current run commands use main memory.

Leaderboard scores come from independent experiment reports. Completing 340
runs with the current protocol does not establish reproduction of a leaderboard
entry; compare the model, memory and code configuration too. Historical scores
retain their original sources.

Reported and Historical Target50 Results
----------------------------------------

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

Historical Codex Reproduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The archived reproduction contains all 340 cells and reports the following
task-level aggregates. These historical values do not describe a new v2 run:

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

The `archived per-task table
<https://github.com/RLinf/RPent/blob/57088f6df30b227f2229ead985aa75403c0ce291/robots/robocasa/eval/target50_codex_results.md>`_
contains the success count and accuracy for every task. This historical record is
task-level aggregate data; it does not include per-seed traces, raw trajectories,
or failure classifications and therefore is not a per-cell audit artifact.
