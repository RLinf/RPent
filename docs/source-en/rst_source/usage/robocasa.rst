RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ is the kitchen-scale, long-horizon
manipulation environment. In RPent it is driven by the **RLDX-1** VLA
policy, served over HTTP RPC by default (matching LIBERO); a
pickle-framed socket transport is also supported. See
``robots/robocasa/vla_server.py`` and ``robots/robocasa/robot_spec.py``
for the wire/transport selection.

.. note::

   The public Target50 protocol is frozen in
   ``robots/robocasa/eval/target50.json``. RPent uses ordinary single-task
   ``rpent --robot robocasa`` commands for its 340 cells.

Installation
------------

RLDX-1 requires Python ``3.10``. Create a dedicated environment and install
the complete RoboCasa365 stack with ``.[robocasa]``:

.. code-block:: bash

   uv venv --python 3.10
   source .venv/bin/activate
   uv pip install -e ".[robocasa]" \
      --constraint robots/robocasa/eval/target50-constraints.txt \
      --override robots/robocasa/eval/target50-overrides.txt \
      --torch-backend=cu126
   uv pip check

The RoboCasa-specific constraints file pins the compatibility-sensitive
package versions validated for Target50 reproduction without narrowing
RPent's shared LIBERO or RoboTwin dependencies. The companion override file
resolves the formal environment directly to the immutable Robosuite revision
while the ordinary ``robocasa`` extra tracks its maintained ``rpent`` branch.
The command also lets uv select the official CUDA wheel without treating the
PyTorch wheel index as a general package index.
Passing that index through ``--index`` can make uv select stale, unrelated
packages under its default first-index strategy. The ``cu126`` command above
is the validated CUDA installation; use another supported Torch backend only
when required by the host.

For networks closer to Chinese mirrors:

.. code-block:: bash

   uv pip install -e ".[robocasa]" \
      --constraint robots/robocasa/eval/target50-constraints.txt \
      --override robots/robocasa/eval/target50-overrides.txt \
      --default-index https://mirrors.aliyun.com/pypi/simple \
      --torch-backend=cu126

.. note::

   flash-attn is optional; RLDX-1 falls back to PyTorch SDPA without it.
   For a faster policy forward pass, install the prebuilt wheel — PyPI
   ships only an sdist, so a plain ``pip install flash-attn`` compiles for
   10-20 minutes:

   .. code-block:: bash

      uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

   That wheel carries SM_80 and SM_90 kernels only; on Blackwell
   (``sm_120``) build from source or stay on SDPA.

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

Re-run with ``--skip-existing`` to leave downloaded folders alone.

**Navigation camera**

The ``robocasa`` extra installs the ``rpent`` branch of ``RLinf/robosuite``,
which provides the Omron base's fixed ``navview`` camera. Its composed MuJoCo
name is ``mobilebase0_navview``. Navigation RGB-D and world-map rendering
validate the camera when they first request it, and report an error if it is
missing. No manual ``site-packages`` XML patch is required.
Target50 freezes the resolved Robosuite revision at
``97cfbde4b68d8ec43dad20cf4747297866a6ca2e``. The Target50 override in the
installation command above selects that exact revision.

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

**Task memory**

Select automatic synchronization with ``--memory-profile hf`` (the default).
Before every such ordinary run, RPent's shared memory manager synchronizes the
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

RoboCasa never asks the planner to use global memory or another task's memory.
Seven Composite-Unseen tasks have no task memory and remain in the evaluation:
``HeatKebabSandwich``, ``PanTransfer``, ``PortionHotDogs``,
``SeparateFreezerRack``, ``WaffleReheat``, ``WashFruitColander``, and
``WeighIngredients``. They continue from live observations. Memory is strategy
evidence; historical coordinates, poses, pixels, and subtask prompts must not
replace current localization or the live task language.

Ordinary runs synchronize Hugging Face ``main``. Formal Target50 runs use the
immutable memory snapshot
``551fc3157b3e56b40a3d3a3b4c7ff81721ebe89b``:

.. code-block:: bash

   hf download RLinf/RPent-memory \
      --repo-type dataset \
      --revision 551fc3157b3e56b40a3d3a3b4c7ff81721ebe89b \
      --include "robocasa/**" \
      --local-dir ./target50-memory

Select the local profile and pass the directory containing the frozen results
corpus:

.. code-block:: bash

   rpent --robot robocasa --task-name OpenDrawer --seed 1 \
         --vla-model-path ./checkpoints/rldx-1-ft-rc365 \
         --planner claude_code --model claude-opus-4-8 \
         --memory-profile local \
         --memory-dir ./target50-memory/robocasa

Harness VLA Target50 reproduction protocol
-------------------------------------------

``robots/robocasa/eval/target50.json`` is the canonical manifest for reproducing
Harness VLA on RoboCasa Target50. It freezes the ``target`` environment split,
dependency revisions, memory scope, task and seed matrix, cell time limits,
success source, and retry policy. Its protocol ID is
``robocasa-harness-vla-v1``:

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

Published Target50 results
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

Troubleshooting
---------------

- If navigation RGB-D or world-map rendering reports a missing
  ``mobilebase0_navview``, reinstall ``.[robocasa]`` to refresh the
  ``RLinf/robosuite`` ``rpent`` branch. Do not patch installed XML files
  manually.
- If ``read_text_file`` reports a missing current-task result, check the
  ``memory/robocasa/results/`` corpus or the selected local directory.
  RPent does not fall back to another task's memory.
- Environment and VLA startup failures are recorded in
  ``<output_dir>/env_server.log`` and ``<output_dir>/vla_server.log``.
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
