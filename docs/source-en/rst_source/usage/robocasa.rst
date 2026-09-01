RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ is the kitchen-scale, long-horizon
manipulation environment. In RPent it is driven by the **RLDX-1** VLA
policy, served over HTTP RPC by default (matching LIBERO); a
pickle-framed socket transport is also supported. See
``robots/robocasa/vla_server.py`` and ``robots/robocasa/robot_spec.py``
for the wire/transport selection.

.. note::

   This page documents ordinary single-task ``rpent --robot robocasa`` runs.
   It does not define the full Atomic/Seen/Unseen benchmark reproduction
   protocol.

Installation
------------

RLDX-1 requires Python ``3.10``. Create a dedicated environment and install
the complete RoboCasa365 stack with ``.[robocasa]``:

.. code-block:: bash

   uv venv --python 3.10
   uv pip install -e ".[robocasa]" \
      "torch==2.7.0" "torchvision==0.22.0" \
      --torch-backend=cu126

This pins the Torch pair used by RLDX-1 and lets uv select the official CUDA
wheel without treating the PyTorch wheel index as a general package index.
Passing that index through ``--index`` can make uv select stale, unrelated
packages under its default first-index strategy. The ``cu126`` command above
is the validated CUDA installation; use another supported Torch backend only
when required by the host.

For networks closer to Chinese mirrors:

.. code-block:: bash

   uv pip install -e ".[robocasa]" \
      "torch==2.7.0" "torchvision==0.22.0" \
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

One command writes ``macros_private.py`` and downloads the kitchen assets
(~10 GB). Point it outside ``site-packages`` so the assets survive
reinstalls:

.. code-block:: bash

   robocasa-download-assets --assets-path ~/.robocasa/assets -y

It prints the environment variables to export afterwards; add them to the
shell that launches ``rpent``:

.. code-block:: bash

   export ROBOCASA_MACROS_PATH=~/.robocasa/macros_private.py
   export ROBOCASA_ASSETS_PATH=~/.robocasa/assets

Re-run with ``--skip-existing`` to leave downloaded folders alone.

**Navigation camera**

The ``robocasa`` extra installs the ``rpent`` branch of ``RLinf/robosuite``,
which provides the Omron base's fixed ``navview`` camera. Its composed MuJoCo
name is ``mobilebase0_navview``. RPent does not run a separate reset-time
camera preflight; a missing camera fails when navigation RGB-D or world-map
rendering first requests it. Reinstall ``.[robocasa]`` to refresh that branch;
no manual ``site-packages`` XML patch is required. During this PR's validation,
the branch resolved to commit ``97cfbde4b68d8ec43dad20cf4747297866a6ca2e``;
record the resolved commit when reporting experiments.

**RLDX-1 checkpoint**

The ``--vla-model-path`` flag on the run commands below expects a
local path to the ``RLDX-1-FT-RC365`` checkpoint (the RoboCasa365
fine-tune). Download it from HuggingFace:

.. code-block:: bash

   hf download RLWRLD/RLDX-1-FT-RC365 --local-dir ./checkpoints/rldx-1-ft-rc365

If the download is slow, use the HF mirror:

.. code-block:: bash

   HF_ENDPOINT=https://hf-mirror.com hf download RLWRLD/RLDX-1-FT-RC365 --local-dir ./checkpoints/rldx-1-ft-rc365

**Task memory**

Default evaluation uses RPent's public resource sync to download the
``robocasa/**`` subtree from the ``RLinf/RPent-memory`` Hugging Face dataset
into ``resources/robocasa``. The current task may use only these task-matched
files:

.. code-block:: text

   resources/robocasa/results/<Task>_s0.json
   resources/robocasa/results/recipe_<Task>_s0.jsonl
   resources/robocasa/results/<Task>.md  # optional

The JSON/JSONL pair contains reviewed seed-0 evidence. The optional Markdown
file contains task-specific exploration memory and may summarize multiple
attempts; 16 Composite-Seen and 9 Composite-Unseen tasks currently provide one.
The prompt requires the planner to read every current-task file that exists
before acting. RPent makes those files available through ``read_text_file``
but does not inject their contents into the prompt.

RoboCasa never asks the planner to use global memory or another task's memory.
The runtime does not preflight corpus completeness. If a current-task file is
absent, ``read_text_file`` reports it as missing and the planner continues with
the available task files and live observations. The published default corpus
is validated separately. To use reviewed local files instead, select the local
profile and pass the directory containing this results corpus:

.. code-block:: bash

   rpent --robot robocasa --task-name OpenDrawer --seed 1 \
         --vla-model-path /path/to/rldx --planner claude_code \
         --memory-profile local --memory-dir /path/to/robocasa-results

Available task list
-------------------

The 50 tasks used in RPent split into three groups:

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

The RoboCasa CLI flags are registered by ``robots/robocasa/__init__`` and
are visible under ``rpent --robot robocasa --help``:

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer \
         --split target \
         --seed 0 \
         --vla-model-path /path/to/rldx \
         --planner claude_code \
         --model claude-opus-4-8

RoboCasa does not select a planner implementation; any planner supported by
RPent can run this robot. See :doc:`configure_planner` for configuration.

.. note::

   Use ``--env-endpoint`` / ``--vla-endpoint`` to point at already-running
   servers (``[protocol://]host:port``); when omitted, RPent spawns the env
   and VLA daemons in-process and writes their logs to
   ``<output_dir>/env_server.log`` and ``<output_dir>/vla_server.log``.

Troubleshooting
---------------

- If navigation RGB-D or world-map rendering reports a missing
  ``mobilebase0_navview``, reinstall ``.[robocasa]`` to refresh the
  ``RLinf/robosuite`` ``rpent`` branch. Do not patch installed XML files
  manually.
- If ``read_text_file`` reports a missing current-task result, check the
  Hugging Face ``robocasa/results`` corpus or the selected local directory.
  RPent does not preflight the corpus or fall back to another task.
- Environment and VLA startup failures are recorded in
  ``<output_dir>/env_server.log`` and ``<output_dir>/vla_server.log``.

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
  fields, an annotation, and a session id used by ``reset_session`` /
  ``predict``.
