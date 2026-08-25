RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ is the kitchen-scale, long-horizon
manipulation environment. In RPent it is driven by the **RLDX-1** VLA
policy, served over HTTP RPC by default (matching LIBERO); a
pickle-framed socket transport is also supported. See
``robots/robocasa/vla_server.py`` and ``robots/robocasa/robot_spec.py``
for the wire/transport selection.

.. note::

   The current code does not yet fully match the results shown at
   `harnessvla.github.io <https://harnessvla.github.io>`_; a full
   reproduction will be released later.

Installation
------------

RoboCasa365 is part of ``.[full]``. To install it on its own — RLDX-1
requires Python ``3.10``:

.. code-block:: bash

   uv venv --python 3.10
   uv pip install -e ".[robocasa]"

To avoid a CUDA-build mismatch between PyPI's torch wheel and the local
driver, pass ``--index`` to pin a PyTorch CUDA index, e.g.
``uv pip install -e ".[robocasa]" --index https://download.pytorch.org/whl/cu126``;
on CUDA 13-only hosts use ``cu130``.

For networks closer to Chinese mirrors:

.. code-block:: bash

   uv pip install -e ".[robocasa]" \
      --default-index https://mirrors.aliyun.com/pypi/simple \
      --index https://pypi.tuna.tsinghua.edu.cn/simple \
      --index https://mirrors.aliyun.com/pytorch-wheels/cu126

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

**RLDX-1 checkpoint**

The ``--vla-model-path`` flag on the run commands below expects a
local path to the ``RLDX-1-FT-RC365`` checkpoint (the RoboCasa365
fine-tune). Download it from HuggingFace:

.. code-block:: bash

   hf download RLWRLD/RLDX-1-FT-RC365 --local-dir ./checkpoints/rldx-1-ft-rc365

If the download is slow, use the HF mirror:

.. code-block:: bash

   HF_ENDPOINT=https://hf-mirror.com hf download RLWRLD/RLDX-1-FT-RC365 --local-dir ./checkpoints/rldx-1-ft-rc365

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

.. note::

   Use ``--env-endpoint`` / ``--vla-endpoint`` to point at already-running
   servers (``[protocol://]host:port``); when omitted, RPent spawns the env
   and VLA daemons in-process and writes their logs to
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
