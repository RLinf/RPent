RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ is the kitchen-scale, long-horizon
manipulation environment. In RPent it is driven by the **RLDX-1** VLA
policy, served over HTTP RPC by default (matching LIBERO); a
pickle-framed socket transport is also supported. See
``robots/robocasa/vla_server.py`` and ``robots/robocasa/__init__.py``
for the wire/transport selection.

Installation
------------

RoboCasa365 is not part of ``.[full]`` and needs its own virtualenv. The
``.[robocasa]`` extra pulls three things on top of the RLinf runtime:

- ``rlinf-robocasa365`` — the simulator, published to PyPI from the
  ``rlinf`` branch of `RLinf/robocasa <https://github.com/RLinf/robocasa/tree/rlinf>`_.
  It brings its own MuJoCo, NumPy and SciPy, so RPent does not restate
  those versions. The fork is patched to load
  ``macros_private`` and ``assets`` from env vars, so the wheel install
  needs no local clone.
- ``rlinf-rldx`` — the RLDX-1 VLA policy, published to PyPI from the
  ``rpent`` branch of `RLinf/RLDX-1 <https://github.com/RLinf/RLDX-1/tree/rpent>`_.
  It declares its own Python, torch, torchvision, transformers, numpy
  and flash-attn requirements.
- ``robosuite`` from git — the only direct reference left. RoboCasa365
  needs three APIs that are on the robosuite development line but not in
  the 1.5.2 release (``load_model_on_init``,
  ``ManipulationTask(enable_multiccd=...)`` and the
  ``JOINT_VELOCITY_LEGACY`` mobile-base controller), and master reports
  ``__version__ == "1.5.2"`` as well, so no version specifier can select
  it. Pinned to a commit on the RLinf fork rather than upstream
  ``@master``, so the resolved build is reproducible.

.. note::

   ``.[robocasa]`` cannot share a virtualenv with ``.[full]``. RLDX-1's
   backbone imports ``transformers.models.qwen3_vl``, which first appears
   in transformers 4.57, while ``rlinf-openpi`` requires a 4.53-based
   transformers fork that does not contain it.

   LIBERO itself is no longer the obstacle: ``rpent-libero`` and
   ``rpent-liberopro`` are ports to robosuite 1.5, verified to reproduce
   the robosuite 1.4 observation contract and action space exactly, so
   ``.[robocasa]`` and the LIBERO extras can coexist as long as
   ``openpi`` is not in the same environment.

Everything, including PyTorch and the robosuite pin, comes from the
extra. RLDX-1 requires Python ``3.10``:

.. code-block:: bash

   uv venv --python 3.10
   uv pip install -e ".[robocasa]"

.. note::

   flash-attn is **not** required. RLDX-1 falls back to PyTorch SDPA
   when it is absent. To opt in for a faster policy forward pass,
   install the prebuilt wheel afterwards — PyPI ships only an sdist, so
   a plain ``pip install flash-attn`` would compile for 10-20 minutes:

   .. code-block:: bash

      uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

   That wheel carries SM_80 and SM_90 kernels only; on Blackwell
   (``sm_120``) build from source or stay on SDPA.

**Post-install setup**

One command writes ``macros_private.py`` and downloads the kitchen assets
(~10 GB). Point it somewhere outside ``site-packages`` so the assets
survive reinstalls and can be shared between virtualenvs:

.. code-block:: bash

   robocasa-download-assets --assets-path ~/.robocasa/assets -y

It prints the two environment variables to export afterwards; add them to
the shell that launches ``rpent``:

.. code-block:: bash

   export ROBOCASA_MACROS_PATH=~/.robocasa/macros_private.py
   export ROBOCASA_ASSETS_PATH=~/.robocasa/assets

Re-running with ``--skip-existing`` leaves already-downloaded folders
alone. Use ``--no-assets`` to only write the macros file, ``--type`` to
fetch a subset, and ``--assets-path`` unset to keep everything inside the
package.

**RLDX-1 checkpoint**

The ``--vla-model-path`` flag on the run commands below expects a
local path to the ``RLDX-1-FT-RC365`` checkpoint (the RoboCasa365
fine-tune). Download it from HuggingFace:

.. code-block:: bash

   huggingface-cli download RLWRLD/RLDX-1-FT-RC365 --local-dir ./checkpoints/rldx-1-ft-rc365

If the download is slow, use the HF mirror:

.. code-block:: bash

   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download RLWRLD/RLDX-1-FT-RC365 --local-dir ./checkpoints/rldx-1-ft-rc365

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

Pass any of these to ``--robocasa-env``. The full RoboCasa catalog is
larger; see the `RoboCasa <https://robocasa.ai>`_ upstream.

Running a task
--------------

The RoboCasa CLI flags are registered by ``robots/robocasa/__init__`` and
are visible under ``rpent --env robocasa --help``:

.. code-block:: bash

   rpent --env robocasa \
         --robocasa-env OpenDrawer \
         --robocasa-split target \
         --robocasa-seed 0 \
         --vla-model-path /path/to/rldx \
         --planner claude_code \
         --model claude-opus-4-8

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
