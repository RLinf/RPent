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

RoboCasa365 is not part of ``.[full]``. The ``.[robocasa]`` extra pulls
the full stack — MuJoCo 3.3.1, the ARISE-Initiative robosuite fork, the
pinned lerobot commit, protobuf, and the ``robocasa`` package itself
(a wheel from the ``github.com/qurakchin/robocasa`` fork, branch
``v1.0.1_rlinf``). The fork is patched to load ``macros_private`` and
``assets`` from env vars, so a non-editable wheel install works — no
local clone needed.

**Prerequisites — Python, PyTorch, torchvision, flash-attn**

RLDX-1 hard-pins these versions: Python ``3.10.*``, ``torch==2.7.0``,
``torchvision==0.22.0``, ``transformers==4.57.0``,
``flash-attn==2.7.4.post1``. The steps below are a recommended path —
PyTorch, torchvision, and flash-attn may be installed via any tool
(uv, pip, conda, system packages, …) as long as the versions match
the pins. Then run the main ``.[robocasa]`` install, which pulls in
the rest of the stack.

.. code-block:: bash

   # Create a Python 3.10 venv — 3.11/3.12 will fail dependency resolution.
   uv venv --python 3.10

   # PyTorch + torchvision — cu124 is the recommended default; cu121/cu126/cu128
   # are interchangeable for CUDA 12.x drivers. cu118 (CUDA 11.8) has no
   # prebuilt flash-attn wheel and forces a source build in the next step.
   uv pip install torch==2.7.0 torchvision==0.22.0 \
       --index-url https://download.pytorch.org/whl/cu124

   # flash-attn prebuilt wheel from the upstream GitHub release
   # (cu12 + torch 2.7 + py3.10 + cxx11abi=TRUE). PyPI only ships the
   # sdist, so a bare `uv pip install flash-attn==2.7.4.post1` would
   # build from source — needs nvcc and takes 10-20 min.
   uv pip install \
       https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

Once the prerequisites above are installed, run the main install:

.. code-block:: bash

   # uv sees the flash-attn wheel from above as satisfying the
   # `flash-attn==2.7.4.post1` pin in pyproject.toml (PEP 440 `==`
   # ignores the local-version segment) and skips re-fetch.
   uv pip install -e ".[robocasa]"

**Note**: The prebuilt flash-attn wheel ships SM_80 and SM_90 kernels
only — it imports on Ampere/Hopper but crashes on Blackwell
(``sm_120``). RTX 5090 users must build flash-attn from source; see the
`flash-attn install docs
<https://github.com/Dao-AILab/flash-attention#installation>`_.

**Post-install setup**

After the ``.[robocasa]`` install, RoboCasa still needs a
``macros_private.py`` and the kitchen assets before ``rpent`` can
run:

1. Generate ``macros_private.py`` and export its path:

   .. code-block:: bash

      # Default destination: <repo_root>/.robocasa/macros_private.py
      export ROBOCASA_MACROS_PATH=$PWD/.robocasa/macros_private.py
      python -m robocasa.scripts.setup_macros

   The fork's ``macros.py`` reads ``$ROBOCASA_MACROS_PATH`` at import
   time, so the env var must be set in any shell that launches
   ``rpent`` — add it to your ``.bashrc`` / ``.zshrc``.

2. Download the kitchen assets (10+ GB) and, optionally, relocate
   them outside ``site-packages``:

   .. code-block:: bash

      # Downloads into the wheel's bundled robocasa/models/assets/
      python -m robocasa.scripts.download_kitchen_assets --type all

      # Optional: relocate so assets survive wheel reinstalls and can
      # be shared across venvs
      export ROBOCASA_ASSETS_PATH=$PWD/.robocasa/assets
      WHEEL_ASSETS=$(python -c "import robocasa; print(robocasa.__path__[0])")/models/assets
      mkdir -p "$ROBOCASA_ASSETS_PATH"
      mv "$WHEEL_ASSETS"/* "$ROBOCASA_ASSETS_PATH"/

   With ``ROBOCASA_ASSETS_PATH`` unset, robocasa falls back to the
   wheel's bundled ``models/assets/`` — so the download alone is
   enough to run. Export the var only if you relocated the files.

3. (Optional) Sanity-check the imports:

   .. code-block:: bash

      python -c "import robosuite, robocasa; print(robosuite.__version__, robocasa.__path__[0])"

See :doc:`../installation` for the install-time defaults.

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
