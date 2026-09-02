BEHAVIOR
========

`BEHAVIOR-1K <https://behavior.stanford.edu/>`_ provides long-horizon household
tasks in OmniGibson. RPent exposes ``turning_on_radio`` and
``picking_up_trash`` as a standard sibling robot plugin under
``robots/behavior``.

The integration follows the same lightweight contract as LIBERO, RoboCasa, and
RoboTwin: ``get_robot_spec()`` supplies CLI/config/runtime hooks and
``get_toolkit()`` supplies the public tools. BEHAVIOR-specific lifecycle code
stays inside ``robots/behavior``. The common ``--explore`` loop remains a
LIBERO feature; BEHAVIOR uses ``--behavior-mode explore`` and its own outer
harness.

Installation status
-------------------

BEHAVIOR is source-editable and uses two independent Python 3.10 environments:

- the **RPent venv** runs the CLI, planner, Dashboard, and MemoryManager;
- the **BEHAVIOR venv** runs RLinf, OmniGibson, Isaac Sim, and Pi0.5.

The ``.[behavior]`` extra installs only RPent-side dependencies. It does not
install the complete simulator, assets, or checkpoint, and a normal wheel does
not promise a directly runnable BEHAVIOR stack. From a source checkout, run:

.. code-block:: bash

   export RPENT_REPRO_ROOT="$PWD/.behavior-runtime"
   export UV_CACHE_DIR="$RPENT_REPRO_ROOT/uv-cache"
   bash scripts/install_behavior_runtime.sh

The installer keeps RPent editable in both venvs, clones the reviewed RLinf
revision, invokes the official RLinf BEHAVIOR installer, applies the reviewed
CUDA/OpenPI compatibility pins, verifies critical imports and CUDA, and writes
freezes plus source identities under ``$RPENT_REPRO_ROOT/manifests``. Use a new
``RPENT_REPRO_ROOT`` for a fresh install; the script refuses to overwrite a
wrong or dirty RLinf checkout.

Simulator assets
----------------

Accept the BEHAVIOR/OmniGibson licences, choose a dedicated data root, and run
the three official download functions from the BEHAVIOR venv:

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   export BEHAVIOR_PYTHON="$RPENT_REPRO_ROOT/venvs/behavior/bin/python"
   mkdir -p "$OMNIGIBSON_DATA_PATH"

   "$BEHAVIOR_PYTHON" -c \
     "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
   "$BEHAVIOR_PYTHON" -c \
     "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
   "$BEHAVIOR_PYTHON" -c \
     "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"

The final data root must contain:

.. code-block:: text

   BEHAVIOR-1K-datasets/
     2025-challenge-task-instances/
     behavior-1k-assets/
       scenes/
     omnigibson-robot-assets/
     omnigibson.key

Pi0.5 checkpoint
----------------

Download the reviewed checkpoint into a directory outside the source tree:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32
   "$RPENT_REPRO_ROOT/venvs/rpent/bin/hf" download \
     RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 \
     --local-dir "$PI05_CHECKPOINT_PATH"

``scripts/verify_behavior_assets.sh`` verifies the required OmniGibson layout
and the source-controlled checkpoint size/SHA binding. The shared #136 Pi0.5
component receives head, left-wrist, right-wrist, and raw R1Pro proprio data.
The raw RPC result is ``[1, 32, 23]``; the common client returns ``[32, 23]``.

.. code-block:: bash

   scripts/verify_behavior_assets.sh

Task identity
-------------

Use ``--task-name`` and ``--public-seed``. Public seeds map to fixed official
activity instances through ``robots/behavior/task_specs.py``.

.. list-table::
   :header-rows: 1
   :widths: 24 42 16 18

   * - Task
     - Instruction
     - Explore seeds
     - Eval seeds
   * - ``turning_on_radio``
     - Turn on the radio receiver on the living-room table.
     - ``0``
     - ``1``-``9``
   * - ``picking_up_trash``
     - Put the three living-room soda cans into the kitchen trash can.
     - ``0``-``9``
     - ``10``-``19``

One evaluation run
------------------

Bind each CUDA child to one physical GPU explicitly:

.. code-block:: bash

   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent" --robot behavior \
     --task-name turning_on_radio --public-seed 1 \
     --behavior-mode eval \
     --planner codex --model gpt-5.5 \
     --behavior-repo "$RPENT_REPRO_ROOT/RLinf" \
     --behavior-python "$RPENT_REPRO_ROOT/venvs/behavior/bin/python" \
     --activity-instance-dir \
       "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --behavior-env-cuda-device 0 \
     --behavior-model-cuda-device 1 \
     --memory-profile local \
     --memory-dir /path/to/behavior-memory

The first environment load can take several minutes. The environment and VLA
are separate processes, and each receives only its explicitly selected GPU.

Official MemoryManager
----------------------

BEHAVIOR uses the same Markdown/YAML ``MemoryManager`` format and common memory
tools as the other robots. It does not load, migrate, or silently fall back to
the former DINO episode catalog.

- Eval creates one ``MemoryManager`` with ``read_only`` access.
- Explore creates one ``MemoryManager`` with ``inbox_write`` access scoped to
  ``<memory-dir>/_inbox/<recipe-tag>``.
- ``MEMORY.md``, ``global/``, ``suite/``, ``task/``, ``_inbox/``, and
  ``_merged/`` retain their standard RPent meanings.

An absent or empty corpus is valid, but it contains no advice. Pass the same
explicit ``--memory-dir`` to runs that should share reviewed memory.

For repeated Explore attempts, use the BEHAVIOR-owned harness. It launches a
fresh RPent process and episode for every attempt, points every attempt at one
official corpus, and calls the existing ``MemoryManager.merge_memory()`` after
the run. The planner cannot reset inside an invocation.

.. code-block:: bash

   python -m robots.behavior.harness explore \
     --attempts 3 \
     --output-dir /path/to/behavior-explore \
     --memory-dir /path/to/behavior-memory \
     -- \
     --task-name picking_up_trash --public-seed 0 \
     --planner codex --model gpt-5.5

Use ``--no-auto-merge-memory`` when review policy requires preserving the inbox
without publication. Task audit/recipe pairs are promoted only when the
terminal receipt carries official success.

Runtime and Dashboard
---------------------

The runtime has three component roles:

- ``env``: the task-scoped official BEHAVIOR/OmniGibson environment;
- ``vla``: the shared ``rpent/robots/components/pi05_vla_server.py`` service;
- ``memory``: the task-scoped official MemoryManager.

Start a Dashboard Session with:

.. code-block:: bash

   TASK_NAME=turning_on_radio PUBLIC_SEED=1 \
     BEHAVIOR_MEMORY_DIR=/path/to/behavior-memory \
     scripts/run_behavior_dashboard.sh

The Dashboard uses the common Start Session flow and head/left-wrist/right-
wrist camera views. BEHAVIOR does not add robot-local manual buttons, a manual
control backend, or ``env.dashboard_*`` RPC methods. Planner primitives such as
``pi0_nav_pick``, ``observe``, ``navigate_to``, ``move_to``, ``press``,
``open``, and ``close`` remain available according to the active tool schema
and backend capabilities.

The main logs are:

.. code-block:: text

   <output-dir>/run.log
   <output-dir>/behavior_vla_server.log
   <output-dir>/tasks/<task-run>/behavior_env_server.log
   <output-dir>/tasks/<task-run>/episode.mp4
   <output-dir>/tasks/<task-run>/terminal_receipt.json

Success and diagnostics
-----------------------

Official task success is exactly the current episode's
``info["done"]["success"] is True``. Reward, ``terminated``, ``truncated``,
primitive success, screenshots, video, and process exit do not substitute for
it. The receipt records that raw evidence; it cannot manufacture success.

Run the lightweight source check before starting the simulator:

.. code-block:: bash

   python -m robots.behavior.selfcheck

The self-check validates plugin discovery, CLI/config derivation, task mapping,
memory profile, and public tool count. It does not load assets, start a GPU
service, execute an action, or establish task success.
