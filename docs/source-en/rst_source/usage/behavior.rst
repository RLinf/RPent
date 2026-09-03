BEHAVIOR
========

`BEHAVIOR-1K <https://behavior.stanford.edu/>`_ provides long-horizon household
tasks in OmniGibson. RPent exposes ``turning_on_radio`` and
``picking_up_trash`` as a standard sibling robot plugin under
``robots/behavior``.

The integration follows the same lightweight contract as LIBERO, RoboCasa, and
RoboTwin: ``get_robot_spec()`` supplies CLI/config/runtime hooks and
``get_toolkit()`` supplies the public tools. BEHAVIOR-specific lifecycle code
stays inside ``robots/behavior``. The common ``--explore`` entry point supports
BEHAVIOR while preserving its one-attempt-per-session environment lifecycle.

Installation status
-------------------

BEHAVIOR is source-editable and uses two independent Python 3.10 environments:

- the **RPent venv** runs the CLI, planner, Dashboard, and MemoryManager;
- the **BEHAVIOR venv** runs RLinf, OmniGibson, Isaac Sim, and Pi0.5.

The ``.[behavior]`` extra installs only RPent-side dependencies. It does not
install the complete simulator, assets, or checkpoint, and a normal wheel does
not promise a directly runnable BEHAVIOR stack. From a source checkout, run:

.. code-block:: bash

   python -m pip install -e ".[behavior]"
   export RPENT_REPRO_ROOT="$PWD/.behavior-runtime"
   export UV_CACHE_DIR="$RPENT_REPRO_ROOT/uv-cache"
   behavior-install-runtime

The installer keeps RPent editable in both venvs, clones the reviewed RLinf
revision, invokes the official RLinf BEHAVIOR installer, applies the reviewed
CUDA/OpenPI compatibility pins, verifies critical imports and CUDA, and writes
freezes plus source identities under ``$RPENT_REPRO_ROOT/manifests``. Use a new
``RPENT_REPRO_ROOT`` for a fresh install; the script refuses to overwrite a
wrong or dirty RLinf checkout.

Simulator assets
----------------

Accept the BEHAVIOR/OmniGibson licences, choose a dedicated data root, and use
the standard asset command. It invokes the three official OmniGibson download
functions in the BEHAVIOR venv rather than importing OmniGibson into the RPent
environment:

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   export BEHAVIOR_PYTHON="$RPENT_REPRO_ROOT/venvs/behavior/bin/python"
   behavior-download-assets --accept-license --skip-existing

Omit ``--accept-license`` to let the official downloader display its
interactive licence prompt. The flag is an explicit non-interactive
confirmation; do not use it unless you accept the licence terms.

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

``behavior-download-assets --verify`` verifies the required OmniGibson layout
and the source-controlled checkpoint size/SHA binding. The shared #136 Pi0.5
component receives head, left-wrist, right-wrist, and raw R1Pro proprio data.
The raw RPC result is ``[1, 32, 23]``; the common client returns ``[32, 23]``.

.. code-block:: bash

   behavior-download-assets --verify

DINOv2 configuration
--------------------

BEHAVIOR keeps a reviewed `DINOv2 <https://github.com/facebookresearch/dinov2>`_
ViT-S/14 deployment for whole-image embeddings and episode-memory retrieval.
Provide a DINOv2 source archive and the ``dinov2_vits14_pretrain.pth`` weights:

.. code-block:: bash

   export DINOV2_SOURCE_ARCHIVE=/path/to/dinov2-source.tar.gz
   export DINOV2_WEIGHTS=/path/to/dinov2_vits14_pretrain.pth

   curl -L \
     https://github.com/facebookresearch/dinov2/archive/7764ea0f912e53c92e82eb78a2a1631e92725fc8.tar.gz \
     -o "$DINOV2_SOURCE_ARCHIVE"
   curl -L \
     https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
     -o "$DINOV2_WEIGHTS"

DINOv2 is the shared visual-memory component. It is not a segmentation model,
does not replace SAM3 masks, and does not replace current public observations
or MemoryManager Markdown/YAML material. The accepted DINOv2 source revision
and both asset SHA-256 identities are pinned in
``robots/behavior/dino_v2/encoder.py``; the runtime rejects mismatched assets.

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
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS" \
     --memory-profile local \
     --memory-dir /path/to/behavior-memory \
     --behavior-memory-dir /path/to/reviewed-behavior-episode-memory

The first environment load can take several minutes. The environment, VLA, and
DINO are separate processes, and each receives only its explicitly selected GPU.

Official MemoryManager
----------------------

BEHAVIOR uses the same Markdown/YAML ``MemoryManager`` format and common memory
tools as the other robots. The DINO episode-memory catalog is a separate visual
experience retrieval source; when configured, its advisory is attached to
public tool receipts and remains historical guidance only.

- Eval creates one ``MemoryManager`` with ``read_only`` access.
- Explore creates one ``MemoryManager`` with ``inbox_write`` access scoped to
  ``<memory-dir>/_inbox/<recipe-tag>``.
- ``MEMORY.md``, ``global/``, ``suite/``, ``task/``, ``_inbox/``, and
  ``_merged/`` retain their standard RPent meanings.

An absent or empty corpus is valid, but it contains no advice. Pass the same
explicit ``--memory-dir`` to runs that should share reviewed memory.

Use ``--behavior-memory-dir`` only for the reviewed DINO episode-memory catalog.
Omitting it selects a legal empty episode catalog and does not download or
silently substitute task-specific memory.

Use the standard RPent Explore entry point for a bounded sequence of sessions:

.. code-block:: bash

   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent" --robot behavior \
     --behavior-mode explore \
     --explore \
     --explore-sessions 3 \
     --task-name picking_up_trash --public-seed 0 \
     --output-dir /path/to/behavior-explore \
     --memory-dir /path/to/behavior-memory \
     --planner codex --model gpt-5.5 \
     --behavior-repo "$RPENT_REPRO_ROOT/RLinf" \
     --behavior-python "$RPENT_REPRO_ROOT/venvs/behavior/bin/python" \
     --activity-instance-dir \
       "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --behavior-env-cuda-device 0 \
     --behavior-model-cuda-device 1 \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS"

For BEHAVIOR, one session is exactly one attempt. Each session starts a fresh
environment sidecar, episode, and ``sessions/session_NNN`` output directory;
the VLA and DINO sidecars stay shared across sessions. The planner cannot reset
inside an invocation, and ``--explore-attempts-per-session`` values above zero
are rejected.

``robots.behavior.harness`` remains available as the strengthened path when
every attempt must run in a fully isolated RPent process and results must be
aggregated across attempts. On that harness path, task audit/recipe pairs are
promoted only when the terminal receipt carries official success.

Runtime and Dashboard
---------------------

The runtime has four component roles:

- ``env``: the task-scoped official BEHAVIOR/OmniGibson environment;
- ``vla``: the shared ``rpent/robots/components/pi05_vla_server.py`` service;
- ``dino``: the shared ``robots/behavior/dino_v2/server.py`` episode-memory
  embedding service;
- ``memory``: the task-scoped official MemoryManager.

Start a Dashboard Session with:

.. code-block:: bash

   export RPENT_BEHAVIOR_PYTHON="$RPENT_REPRO_ROOT/venvs/behavior/bin/python"
   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent" \
     --robot behavior --dashboard \
     --task-name turning_on_radio --public-seed 1 \
     --behavior-mode eval \
     --behavior-repo "$RPENT_REPRO_ROOT/RLinf" \
     --behavior-python "$RPENT_BEHAVIOR_PYTHON" \
     --activity-instance-dir \
       "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS" \
     --memory-profile local --memory-dir /path/to/behavior-memory \
     --output-dir /path/to/behavior-dashboard-run

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
   <output-dir>/behavior_dino_server.log
   <output-dir>/tasks/<task-run>/behavior_env_server.log
   <output-dir>/tasks/<task-run>/episode.mp4
   <output-dir>/tasks/<task-run>/terminal_receipt.json
