BEHAVIOR
========

`BEHAVIOR-1K <https://behavior.stanford.edu/>`_ provides long-horizon household
tasks in OmniGibson. Run ``turning_on_radio`` or ``picking_up_trash`` with
RPent's BEHAVIOR integration.

Overview
--------

Check the task, model, and runtime requirements before installing the simulator.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      Pi0.5 for 32-step action chunks.

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``.

   .. grid-item-card:: Tasks

      ``turning_on_radio`` and ``picking_up_trash``.

   .. grid-item-card:: Hardware

      Linux, NVIDIA GPU; two Python 3.10 environments; CUDA and Isaac Sim.

Tasks
~~~~~

Use ``--task-name`` and ``--public-seed``. Public seeds map to fixed official
activity instances through ``robots/behavior/task_specs.py``.

.. list-table::
   :header-rows: 1
   :widths: 24 42 16 18

   * - Task
     - Goal
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

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The planner's observations and tools differ from the inputs and actions of Pi0.5.

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - Head and left/right wrist RGB views, depth/world coordinates, and robot state. Pi0.5 receives three camera views and R1Pro proprioception.
   * - Action
     - The planner calls Pi0.5 or motion primitives. Pi0.5 predicts 32 steps of 23-value actions; the environment may stop a chunk early on success or truncation.
   * - Reward / success
     - Only the current episode's ``info["done"]["success"] is True`` establishes official task success. Reward, termination, and planner ``finish`` do not.
   * - Task prompt
     - The selected task's pinned instruction comes from ``robots/behavior/task_specs.py``.

Installation and Resources
--------------------------

Use a Linux host with an NVIDIA GPU and a compatible CUDA driver. Install
``git``, ``bash``, ``curl``, and ``nvidia-smi``, and reserve space
for the simulator, licensed data, and model weights. RLinf installs its system
dependencies using non-interactive ``sudo``. On a host where they are already
installed, set ``RLINF_NO_ROOT=1`` to pass RLinf's ``--no-root`` option instead.
On a slow upstream connection, ``RLINF_USE_MIRROR=1`` passes RLinf's
``--use-mirror`` option and uses the same Git mirror for cuRobo.

The source-checkout installer creates two independent Python 3.10 environments:

- the **RPent venv** runs the CLI, planner, Dashboard, and MemoryManager;
- the **BEHAVIOR venv** runs RLinf, OmniGibson, Isaac Sim, and Pi0.5.

For a fresh setup, run the installer from a source checkout:

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   export RPENT_REPRO_ROOT="$PWD/../rpent-behavior-runtime"
   export UV_CACHE_DIR="$RPENT_REPRO_ROOT/uv-cache"
   bash robots/behavior/install_behavior_runtime.sh
   export RPENT_PYTHON="$RPENT_REPRO_ROOT/venvs/rpent/bin/python"
   export BEHAVIOR_PYTHON="$RPENT_REPRO_ROOT/venvs/behavior/bin/python"
   "$RPENT_PYTHON" -m robots.behavior.selfcheck

The runtime root above is outside the source checkout. The installer installs
RPent editable in its planner venv and runs ENV, VLA, and DINO from this
checkout in the separate simulator venv. It uses the pinned RLinf checkout's
``embodied --model openpi --env behavior`` installer without repinning its
Torch, OpenPI, or LeRobot versions, adds cuRobo for motion planning, and writes
package freezes and source identities under ``$RPENT_REPRO_ROOT/manifests``.
Use a new ``RPENT_REPRO_ROOT`` for a fresh installation; it will not overwrite
a dirty or different RLinf checkout. A successful selfcheck verifies source
contracts, not GPU services or task execution.

Motion planning uses NVlabs/cuRobo v0.8.0 at commit
``4ea77366ca48ee453e7df139e39fa6532af49f3b`` in the BEHAVIOR venv.
Install it against the RLinf-selected runtime without changing that runtime's
Torch, NumPy, or Isaac Sim versions; an unconstrained resolver can upgrade
NumPy to 2, which is incompatible with this simulator stack.

For a non-editable RPent installation, set ``RPENT_REPO_ROOT`` to a writable
checkout at the same revision. Local-memory runs also need ``--memory-profile local``
and a writable ``--memory-dir``; HF-profile Eval does not accept
``--memory-dir``. The wheel includes ``robots/behavior``, but ``.[behavior]``
installs only RPent-side dependencies, not OmniGibson, licensed data, or the
Pi0.5 checkpoint.

Simulator Assets
~~~~~~~~~~~~~~~~

Accept the BEHAVIOR/OmniGibson licences, choose a dedicated data root, and use
the standard asset command. It invokes the three official OmniGibson download
functions in the BEHAVIOR venv rather than importing OmniGibson into the RPent
environment:

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH="$RPENT_REPRO_ROOT/omnigibson-data"
   "$RPENT_PYTHON" -m robots.behavior.assets_cli --accept-license

Omit ``--accept-license`` to let the official downloader display its
interactive licence prompt. The flag is an explicit non-interactive
confirmation; do not use it unless you accept the licence terms.
The downloader writes ``.rpent-behavior-download.json`` with a path/size
inventory after all three official downloads complete. ``--skip-existing``
and ``--verify`` accept only a matching
inventory; an older data root without this record needs a complete download
into a fresh root. This inventory is not a content SHA-256 check.

The final data root must contain:

.. code-block:: text

   <OMNIGIBSON_DATA_PATH>/
     2025-challenge-task-instances/
     behavior-1k-assets/
       scenes/
     omnigibson-robot-assets/
     omnigibson.key

Pi0.5 Checkpoint
~~~~~~~~~~~~~~~~

Download the pinned checkpoint into a directory outside the source tree:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH="$RPENT_REPRO_ROOT/downloads/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32"
   "$RPENT_REPRO_ROOT/venvs/rpent/bin/hf" download \
     RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 \
     --local-dir "$PI05_CHECKPOINT_PATH"

The shared Pi0.5 component receives head, left-wrist, right-wrist, and raw
R1Pro proprio data. The raw RPC result is ``[1, 32, 23]``; the client returns
``[32, 23]``. The checkpoint validator checks the pinned file sizes and SHA-256
identities before startup.

DINOv2 Derived Cache
~~~~~~~~~~~~~~~~~~~~

BEHAVIOR uses a pinned `DINOv2 <https://github.com/facebookresearch/dinov2>`_
ViT-S/14 deployment for whole-image embeddings over the official Memory
corpus.
Provide a DINOv2 source archive and the ``dinov2_vits14_pretrain.pth`` weights:

.. code-block:: bash

   export DINOV2_SOURCE_ARCHIVE="$RPENT_REPRO_ROOT/downloads/dinov2-source.tar.gz"
   export DINOV2_WEIGHTS="$RPENT_REPRO_ROOT/downloads/dinov2_vits14_pretrain.pth"

   curl -fL \
     https://github.com/facebookresearch/dinov2/archive/7764ea0f912e53c92e82eb78a2a1631e92725fc8.tar.gz \
     -o "$DINOV2_SOURCE_ARCHIVE"
   curl -fL \
     https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
     -o "$DINOV2_WEIGHTS"

The source revision and both file SHA-256 identities are pinned in
``robots/behavior/dino_v2/encoder.py``.

Check the complete asset layout and pinned checkpoint/DINO file identities
before starting a task:

.. code-block:: bash

   "$RPENT_PYTHON" -m robots.behavior.assets_cli --verify

Run a Task
----------

Configure a planner with :doc:`../guides/configure_planner` and check its
connection. Replace GPU ordinal ``0`` with an available physical GPU with
enough free memory for ENV, Pi0.5, and DINO. ``--cuda-device`` selects it for
all three; use the component-specific flags to split them across GPUs if needed.
This example runs one Eval episode with an empty local memory corpus, so it
needs no HF memory download:

.. code-block:: bash

   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent-check-llm" \
     --planner codex
   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent" --robot behavior \
     --task-name turning_on_radio --public-seed 1 \
     --planner codex --reasoning-effort low \
     --memory-profile local \
     --memory-dir "$RPENT_REPRO_ROOT/memory/behavior" \
     --output-dir "$RPENT_REPRO_ROOT/runs/first-eval" \
     --behavior-repo "$RPENT_REPRO_ROOT/RLinf" \
     --behavior-python "$RPENT_REPRO_ROOT/venvs/behavior/bin/python" \
     --activity-instance-dir \
       "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --cuda-device 0 \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS"

The first environment load can take several minutes. Proceed only if the
planner check succeeds; it checks credentials and connectivity, not the
simulator or the selected reasoning effort. Choose a new output directory for
later runs.

View Results
------------

For the Eval command above, inspect these paths after the run:

.. code-block:: text

   <output-dir>/run.log
   <output-dir>/transcript_*.json
   <output-dir>/states.json
   <output-dir>/episode.mp4
   <output-dir>/terminal_receipt.json       (when finish was called)
   <output-dir>/behavior_env_server.log
   <output-dir>/behavior_vla_server.log
   <output-dir>/behavior_dino_server.log

``states.json`` indexes step observations and camera images; the transcript
records planner messages. After ``finish``, ``terminal_receipt.json`` records
``task_success``. Its nested ``official_success_receipt`` is ``null`` unless
the current episode reported ``info["done"]["success"] is True``. On success,
that nested receipt includes a SHA-256 digest for integrity checking, not proof
of simulator authenticity. A video, clean process exit, reward, or planner
``finish`` cannot establish official success. The video is limited to the first
2,000 recorded frames, so it may not cover the full episode.

Explore keeps ``run.log``, the transcript, and server logs at the run root,
but writes ``states.json``, ``episode.mp4``, and
``terminal_receipt.json`` under ``<output-dir>/sessions/session_001/`` (then
``session_002/``, and so on). Dashboard creates each TaskRun under
``<output-dir>/tasks/0001_<task-name>_s<seed>/``; its task state and video live
there, and an Explore TaskRun adds ``sessions/session_001/``. See
:doc:`../guides/dashboard` for the shared viewer.

Task Memory
-----------

BEHAVIOR uses the shared Markdown/YAML ``MemoryManager`` corpus and memory
tools; see :doc:`../guides/memory` for memory profiles and exploration. Eval
uses read-only memory (HF by default, or local with
``--memory-profile local``); Explore writes drafts to its local inbox. An empty
corpus is valid but provides no advice. Use the same ``--memory-dir`` for local
runs that should share published experience.

DINO derives an optional cache under ``<memory-dir>/dino_v2/`` from published
``task-specific`` audit/recipe pairs and their verified evidence. Only head RGB
frames are embedded; wrist frames remain diagnostic evidence. Valid success
evidence is published with the task audit/recipe pair through MemoryManager.
When published sources exist, a query rebuilds a missing or stale cache. Its
matches are historical advice in tool results, not current observations or
another corpus.
To rebuild the cache ahead of a run, use the BEHAVIOR interpreter; do not edit
cache files by hand:

.. code-block:: bash

   "$BEHAVIOR_PYTHON" -m robots.behavior.build_memory_cli \
     --memory-dir "$RPENT_REPRO_ROOT/memory/behavior" \
     --source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --weights "$DINOV2_WEIGHTS" \
     --cuda-device 0

Exploration Mode
----------------

Use the standard RPent Explore entry point for a bounded sequence of sessions:

.. code-block:: bash

   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent" --robot behavior \
     --explore \
     --explore-sessions 3 \
     --task-name picking_up_trash --public-seed 0 \
     --output-dir "$RPENT_REPRO_ROOT/runs/behavior-explore" \
     --memory-dir "$RPENT_REPRO_ROOT/memory/behavior" \
     --planner codex --reasoning-effort low \
     --behavior-repo "$RPENT_REPRO_ROOT/RLinf" \
     --behavior-python "$RPENT_REPRO_ROOT/venvs/behavior/bin/python" \
     --activity-instance-dir \
       "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --cuda-device 0 \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS"

In BEHAVIOR Explore, one session is exactly one attempt. Each session starts a fresh
environment sidecar, episode, and ``sessions/session_NNN`` output directory;
the VLA and DINO sidecars stay shared across sessions. The planner cannot reset
inside an invocation, and ``--explore-attempts-per-session`` values above zero
are rejected.

Dashboard
---------

The Dashboard Session uses three services and one memory manager:

- ``env``: the task-scoped official BEHAVIOR/OmniGibson environment;
- ``vla``: the session-shared ``robots/behavior/vla_server.py`` adapter built
  on the common Pi0.5 facade;
- ``dino``: the shared ``robots/behavior/dino_v2/server.py`` episode-memory
  embedding service;
- ``memory``: a task-scoped MemoryManager, not a separate service.

Start a Dashboard Session with:

.. code-block:: bash

   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent" \
     --robot behavior --dashboard \
     --planner codex --reasoning-effort low \
     --behavior-repo "$RPENT_REPRO_ROOT/RLinf" \
     --behavior-python "$RPENT_REPRO_ROOT/venvs/behavior/bin/python" \
     --activity-instance-dir \
       "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS" \
     --cuda-device 0 \
     --memory-profile local \
     --memory-dir "$RPENT_REPRO_ROOT/memory/behavior" \
     --output-dir "$RPENT_REPRO_ROOT/runs/behavior-dashboard"

Open the printed Dashboard URL. Once the shared VLA and DINO services are
ready, submit this task in the Dashboard:

.. code-block:: text

   /rpent-task turning_on_radio 1

This starts a task-scoped ENV. Press Ctrl+C in the launch terminal to stop the
Session; closing the browser does not stop its services. See
:doc:`../guides/dashboard` for the shared Session workflow.

The Dashboard uses the common Start Session flow and head/left-wrist/right-
wrist camera views. It has no manual robot-control buttons. BEHAVIOR adds nine
robot-specific tools alongside the common memory and ``finish`` tools:
``pi0_nav_pick``, ``observe``,
``pixel_to_world``, ``navigate_to``, ``move_to``, ``rotate_wrist``, ``close``,
``open``, and ``press``. ``move_to(hand=both)`` coordinates both arms through
cuRobo collision-checked trajectories; wrist rotation uses the same planner.
Navigation executes a bounded straight base segment or rotation and rejects
obstructed paths. RGB-D projections use the current physical camera frame;
R1Pro has no movable head camera, so non-center ``head_view`` presets are
rejected. ``press`` advances an already aligned hand at most 2 cm for at most
10 seconds, stopping on external contact or episode end.
Contact is not verified button contact; visual hand checks remain unverified.
Planning, collision, tracking and duration failures are reported explicitly.
Only raw ``info["done"]["success"]`` establishes task success. Motion primitives
return their final observation for the next policy call and streaming video;
VLA chunks additionally record each returned environment frame.
