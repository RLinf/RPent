BEHAVIOR
========

`BEHAVIOR-1K <https://behavior.stanford.edu/>`_ provides long-horizon household
tasks in OmniGibson. RPent exposes ``turning_on_radio`` and
``picking_up_trash`` from the sibling source package ``robots/behavior``.

Like LIBERO, RoboCasa, RoboTwin, and Franka, the robot package provides
``get_robot_spec()`` for CLI/config/runtime hooks and ``get_toolkit()`` for the
public tools. BEHAVIOR-specific lifecycle code stays inside
``robots/behavior``. The shared ``--explore`` entry point supports BEHAVIOR
while preserving its one-attempt-per-session environment lifecycle.

Installation status
-------------------

BEHAVIOR is a source-checkout robot plugin and uses two independent Python
3.10 environments:

- the **RPent venv** runs the CLI, planner, Dashboard, and MemoryManager;
- the **BEHAVIOR venv** runs RLinf, OmniGibson, Isaac Sim, and Pi0.5.

The ordinary ``rpent`` package ships the framework package only. The
``.[behavior]`` extra installs lightweight RPent-side helpers, but it does not
package ``robots/behavior`` and does not install the complete simulator,
assets, or checkpoint. Run the BEHAVIOR setup from an RPent source checkout:

.. code-block:: bash

   python -m pip install -e ".[behavior]"
   export RPENT_REPRO_ROOT="$PWD/.behavior-runtime"
   export UV_CACHE_DIR="$RPENT_REPRO_ROOT/uv-cache"
   python -m robots.behavior.install_runtime

The installer keeps RPent editable only in the RPent venv; ENV, VLA, and DINO
load the same source checkout without installing RPent in the simulator venv.
It clones the reviewed RLinf revision, invokes the official RLinf BEHAVIOR installer,
applies the reviewed CUDA/OpenPI/LeRobot compatibility pins inside the
BEHAVIOR runtime venv, verifies critical imports and CUDA, and writes freezes
plus source identities under ``$RPENT_REPRO_ROOT/manifests``. Use a new
``RPENT_REPRO_ROOT`` for a fresh install; the script refuses to overwrite a
wrong or dirty RLinf checkout.

Motion planning uses NVlabs/cuRobo v0.8.0 at commit
``4ea77366ca48ee453e7df139e39fa6532af49f3b`` in the BEHAVIOR venv.
The installer applies constraints before its final repin, retaining NumPy
1.26.4, Torch 2.5.1+cu124 and Isaac Sim 4.5.0.0. Do not install cuRobo with
an unconstrained resolver: a NumPy 2 upgrade is incompatible with this stack.

Simulator assets
----------------

Accept the BEHAVIOR/OmniGibson licences, choose a dedicated data root, and use
the standard asset command. It invokes the three official OmniGibson download
functions in the BEHAVIOR venv rather than importing OmniGibson into the RPent
environment:

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   export BEHAVIOR_PYTHON="$RPENT_REPRO_ROOT/venvs/behavior/bin/python"
   python -m robots.behavior.assets_cli --accept-license --skip-existing

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

``python -m robots.behavior.assets_cli --verify`` verifies the required
OmniGibson layout and the source-controlled checkpoint size/SHA binding. The
shared #136 Pi0.5 component receives head, left-wrist, right-wrist, and raw
R1Pro proprio data. The raw RPC result is ``[1, 32, 23]``; the common client
returns ``[32, 23]``.

.. code-block:: bash

   python -m robots.behavior.assets_cli --verify

DINOv2 derived cache
--------------------

BEHAVIOR keeps a reviewed `DINOv2 <https://github.com/facebookresearch/dinov2>`_
ViT-S/14 deployment for whole-image embeddings over the official Memory
corpus.
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

The DINO cache is rebuilt from published ``task-specific`` audit/recipe pairs and
their verified evidence in ``task-specific/artifacts/<cell>/``. Only head RGB frames
are embedded; wrist frames remain diagnostic evidence. Successful runs prepare
evidence locally, then MemoryManager publishes it with the task pair under the
same merge lock. ``dino_v2/`` is disposable derived data, not another corpus.
Use the BEHAVIOR interpreter to rebuild it; do not edit cache files by hand.

.. code-block:: bash

   "$BEHAVIOR_PYTHON" -m robots.behavior.build_memory_cli \
     --memory-dir /path/to/memory/behavior \
     --source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --weights "$DINOV2_WEIGHTS" \
     --cuda-device 1

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

The first environment load can take several minutes. The environment, VLA, and
DINO are separate processes, and each receives only its explicitly selected GPU.

Official MemoryManager
----------------------

BEHAVIOR uses the same Markdown/YAML ``MemoryManager`` format and common memory
tools as the other robots. Normal Eval follows the shared default
``--memory-profile hf``. Explore uses local memory and writes only through the
shared inbox flow. The DINO cache is derived from reviewed Memory corpus
records; when configured, its advisory is attached to public tool receipts and
remains historical guidance only.

- Eval creates one ``MemoryManager`` with ``read_only`` access from the default
  HF corpus unless ``--memory-profile local`` is explicitly requested.
- Explore creates one ``MemoryManager`` with ``inbox_write`` access scoped to
  ``<memory-dir>/_internal/inbox/<recipe-tag>``.
- ``MEMORY.md``, ``global/``, ``task-family/``, and ``task-specific/`` hold the
  published corpus. Solved audit/recipe pairs are copied to ``task-specific/`` using the
  shared MemoryManager names, including ``<tag>_recipe.jsonl`` for recipes.
- When merge processes a valid root-level draft, the cell inbox is archived to
  ``_internal/merged/<recipe-tag>``. An inbox containing only invalid drafts
  stays in place; conflicting prose is archived under ``_internal/conflicts/``.

An absent or empty corpus is valid, but it contains no advice. Pass the same
explicit ``--memory-dir`` only to local runs that should share reviewed memory.

Use the standard RPent Explore entry point for a bounded sequence of sessions:

.. code-block:: bash

   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent" --robot behavior \
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
     --behavior-repo "$RPENT_REPRO_ROOT/RLinf" \
     --behavior-python "$RPENT_BEHAVIOR_PYTHON" \
     --activity-instance-dir \
       "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS" \
     --output-dir /path/to/behavior-dashboard-run

The Dashboard uses the common Start Session flow and head/left-wrist/right-
wrist camera views. BEHAVIOR does not add robot-local manual buttons, a manual
control backend, or ``env.dashboard_*`` RPC methods. The public contract
registers nine executable planner primitives: ``pi0_nav_pick``, ``observe``,
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

The main logs are:

.. code-block:: text

   <output-dir>/run.log
   <output-dir>/behavior_vla_server.log
   <output-dir>/behavior_dino_server.log
   <output-dir>/tasks/<task-run>/behavior_env_server.log
   <output-dir>/tasks/<task-run>/episode.mp4
   <output-dir>/tasks/<task-run>/terminal_receipt.json
