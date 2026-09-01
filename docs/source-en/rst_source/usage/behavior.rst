BEHAVIOR
========

`BEHAVIOR-1K <https://behavior.stanford.edu/>`_ is a benchmark for
long-horizon household activities in photorealistic, interactive environments.
RPent currently exposes two reviewed task families, ``turning_on_radio`` and
``picking_up_trash``, through the source-editable ``robots/behavior``
integration. The default VLA is **Pi0.5**, served by
``robots/behavior/vla_server.py``.

Installation
------------

Use Ubuntu 22.04, an NVIDIA RTX GPU supported by Isaac Sim 4.5, and a current
NVIDIA driver. The source installer checks the host commands and builds two
independent Python 3.10 environments:

- the **RPent environment** runs the CLI, planner, and Dashboard;
- the **BEHAVIOR environment** runs RLinf, OmniGibson, Isaac Sim, Pi0.5,
  DINOv2, and every BEHAVIOR sidecar process.

Do not merge these environments. Isaac Sim and OpenPI require compatibility
pins that are intentionally different from the general RPent dependency set.

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent

   export RPENT_REPRO_ROOT="$PWD/.behavior-runtime"
   bash scripts/install_behavior_runtime.sh

The installer pins ``uv`` and the reviewed RLinf revision, invokes the official
RLinf BEHAVIOR installer, and then performs one final compatibility repin. In
particular, FastAPI/Pydantic are restored after Isaac installation and the
OpenPI transformer replacement is copied only after the final Transformers
version is installed. No package install runs after that replacement.

The complete package freezes, source revisions, installation log, and
``uv pip check`` report are written below ``$RPENT_REPRO_ROOT``. The BEHAVIOR
environment installs RPent editable so the directly launched sidecar scripts
can import its source. Consequently, ``pip check`` also sees planner-only
package metadata that asks for newer Pydantic/Starlette versions, although
those planner packages run from the separate RPent environment. The report can
also contain the reviewed upstream conflicts around ``rlinf-openpi`` and
``lerobot`` torch/torchvision/torchcodec, ``tensorflow-addons`` typeguard, and
``tensorflow-metadata`` protobuf pins. Do not resolve this report by upgrading
packages after the final repin. The installer separately requires the exact
reviewed versions, critical imports, a CUDA tensor smoke, and the BEHAVIOR
self-check to pass.

BEHAVIOR assets
---------------

The policy checkpoint is not a replacement for the OmniGibson dataset. Prepare
the full licensed BEHAVIOR-1K data root from inside the BEHAVIOR environment:

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

The BEHAVIOR task archive is larger than 30 GB. After extraction, the data root
must contain all four entries below; missing ``scenes`` causes environment
startup to fail, and missing ``omnigibson.key`` prevents encrypted USD assets
from loading.

.. code-block:: text

   BEHAVIOR-1K-datasets/
     2025-challenge-task-instances/
     behavior-1k-assets/
       scenes/
     omnigibson-robot-assets/
     omnigibson.key

VLA configuration
-----------------

Download the BEHAVIOR Pi0.5 checkpoint
`RLinf-Pi05-BEHAVIOR-1K-PT50-CS32
<https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_, then point
``PI05_CHECKPOINT_PATH`` at the downloaded directory:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/your/pi05-behavior-model
   hf download RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 \
     --local-dir "$PI05_CHECKPOINT_PATH"

The policy receives three RGB views and the compact R1Pro state. Its
``predict`` RPC returns a batched ``[1, T, 23]`` tensor, and the executor
consumes each ``[T, 23]`` action chunk. Keep the checkpoint directory outside
the Python package and bind every run explicitly with ``PI05_CHECKPOINT_PATH``
or ``--policy-checkpoint``.

DINOv2 configuration
---------------------

BEHAVIOR uses a reviewed `DINOv2 <https://github.com/facebookresearch/dinov2>`_
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

DINOv2 occupies the shared visual-memory component role in the BEHAVIOR
runtime. It is not a segmentation model and does not replace SAM3 masks;
current target localization uses fresh observations and the public geometry
tools. The accepted DINOv2 source revision and both asset SHA-256 identities
are pinned in ``robots/behavior/dino_v2/encoder.py``; the runtime
rejects assets that do not match that public contract.

Task selection
--------------

A BEHAVIOR run uses the following task settings:

- ``--task-name`` selects ``turning_on_radio`` or ``picking_up_trash``.
- ``--public-seed`` selects a stable public seed that maps to one official
  BEHAVIOR activity instance.
- ``--behavior-mode`` selects ``eval`` or ``explore``. Evaluation is the
  default. Explore attempts are launched by the outer harness described below.
- ``--max-episode-steps`` sets the episode step budget.

``--task`` and ``--seed`` are compatibility aliases for ``--task-name`` and
``--public-seed``. New commands should use the explicit BEHAVIOR names.

.. _behavior-core-tasks:

Core BEHAVIOR tasks
~~~~~~~~~~~~~~~~~~~

The public seed split is part of the source-controlled task specification.
Explore and Eval use disjoint official activity instances.

.. list-table::
   :header-rows: 1
   :widths: 24 38 18 20

   * - Task
     - Instruction
     - Explore seeds
     - Eval seeds
   * - ``turning_on_radio``
     - Turn on the radio receiver on the living-room table.
     - ``0``
     - ``1``-``9``
   * - ``picking_up_trash``
     - Put the three soda cans from the living room into the kitchen trash can.
     - ``0``-``9``
     - ``10``-``19``

The complete public-seed-to-instance mapping is defined in
``robots/behavior/task_specs.py``. Native activity instance IDs are deployment
details and should not be substituted for public seeds on the CLI.

Verify assets and run
---------------------

Validate the complete data tree, policy checkpoint contract, and both pinned
DINOv2 SHA-256 identities before starting Isaac Sim:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/your/pi05-behavior-model
   export DINOV2_SOURCE_ARCHIVE=/path/to/dinov2-source.tar.gz
   export DINOV2_WEIGHTS=/path/to/dinov2_vits14_pretrain.pth

   scripts/verify_behavior_assets.sh

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

The first environment load commonly takes 1.5 to 5 minutes. xFormers,
deprecation, audio, and headless GLFW warnings can be non-fatal; use the
component logs described in the Dashboard section to distinguish warnings from
startup failure. To switch planners, see :doc:`configure_planner`.

Exploration and local-memory evaluation
---------------------------------------

RPent supports two BEHAVIOR run modes:

- **Exploration** is a memory-generation workflow. The outer harness may run
  multiple attempts, but every attempt owns a fresh RPent process, planner
  invocation, environment server, and episode. BEHAVIOR does not reset an
  episode inside one planner invocation.
- **Evaluation** is the default, single-attempt path. It reads an explicitly
  reviewed episode-memory catalog when ``--behavior-memory-dir`` is provided
  and does not retry the episode.

Use an Eval seed for local-memory evaluation:

.. code-block:: bash

   rpent --robot behavior \
     --task-name turning_on_radio --public-seed 1 \
     --behavior-mode eval \
     --planner codex --model gpt-5.5 \
     --behavior-memory-dir /path/to/reviewed-behavior-memory \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS"

Omitting ``--behavior-memory-dir`` selects a legal empty episode catalog. It
does not download or silently substitute task-specific memory.

Launch repeated Explore attempts through the BEHAVIOR-owned outer harness:

.. code-block:: bash

   python -m robots.behavior.harness explore \
     --attempts 3 \
     --output-dir /path/to/behavior-explore \
     -- \
     --task-name picking_up_trash --public-seed 0 \
     --planner codex --model gpt-5.5 \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS"

Explore artifacts can be reviewed and promoted into recipes, task memory, or
DINO-indexed episode memory. Candidate Explore evidence must remain separate
from held-out Eval artifacts. A successful run is recognized only from the
official raw ``info["done"]["success"]`` value recorded in the terminal
receipt; planner or primitive completion is not a substitute.

What runs where
---------------

- **env_server** (``robots/behavior/env_server.py``) owns the official
  BEHAVIOR/OmniGibson environment. It exposes reset, observation, action,
  camera rendering, and official success receipts over RPent RPC.
- **vla_server** (``robots/behavior/vla_server.py``) owns the Pi0.5 BEHAVIOR
  checkpoint and exposes ``predict`` over RPent RPC.
- **dino_server** (``robots/behavior/dino_v2/server.py``) owns the DINOv2-S/14
  encoder and serves episode-memory embeddings.
- **toolkit** (``robots/behavior/toolkit.py``) defines the public tools the
  planner can call and records observations, action traces, and terminal
  receipts.

The environment process has its own GPU binding. VLA and DINOv2 share the model
GPU by default. Each local CUDA child receives one explicit physical
``CUDA_VISIBLE_DEVICES`` value.

Tools the planner can call
--------------------------

BEHAVIOR tools fall into three groups. The active toolkit schema remains the
source of truth for a run.

**VLA-backed action:**

- ``pi0_nav_pick(instruction, chunks)`` uses Pi0.5 for navigation and grasping.

**Observation and analytic actions:**

- ``observe(...)`` reads fresh head or wrist-camera observations.
- ``pixel_to_world(...)`` back-projects a fresh image pixel into the scene.
- ``navigate_to(...)`` plans mobile-base motion.
- ``move_to(...)`` and ``move_both_to(...)`` plan one-arm or dual-arm motion.
- ``rotate_wrist(...)`` changes wrist orientation.
- ``close(...)`` and ``open(...)`` control the grippers.
- ``press(...)`` executes a guarded contact action.

The public schema describes the reviewed planner route, but a deployment may
report ``manual_motion_unavailable`` when its official RLinf backend has no
reviewed manual-motion adapter. In that case these manual motion tools must not
be treated as executable fallbacks; ``pi0_nav_pick`` remains the validated
motion entrypoint for that deployment.

**Safety, state, and termination:**

- ``get_prepared_motion_status(...)`` reads prepared-motion execution status.
- ``save_robot_state_checkpoint(...)`` records a planner-visible state marker.
- ``finish(status, summary)`` ends the planner run and writes its receipt.

Physical action tools advance the environment. Observation, status, and state
checkpoint tools do not by themselves establish task success.

Live dashboard
--------------

Add ``--dashboard`` to start a long-lived local Dashboard Session. The VLA and
DINOv2 services are shared across TaskRuns, while every TaskRun receives a
fresh environment:

.. code-block:: bash

   TASK_NAME=turning_on_radio PUBLIC_SEED=1 \
     scripts/run_behavior_dashboard.sh

Open the printed URL, confirm the Session configuration, and start a TaskRun
from the page with:

.. code-block:: text

   /rpent-task turning_on_radio 1

The Dashboard shows planner reasoning, the head and wrist-camera frames, and
the action timeline. A new ``/rpent-task`` starts a fresh environment. The
Dashboard does not change the official success definition. Use
``--dashboard-language zh-cn`` for the Chinese UI.

The launcher prints the exact output directory. Diagnose component startup in:

.. code-block:: text

   <output-dir>/run.log
   <output-dir>/behavior_vla_server.log
   <output-dir>/behavior_dino_server.log
   <output-dir>/tasks/<task-run>/behavior_env_server.log

Bringing your own VLA
---------------------

If you have a BEHAVIOR-compatible VLA that is not Pi0.5, swap the model client
without changing the environment by:

1. Exposing the same ``predict`` RPC contract and returning a finite
   ``[1, T, 23]`` action tensor in the BEHAVIOR policy layout.
2. Pointing RPent at it with ``--vla-endpoint [protocol://]host:port``.
3. Updating ``robots/behavior/toolkit.py`` only if the public tool surface must
   change.

See :doc:`../development/add_primitive` for the tool-extension walkthrough.

Reproducing results
-------------------

The BEHAVIOR workflow and benchmark recipe are still under active exploration;
RPent does not claim a BEHAVIOR benchmark success rate at this stage.

For a reproducible run, record the RPent commit, pinned RLinf/OmniGibson/Isaac
environment, policy checkpoint digest, DINOv2 source and weight digests,
task/public-seed mapping version, planner and model, GPU bindings, and complete
output directory. Before a full run, verify the lightweight RPent contract:

.. code-block:: bash

   python -m robots.behavior.selfcheck

The self-check validates plugin import, RobotSpec/CLI parsing, task/seed
mapping, and the public tool count. It does not render the prompts, start the
simulator, or establish task success. A runtime result is reportable as
successful only when ``terminal_receipt.json`` contains an
``official_success_receipt`` whose ``source`` is
``info["done"]["success"]`` and whose ``raw_done.success`` is ``true``.
