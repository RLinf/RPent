BEHAVIOR
========

`BEHAVIOR-1K <https://behavior.stanford.edu/>`_ support is maintained as a
source-editable RPent robot plugin for long-horizon household manipulation. The
normal ``rpent`` wheel still packages only ``rpent*`` modules. It does not ship
``robots/behavior``, OmniGibson or Isaac Sim, the official BEHAVIOR dataset,
large DINOv2 assets, policy checkpoints, or recorded episode memory.

Install boundary
----------------

Install the stable RPent-side dependencies with:

.. code-block:: bash

   pip install -e ".[behavior]"

The ``behavior`` extra covers the common RPent runtime pieces used by the
BEHAVIOR plugin: RLinf, OpenPI, PyTorch/TorchVision for Pi0.5 and DINOv2 image
encoders, Pillow/ImageIO video helpers, and RPent's HTTP/socket RPC stack. It is
not included in ``.[full]`` because BEHAVIOR also depends on a pinned source
checkout, official simulator assets, and heavyweight runtime resources that are
managed outside the wheel.

Use the pinned upstream BEHAVIOR installation instructions for OmniGibson,
Isaac Sim, BEHAVIOR data, robot assets, and environment variables such as
``OMNI_KIT_ACCEPT_EULA`` and the BEHAVIOR asset root. After the source tree and
resources are installed, run the plugin self-check:

.. code-block:: bash

   python -m robots.behavior.selfcheck

The self-check verifies the RPent-side plugin import, task/seed mapping, prompt
contract, and public tool count. It does not start OmniGibson or validate the
official assets, DINO weights, or policy checkpoint. Verify those heavyweight
runtime resources with the pinned upstream setup checks and a bounded smoke run;
do not copy them into RPent package data.

Runtime scope
-------------

The current RPent BEHAVIOR runtime is scoped to the reviewed Radio and Trash
task surfaces:

- ``turning_on_radio`` for radio button manipulation.
- ``picking_up_trash`` for soda-can disposal into the kitchen trash can.

Other BEHAVIOR tasks may be useful for development, but they are outside this
documented runtime contract until they receive their own task specs, prompts,
memory, and receipts.

Minimal evaluation
------------------

Run evaluation from the source checkout that contains ``robots/behavior``:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/pi05-b1kpt50-cs32
   export BEHAVIOR_ENV_GPU=2
   export BEHAVIOR_MODEL_GPU=7

   rpent --robot behavior \
     --task-name turning_on_radio \
     --public-seed 1 \
     --behavior-mode eval \
     --model gpt-5.5 \
     --behavior-env-cuda-device "$BEHAVIOR_ENV_GPU" \
     --behavior-model-cuda-device "$BEHAVIOR_MODEL_GPU" \
     --dino-source-archive /path/to/dinov2-source.tar.gz \
     --dino-weights /path/to/dinov2_vits14_pretrain.pth \
     --behavior-memory-dir /path/to/reviewed-episode-catalog \
     --output-dir /path/to/behavior-eval

Evaluation is the formal, single-pass measurement path. It must preserve the
raw action trace and final artifacts. Official task success is the raw
BEHAVIOR bit, ``info["done"]["success"]`` as recorded in
``info_done.success``. Treat planner progress, primitive success,
``task_success``, workflow sealing, terminal receipts, and public publication
state as separate claims.

Independent Explore harness
---------------------------

Explore is a separate memory-generation workflow. It may run repeated attempts,
fresh planner sessions, and local memory review, but it is not the held-out
success-rate measurement:

.. code-block:: bash

   export BEHAVIOR_ENV_GPU=2
   export BEHAVIOR_MODEL_GPU=7

   python -m robots.behavior.harness explore \
     --attempts 3 \
     --output-dir /path/to/behavior-explore \
     -- \
     --task-name picking_up_trash \
     --public-seed 0 \
     --model gpt-5.5 \
     --behavior-env-cuda-device "$BEHAVIOR_ENV_GPU" \
     --behavior-model-cuda-device "$BEHAVIOR_MODEL_GPU" \
     --dino-source-archive /path/to/dinov2-source.tar.gz \
     --dino-weights /path/to/dinov2_vits14_pretrain.pth \
     --behavior-memory-dir /path/to/reviewed-episode-catalog

Explore output can seed reviewed recipes, task memory, and episode memory, but
it must keep candidate/development evidence separate from formal evaluation
artifacts.

Dashboard
---------

The standard RPent Dashboard launcher supports BEHAVIOR and reuses the shared
VLA and DINO components across TaskRuns while giving each TaskRun a fresh env:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/pi05-b1kpt50-cs32
   export BEHAVIOR_ENV_GPU=2
   export BEHAVIOR_MODEL_GPU=7

   rpent --robot behavior --dashboard \
     --model gpt-5.5 \
     --behavior-env-cuda-device "$BEHAVIOR_ENV_GPU" \
     --behavior-model-cuda-device "$BEHAVIOR_MODEL_GPU" \
     --dino-source-archive /path/to/dinov2-source.tar.gz \
     --dino-weights /path/to/dinov2_vits14_pretrain.pth

Start a TaskRun from the page with:

.. code-block:: text

   /rpent-task turning_on_radio 1

The lower-level BEHAVIOR Dashboard module is also available for direct manual
control and debugging:

.. code-block:: bash

   export BEHAVIOR_ENV_GPU=2
   export BEHAVIOR_MODEL_GPU=7

   python -m robots.behavior.dashboard \
     --task-name turning_on_radio \
     --public-seed 1 \
     --behavior-env-cuda-device "$BEHAVIOR_ENV_GPU" \
     --behavior-model-cuda-device "$BEHAVIOR_MODEL_GPU" \
     --dino-source-archive /path/to/dinov2-source.tar.gz \
     --dino-weights /path/to/dinov2_vits14_pretrain.pth

The Dashboard is for observing and steering a BEHAVIOR run. It does not change
the official success definition.

The env process has its own GPU binding. VLA and DINO intentionally share the
model GPU, matching the shared VLA/SAM3 component pattern used by LIBERO. Every
local CUDA child still receives one explicit physical ``CUDA_VISIBLE_DEVICES``
value. ``--cuda-device`` remains a shared fallback when both component-specific
flags should resolve to the same physical GPU.

What runs where
---------------

- **env_server** (``robots/behavior/env_server.py``) owns the official
  BEHAVIOR/OmniGibson process through the pinned source checkout and exposes
  reset, observation, action, Dashboard-control, and raw success receipts over
  RPent RPC.
- **vla_server** (``robots/behavior/vla_server.py``) owns the Pi0.5 BEHAVIOR
  checkpoint and returns BEHAVIOR ``[T,23]`` actions through the RPent runtime
  contract.
- **dino_server** (``robots/behavior/dino_server.py``) owns DINOv2 image
  embeddings for episode-memory retrieval.
- **toolkit** (``robots/behavior/toolkit.py``) exposes only public planner
  tools and records public observations, action traces, and terminal receipts.

Tools the planner can call
--------------------------

BEHAVIOR tools are task-scoped. The current public surface contains:

- VLA-backed action: ``pi0_nav_pick(instruction, chunks)``.
- Observation and geometry: ``observe(...)`` and ``pixel_to_world(...)``.
- Analytic motion and gripper actions: ``move_to(...)``, ``move_both_to(...)``,
  ``rotate_wrist(...)``, ``close(...)``, ``open(...)``, ``press(...)``, and
  ``navigate_to(...)``.
- Safety and receipts: ``get_prepared_motion_status(...)``,
  ``save_robot_state_checkpoint(...)``, and ``finish(status, summary)``.

Tool availability can narrow when a runtime component is intentionally absent;
the active toolkit schema is the source of truth for a run.

VLA and DINO components
-----------------------

The BEHAVIOR policy path uses the shared Pi0.5 profile
``pi05-b1kpt50-cs32``. Point ``PI05_CHECKPOINT_PATH`` at the validated local
checkpoint and keep task-specific registries from silently replacing it.

DINOv2 visual retrieval uses a reviewed local DINOv2-S/14 deployment for image
embedding and episode-memory lookup. The DINO source archive and weights are
runtime assets, not wheel data. Keep their digests in the resource binding or a
separate deployment audit record.

Episode memory
--------------

BEHAVIOR memory is runtime data. It may include global task notes, reviewed
recipes, DINO-indexed episode memory, and run receipts. Keep it outside the
Python package and bind each run to the memory revision it actually used.

Receipts and raw success
------------------------

For every Eval or Explore run, inspect the public tool records and
``terminal_receipt.json``. A success claim must be backed by the raw
``info["done"]["success"]`` evidence carried by its official receipt; planner
status and local primitive completion are not substitutes.

Troubleshooting
---------------

- ``ModuleNotFoundError: robots.behavior`` means the BEHAVIOR source plugin is
  not on ``PYTHONPATH`` or was not installed editable.
- OmniGibson or Isaac startup failures should be fixed from the upstream pinned
  install guide, not by adding simulator packages to the ``behavior`` extra.
- Missing ``PI05_CHECKPOINT_PATH`` or a digest mismatch should fail before VLA
  execution. Re-run ``python -m robots.behavior.selfcheck`` after changing
  checkpoints.
- Video or frame extraction failures often mean the ImageIO ffmpeg backend is
  missing; reinstall the ``behavior`` extra in the active environment.
- If a run reports progress but no official success receipt, classify it as a
  non-success unless the raw trace contains ``info_done.success=true``.

Known smoke-test boundary
-------------------------

Short BEHAVIOR smoke runs prove that the source checkout, simulator process,
RPC wiring, image path, and Pi0.5 call path can start. They are not held-out
evaluation, do not establish benchmark success rate, and must not be reported
as official task completion without the raw BEHAVIOR success bit and receipt.
