RoboDojo Backend Installation & Reproduction
=============================================

This guide explains how to set up, from scratch, the complete environment
required to reproduce this integration (``rpent --env robodojo``). The steps
below were validated end-to-end on a Linux x86_64 workstation (NVIDIA RTX PRO
6000 Blackwell, driver 580.173.02, Ubuntu 24.04).

Runtime composition
-------------------

The integration keeps three isolated runtimes, plus large assets and one
planner credential; RPent only orchestrates them and never mixes them:

.. list-table::
   :header-rows: 1

   * - Runtime
     - Python
     - Contents
     - Purpose
   * - RPent venv
     - 3.11
     - rpent itself + SAM3
     - agent loop / tools / memory
   * - robodojo-sim (conda)
     - 3.11
     - Isaac Sim 5.1 / IsaacLab 0.54.3 / CuRobo
     - simulation
   * - Pi_05 policy env (uv)
     - 3.11
     - OpenPI (via XPolicyLab)
     - policy server

Prerequisites
-------------

* Linux x86_64, NVIDIA GPU (Blackwell requires cu128 torch), ~100-180 GB disk;
* RoboDojo official checkpoint (``RoboDojo-sim-arx_x5-joint-0``, ~44.7 GB)
  and assets (~28.5 GB) from ModelScope;
* SAM3 checkpoint (~3.45 GB) plus the CLIP BPE vocabulary.

RPent + SAM3
------------

.. code-block:: bash

   uv pip install -e ".[rlinf,openpi,libero-pro,sam3]"   # full install
   # RoboDojo only: uv pip install -e ".[sam3]"
   # Pin mcp>=1.23,<2; on Blackwell use torch==2.7.1+cu128 / torchvision==0.22.1+cu128
   # Place the CLIP BPE vocabulary (bpe_simple_vocab_16e6.txt.gz) next to the
   # sam3 package and point SAM3_CHECKPOINT_PATH at sam3.pt

RoboDojo sources
----------------

Clone the official RoboDojo repository (including the XPolicyLab submodule)
and pin it to the validated commit (see the integration log and
``config/runtime.env`` wiring).

Simulation environment (Isaac Sim / IsaacLab / CuRobo)
------------------------------------------------------

* ``isaacsim[all,extscache]==5.1.0``;
* the pinned IsaacLab 0.54.3 fork, with ``isaacsim.asset.importer.urdf==2.4.31``
  materialized into the Kit extension cache (Isaac Sim wheels ship 2.4.30);
* CuRobo (``cuda_core``): pin ``viser==0.1.34``, ``tyro==0.9.0``,
  ``websockets==12.0``; set ``TORCH_CUDA_ARCH_LIST=12.0`` on Blackwell.

RoboDojo assets
---------------

Clone the ModelScope dataset repository (shared by assets and checkpoints):
14,506 LFS files (~41 GB declared). 9,224 eval-layout JSONs are stored as LFS
pointers without a ``filter=lfs`` attribute, so a plain ``git lfs checkout``
will not materialize them — use the SHA-256-verified scoped materializer
(see the integration log, section on LFS).

Pi_05 policy environment (OpenPI via XPolicyLab)
------------------------------------------------

Run ``uv sync --locked --no-dev --group lerobot`` in the vendored OpenPI, add
the minimal pinned extras, and install XPolicyLab editable; place the
checkpoint and ``dataset_stats.json`` in the policy directory layout expected
by the checkpoint convention.

Wiring & smoke test
-------------------

``spec.py`` reads ``config/runtime.env`` under the RoboDojo workspace
(``ROBODOJO_SIM_ENV``, ``ROBODOJO_PI05_ENV``, ``ROBODOJO_SOURCE_ROOT``, ...);
``SAM3_CHECKPOINT_PATH`` points at sam3.pt. Smoke chain:

.. code-block:: text

   robodojo.sh doctor → official Pi_05 debug gate → bare eval → rpent --env robodojo ...

A single run uses ~45 GB VRAM. Full command sequences, troubleshooting, and
known pins are in the integration log and
``robots/robodojo/guides/interface.md``.
