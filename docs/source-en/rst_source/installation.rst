Installation
============

RPent installs with a single ``pip install``. The optional-dependency
extras pull the forked `RLinf <https://github.com/RLinf/RLinf>`_ runtime,
openpi, and the LIBERO simulator as git dependencies, so there is no
longer a separate RLinf clone or setup script.

Prerequisites
-------------

- Linux with an NVIDIA GPU (LIBERO / RoboCasa render on EGL).
- CUDA 12.x drivers matching your GPU.
- Python 3.10–3.11.
- ``git``, ``bash``, and a working C toolchain for MuJoCo / robosuite.

You will also want:

- An API key for at least one LLM provider — Anthropic, OpenAI, or an
  OpenAI-compatible chat endpoint — for the reasoning brain.
- A VLA checkpoint. For LIBERO / Pi0.5 the recommended checkpoint lives
  at `HuggingFace: rlinf-pi05-libero-130-fullshot-sft
  <https://huggingface.co/datasets/RLinf/rlinf-pi05-libero-130-fullshot-sft>`_.

1. Install RPent with pip
-------------------------

Clone RPent (for the CLI and run configs) and install with the extra for
the stack you want:

.. code-block:: bash

   git clone https://github.com/RLinf/RPent rpent && cd rpent
   pip install -e ".[full]"

``.[full]`` is the default end-to-end stack — the openpi Pi0.5 VLA and
the LIBERO-PRO simulator on top of the RLinf runtime.

Available extras:

.. list-table::
   :header-rows: 1

   * - Extra
     - Installs
   * - ``.[full]``
     - ``rlinf`` + ``openpi`` + ``libero-pro`` — the default run stack
   * - ``.[libero-pro]``
     - Base LIBERO + LIBERO-PRO simulator only
   * - ``.[libero-plus]``
     - Base LIBERO + LIBERO-plus simulator
   * - ``.[libero]``
     - Base LIBERO only
   * - ``.[openpi]``
     - openpi VLA only
   * - ``.[rlinf]``
     - RLinf runtime only

2. (Optional) RoboCasa
----------------------

RoboCasa (kitchen-scale, long-horizon manipulation) has its own
one-shot setup script:

.. code-block:: bash

   bash scripts/setup_robocasa.sh

See `docs/SETUP_ROBOCASA.zh.md
<https://github.com/RLinf/RPent/blob/main/docs/SETUP_ROBOCASA.zh.md>`_
for the full RoboCasa365 + RLDX-1 walkthrough (assets, controller
configs, VLA checkpoints).

3. (Optional) Real-world robot dependencies
-------------------------------------------

Franka and SO-101 support is being rolled in; when it lands, each
robot's driver ships as a package under ``robots/<name>/`` with its own
``README.md`` describing the SDK / firmware requirements. See
:doc:`usage/franka` and :doc:`usage/so101` for the current status.

Verifying the install
---------------------

The quickest way to confirm everything is wired correctly is to run one
LIBERO task end-to-end — see :doc:`quickstart`. If that succeeds, the
env server, VLA server, and reasoning brain are all healthy.

If something breaks:

- The env server writes its stdout / stderr to
  ``<output_dir>/env_server.log``.
- The VLA server writes to ``<output_dir>/vla_server.log``.
- The agent's own run log lives at ``<output_dir>/run.log``.

The three logs are always in that per-run scratch directory, so a
failed run is self-contained and easy to inspect.
