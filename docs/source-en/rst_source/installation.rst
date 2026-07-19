Installation
============

RPent runs on top of a forked branch of
`RLinf <https://github.com/RLinf/RLinf>`_, which supplies the LIBERO /
RoboCasa simulators and the VLA training / inference stack. The two
repositories are meant to live side by side in a single ``workspace/``
directory.

Prerequisites
-------------

- Linux with an NVIDIA GPU (LIBERO / RoboCasa render on EGL).
- CUDA 12.x drivers matching your GPU.
- Python 3.10+.
- `uv <https://github.com/astral-sh/uv>`_ (used to sync RPent's extra
  dependencies on top of the RLinf-provided virtualenv).
- ``git``, ``bash``, and a working C toolchain for MuJoCo / robosuite.

You will also want:

- An API key for at least one LLM provider — Anthropic, OpenAI, or an
  OpenAI-compatible chat endpoint — for the reasoning brain.
- A VLA checkpoint. For LIBERO / Pi0.5 the recommended checkpoint lives
  at `HuggingFace: rlinf-pi05-libero-130-fullshot-sft
  <https://huggingface.co/datasets/RLinf/rlinf-pi05-libero-130-fullshot-sft>`_.

1. Clone RLinf and RPent side by side
-------------------------------------

.. code-block:: bash

   mkdir workspace && cd workspace
   # RPent depends on a forked branch of RLinf; the fork will be merged
   # back to RLinf main after more iterations.
   git clone https://github.com/jx-qiu/RLinf -b feature/physicalagent rlinf
   git clone https://github.com/RLinf/RPent rpent

The exact layout you should end up with:

.. code-block:: text

   workspace/
   ├── rlinf/     # RLinf fork (simulators + VLA infrastructure)
   └── rpent/     # RPent (the agent framework — this repo)

2. Create the RLinf virtualenv (LIBERO + openpi)
------------------------------------------------

RLinf ships a single-command installer that builds a virtualenv with
LIBERO + openpi (the Pi0.5 runtime) pre-wired:

.. code-block:: bash

   cd rlinf
   bash requirements/install.sh embodied \
     --env libero --model openpi --use-mirror \
     --venv ../.venv-opi-libero
   cd ..
   source .venv-opi-libero/bin/activate

The installer places the virtualenv one level above ``rlinf/`` so it can
be reused by RPent. Activate it *before* the next step.

3. Install RPent's extra dependencies
-------------------------------------

With the RLinf virtualenv active, sync RPent's extras on top of it and
run the LIBERO PRO/PLUS setup script:

.. code-block:: bash

   cd rpent
   uv sync --active --inexact
   bash scripts/install_libero_pro_plus.sh

``uv sync --active --inexact`` adds RPent's own Python dependencies
(pydantic-ai, the Claude Agent SDK, the Codex SDK, FastAPI for the
dashboard, …) without disturbing the RLinf packages that share the
environment. ``install_libero_pro_plus.sh`` patches the LIBERO
distribution to expose the ``pro`` and ``plus`` task variants used
throughout RPent.

4. (Optional) RoboCasa
----------------------

RoboCasa (kitchen-scale, long-horizon manipulation) has its own
one-shot setup script:

.. code-block:: bash

   bash scripts/setup_robocasa.sh

See `docs/SETUP_ROBOCASA.zh.md
<https://github.com/RLinf/RPent/blob/main/docs/SETUP_ROBOCASA.zh.md>`_
for the full RoboCasa365 + RLDX-1 walkthrough (assets, controller
configs, VLA checkpoints).

5. (Optional) Real-world robot dependencies
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
