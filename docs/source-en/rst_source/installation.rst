Installation
============

RPent installs with a single ``pip install``. The optional-dependency
extras pull openpi and the LIBERO simulators.

Prerequisites
-------------

- Linux with an NVIDIA GPU (LIBERO renders on EGL).
- CUDA 12.x drivers matching your GPU.
- Python 3.10–3.11.
- ``git``, ``bash``, and a working C toolchain for MuJoCo / robosuite.

You will also want:

- An API key for at least one LLM provider — Anthropic, OpenAI, or an
  OpenAI-compatible chat endpoint — for the planner.
- A VLA checkpoint. For LIBERO / Pi0.5 the recommended checkpoint lives
  at `HuggingFace: RLinf-Pi05-LIBERO-130-fullshot-SFT
  <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT>`_.

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

2. Download the LIBERO simulator assets
---------------------------------------

The Python packages installed with pip do not include the large resource
files required to run LIBERO. Choose one command based on the extra
installed above. For the recommended ``.[full]`` extra, run the second
command:

.. code-block:: bash

   libero-download-assets --skip-existing      # .[libero]
   liberopro-download-assets --skip-existing   # .[libero-pro] / .[full]
   liberoplus-download-assets --skip-existing  # .[libero-plus]

These resources usually need to be downloaded only once;
``--skip-existing`` skips files that are already present.

.. tip::

   If your connection to Hugging Face is slow, download through the
   mirror by prefixing the command with ``HF_ENDPOINT``:

   .. code-block:: bash

      HF_ENDPOINT=https://hf-mirror.com liberopro-download-assets --skip-existing

3. (Optional) Real-world robot dependencies
-------------------------------------------

Franka and SO-101 support is being rolled in; when it lands, each
robot's driver ships as a package under ``robots/<name>/`` with its own
``README.md`` describing the SDK / firmware requirements. See
:doc:`usage/franka` and :doc:`usage/so101` for the current status.

Checking the installation
-------------------------

The quickest way to confirm everything is wired correctly is to run one
LIBERO task end-to-end — see :doc:`quickstart`. If that succeeds, the
env server, VLA server, and agent are all healthy.

If something breaks:

- The env server log is at ``<output_dir>/env_server.log``.
- The VLA server writes to ``<output_dir>/vla_server.log``.
- The agent's own run log lives at ``<output_dir>/run.log``.

The three logs are always in that per-run scratch directory, so a
failed run is self-contained and easy to inspect.
