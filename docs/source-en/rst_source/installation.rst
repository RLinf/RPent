Installation
============

RPent installs with a single ``pip install`` and provides several optional
dependency combinations.

Prerequisites
-------------

- Linux with an NVIDIA GPU (LIBERO and RoboTwin render on EGL).
- CUDA 12.x drivers matching your GPU. RoboTwin also requires a matching CUDA
  toolkit/NVCC, compiler toolchain, and the GL/EGL/Vulkan development libraries
  used by its headless renderer and native extensions.
- Python 3.10–3.12 (see ``pyproject.toml``'s ``requires-python``). For RoboTwin,
  use Python 3.11, which is the version covered by the full runtime validation.
- ``git``, ``bash``, and a working C toolchain for MuJoCo / robosuite.

You will also want:

- An API key for at least one LLM provider — Anthropic, OpenAI, or an
  OpenAI-compatible chat endpoint — for the planner.

1. Install RPent with pip
-------------------------

Clone RPent (for the CLI and run configs) and install with the extra for
the stack you want:

.. code-block:: bash

   git clone https://github.com/RLinf/RPent rpent && cd rpent
   pip install -e ".[full]"

``.[full]`` is the default end-to-end stack — the openpi Pi0.5 VLA, the
LIBERO-PRO and RoboCasa365 simulators, and SAM 3.0 on top of the RLinf
runtime.

Available extras:

.. list-table::
   :header-rows: 1

   * - Extra
     - Installs
   * - ``.[full]``
     - ``rlinf`` + ``openpi`` + ``libero-pro`` + ``robocasa`` + ``sam3`` — the default run stack
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
   * - ``.[robocasa]``
     - RoboCasa365 simulator + the RLDX-1 VLA; see :doc:`usage/robocasa`
   * - ``.[robotwin]``
     - RoboTwin simulation and LingBot inference dependencies
   * - ``.[sam3]``
     - SAM 3.0 only

``.[robotwin]`` installs the complete supported Python runtime and does not
require running RLinf's unified installer or cloning RoboTwin. The large
RoboTwin assets and LingBot checkpoint remain external resources and are
prepared with the commands in :doc:`usage/robotwin`.

During the pre-release period, this extra installs several dependencies from
fixed Git commits, so the installation needs access to GitHub. These will be
replaced by released package versions after publication.

2. Download the assets required to run LIBERO
---------------------------------------------

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
robot's env package will live under ``robots/<name>/`` with its own
``README.md`` describing the SDK / firmware requirements. See
:doc:`usage/franka` and :doc:`usage/so101` for the current status.

Checking the installation
-------------------------

The quickest way to confirm everything is wired correctly is to run one
LIBERO task end-to-end — see :doc:`quickstart`. If that succeeds, the
env server, VLA server, SAM3 server, and agent are all healthy.

If something breaks:

- The env server log is at ``<output_dir>/env_server.log``.
- The VLA server writes to ``<output_dir>/vla_server.log``.
- The SAM3 server writes to ``<output_dir>/sam3_server.log``.
- The agent's own run log lives at ``<output_dir>/run.log``.

These logs are always in that per-run scratch directory, so a
failed run is self-contained and easy to inspect.
