Installation
============

RPent installs with a single ``pip install`` and provides several optional
dependency combinations.

Prerequisites
-------------

- Linux with an NVIDIA GPU (LIBERO renders on EGL).
- CUDA 12.x drivers matching your GPU.
- Python 3.10–3.12 (see ``pyproject.toml``'s ``requires-python``).
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

``.[full]`` is the default end-to-end stack — the openpi Pi0.5 VLA,
the LIBERO-PRO simulator, and SAM 3.0 on top of the RLinf runtime.

Available extras:

.. list-table::
   :header-rows: 1

   * - Extra
     - Installs
   * - ``.[full]``
     - ``rlinf`` + ``openpi`` + ``libero-pro`` + ``sam3`` — the default run stack
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
     - RLinf + RoboCasa365 simulator
   * - ``.[sam3]``
     - SAM 3.0 only

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

3. (Optional) Install the RoboCasa365 stack
-------------------------------------------

The ``.[robocasa]`` extra installs the full RoboCasa365 stack — MuJoCo
3.3.1, the ARISE-Initiative robosuite fork, the pinned lerobot commit,
protobuf, and the ``robocasa`` package itself (a wheel from the
``github.com/qurakchin/robocasa`` fork, branch ``v1.0.1_rlinf``). The
fork is patched so that ``macros_private`` and ``assets`` are loaded
from env vars (``ROBOCASA_MACROS_PATH``, ``ROBOCASA_ASSETS_PATH``),
which means a non-editable wheel install works — no local clone needed.

.. code-block:: bash

   # install rpent + rlinf + robocasa wheel + all deps
   uv pip install -e ".[robocasa]"

   # write macros_private.py + print env-var hints
   bash scripts/robocasa/install_robocasa.sh

After the helper script runs, set the two env vars it prints before
launching ``rpent``:

- ``ROBOCASA_MACROS_PATH`` — defaults to ``<repo_root>/.robocasa/macros_private.py``
  (the helper writes the file to ``.robocasa/`` in the project, not ``$HOME``).
- ``ROBOCASA_ASSETS_PATH`` — the 19G kitchen assets directory. The
  ``robocasa`` wheel ships only the 3.4M base assets; the large
  kitchen assets must come from elsewhere (e.g. an existing RoboCasa
  checkout, or downloaded separately).

4. (Optional) Real-world robot dependencies
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
