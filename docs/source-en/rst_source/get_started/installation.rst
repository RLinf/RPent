Installation
============

Install an isolated RPent environment for a robot, simulator, or standalone
model service from a source checkout, and verify that its dependencies work.
For a complete first LIBERO-PRO run with assets and checkpoints, follow
:doc:`quickstart`.

Requirements
------------

- Linux ``x86_64`` and `uv <https://docs.astral.sh/uv/getting-started/installation/>`_.
- Python 3.10–3.12. ``rpent-install`` automatically selects the newest supported
  minor accepted by all requested components and asks ``uv`` to provision it.
- An NVIDIA GPU and driver compatible with the selected component's PyTorch build.

Common Environments
-------------------

Installing a ``robot-*`` component pulls in the planner layer, tools,
simulator, and required model services into one virtual environment. Check each
platform's guide for extra system prerequisites or asset downloads.

.. list-table::
   :header-rows: 1
   :widths: 20 46 34

   * - Environment
     - Install command
     - Python & guide
   * - LIBERO / LIBERO-PRO
     - ``./rpent-install.sh install robot-libero --venv .venv-libero``
     - 3.12 · :doc:`../simulators/libero`
   * - LIBERO-plus
     - ``./rpent-install.sh install robot-libero_plus --venv .venv-libero-plus``
     - 3.12 · :doc:`../simulators/libero`
   * - RoboCasa365
     - ``./rpent-install.sh install robot-robocasa --venv .venv-robocasa``
     - 3.10 · :doc:`../simulators/robocasa`
   * - RoboTwin
     - ``./rpent-install.sh install robot-robotwin --venv .venv-robotwin``
     - 3.11 · :doc:`../simulators/robotwin`
   * - RoboLab
     - ``./rpent-install.sh install robot-robolab --venv .venv-robolab-policy``
     - 3.11 · :doc:`../simulators/robolab`
   * - RoboDojo
     - ``./rpent-install.sh install robot-robodojo --venv .venv-robodojo``
     - 3.11 · :doc:`../simulators/robodojo`
   * - EFE (host debug)
     - ``./rpent-install.sh install robot-efe --venv .venv-efe``
     - 3.12 · :doc:`../simulators/efe` (Harbor trials use ``$EFE_ROOT/.venv``)
   * - Single-Arm Franka
     - ``./rpent-install.sh install robot-franka --venv .venv-franka-real``
     - 3.12 · :doc:`../real_world_robots/franka`
   * - Dual-Arm Franka
     - ``./rpent-install.sh install robot-dual_franka --venv .venv-dual-franka-real``
     - 3.12 · :doc:`../real_world_robots/dual_franka`
   * - YAM (control machine)
     - ``./rpent-install.sh install robot-yam --venv .venv-yam-real``
     - 3.11 · :doc:`../real_world_robots/yam`

.. dropdown:: Standalone Model and Developer Environments

   .. list-table::
      :header-rows: 1
      :widths: 20 46 34

      * - Environment
        - Install command
        - Python & guide
      * - Standalone Pi0.5 VLA
        - ``./rpent-install.sh install model-openpi infra-rlinf --venv .venv-openpi``
        - 3.12 · :doc:`../guides/advanced_deployment`
      * - Standalone Molmo
        - ``./rpent-install.sh install model-molmo --venv .venv-molmo``
        - 3.12 · :doc:`../guides/flash`
      * - Minimal core
        - ``./rpent-install.sh install infra-minimal --venv .venv-minimal``
        - 3.12 · CLI & discovery check
      * - Development tools
        - ``./rpent-install.sh install infra-dev --venv .venv-dev``
        - 3.12 · Lint & test tools
      * - Full unit test suite
        - ``./rpent-install.sh install infra-unit_cpu --venv .venv-unit-cpu``
        - 3.12 · Adds CPU Torch for CI tests

Install an Environment
----------------------

From a fresh checkout, use ``./rpent-install.sh`` to bootstrap the installer
and create a virtual environment:

.. code-block:: bash

   ./rpent-install.sh install robot-libero --venv .venv-libero
   source .venv-libero/bin/activate

Once activated, ``rpent-install`` is available directly inside the environment:

- **Compose components:** Pass multiple names to share one environment (for
  example, ``model-openpi infra-rlinf`` for a standalone Pi0.5 VLA host, or
  ``robot-libero infra-dev`` for development with test tools).
- **Select Python:** Pass ``--python <version>`` (such as ``3.11``) to choose a
  specific supported Python version.
- **Preview requirements:** Pass ``--dry-run`` to inspect the generated
  requirements without installing.

PyPI Mirrors and CUDA PyTorch Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``--index-url`` to install through a PyPI mirror, or ``--torch-backend``
to select a PyTorch wheel index (``auto``, ``cpu``, ``cu124``, ``cu126``,
``cu128``):

.. code-block:: bash

   ./rpent-install.sh install robot-libero --venv .venv-libero \
     --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
     --torch-backend cu126

You can also set ``UV_INDEX_URL``, ``UV_DEFAULT_INDEX``, or
``UV_TORCH_BACKEND`` in the shell environment before invoking the installer.

.. dropdown:: Migrating from Legacy ``pyproject.toml`` Extras

   Per-robot ``pyproject.toml`` extras (``uv pip install -e ".[...]"``) are
   replaced by ``rpent-install`` components:

   .. list-table::
      :header-rows: 1
      :widths: 36 64

      * - Former extra
        - Current command
      * - ``.[libero]``, ``.[libero-pro]``
        - ``./rpent-install.sh install robot-libero --venv .venv-libero``
      * - ``.[libero-plus]``
        - ``./rpent-install.sh install robot-libero_plus --venv .venv-libero-plus``
      * - ``.[robocasa]``
        - ``./rpent-install.sh install robot-robocasa --venv .venv-robocasa``
      * - ``.[robotwin]``
        - ``./rpent-install.sh install robot-robotwin --venv .venv-robotwin``
      * - ``.[robolab]``
        - ``./rpent-install.sh install robot-robolab --venv .venv-robolab-policy``
      * - ``.[robodojo]``
        - ``./rpent-install.sh install robot-robodojo --venv .venv-robodojo``
      * - ``.[robodojo-sim]``
        - ``./rpent-install.sh install sim-robodojo --venv .venv-robodojo-sim``
      * - ``.[franka]``
        - ``./rpent-install.sh install robot-franka --venv .venv-franka-real``
      * - ``.[yam]``
        - ``./rpent-install.sh install robot-yam --venv .venv-yam-real``
      * - ``.[flywheel]``
        - ``./rpent-install.sh install tool-flywheel --venv .venv-flywheel``
      * - ``.[molmo]``
        - ``./rpent-install.sh install model-molmo --venv .venv-molmo``
      * - ``.[sam3]``
        - ``./rpent-install.sh install model-sam3 --venv .venv-sam3``
      * - ``.[rlinf]``
        - ``./rpent-install.sh install infra-rlinf --venv .venv-rlinf``
      * - ``.[test]``
        - ``./rpent-install.sh install infra-dev --venv .venv-dev``

.. dropdown:: Environment Reuse, Overlays, and Wheel Scope

   - **Default path:** omitting ``--venv`` installs into
     ``.rpent/venvs/<components joined by +>`` inside the checkout.
   - **Reuse and overlays:** re-running ``install`` for the same component
     closure updates packages in place and preserves overlaid packages (such as
     ``modelscope``, ``pre-commit``, or editable RLinf checkouts). Pass
     ``--force`` to recreate the virtual environment cleanly; if recreation
     fails, the previous environment is restored.
   - **Foreign virtual environments:** ``rpent-install`` refuses to overwrite a
     non-empty non-venv directory, and it will not write into a virtualenv
     created by another tool unless ``--force`` is passed. For externally
     managed environments, run ``uv pip install -e .`` on service-only hosts
     (such as the YAM inference machine in :doc:`../real_world_robots/yam`), or
     ``uv pip install -e ".[planner]"`` when running the ``rpent`` CLI (such as
     Harbor in :doc:`../simulators/efe`).
   - **PyPI wheel vs. checkout:** ``pip install rpent`` installs the core
     runtime only (``rpent[planner]`` adds planner adapters and the dashboard)
     without robot integrations or simulator/model components. Run robots from a
     source checkout built with ``./rpent-install.sh``.

.. dropdown:: Contributor Tools: List and Validate Components

   Inspect all available components and validate manifests before installing:

   .. code-block:: bash

      ./rpent-install.sh list
      ./rpent-install.sh check --all

   ``check`` computes each component's dependency set without downloading
   packages, verifying Python ranges, exact-pin conflicts, and mutual-exclusion
   rules. Passing several component names checks whether they can share one
   environment.

   Add ``--resolve`` to run ``uv pip compile`` against Linux ``x86_64``
   (default ``manylinux_2_31``, or ``manylinux_2_35`` when required by a
   component such as ``sim-robodojo``) and discard the temporary output.
   Checking environments that build ``nvidia-curobo`` requires a Linux host
   with ``CUDA_HOME`` set.

Diagnose an Environment
-----------------------

Verify that an installed virtual environment matches its manifests, passes
``uv pip check``, and can import every component in its closure:

.. code-block:: bash

   ./rpent-install.sh doctor --venv .venv-libero
   ./rpent-install.sh doctor --venv .venv-libero --fingerprint

``--fingerprint`` prints the stored ``rpent-install.json`` metadata and its
hash without running checks. A healthy environment prints ``ok`` lines for
targets, installation inputs, package set, ``uv pip check``, and each component
import probe:

.. code-block:: text

   ok   targets: robot-libero
   ok   installed at: 2026-11-28T00:00:00Z
   ok   installation inputs unchanged
   ok   python: 3.12.11
   ok   platform: linux-x86_64
   ok   installed package set matches the fingerprint
   ok   uv pip check
   ok   probe robot-libero: import robots.libero

If you overlaid extra packages into the environment (such as ``modelscope`` or
a custom CUDA PyTorch wheel) and both ``uv pip check`` and all component probes
succeed, ``doctor`` reports a ``warn`` line instead of failing.

FAQ
---

CUDA Unavailable or Model Server Exits Immediately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Components such as ``model-sam3``, ``model-rldx``, and ``robot-robocasa`` do not
pin a specific PyTorch CUDA build. Without ``--torch-backend``, ``uv`` may pull
the newest CUDA 13.x wheel, which cannot use the GPU on older NVIDIA drivers
(for example, driver 550). Run ``nvidia-smi`` to check the CUDA version
supported by your driver and pass a matching backend when installing (for
example, ``--torch-backend cu126`` or ``UV_TORCH_BACKEND=cu126``).

LIBERO / LIBERO-PRO Missing Assets After Setting Config Path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``UV_LINK_MODE=symlink`` is enabled, ``rpent-liberopro`` and
``rpent-libero`` files inside the virtual environment are symlinks into the
read-only ``uv`` cache, so relative asset lookups resolve into the cache
directory instead of the environment. Recreate the environment in ``copy``
mode, or materialize just the affected package from its ``uv pip freeze`` line:

.. code-block:: bash

   # Option 1: recreate the environment without symlink mode
   UV_LINK_MODE=copy ./rpent-install.sh install robot-libero --venv .venv-libero --force

   # Option 2: reinstall only the affected package in copy mode
   uv pip install --link-mode copy --reinstall-package <package> --no-deps "<requirement>"

Slow Installs, GitHub Clone Stalls, or Cache Lock Timeouts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Cache misses:** When sharing a ``uv`` cache across users or jobs, keep the
  same PyPI mirror URL; switching mirrors misses the existing cache and forces
  large wheels to re-download.
- **Lock timeouts:** Running multiple installs for the same Python version
  against a shared cache at the same time can hit cache lock timeouts. Run them
  sequentially and raise ``UV_LOCK_TIMEOUT`` (for example,
  ``UV_LOCK_TIMEOUT=1200``).
- **Slow GitHub clones:** Pass Git's temporary ``insteadOf`` environment
  variables before running the installer to route Git fetches through a mirror
  without changing global Git config. Include ``https://`` in
  ``<mirror-prefix>`` and omit any trailing slash (for example,
  ``https://gh-proxy.com``; a trailing slash produces an invalid
  ``//https://github.com/`` URL):

  .. code-block:: bash

     GIT_CONFIG_COUNT=1 \
     GIT_CONFIG_KEY_0="url.<mirror-prefix>/https://github.com/.insteadOf" \
     GIT_CONFIG_VALUE_0="https://github.com/" \
     ./rpent-install.sh install robot-libero --venv .venv-libero
