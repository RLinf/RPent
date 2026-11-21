Installation
============

An RPent environment is the set of components you name on the command line,
plus the closure of their ``requires``. Each component declares its own
dependencies and Python range in ``component.toml``.
``rpent-install`` creates the virtual environment, installs that closure with
the core ``rpent`` package, runs an import probe for every component it
installs, and writes an installation fingerprint.

Requirements
------------

- Linux ``x86_64`` and `uv <https://docs.astral.sh/uv/getting-started/installation/>`_.
- A Python minor version every selected component supports. ``rpent-install``
  uses the newest of 3.12, 3.11, and 3.10 that the whole closure accepts and
  asks uv to install it; the interpreter that runs the command does not decide
  the environment's version.
- A GPU and driver compatible with the selected component's PyTorch build.

Components
----------

The name prefix says what a component is and where its code lives:

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Prefix
     - Location
     - Contents
   * - ``robot-``
     - ``robots/<name>/``
     - A robot slice: native translation, tools, and its runtime wiring.
   * - ``sim-``
     - ``components/sim/<name>/``
     - A simulator service that robots and models share.
   * - ``model-``
     - ``components/models/<name>/``
     - A model service: VLA, grounding, or action-model backend.
   * - ``tool-``
     - ``components/tools/<name>/``
     - A tool that depends only on the core, such as the flywheel exporter, or
       a capability library a robot slice adapts, such as the RGB-D tools.
   * - ``infra-``
     - ``components/infra/<name>/``
     - Packages an environment needs without owning code (RLinf runtime, test
       tools, CPU Torch).

Every component is one directory holding its ``component.toml``. A component
that installs packages for code it does not own contributes only that manifest:
the ``infra-`` components are all of this shape, and ``sim-libero_plus``
installs the LIBERO-plus package whose environment server ships with
``sim-libero``.

The core package installs in two layers. Plain ``rpent`` is the runtime layer:
component and model contracts, RPC, model clients, and the installer, so a
simulator or model environment can hold it without the planner stack.
``rpent[planner]`` adds the planner adapters, execution sessions, prompts, and
the dashboard; the tool contracts and toolkit live in the runtime layer. A
component that imports the planner declares ``planner = true`` in its
``component.toml``, and every environment holding that component installs the
planner layer as well; without the flag an environment gets the runtime layer
only. ``install`` fails with a named component list if a
component sets the flag and the checkout's ``pyproject.toml`` has no ``planner``
extra.

The dependency direction is ``rpent`` ← ``components/sim`` ←
``components/models`` ← ``robots``; ``components/tools`` depends only on
``rpent`` and, for a capability library, the model clients' result types. A
robot environment therefore pulls in its simulator, model services, and tools
automatically:

.. code-block:: bash

   rpent-install list

``robot-libero`` requires ``sim-libero``, ``model-openpi``, ``model-sam3``,
``tool-flywheel``, and ``tool-rgbd``; ``robot-robocasa`` requires
``sim-robocasa`` and ``model-rldx``; ``robot-robodojo`` requires ``sim-robodojo``, ``model-sam3``,
``model-openpi``, ``model-awomo05``, and ``model-vpp2``. Model services no
robot requires (``model-molmo``, ``model-cosmos3``, ``model-cosmos_policy``,
``model-fast_wam``, ``model-apxinf``, ``model-uniwam``, ``model-xpolicylab``)
install in their own environment.

A ``robot-`` component is installed in one step: its environment holds the
planner, the tools, the simulator, and the model servers the robot needs, so
``install robot-libero`` needs no second environment for the models or the
exporter. Every other component installs in an environment of its own with the
core ``rpent`` package, and a robot environment reaches it through the
component's client instead of importing it. ``robot-yam`` is the exception: its
slice carries the YAM runtime and connects to a Pi0.5 policy installed
separately as ``model-openpi``.

List and check components
-------------------------

.. code-block:: bash

   rpent-install list
   rpent-install check --all

``check`` reads the manifests, computes each component's ``requires`` closure,
and reports unknown names, unsupported Python versions, conflicting exact pins,
and the conflicts components declare with ``env_conflicts``. Naming several
components checks the environment they form together, which is the one
``install`` creates; ``--all`` checks each component in an environment of its
own. ``check`` downloads nothing. ``--resolve`` additionally uses
``uv pip compile`` to prove the declared dependencies have a solution for the
Linux ``x86_64`` target and discards the compiled file afterwards.

Install an environment
----------------------

From a checkout, install a component into its own virtual environment:

.. code-block:: bash

   rpent-install install robot-libero --venv .venv-libero
   source .venv-libero/bin/activate

On a fresh checkout the ``rpent-install`` console script does not exist yet.
``rpent-install.sh`` bootstraps it without installing RPent first:

.. code-block:: bash

   ./rpent-install.sh install robot-libero --venv .venv-libero

Name several components to compose one environment. A non-robot component
installs on its own host this way, and the robot environment connects to it
over its client:

.. code-block:: bash

   rpent-install install model-openpi --venv .venv-openpi
   rpent-install install sim-libero_plus model-openpi model-sam3 --venv .venv-libero-plus

``rpent-install`` runs from a checkout, where it reads the ``component.toml``
manifests under ``robots/`` and ``components/``. A non-editable ``pip install
rpent`` provides the CLI; run a specific robot or model service from a source
checkout, where the manifests live.

An environment contains exactly the components you named and their closure.
What it contains decides whether an existing venv can be reused, so naming a
component that pulls another in through ``requires`` and naming both describe
the same environment, and a component whose packages changed describes the same
environment too. A venv holding a different set of components is refused;
install that set under a different ``--venv``, or pass ``--force`` to replace
it. ``--force`` rebuilds the environment at that path, so it never deletes a
directory that is not a virtualenv. A venv that already holds packages but was
not created by ``rpent-install`` is refused too: its packages belong to whoever
installed them, and only an empty one (a fresh ``uv venv``) is used as it is.
An environment left by a failed install keeps the rendered requirements, so
rerunning the same command adopts and finishes it. The rendered requirements
and the installed package versions are written into the environment itself, so
they never outlive it. Without ``--venv``, a target set installs into
``.rpent/venvs/<components joined by +>`` under the checkout;
``.rpent/`` is local state and is ignored by git, so later ``install`` and
``doctor`` runs find the same environment.

Re-running the command reuses the environment and installs what the manifests
now declare; a package an earlier manifest added stays installed until
``--force`` rebuilds the environment. The installer generates requirements from
the manifests, then lets ``uv pip install`` resolve them. Direct version constraints and Git revisions
remain in the manifests; transitive versions may differ between fresh installs
as upstream packages change. ``--dry-run`` prints the generated requirements
without installing.

Any number of components may contribute packages. Index settings
(``index-url``, ``extra-index-urls``, ``index-strategy``) are merged across the
closure; two components that disagree on them make the environment invalid
rather than silently picking one.

Common environments
-------------------

.. code-block:: bash

   rpent-install install infra-minimal --venv .venv-minimal
   rpent-install install infra-dev --venv .venv-dev
   rpent-install install infra-unit_cpu --venv .venv-unit-cpu
   rpent-install install robot-libero --venv .venv-libero
   rpent-install install sim-libero_plus model-openpi model-sam3 --venv .venv-libero-plus
   rpent-install install robot-robocasa --venv .venv-robocasa
   rpent-install install robot-robotwin --venv .venv-robotwin
   rpent-install install robot-franka --venv .venv-franka
   rpent-install install robot-dual_franka --venv .venv-dual-franka
   rpent-install install robot-yam --venv .venv-yam
   rpent-install install robot-robodojo --venv .venv-robodojo
   rpent-install install model-molmo --venv .venv-molmo
   rpent-install install model-openpi --venv .venv-openpi

``infra-minimal`` is the core environment: RPent and its own dependencies with
no robot, simulator, or model package, which is enough to verify CLI and robot
discovery. ``infra-dev`` adds the test tools. ``infra-unit_cpu`` adds
those tools and a CPU Torch-built model stack for the unit suite. The robot
environments cover the supported real robots and simulators; see each
simulator or robot page for platform preparation, checkpoints, and assets.
``sim-libero_plus`` replaces ``sim-libero`` rather than adding to it: LIBERO-plus
needs robosuite<1.5, so the two package sets cannot share an environment.

Diagnose an environment
-----------------------

.. code-block:: bash

   rpent-install doctor --venv .venv-libero
   rpent-install doctor --venv .venv-libero --fingerprint

``doctor`` compares the stored fingerprint with every recorded installation
input (each component's manifest hash, its override files, and the repository's
``[tool.uv]`` settings), checks the recorded Python version and platform, runs
``uv pip check`` and the component probes, and compares the installed package
set with its recorded freeze hash. ``uv`` resolves from the checkout, so the
compared settings do not depend on the directory the command is invoked from;
user-level ``uv`` configuration (``~/.config/uv``) is outside the comparison.
A venv installed for a different set of components is reported as needing a
reinstall rather than as drifted. ``--fingerprint`` prints the stored
fingerprint and its hash instead of running the checks.

``doctor`` describes the environment ``rpent-install`` created from the
manifests. Some robot and simulator guides add packages to that environment
afterwards, such as an editable RLinf checkout, an SDK, or an adapter package.
Those additions change the installed package set, so ``doctor`` reports the
environment as drifted, and rerunning ``install`` re-records the new set. Run
``doctor`` on an environment built from the manifests when the fingerprint
should attest to the component closure alone.

For ``check --resolve``, ``linux-x86_64`` targets ``manylinux_2_31`` (glibc
2.31, Ubuntu 20.04 or newer): current GPU wheels are published for newer
baselines, so a 2.17 target cannot resolve them. A component whose packages are
published only for a newer baseline raises the target for its whole closure
with ``manylinux = "2_35"`` in its manifest; ``sim-robodojo`` does this for
Isaac Sim. Checking a configuration that
builds cuRobo with ``--resolve`` needs a Linux host with ``CUDA_HOME`` set
because cuRobo builds metadata during resolution. A failed resolution is
reported without stopping checks of the remaining components.
