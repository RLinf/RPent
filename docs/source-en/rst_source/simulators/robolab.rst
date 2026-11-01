RoboLab
=======

Run an Isaac Lab RoboLab benchmark task through RPent's observation and action
tools. The integration serves one DROID joint-position environment per run.
RoboLab's own evaluator supplies the success result. The local six-task
evaluation is a subset experiment, not a full RoboLab benchmark result.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      Pi0.5 DROID joint-position policy, or native actions without a model.

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``.

   .. grid-item-card:: Tasks

      RoboLab benchmark task classes, including ``BananaInBowlTask``.

   .. grid-item-card:: Hardware

      Linux, NVIDIA GPU, Python 3.11, Isaac Sim 5.0 / Isaac Lab 2.2.0.

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Selection
     - Scope
   * - ``--task BananaInBowlTask``
     - One installed RoboLab benchmark task, one simulation environment.
   * - ``--seed 0``
     - Seed passed to the native environment configuration.

Use a task class supplied by the installed RoboLab version. RPent does not
bundle RoboLab task definitions or scene assets.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - External and wrist RGB images, eight DROID joint/gripper state values, and the task instruction. The policy receives 224-pixel images; saved planner images use the native camera resolution.
   * - Action
     - ``pi05_act`` executes 1–15 actions from one Pi0.5 prediction. ``execute_action`` sends one native eight-value DROID joint-position action.
   * - Reward / success
     - RoboLab's native per-environment result sets ``state.success`` in ``view_env_state``. The planner's ``finish`` status is separate.
   * - Task prompt
     - The selected environment's instruction, also supplied to Pi0.5.

Installation and Resources
--------------------------

Use Linux with a compatible NVIDIA driver, Python 3.11, ``uv``, Git with Git
LFS, and ``ffmpeg``. Run the installer from the RPent root:

.. code-block:: bash

   bash script/install_robolab.sh

The script creates ``.venv-robolab-policy`` and ``.venv-robolab-sim``, installs
RPent's ``robolab`` extra, and downloads RoboLab revision ``b6cfcf6`` and its
LFS assets to the sibling ``../RoboLab-runtime`` directory. To select another
external directory, pass it as the first argument. Existing checkouts must
already match the required revision. Checkpoints are supplied separately.

The installer checks both environments and loads the required graphics libraries.
On Ubuntu 22.04 x86_64, missing EGL, GLU, OpenGL, and Vulkan libraries are
installed under ``.venv-robolab-sim/system-libs`` using ``curl``, ``dpkg-deb``,
and ``sha256sum``; no sudo is needed and the NVIDIA driver is not changed.
On other distributions, install missing graphics libraries using the system's
package manager. Download concurrency and timeout settings can be overridden
with the ``UV_*`` variables listed by ``bash script/install_robolab.sh --help``.

Manual Installation
~~~~~~~~~~~~~~~~~~~

For manual installation, use the same ``uv venv`` and optional-extra workflow
as the other simulators:

.. code-block:: bash

   uv venv --python 3.11 .venv-robolab-policy
   source .venv-robolab-policy/bin/activate
   uv pip install -e ".[robolab]"

RoboLab locates its USD assets beside its source package. Install its source
checkout and ``openpi-client`` in a separate Isaac environment. Keep
the checkout outside the RPent repository; ``git lfs pull`` downloads the
assets required by the selected tasks. These commands use Isaac Sim 5.0:

.. code-block:: bash

   export ROBOLAB_SOURCE="$HOME/.cache/rpent/RoboLab"
   git lfs install
   mkdir -p "$(dirname "$ROBOLAB_SOURCE")"
   GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/RoboLab.git "$ROBOLAB_SOURCE"
   git -C "$ROBOLAB_SOURCE" checkout b6cfcf6
   git -C "$ROBOLAB_SOURCE" lfs pull
   export RPENT_ROOT="$PWD"
   uv venv --python 3.11 .venv-robolab-sim
   (
     cd "$ROBOLAB_SOURCE"
     UV_PROJECT_ENVIRONMENT="$RPENT_ROOT/.venv-robolab-sim" \
       uv sync --python 3.11 --extra isaac50 --no-dev --locked
     uv pip install --python "$RPENT_ROOT/.venv-robolab-sim/bin/python" \
       -e ".[isaac50]" openpi-client
   )

Complete the manual installation with graphics and dependency checks:

.. code-block:: bash

   bash script/install_robolab_graphics.sh
   uv pip check --python .venv-robolab-policy/bin/python
   uv pip check --python .venv-robolab-sim/bin/python

Install the full RPent package only in the policy environment. The simulator
loads RPent's environment server and shared RPC modules from this checkout;
its launcher supplies ``PYTHONPATH``. Installing RPent's planner dependencies
in the Isaac environment conflicts with Isaac Sim's pinned dependencies.

Checkpoint
~~~~~~~~~~

The simulator reads the installed ``robolab`` package and its adjacent
``assets/`` directory. It does not read RPent's ``ref-repo/`` or ``data/``.
Set the Pi0.5 checkpoint path separately, outside the repository if desired:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/absolute/path/to/pi05_droid_jointpos_torch

The checkpoint must match the DROID joint-position Pi0.5 model (15-action
prediction horizon, eight environment action values). The policy environment
uses RLinf's ``pi05_droid_polaris`` model configuration. Supply a compatible
PyTorch checkpoint containing ``model.safetensors`` and
``assets/droid/norm_stats.json``. The installer does not download or convert
model weights, and a JAX checkpoint cannot be passed directly to this loader.
The repository does not require a pre-existing local dataset or checkpoint
folder; use ``--pi05-checkpoint`` or ``PI05_CHECKPOINT_PATH`` for your own path.

Run a Task
----------

Activate the policy environment and check the local Codex login. The bundled
CLI uses the current user's login; no API key is needed for this login method.
See :doc:`../guides/configure_planner` for other planners and authentication modes.

.. code-block:: bash

   source .venv-robolab-policy/bin/activate
   export CODEX_BIN="$(python -c 'from codex_cli_bin import bundled_codex_path; print(bundled_codex_path())')"
   "$CODEX_BIN" login
   "$CODEX_BIN" login status
   rpent-check-llm --planner codex --timeout-s 90 --json

After the connection check succeeds, run a task. Set ``--cuda-device`` to a
physical GPU with enough free memory; the example uses GPU 0. The planner uses
your configured Codex model unless you pass ``--model``.


.. code-block:: bash

   rpent --robot robolab --task BananaInBowlTask --seed 0 \
     --sim-python "$PWD/.venv-robolab-sim/bin/python" \
     --cuda-device 0 --planner codex \
     --output-dir ./logs/first-robolab

The convenience script checks the current Codex login before launching the task:

.. code-block:: bash

   bash script/run_robolab.sh "$PI05_CHECKPOINT_PATH" BananaInBowlTask 0 ./logs/first-robolab

Run this command from a terminal with the same GPU visibility intended for the
simulation. The CLI form above exposes explicit physical GPU selection.
The scripts use the existing Codex configuration and login.

For an environment-only run without the checkpoint, add
``--policy-backend none``. It exposes ``execute_action`` and
``view_env_state``; ``pi05_act`` is then unavailable.

For a Dashboard session, replace ``--task BananaInBowlTask --seed 0`` with
``--dashboard``. Claim a task with ``/rpent-task BananaInBowlTask 0``. The
Dashboard shares Pi0.5 across tasks and creates a fresh environment for each
task; it requires the Pi0.5 backend.

View Results
------------

``logs/first-robolab/states.json`` contains the RPent step records. The
``main.png`` and ``wrist.png`` artifact directories hold the observed images.
``result.json`` records native task success separately from the planner result;
its ``success`` is ``null`` when RoboLab has no terminal result.
``env_server.log`` and ``vla_server.log`` contain the component logs. Native
RoboLab output is under ``logs/first-robolab/native/``.

Read the last ``view_env_state`` result: ``state.success`` is ``true`` only
when RoboLab reports success. ``state.success`` remains ``null`` before a
native terminal result; a step-limit episode is not a success.

Task Memory
-----------

RoboLab uses a local memory directory by default. To reuse your own task notes,
pass ``--memory-profile local --memory-dir /path/to/memory``. No RoboLab memory
corpus is downloaded.

Evaluate a Task Subset
----------------------

The evaluation script runs six tasks serially, with environment seeds 0, 1,
and 2: ``BananaInBowlTask``, ``BowlStackingLeftOnRightTask``,
``BowlStackingRightOnLeftTask``, ``BananaThenRubiksCubeTask``,
``ReorientRedMugTask``, and ``StackYellowOnRedTask``. Supply the checkpoint
explicitly; results default to ``logs/robolab_eval/unaligned_6tasks_3seeds/``.
Preview the commands before starting:

.. code-block:: bash

   bash script/eval_robolab_subset.sh \
     --checkpoint "$PI05_CHECKPOINT_PATH" --gpu 0 \
     --output-dir ./logs/robolab-subset --dry-run

Remove ``--dry-run`` to execute. ``--tasks`` and ``--seeds`` select a smaller
subset. The defaults are 400 planner turns, a 3600-second planner budget per
episode, and ``low`` reasoning effort. Simulator startup is outside this
planner budget. Each attempt gets empty local RPent memory. Codex session
storage follows the user configuration. Environment seeds do not fix policy
sampling randomness. The policy uses the upstream RLinf implementation,
10 denoising steps, two images, and up to 15 executed actions per prediction;
the planner may shorten chunks or issue direct actions.

``manifest.json`` records configuration, package versions, and source hashes.
``summary.json`` and ``summary.csv`` separate native success/failure, unfinished
runs, budget exhaustion, errors, and interruptions. Success rates use completed
evaluation episodes, including unfinished and budget-exhausted episodes, as
the denominator; runtime errors and interruptions are reported separately.
The same command skips completed episodes and retries errors or interruptions
in a new attempt directory. It stops on a runtime error. Configuration or
source changes require a new output directory to avoid mixing results.

Diagnostics
-----------

If the environment does not start, inspect ``env_server.log`` and verify that
``--sim-python`` points to the Isaac environment with an editable RoboLab
installation and a populated ``assets/`` directory. If the policy does not
start, inspect ``vla_server.log`` and the checkpoint path. Keep Isaac Sim and
the policy in separate Python environments when their dependencies conflict.

For Vulkan initialization errors or missing ``libGLU.so.1``, rerun
``bash script/install_robolab_graphics.sh`` and inspect the Isaac log. The
launcher adds these local libraries automatically. GPU rendering and physics
are assigned to the selected physical device with multi-GPU rendering disabled;
Isaac may still create small CUDA contexts on other GPUs. Check actual memory
use with ``nvidia-smi`` on shared machines.

If a wheel download times out, rerun the installer with the same external
checkout and environments; downloaded caches are retained. For dependency
conflicts, inspect ``uv pip check`` in each environment separately. Codex
connection errors belong to the planner log and may be retried; a clean process
exit alone does not establish native task success.
