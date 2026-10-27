RoboLab
=======

Run an Isaac Lab RoboLab benchmark task through RPent's observation and action
tools. The integration serves one DROID joint-position environment per run.
RoboLab's own evaluator supplies the success result; RPent benchmark success
rates have not been established.

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

Create a policy environment from the RPent root, following the same ``uv venv``
and optional-extra workflow as the other simulators:

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
   git clone https://github.com/NVlabs/RoboLab.git "$ROBOLAB_SOURCE"
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

Install the full RPent package only in the policy environment. The simulator
loads RPent's environment server and shared RPC modules from this checkout;
its launcher supplies ``PYTHONPATH``. Installing RPent's planner dependencies
in the Isaac environment conflicts with Isaac Sim's pinned dependencies.

The simulator reads the installed ``robolab`` package and its adjacent
``assets/`` directory. It does not read RPent's ``ref-repo/`` or ``data/``.
Set the Pi0.5 checkpoint path separately, outside the repository if desired:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/absolute/path/to/pi05_droid_jointpos_torch

The checkpoint must match the DROID joint-position Pi0.5 model (15-action
prediction horizon, eight environment action values). The policy environment
uses RLinf's ``pi05_droid_polaris`` model configuration.

Run a Task
----------

With the policy environment active and the planner configured, run:

.. code-block:: bash

   rpent --robot robolab --task BananaInBowlTask --seed 0 \
     --sim-python "$PWD/.venv-robolab-sim/bin/python" \
     --planner claude_code --model claude-opus-4-8 \
     --output-dir ./logs/first-robolab

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

Diagnostics
-----------

If the environment does not start, inspect ``env_server.log`` and verify that
``--sim-python`` points to the Isaac environment with an editable RoboLab
installation and a populated ``assets/`` directory. If the policy does not
start, inspect ``vla_server.log`` and the checkpoint path. Keep Isaac Sim and
the policy in separate Python environments when their dependencies conflict.
