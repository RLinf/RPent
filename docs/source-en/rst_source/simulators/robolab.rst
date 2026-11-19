RoboLab
=======

Run an Isaac Lab RoboLab benchmark task through RPent's observation and action
tools using RLinf's RoboLab environment. The integration serves one DROID
joint-position environment per run.
RoboLab's own evaluator supplies the success result.

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

Use two Python 3.11 environments on Linux. RPent and Pi0.5 run in the policy environment. Isaac Sim 5.0 and Isaac Lab 2.2.0 run in the simulator environment because they require a different PyTorch version.

Install RPent and the policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the RPent repository root, run:

.. code-block:: bash

   uv venv --python 3.11 .venv-robolab-policy
   source .venv-robolab-policy/bin/activate
   uv pip install -e ".[robolab]"

Reuse an existing simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already run RoboLab tasks in an RLinf simulator environment, set the path below to its Python interpreter. The run command uses it via ``--sim-python``. Continue with the checkpoint section.

.. code-block:: bash

   export SIM_PYTHON=/path/to/robolab-env/bin/python

Install a new simulator
~~~~~~~~~~~~~~~~~~~~~~~

For a fresh installation, prepare a compatible NVIDIA driver, Git LFS, ffmpeg, and the EGL, GLU, OpenGL and Vulkan runtime libraries. Follow the `RLinf RoboLab guide <https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/robolab.html>`_ for system prerequisites. From the RPent root, run:

.. code-block:: bash

   git lfs install
   git clone https://github.com/RLinf/RLinf.git "$HOME/RLinf"
   cd "$HOME/RLinf"
   bash requirements/install.sh embodied --env robolab --install-rlinf
   git -C .venv/robolab lfs pull
   export SIM_PYTHON="$PWD/.venv/bin/python"
   cd -

This example places RLinf in your home directory. The installer creates ``.venv`` inside that checkout. Keep the checkout available. If you set ``ROBOLAB_PATH``, pull the LFS assets in that directory. If the system dependencies are already installed, you can add ``--no-root`` to the installer command. Check that the simulator can import the adapter:

.. code-block:: bash

   "$SIM_PYTHON" -c 'from rlinf.envs.sim.robolab.robolab_env import RoboLabEnv'

Checkpoint
~~~~~~~~~~

Download the public
`RLinf-Pi05-RoboLab-droid-jointpos checkpoint <https://huggingface.co/RLinf/RLinf-Pi05-RoboLab-droid-jointpos>`_
from Hugging Face in the activated policy environment:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH="$HOME/.cache/rpent/checkpoints/RLinf-Pi05-RoboLab-droid-jointpos"
   hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos \
     --local-dir "$PI05_CHECKPOINT_PATH"

The directory contains ``model.safetensors``, ``config.json``, and
``assets/droid/norm_stats.json``. Keep the directory structure intact.
The policy uses RLinf's ``pi05_droid_polaris`` configuration: two images,
10 denoising steps, a 15-action prediction horizon, and eight environment
action values. This loader accepts the PyTorch checkpoint; a JAX checkpoint
cannot be used directly. Set ``PI05_CHECKPOINT_PATH`` in each new shell, or
pass the directory with ``--am-checkpoint``.

Run a Task
----------

Configure a planner using :doc:`../guides/configure_planner`. The example
uses Codex with the current user's login and configured model; this login
method does not require an API key. From the RPent root, run the following in the shell where you set
``SIM_PYTHON`` and ``PI05_CHECKPOINT_PATH``.

.. code-block:: bash

   source .venv-robolab-policy/bin/activate
   rpent-check-llm --planner codex
   rpent --robot robolab --task BananaInBowlTask --seed 0 \
     --sim-python "$SIM_PYTHON" \
     --cuda-device 0 --planner codex \
     --output-dir ./logs/first-robolab

Action-model deployment uses the shared ``--am-checkpoint``, ``--am-python``,
``--am-root``, and ``--am-cuda-device`` options. To reuse a running model
worker, pass ``--am-endpoint`` instead of local deployment options.

The example shares GPU 0 between the simulator and the model. To run the model on GPU 0 and the simulator on GPU 1, use ``--cuda-device 0 --env-cuda-device 1``. To connect to a running simulator, use ``--env-endpoint`` with its task and seed.

For an environment-only run without a checkpoint, add
``--policy-backend none``. It exposes ``execute_action`` and
``view_env_state``; ``pi05_act`` is unavailable.

For a Dashboard session, replace ``--task BananaInBowlTask --seed 0`` with
``--dashboard``. Claim a task with ``/rpent-task BananaInBowlTask 0``. The
Dashboard shares Pi0.5 across tasks and creates a fresh environment for each
task. With ``--policy-backend none``, it starts only the environment and
exposes native actions without a VLA component or ``pi05_act``.

View Results
------------

``logs/first-robolab/states.json`` contains the RPent step records. The
``main.png`` and ``wrist.png`` artifact directories hold the observed images.
``result.json`` records native task success separately from the planner result;
its ``success`` is ``null`` when RoboLab has no terminal result.
``env_server.log`` and, when a local model is launched, ``vla_server.log``
contain the component logs. Native RoboLab output is under
``logs/first-robolab/native/``.

Read the last ``view_env_state`` result: ``state.success`` is ``true`` only
when RoboLab reports success. It remains ``null`` before a native terminal
result; a step-limit episode is not a success.

Task Memory
-----------

No published RoboLab task-memory corpus is currently provided. RPent creates
a local directory for RoboLab memory, which is initially empty on a fresh
installation. It does not download a memory corpus by default. To reuse your own task
notes, pass ``--memory-profile local --memory-dir /path/to/memory``.
Alternatively, ``--memory-repo owner/repo@revision`` downloads the ``robolab/``
subtree of an explicit Hugging Face dataset into RPent's memory cache. Do not
combine it with ``--memory-profile local`` or ``--memory-dir``.
RoboLab does not support RPent exploration mode.

Diagnostics
-----------

If the environment does not start, inspect ``env_server.log`` and verify that
``--sim-python`` points to the RLinf-installed simulator environment with
an importable ``rlinf.envs.sim.robolab.robolab_env``, an editable RoboLab
installation, and a populated ``assets/`` directory. If the policy does not
start, inspect ``vla_server.log`` and the checkpoint path.

For Vulkan initialization errors or a missing ``libGLU.so.1``, check the NVIDIA driver and graphics libraries. Under a scheduler or in a container, omit the example's ``--cuda-device 0`` to inherit the assigned GPUs. If the device ordinal is invalid or rendering uses another GPU, compare the device in the Isaac log with those visible inside the container.

If a wheel download times out, rerun the RLinf installer with the same
``--venv`` path and cache. For dependency conflicts, inspect ``uv pip check`` in
each environment separately. A clean process exit alone does not establish
native task success.
