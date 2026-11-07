RoboLab
=======

Run an Isaac Lab RoboLab benchmark task through RPent's observation and action
tools. The integration serves one DROID joint-position environment per run.
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

Use Linux with a compatible NVIDIA driver, Python 3.11, ``uv``, Git, Git LFS,
and ``ffmpeg``. Isaac Sim also requires EGL, GLU, OpenGL, and Vulkan runtime
libraries. On Ubuntu 22.04, install the system packages if they are missing:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install git-lfs ffmpeg libegl1 libglu1-mesa libopengl0 libvulkan1

Use the system package manager for other distributions. If you cannot install
system packages, ask the machine administrator to provide these libraries.
From the RPent root, create the policy environment and install the optional
extra, following the :doc:`../get_started/quickstart` workflow:

.. code-block:: bash

   uv venv --python 3.11 .venv-robolab-policy
   source .venv-robolab-policy/bin/activate
   uv pip install -e ".[robolab]"

RoboLab reads USD assets beside its source package. Install RoboLab revision
``b6cfcf6`` and its LFS assets outside the RPent repository, in a separate
Isaac Sim 5.0 / Isaac Lab 2.2.0 environment:

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

Install the full RPent package only in the policy environment. The simulator
loads the environment server and shared RPC modules from this checkout using
the launcher's ``PYTHONPATH``. Installing planner dependencies in the simulator
environment conflicts with Isaac Sim's pinned versions. Keep the Isaac extra
in the second install command so the resolver preserves those constraints.
Check both environments after installation:

.. code-block:: bash

   uv pip check --python .venv-robolab-policy/bin/python
   uv pip check --python .venv-robolab-sim/bin/python

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
method does not require an API key. Run from the RPent root with the checkpoint
path configured above. Choose a physical GPU with enough free memory using
``--cuda-device``; the example uses GPU 0.

.. code-block:: bash

   source .venv-robolab-policy/bin/activate
   rpent-check-llm --planner codex
   rpent --robot robolab --task BananaInBowlTask --seed 0 \
     --sim-python "$PWD/.venv-robolab-sim/bin/python" \
     --cuda-device 0 --planner codex \
     --output-dir ./logs/first-robolab

Action-model deployment uses the shared ``--am-checkpoint``, ``--am-python``,
``--am-root``, and ``--am-cuda-device`` options. To reuse a running model
worker, pass ``--am-endpoint`` instead of local deployment options.

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
installation. It does not download a memory corpus. To reuse your own task
notes, pass ``--memory-profile local --memory-dir /path/to/memory``.
RoboLab does not support RPent exploration mode.

Evaluate a Task Subset
------------------------

After configuring the checkpoint and planner, run the following commands from
the RPent root to evaluate six tasks serially with environment seeds 0, 1,
and 2. Select the available GPU before starting. Each invocation creates a
fresh output directory and each episode starts with empty local memory:

.. code-block:: bash

   (
     set -e
     mkdir -p ./logs
     eval_root="$(mktemp -d "$PWD/logs/robolab-subset-XXXXXX")"
     for task in BananaInBowlTask BowlStackingLeftOnRightTask \
       BowlStackingRightOnLeftTask BananaThenRubiksCubeTask \
       ReorientRedMugTask StackYellowOnRedTask; do
       for seed in 0 1 2; do
         episode_dir="$eval_root/${task}_s${seed}"
         rpent --robot robolab --task "$task" --seed "$seed" \
           --sim-python "$PWD/.venv-robolab-sim/bin/python" \
           --cuda-device 0 --planner codex --reasoning-effort low \
           --max-turns 400 --planner-timeout-s 3600 \
           --memory-profile local --memory-dir "$episode_dir/memory" \
           --output-dir "$episode_dir"
       done
     done
   )

Each episode has 400 planner turns and a 3600-second planner budget; simulator
startup is outside this budget. Environment seeds do not fix policy sampling
randomness or planner decisions. The planner may shorten action chunks or
issue direct actions. This is a subset evaluation, not a full benchmark.

Inspect each episode's ``result.json`` and logs. Count native success only
when ``success`` is ``true``; report ``false`` and ``null`` separately. Use
all 18 planned episodes as the denominator and list missing results, runtime
errors, and interruptions explicitly. The loop stops on a nonzero process
exit and does not retry or generate an aggregate report. To retry an episode,
run its task and seed with fresh output and memory directories, retaining the
failed attempt. Record the code and resource revisions, package versions,
planner model, and retry policy alongside results for comparisons.

Diagnostics
-----------

If the environment does not start, inspect ``env_server.log`` and verify that
``--sim-python`` points to the Isaac environment with an editable RoboLab
installation and a populated ``assets/`` directory. If the policy does not
start, inspect ``vla_server.log`` and the checkpoint path.

For Vulkan initialization errors or missing ``libGLU.so.1``, check the NVIDIA
driver and system graphics libraries listed above, then inspect the Isaac log.
Rendering and physics use the selected physical GPU with multi-GPU rendering
disabled; Isaac may still create small CUDA contexts on other GPUs. Check
actual memory use with ``nvidia-smi`` on shared machines.

If a wheel download times out, retry the failed ``uv`` command using the same
environment and cache. For dependency conflicts, inspect ``uv pip check`` in
each environment separately. A clean process exit alone does not establish
native task success.
