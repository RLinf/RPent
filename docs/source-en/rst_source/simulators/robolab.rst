RoboLab
=======

Run a RoboLab benchmark task in Isaac Lab through RPent's observation and
action tools.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      Pi0.5 DROID joint-position policy, or native actions without a model.

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``, ``gemini_cli``.

   .. grid-item-card:: Tasks

      RoboLab-120, 120 tasks from the installed RoboLab version.

   .. grid-item-card:: Hardware

      Ubuntu 22.04+, an NVIDIA GPU with a working Vulkan driver (48 GB+ VRAM
      recommended), Python 3.11, Isaac Sim 5.0 / Isaac Lab 2.2.0.

Tasks
~~~~~

RoboLab-120 is RoboLab's benchmark suite of 120 manipulation tasks spanning
pick-and-place, stacking, rearrangement, and tool use. Task classes live under
``robolab/tasks/benchmark/``; ``ls <checkout>/robolab/tasks/benchmark`` lists
them, and the class name is what ``--task`` takes. The `task catalog
<https://github.com/NVlabs/RoboLab/blob/main/robolab/tasks/README.md>`_ has
each task's instruction, difficulty, and episode length.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Selection
     - Scope
   * - Suite and episode limit
     - RoboLab-120, 120 tasks, with limits set by each task rather than by
       RPent; they range from 20 to 300 seconds.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - The DROID embodiment (fixed-base Franka arm, Robotiq 2F-85 gripper, wrist camera, high-gain joint control) supplies external and wrist RGB images, eight joint and gripper state values, and the task instruction. The policy receives 224×224 images; saved planner images keep the camera resolution.
   * - Action
     - ``pi05_act`` executes 1–15 actions, 5 by default, from one Pi0.5 prediction. ``execute_action`` sends one eight-value joint-position action: seven arm joints plus the gripper.
   * - Reward / success
     - RoboLab's native result sets ``state.success`` in ``view_env_state``; the planner's ``finish`` status is separate. See `View Results`_.
   * - Task prompt
     - The selected environment's instruction, also supplied to Pi0.5.

Installation and Resources
--------------------------

The verified setup uses two Python 3.11 environments: RPent and Pi0.5 in the
policy environment, Isaac Sim 5.0 and Isaac Lab 2.2.0 in the simulator
environment. With a working RoboLab simulator environment, set ``SIM_PYTHON``
to its interpreter and skip to `Model Weights`_.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Requirement
     - Value
   * - Operating system
     - Ubuntu 22.04 or later.
   * - GPU and driver
     - NVIDIA GPU with a working Vulkan driver. Upstream RoboLab requires an RTX
       GPU and recommends 48 GB+ VRAM; RLinf evaluated an A800 80 GB with
       driver 570.172.08.
   * - Disk
     - About 8 GB including about 7 GB of assets, excluding Isaac Sim and the
       two environments.
   * - Software
     - Python 3.11, ``uv``, Git LFS, ffmpeg, and the EGL, GLU, OpenGL, and
       Vulkan runtime libraries.

On Ubuntu, install the system packages and `uv
<https://docs.astral.sh/uv/getting-started/installation/>`_:

.. code-block:: bash

   sudo apt update
   sudo apt install -y git-lfs ffmpeg \
     libegl1 libglu1-mesa libgl1 libvulkan1 vulkan-tools
   git lfs install
   curl -LsSf https://astral.sh/uv/install.sh | sh

Install NVIDIA's proprietary driver following the Isaac Sim 5.0 requirements
linked from the `RLinf RoboLab guide
<https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/robolab.html>`_.

Install RPent and the policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv venv --python 3.11 .venv-robolab-policy
   source .venv-robolab-policy/bin/activate
   uv pip install -e ".[robolab]"

The ``robolab`` extra provides RPent, RLinf's Pi0.5 loader, and the OpenPI
client the policy service imports; Isaac Sim and RoboLab come from the
simulator environment below.

Install the simulator
~~~~~~~~~~~~~~~~~~~~~

RLinf's installer clones RoboLab at a pinned revision, installs Isaac Sim 5.0,
Isaac Lab 2.2.0, and an editable RoboLab with its scene assets into a new
``.venv``. Set ``ROBOLAB_PATH`` to an existing RoboLab checkout to skip the
clone and the pin.

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git "$HOME/RLinf"
   cd "$HOME/RLinf"
   bash requirements/install.sh embodied --env robolab --install-rlinf
   ROBOLAB_DIR="${ROBOLAB_PATH:-$PWD/.venv/robolab}"
   git -C "$ROBOLAB_DIR" lfs pull
   export SIM_PYTHON="$PWD/.venv/bin/python"
   "$SIM_PYTHON" -c 'from rlinf.envs.sim.robolab.robolab_env import RoboLabEnv'
   cd -

Model Weights
-------------

In the activated policy environment, download the public
`RLinf-Pi05-RoboLab-droid-jointpos checkpoint <https://huggingface.co/RLinf/RLinf-Pi05-RoboLab-droid-jointpos>`_
from Hugging Face:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH="$HOME/.cache/rpent/checkpoints/RLinf-Pi05-RoboLab-droid-jointpos"
   hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos \
     --local-dir "$PI05_CHECKPOINT_PATH"

This is RoboLab's ``pi05_droid_jointpos`` checkpoint, converted to PyTorch by
RLinf and loaded with the ``pi05_droid_polaris`` configuration. The loader
cannot read a JAX checkpoint. Set ``PI05_CHECKPOINT_PATH`` in each new shell,
or pass the directory with ``--am-checkpoint``.

Run a Task
----------

Configure a planner with :doc:`../guides/configure_planner`; every option is
listed in :doc:`../guides/cli`. The example uses Codex with the current user's
login, which needs no API key. With ``SIM_PYTHON`` and ``PI05_CHECKPOINT_PATH``
set:

.. code-block:: bash

   source .venv-robolab-policy/bin/activate
   rpent-check-llm --planner codex
   rpent --robot robolab --task BananaInBowlTask --seed 0 \
     --sim-python "$SIM_PYTHON" \
     --cuda-device 0 --planner codex \
     --output-dir ./logs/first-robolab

``rpent-check-llm`` exits 0 when the planner answers and shows a ``FAILED``
status otherwise. The run then starts the simulator and the Pi0.5 service as
child processes and hands the task to the planner; components addressed with
``--env-endpoint`` or ``--am-endpoint`` are not started locally. Isaac Sim
needs several minutes to build its rendering context on the first start, and
RPent waits up to 900 seconds for the environment and 1800 for the model
service before failing with the name of the component that did not connect.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Option
     - What it sets
   * - ``--task``
     - RoboLab task class; omit it with ``--dashboard`` (see
       :doc:`../guides/dashboard`).
   * - ``--seed``
     - Random seed for the native environment, ``0`` by default. Whether changing the
       seed changes the scene layout depends on the task.
   * - ``--sim-python``
     - Python of the simulator environment; do not point it at the policy
       environment.
   * - ``--output-dir``
     - Run directory; defaults to
       ``<repo>/logs/<timestamp>_robolab_<task>_s<seed>``.
   * - ``--cuda-device`` / ``--env-cuda-device``
     - GPU for the model service and the simulator; the second option overrides
       the simulator's, as in ``--cuda-device 0 --env-cuda-device 1``.
   * - ``--env-endpoint``
     - URL of a running simulator service; its task and seed must match
       ``--task`` and ``--seed``.

Model-service deployment uses the shared ``--am-checkpoint``, ``--am-python``,
``--am-root``, and ``--am-cuda-device`` options, or ``--am-endpoint`` for a
running model worker. For an environment-only run, add ``--policy-backend
none``: ``execute_action`` and ``view_env_state`` remain available,
``pi05_act`` does not.

View Results
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``result.json``
     - One record per run: ``task``, ``seed``, ``success``, ``elapsed_s``, the
       planner backend and model, and two flags. ``environment_result_available``
       is ``false`` when the episode ended without a terminal native result;
       ``agent_error_present`` is ``true`` when the planner raised an error.
   * - ``states.json``, ``main.png``, ``wrist.png``, ``native/``
     - RPent step records with each tool call and its result, the images saved
       for the planner, and RoboLab's own episode output.
   * - ``env_server.log``, ``vla_server.log``, ``run.log``, ``transcript_*.json``
     - Service and planner logs; ``vla_server.log`` exists only for a local model.

Read ``state.success`` in the last ``view_env_state`` result. RoboLab reports
``true`` for success, ``false`` for failure, and ``null`` while the native
outcome is unknown. The run summary records this value in ``result.json``.

Task Memory
-----------

RPent creates an empty ``memory/robolab/`` under the RPent checkout and loads
memory from there by default; no RoboLab task-memory corpus is published. To
reuse your own notes, pass ``--memory-profile local --memory-dir
/path/to/memory``. ``--memory-repo owner/repo@revision`` downloads the
``robolab/`` subtree of a Hugging Face dataset instead and cannot be combined
with the local options. RoboLab does not support exploration mode; see
:doc:`the memory guide <../guides/memory>` for sources and the file layout.

Troubleshooting
---------------

If the environment does not start, inspect ``env_server.log``; ``--sim-python``
must point to the RLinf-installed simulator environment with an importable
``rlinf.envs.sim.robolab.robolab_env``. If the model does not start, inspect
``vla_server.log``; an unset ``PI05_CHECKPOINT_PATH`` is the common cause.

For Vulkan errors or a missing ``libGLU.so.1``, check the driver and the
graphics libraries in the requirements table above; under a scheduler or in a
container, omit ``--cuda-device 0`` to inherit the assigned GPUs. If a wheel
download times out, rerun the installer with the same ``--venv`` path and
cache, adding ``--no-root`` when system packages are already installed; check
``uv pip check`` in each environment for dependency conflicts.

.. note::

   Page checked on 2026-11-24 against RPent ``3ed1dbc`` and the RLinf installer
   (RoboLab ``b6cfcf6``). The A800 80 GB / driver 570.172.08 setup is what
   RLinf evaluated, not a minimum requirement.
