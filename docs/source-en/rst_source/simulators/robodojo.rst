RoboDojo
========

Run bimanual tabletop tasks with RPent in RoboDojo. The simulator uses Isaac
Sim / IsaacLab with an ARX-X5 dual-arm platform and Pi0.5 as the default action model.
The integration is experimental; success rates across a complete task suite
have not been established.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      Pi0.5

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``

   .. grid-item-card:: Tasks

      Bottle disposal, pen-holder filling, bowl stacking, and other tabletop tasks.

   .. grid-item-card:: Hardware

      Linux, NVIDIA GPU; Python 3.11; Isaac Sim 5.1.

Tasks
~~~~~

The `RoboDojo simulation task catalog <https://robodojo-benchmark.com/doc/sim-tasks/>`_
groups 42 benchmark tasks into five capability categories:

.. list-table::
   :header-rows: 1

   * - Category
     - Tasks
     - Focus
   * - Generalization
     - 12
     - Varied objects and layouts, including random variants.
   * - Memory
     - 6
     - State tracking and sequence recall.
   * - Precision
     - 8
     - Manipulation with tight spatial constraints.
   * - Long-Horizon
     - 8
     - Multi-step tasks with several subgoals.
   * - Open
     - 8
     - Open-ended or language- and image-conditioned tasks.

The catalog also lists one separate DLC task. Use ``--task`` to select a task
from the installed runtime and ``--layout`` to select its scene layout. For
example, ``stack_bowls_random`` is a generalization task, while
``fill_pen_holder`` and ``put_bottles_into_dustbin`` are long-horizon tasks.
These are examples, not the complete task list.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - Head and left/right wrist RGB-D images, camera calibration, and joint states, end-effector poses, and gripper states for both arms. Pi0.5 uses three RGB views, robot state, and task text.
   * - Action
     - The planner calls ``pi0_pick`` or primitives such as ``move_to`` and ``set_gripper``. Pi0.5 outputs 14-dimensional joint and gripper actions for both arms.
   * - Reward / success
     - RoboDojo's native environment determines task success, recorded in the ``success`` field of ``result.json``.
   * - Task prompt
     - The current scene's RoboDojo task description. ``pi0_pick`` uses this instruction when its ``prompt`` argument is omitted.

Installation and Resources
--------------------------

For NVIDIA driver recommendations, see the `Isaac Sim 5.1 system requirements
<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_.
The host also needs the CUDA toolkit/NVCC and a compiler toolchain to build
cuRobo. From the RPent root, create an environment and install the dependencies:

.. code-block:: bash

   uv venv --python 3.11 .venv-robodojo
   source .venv-robodojo/bin/activate
   uv pip install -e ".[robodojo]" --extra-index-url https://pypi.nvidia.com \
     --override requirements/robodojo-override.txt

.. note::

   Blackwell GPUs need the CUDA 12.8 build of PyTorch. The pinned PyTorch 2.7.0
   also has a CUDA 12.6 build, but that build does not support Blackwell. On
   these GPUs, use the following command for both initial installation and
   reinstallation; ``--torch-backend=cu128`` selects the CUDA 12.8 build:

   .. code-block:: bash

      uv pip install -e ".[robodojo]" --extra-index-url https://pypi.nvidia.com \
        --override requirements/robodojo-override.txt \
        --torch-backend=cu128

   If ``torch==2.7.0+cu126`` is already installed, uv considers it compatible
   with the version requirement and keeps it. Append
   ``--reinstall-package torch --reinstall-package torchvision`` to the command
   above to replace it with the CUDA 12.8 build.

The pinned IsaacLab revision needs an editable installation to locate its
extension configuration. Clone it into a writable directory and install it
in the same environment:

.. code-block:: bash

   git clone --no-checkout https://github.com/yuechen0614/IsaacLab.git /path/to/IsaacLab
   git -C /path/to/IsaacLab checkout afca7b09d60d8beb9c1cb28b43066499940b969b
   uv pip install --no-deps \
     -e /path/to/IsaacLab/source/isaaclab \
     -e /path/to/IsaacLab/source/isaaclab_assets \
     -e /path/to/IsaacLab/source/isaaclab_tasks

Keep this checkout after installation; the runtime still uses it.

Download Simulator Resources
----------------------------

Download the robot, object, material, and layout files from the
`RoboDojo dataset <https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo>`_
and set the resource directory:

.. code-block:: bash

   hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
     --include 'Assets/**' --local-dir /data/robodojo
   export ROBODOJO_ASSETS_ROOT=/data/robodojo

The download should contain ``Assets/Robots``, ``Assets/Object``,
``Assets/Material``, and ``Assets/Eval_Layout``.

Generate a ``*.yml`` configuration beside each ``*_tmp.yml`` template under
``Assets/Robots``, substituting absolute local resource paths. The simulator
reads ``Assets/Robots/x5/curobo.yml`` for x5:

.. code-block:: bash

   python - <<'PY'
   import os
   import runpy
   from robodojo_runtime import source_root

   script = source_root() / "utils/update_embodiment_config_path.py"
   os.chdir(os.environ["ROBODOJO_ASSETS_ROOT"])
   runpy.run_path(str(script), run_name="__main__")
   PY
   test -s "$ROBODOJO_ASSETS_ROOT/Assets/Robots/x5/curobo.yml"

Run this once after the initial download; subsequent task runs do not need it.
Moving the resource directory invalidates the paths saved in these configurations,
so run the command again afterward. Regeneration overwrites existing ``*.yml``
configurations.

Some NVIDIA resources referenced by the scene load online by default.
For offline use or repeated resource-loading failures, see
:ref:`robodojo-local-nvidia-assets` at the end of this page.

VLA Configuration
-----------------

Use ``--policy-backend`` to select the Pi0.5 backend. The default is ``rlinf``;
``xpolicylab`` is available for a separately prepared runtime.

RLinf (Default)
~~~~~~~~~~~~~~~

``--policy-backend rlinf`` uses the RLinf runtime installed above.
``PI05_CHECKPOINT_PATH`` selects its PyTorch weights and normalization statistics.

The released weights use JAX format and need conversion to PyTorch.
This example downloads only the complete ``params/`` directory and matching
``norm_stats.json`` from ``RoboDojo-sim-arx_x5-joint-0/59999``, about 12.4 GB
in total. Training state and other models are unnecessary. The converted
weights are about 7.23 GB; allow extra space for conversion and caches.

.. code-block:: bash

   hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
     --include \
       'ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/params/**' \
       'ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/assets/arx_x5_sim/norm_stats.json' \
     --local-dir /data/robodojo/pi05_jax
   export PI05_JAX_CHECKPOINT=/data/robodojo/pi05_jax/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999

.. note::

   Give the ``--local-dir`` directory a lowercase ``pi05_`` prefix, as in
   ``pi05_jax`` above. The current converter requires lowercase ``pi05`` in
   its input path; the release directory name ``Pi_05`` does not satisfy this.

   If your existing download path has no lowercase ``pi05``, you do not need
   to download again. Create a symlink with a ``pi05_`` prefix to the model
   directory containing ``59999/``, then set ``PI05_JAX_CHECKPOINT`` to
   ``59999`` under that link. For a download rooted at ``/data/robodojo``:

   .. code-block:: bash

      ln -s /data/robodojo/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0 \
        /data/robodojo/pi05_robodojo_arx_x5
      export PI05_JAX_CHECKPOINT=/data/robodojo/pi05_robodojo_arx_x5/59999

Convert the weights in the same environment and set the converted model path:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/data/robodojo/pi05_robodojo_arx_x5_torch

   python -m rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python \
     --checkpoint-dir "$PI05_JAX_CHECKPOINT" \
     --config-name pi05_aloha \
     --output-path "$PI05_CHECKPOINT_PATH"

``--checkpoint-dir`` must point to the directory containing ``params/``.

Add ``--inspect_only`` to view parameter information without converting.

The model also needs the ``norm_stats.json`` file supplied with its weights.
The converter does not write this file to
``$PI05_CHECKPOINT_PATH/RoboDojo/arx_x5/norm_stats.json``, so copy it explicitly:

.. code-block:: bash

   mkdir -p "$PI05_CHECKPOINT_PATH/RoboDojo/arx_x5"
   cp "$PI05_JAX_CHECKPOINT/assets/arx_x5_sim/norm_stats.json" \
     "$PI05_CHECKPOINT_PATH/RoboDojo/arx_x5/norm_stats.json"

XPolicyLab (Optional)
~~~~~~~~~~~~~~~~~~~~~

Use the RLinf setup above unless you need XPolicyLab. The default backend is
``--policy-backend rlinf``, so no XPolicyLab installation is needed. RPent also
provides ``--policy-backend xpolicylab`` for users who already have that
environment or want its Pi0.5 implementation.

XPolicyLab is another software stack for loading and running Pi0.5. Clone its
source separately and prepare its Python environment following the
`Pi0.5 installation instructions
<https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/README.md#installation>`_.
You can reuse the original JAX weights and statistics downloaded above,
but must configure their location as XPolicyLab expects.
After configuring SAM3 and your planner as described below, run a task with
the XPolicyLab checkout and its Python environment:

.. code-block:: bash

   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --planner codex --model <planner-model> \
     --policy-backend xpolicylab --xpolicylab-root /path/to/XPolicyLab \
     --pi05-python /path/to/xpolicylab-env/bin/python \
     --output-dir /path/to/run-output

``--xpolicylab-root`` points to the separate XPolicyLab checkout.
If the Python packages needed for XPolicyLab's Pi0.5 inference are installed in
a separate environment, point ``--pi05-python`` to that environment's Python
interpreter; otherwise omit the option.

SAM3 Configuration
------------------

SAM3 provides object segmentation. Request access on the
`facebook/sam3 model page <https://huggingface.co/facebook/sam3>`_, then
download ``sam3.pt`` and set its path:

.. code-block:: bash

   hf download facebook/sam3 sam3.pt --local-dir /data/sam3
   export SAM3_CHECKPOINT_PATH=/data/sam3/sam3.pt

Alternatively, download the same file from
`ModelScope <https://modelscope.cn/models/facebook/sam3>`_ and set
``SAM3_CHECKPOINT_PATH`` to the local file.

Run a Task
----------

Follow :doc:`../guides/configure_planner` to choose and configure the planner
and model you want to use. The following Codex example checks connectivity in
the terminal that will run the task:

.. code-block:: bash

   rpent-check-llm --planner codex --model <planner-model> --json

Replace ``<planner-model>`` with the selected model, then run the
bottle-disposal task:

.. code-block:: bash

   # Workaround: RPent currently prioritizes a sibling rlinf checkout, which can shadow the installed RoboDojo integration.
   # Point to the active environment's site-packages so child services use its installed RLinf.
   # Remove this setting once RPent's default import-path selection is fixed.
   export RPENT_RLINF_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000  # Recommended: let objects settle; the default is 20 steps
   export OMNI_KIT_ACCEPT_EULA=YES  # Recommended for the first non-interactive start; omit if the Kit license is already accepted
   # Isaac Sim takes tens of seconds to start; the first run also compiles shaders
   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --planner codex --model <planner-model> --reasoning-effort medium \
     --max-turns 40 --planner-timeout-s 1500 \
     --output-dir /path/to/run-output

View Results
------------

The terminal shows service startup, planner output, and tool calls. After the
run, inspect these files in the directory selected by ``--output-dir``:

- ``result.json``: the environment's task verdict and run error status.
- ``states.json``: action records and post-action observations.
- ``videos/``: head, left-wrist, and right-wrist camera videos.
- ``robodojo_env_server.log``, ``robodojo_vla_server.log``, and
  ``sam3_server.log``: service startup and error logs.
- ``run.log`` and ``transcript_*.json``: main-process logs, planner conversations,
  and errors.

Check ``environment_result_available=true`` and
``agent_error_present=false`` first, then inspect whether ``success`` is
``true``. A clean service exit or a planner call to ``finish`` alone does not
establish task success.

Without ``--output-dir``, results go to
``logs/<timestamp>_robodojo_<task>_l<layout>/``. Use different directories for
concurrent runs. For live monitoring, see :doc:`../guides/dashboard`.

Common Options
--------------

RPent starts the simulator, policy, and SAM3 services in the current Python
environment by default. Specify ``--sim-python`` and ``--pi05-python`` only
when the simulator and policy are installed in different environments.

- ``--task`` and ``--layout`` select the task and fixed layout. ``--random``
  ignores ``--layout`` and samples a new random scene on each reset.
- ``--cuda-device`` selects the GPU for locally started services. If omitted,
  services inherit ``CUDA_VISIBLE_DEVICES``.
- ``--env-endpoint``, ``--vla-endpoint``, and ``--sam3-endpoint`` connect to
  existing services. The VLA service must match ``--policy-backend``.
- ``--source-root`` selects a local RoboDojo checkout instead of the packaged code.

Child services inherit the current shell environment. The CLI does not load
``config/runtime.env`` automatically. Run ``rpent --robot robodojo --help``
for the full option list.

Task Memory and Flash Replay
----------------------------

RoboDojo does not currently support Flash Mode or ``--explore``.

.. _robodojo-local-nvidia-assets:

Prepare Local NVIDIA Scene Assets (Optional)
--------------------------------------------

Download NVIDIA scene assets in advance if either of these applies:

- You need to run the simulator on a machine without network access.
- NVIDIA resource URLs are unreliable and the logs show repeated download failures.

RoboDojo scenes also reference NVIDIA 3D models, materials, and textures.
These files are separate from the RoboDojo resources downloaded above and
load over the network by default. Skip this section if scenes load normally;
you do not need a local asset pack or ``ROBODOJO_USD_ASSET_PREFIX``.

The steps below prepare local resources for ``Assets/Isaac/5.0`` paths.
The three parts of the complete asset pack total about 150 GB; this is not
the download size for a single RoboDojo scene. Allow extra space for merging
and extraction. For an offline machine, download the parts on a connected
machine first, then copy them to ``/data/nvidia``.

.. code-block:: bash

   mkdir -p /data/nvidia

   # Download all three parts; interrupted downloads can resume
   wget -c -P /data/nvidia \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.001 \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.002 \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.003

   # Merge and extract, preserving the Assets/Isaac/5.0 directory structure
   cat /data/nvidia/isaac-sim-assets-complete-5.0.0.zip.001 \
       /data/nvidia/isaac-sim-assets-complete-5.0.0.zip.002 \
       /data/nvidia/isaac-sim-assets-complete-5.0.0.zip.003 \
     > /data/nvidia/isaac-sim-assets-complete-5.0.0.zip
   unzip /data/nvidia/isaac-sim-assets-complete-5.0.0.zip -d /data/nvidia

   test -d /data/nvidia/Assets/Isaac/5.0/Isaac
   test -d /data/nvidia/Assets/Isaac/5.0/NVIDIA

Set this variable in the terminal that will run RPent, then run the task:

.. code-block:: bash

   export ROBODOJO_USD_ASSET_PREFIX=/data/nvidia

This changes the matching RoboDojo 5.0 resource URLs to local paths. It does
not make every Isaac Sim resource available offline. If other online resources
still fail to load, such as ``Assets/Isaac/5.1`` paths, prepare assets for that
version and configure local loading separately. See the
`NVIDIA local asset instructions
<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack>`_.
