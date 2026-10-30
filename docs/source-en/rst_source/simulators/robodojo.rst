RoboDojo
========

Run bimanual tabletop tasks with RPent in RoboDojo. The simulator uses
Isaac Sim / IsaacLab with an ARX-X5 dual-arm platform. RPent combines policy
actions, native motion primitives and SAM3 object segmentation to complete tasks.
Choose VPP2 or Awomo05 for policy actions.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      Analytical primitives, VPP2 (default), and Awomo05

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

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - Head and left/right wrist RGB-D images, camera calibration, and joint states, end-effector poses, and gripper states for both arms.
   * - Action
     - ``move_to``, ``rotate_wrist``, ``set_gripper``, ``release``, ``vpp2_act``, and ``awomo05_act``. ``finish`` cannot override the native task verdict.
   * - Reward / success
     - RoboDojo's native environment determines task success, recorded in the ``success`` field of ``result.json``.
   * - Task prompt
     - The current scene's RoboDojo task description, read through ``view_env_state``.

Installation and Resources
--------------------------

For NVIDIA driver recommendations, see the `Isaac Sim 5.1 system requirements
<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_.
The host also needs the CUDA toolkit/NVCC and a compiler toolchain to build
cuRobo. From the RPent root, create an environment and install the dependencies:

.. code-block:: bash

   uv venv --python 3.11 .venv-robodojo
   source .venv-robodojo/bin/activate
   uv pip install -e ".[robodojo-sim,sam3]" --extra-index-url https://pypi.nvidia.com \
     --override requirements/robodojo-override.txt

.. note::

   Blackwell GPUs need the CUDA 12.8 build of PyTorch. The pinned PyTorch 2.7.0
   also has a CUDA 12.6 build, but that build does not support Blackwell. On
   these GPUs, use the following command for both initial installation and
   reinstallation; ``--torch-backend=cu128`` selects the CUDA 12.8 build:

   .. code-block:: bash

      uv pip install -e ".[robodojo-sim,sam3]" --extra-index-url https://pypi.nvidia.com \
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
     --include 'Assets/**' --local-dir /path/to/robodojo
   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo

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

Select ``--vla-backend vpp2|awomo05``; omitting it selects ``vpp2``.
The model implementations and weights are available here:

- VPP2: the official `VPP2 RoboDojo implementation <https://github.com/roboterax/video-prediction-policy-2/tree/72c17acf022494ff63617be86079bb92e92f5bf8>`_
  and the paired 100k weights from `Haodong082399/VPP2 <https://huggingface.co/Haodong082399/VPP2>`_.
- Awomo05: the `official author implementation <https://github.com/Awomo-WestlakeDI/Awomo-0.5>`_,
  the `XPolicyLab RoboDojo evaluation adapter <https://github.com/XPolicyLab/XPolicyLab/blob/562c14cf08b9e6e411e7d76f192740afced77160/policy/Awomo05/README.md>`__
  and the `Auwomo/Awomo-0.5-Robodojo weights and matching statistics <https://huggingface.co/Auwomo/Awomo-0.5-Robodojo>`_.

VPP2 Setup
~~~~~~~~~~

Prepare the VPP2 implementation and paired weights above. Choose a resource
directory and replace ``/path/to/vpp2`` with your own directory:

.. code-block:: bash

   export VPP2_MODEL_PATH=/path/to/vpp2
   git clone --no-checkout https://github.com/roboterax/video-prediction-policy-2.git \
     "$VPP2_MODEL_PATH/policy"
   git -C "$VPP2_MODEL_PATH/policy" checkout 72c17acf022494ff63617be86079bb92e92f5bf8
   hf download Haodong082399/VPP2 --local-dir "$VPP2_MODEL_PATH" \
     --include 'checkpoints/joint2b_s100000/**' \
     --include 'checkpoints/Wan2.1-I2V-14B-480P/**'

The resource root has the following layout. Subdirectories may be symbolic
links to existing resources:

.. code-block:: text

   vpp2/
   ├── policy/src/vpp2/policy.py
   └── checkpoints/
       ├── joint2b_s100000/
       │   ├── action.pt
       │   ├── video.pt
       │   └── dataset_stats.json
       └── Wan2.1-I2V-14B-480P/

Install the model dependencies using the pinned project's
`inference environment instructions
<https://github.com/roboterax/video-prediction-policy-2/blob/72c17acf022494ff63617be86079bb92e92f5bf8/docs/robodojo.md#policy-environment>`_.
These dependencies are separate from ``.[robodojo-sim]``. When they are
compatible with the RPent environment, ``VPP2_MODEL_PATH`` enables automatic
local startup; ``--vla-model-path`` overrides it. The local server loads the
full model on one GPU using RPent's current Python interpreter.

To use a separate model environment, install the policy dependencies and RPent's
core package there, then start the service:

.. code-block:: bash

   python -m robots.robodojo.vpp2_server --model-path /path/to/vpp2 \
     --host 127.0.0.1 --port 9201

Add ``--vla-endpoint socket://127.0.0.1:9201`` to the task command in the
RPent/simulator environment. RPent starts and stops its own model process;
independently started services remain under their operator's control.
For remote deployment, see :doc:`../guides/advanced_deployment`.

Awomo05 Setup
~~~~~~~~~~~~~

Prepare the Awomo05 evaluation implementation and paired weights above.
Choose a resource directory and replace ``/path/to/awomo05`` with your own directory:

.. code-block:: bash

   export AWOMO05_MODEL_PATH=/path/to/awomo05
   git clone --no-checkout https://github.com/RoboDojo-Benchmark/RoboDojo.git \
     "$AWOMO05_MODEL_PATH/policy"
   git -C "$AWOMO05_MODEL_PATH/policy" checkout 2184bf8844ea9d205382c4aefa3a694311418251
   git -C "$AWOMO05_MODEL_PATH/policy" submodule update --init XPolicyLab
   git -C "$AWOMO05_MODEL_PATH/policy/XPolicyLab" fetch origin 562c14cf08b9e6e411e7d76f192740afced77160
   git -C "$AWOMO05_MODEL_PATH/policy/XPolicyLab" checkout 562c14cf08b9e6e411e7d76f192740afced77160
   export AWOMO05_WEIGHTS="$AWOMO05_MODEL_PATH/checkpoints"
   bash "$AWOMO05_MODEL_PATH/policy/XPolicyLab/policy/Awomo05/download_weights.sh"

The resource root has the following layout. Subdirectories may be symbolic
links to existing resources:

.. code-block:: text

   awomo05/
   ├── policy/
   │   ├── env_cfg/arx_x5.yml
   │   └── XPolicyLab/policy/Awomo05/model.py
   └── checkpoints/
       ├── model.pt
       ├── dataset_stats.json
       ├── flux2-klein-base-4b/
       ├── flux2-dev/
       └── Qwen3-VL-4B-Instruct/

Install the model dependencies using the pinned project's
`inference environment instructions <https://github.com/XPolicyLab/XPolicyLab/blob/562c14cf08b9e6e411e7d76f192740afced77160/policy/Awomo05/README.md>`__.
These dependencies are separate from ``.[robodojo-sim]``. When the model
dependencies are compatible with the RPent environment,
``AWOMO05_MODEL_PATH`` enables automatic local startup; ``--vla-model-path``
overrides it. The local server loads the full model on one GPU using RPent's
current Python interpreter.

To use a separate model environment, install the policy dependencies and RPent's
core package there, then start the service:

.. code-block:: bash

   python -m robots.robodojo.awomo05_server --model-path /path/to/awomo05 \
     --host 127.0.0.1 --port 9205

Add ``--vla-endpoint socket://127.0.0.1:9205`` to the task command in the
RPent/simulator environment. RPent starts and stops its own model process;
independently started services remain under their operator's control.
For remote deployment, see :doc:`../guides/advanced_deployment`.

SAM3 Configuration
------------------

SAM3 provides object segmentation. Request access on the
`facebook/sam3 model page <https://huggingface.co/facebook/sam3>`_, then
download ``sam3.pt`` and set its path:

.. code-block:: bash

   hf download facebook/sam3 sam3.pt --local-dir /path/to/sam3
   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt

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

Replace ``<planner-model>`` with the selected model. After configuring VPP2
and SAM3, run the bottle-disposal task at seed group 0, layout 0:

.. code-block:: bash

   # Use the RLinf integration installed in the active environment.
   export RPENT_RLINF_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000  # Recommended: let objects settle; the default is 20 steps
   export OMNI_KIT_ACCEPT_EULA=YES  # Recommended for the first non-interactive start; omit if the Kit license is already accepted
   # Isaac Sim takes tens of seconds to start; the first run also compiles shaders
   rpent --robot robodojo --vla-backend vpp2 \
     --task put_bottles_into_dustbin --seed 0 --layout 0 \
     --planner codex --model <planner-model> --reasoning-effort medium \
     --max-turns 40 --planner-timeout-s 1500 \
     --output-dir /path/to/run-output

View Results
------------

The terminal shows service startup, planner output, and tool calls. After the
run, inspect these files in the directory selected by ``--output-dir``:

- ``result.json``: the environment's task verdict and run error status.
- ``states.json``: commands, execution results and synchronized observation records.
- Per-step RGB, metric depth, calibration and world-xyz artifacts for available cameras.
- ``videos/``: head, left-wrist, and right-wrist camera videos.
- ``robodojo_env_server.log``, ``sam3_server.log``, and ``robodojo_vpp2_server.log``
  or ``robodojo_awomo05_server.log``: locally owned service startup and error logs.
- ``run.log`` and ``transcript_*.json``: main-process logs, planner conversations,
  and errors.

Check ``environment_result_available=true`` and
``agent_error_present=false`` first, then inspect whether ``success`` is
``true``. This field records RoboDojo's native task verdict; ``finish`` ends
the planner loop.

Without ``--output-dir``, results go to
``logs/<timestamp>_robodojo_<task>_s<seed>_l<layout>/``. Use different directories for
concurrent runs. For live monitoring, see :doc:`../guides/dashboard`.

Common Options
--------------

RPent starts the simulator, SAM3 and selected VLA model services in the current
Python environment by default. Specify ``--sim-python`` when the simulator uses
a separate environment.

Use ``--task`` to select a task. For a fixed scene, ``--seed`` selects an
official layout group and ``--layout`` selects a layout within that group;
both default to ``0``. For example,
``--task stack_blocks --seed 0 --layout 2`` uses layout 2 from group 0.
Add ``--random`` to generate a random scene instead; ``--layout`` has no effect
in this mode.

- ``--cuda-device`` selects the GPU for locally started services. If omitted,
  services inherit ``CUDA_VISIBLE_DEVICES``.
- ``--vla-backend`` accepts ``vpp2`` (default) or ``awomo05``.
- ``--vla-model-path`` selects the resource root containing ``policy/`` and
  ``checkpoints/``, overriding ``VPP2_MODEL_PATH`` or ``AWOMO05_MODEL_PATH``
  for the selected backend.
- ``--env-endpoint``, ``--sam3-endpoint`` and ``--vla-endpoint`` connect to
  existing RPent RPC services; the VLA protocol must match the selected backend.
- ``--source-root`` selects a local RoboDojo checkout instead of the packaged code.

Child services inherit the current shell environment. The CLI does not load
``config/runtime.env`` automatically. Run ``rpent --robot robodojo --help``
for the full option list.

Tools
-----

The planner cannot reset an episode; exploration and Flash replay are unavailable.

Initial observations and motion feedback include ``episode_status.step``
(native actions used), ``step_limit`` (total budget), and ``remaining_steps``.
These counters describe the saved observation; the top-level ``step`` is its
record index. Reading historical observations does not consume action steps.

``vpp2_act(instruction, chunks=1)`` runs the specified subtask. Each chunk
predicts 32 absolute dual-arm EE16 targets (xyz, wxyz and gripper per arm)
and executes the first 24 through native EE control. With ``chunks=1``, control
returns to the planner; each additional chunk starts with a new prediction.
VPP2 uses the initial RGB frame and eight history samples spaced by 25 actions.
Both policy and primitive actions update this history; switching subtasks
preserves it. Image reads and ``render`` do not advance the action timeline.
Missing action feedback invalidates policy history until reset; start a new
episode to resume VPP2 when reset is unavailable.

With the Awomo05 backend, ``awomo05_act(chunks=1)`` executes dual-arm
joint targets using the original full task instruction, taken from the initial
observation and held fixed for the episode. Each chunk executes eight steps with per-action observation updates; the next chunk starts fresh
inference. ``chunks`` limits action blocks per call. Model history persists
across calls and includes observations after motion primitives. Cancelled
chunks are not resumed. If inference or execution results are uncertain,
inspect the error and start a new episode rather than retrying.

``view_env_state`` reads saved images and state; ``render`` captures a new
observation. ``sample_world_xyz(view, pixels, step=None, neighborhood=1)``
samples [row,col] pixels, with a default 3-by-3 neighborhood. ``query_world_map``
samples a half-open [r0,c0,r1,c1] image region (default limit 256 points).
Use views ``head``, ``left_wrist`` or ``right_wrist`` from the source RGB step.
Both queries use saved same-frame depth, not a live refresh. Invalid depths
are excluded. These are visible surface coordinates, not object centres.

``segment`` and ``back_project`` also accept ``step``. Segmentation returns
``centroid_rc`` and a foreground ``point_rc``; the centroid can lie outside a
concave mask. Prefer ``point_rc`` and the returned ``step_idx`` for back-projection.
These two tools retain camera names ``cam_head``, ``cam_left_wrist`` and
``cam_right_wrist``.

Describe visible colour, shape and location in ``segment(text_prompt=...)``.
It returns a red-mask overlay; inspect it before using the point, because a
high score does not establish object identity. If segmentation fails, revise
the description or manually query a visible pixel in the same saved frame.
The overlay and geometry are saved with that frame. Use ``back_project``
to obtain world coordinates for the selected foreground pixel.

Motion primitives use native cuRobo pose planning and joint-target execution.
The planner's obstacle model includes a simplified table; choose waypoints
with clearance from scene objects and the other arm, which are not fully
represented in that model.

``move_to(arm, xyz, quat=None, gripper=None, substeps=25)`` uses world metres
and wxyz orientation. Omitted orientation and gripper targets are held.
``substeps`` caps sampled waypoints: 0 keeps the full path, 1 keeps only its
final waypoint. It is not a speed or physics-step count. ``rotate_wrist``
rotates about world Z in degrees. ``final_dist_m`` is measured position error,
not an arrival guarantee. Check ``completed``, ``executed_steps`` and
``stop_reason``; a timed-out mutation may have executed and is not retried.

``set_gripper(arm, val, steps=10)`` uses 0=closed and 1=open. ``release``
defaults to 1. Intermediate values are normalized opening targets rather than
physical widths. Feedback comes from measured joints. Other controller targets
are retained while the gripper moves.

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
machine first, then copy them to ``/path/to/nvidia-assets``.

.. code-block:: bash

   mkdir -p /path/to/nvidia-assets

   # Download all three parts; interrupted downloads can resume
   wget -c -P /path/to/nvidia-assets \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.001 \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.002 \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.003

   # Merge and extract, preserving the Assets/Isaac/5.0 directory structure
   cat /path/to/nvidia-assets/isaac-sim-assets-complete-5.0.0.zip.001 \
       /path/to/nvidia-assets/isaac-sim-assets-complete-5.0.0.zip.002 \
       /path/to/nvidia-assets/isaac-sim-assets-complete-5.0.0.zip.003 \
     > /path/to/nvidia-assets/isaac-sim-assets-complete-5.0.0.zip
   unzip /path/to/nvidia-assets/isaac-sim-assets-complete-5.0.0.zip -d /path/to/nvidia-assets

   test -d /path/to/nvidia-assets/Assets/Isaac/5.0/Isaac
   test -d /path/to/nvidia-assets/Assets/Isaac/5.0/NVIDIA

Set this variable in the terminal that will run RPent, then run the task:

.. code-block:: bash

   export ROBODOJO_USD_ASSET_PREFIX=/path/to/nvidia-assets

This changes the matching RoboDojo 5.0 resource URLs to local paths. It does
not make every Isaac Sim resource available offline. If other online resources
still fail to load, such as ``Assets/Isaac/5.1`` paths, prepare assets for that
version and configure local loading separately. See the
`NVIDIA local asset instructions
<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack>`_.
