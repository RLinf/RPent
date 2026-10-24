RoboDojo
========

Run bimanual tabletop tasks with RPent in RoboDojo. The simulator uses
Isaac Sim / IsaacLab with an ARX-X5 dual-arm platform. Choose VPP2 (the default)
or Awomo05 for policy actions. Native motion primitives and SAM3 segmentation
are available in VPP2 mode; Awomo mixed mode enables them through the
corresponding tool groups.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      VPP2 (default), Awomo05, and native motion primitives

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``; Awomo agent mode: ``api`` or ``codex``.

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
     - Head and left/right wrist RGB-D images, camera calibration, and joint states, end-effector poses, and gripper states for both arms. Depth and calibration are used for geometry queries; Awomo policy execution itself does not require depth.
   * - Action
     - VPP2 mode offers ``vpp2_act`` and native motion primitives (``move_to``, ``move_delta``, ``rotate_wrist``, ``set_gripper``, ``release``, ``hold``). Awomo05 mode offers ``run_policy``; ``--vla-agent`` adds selected motion and perception tools. ``finish`` cannot override the native task verdict.
   * - Reward / success
     - RoboDojo's native environment determines task success, recorded in the ``success`` field of ``result.json``.
   * - Task prompt
     - The current scene's RoboDojo task description, read through ``view_env_state``. Awomo uses the official instruction throughout the episode; VPP2 accepts a subtask instruction.

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

Select ``--vla-backend vpp2|awomo05``; omitting it selects ``vpp2``. Follow the
VPP2 setup below or :ref:`robodojo-awomo-setup`. Awomo does not require VPP2
weights and does not use ``VPP2_MODEL_PATH`` or ``--vla-model-path``.

VPP2 Setup
~~~~~~~~~~

Prepare the official
`VPP2 RoboDojo implementation <https://github.com/roboterax/video-prediction-policy-2/tree/72c17acf022494ff63617be86079bb92e92f5bf8>`_
and the paired 100k weights from
`Haodong082399/VPP2 <https://huggingface.co/Haodong082399/VPP2>`_.
Choose a resource directory and set its location, as in the following example.
Replace ``/path/to/vpp2`` with your own directory:

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

.. _robodojo-awomo-setup:

Awomo05 Setup
~~~~~~~~~~~~~

Use the official `XPolicyLab Awomo05 installation and asset guide
<https://github.com/XPolicyLab/XPolicyLab/blob/562c14cf08b9e6e411e7d76f192740afced77160/policy/Awomo05/README.md>`_.
The model environment is separate from Isaac Sim. The RPent bridge only needs
RPent and XPolicyLab's WebSocket dependencies, so it can run in the RPent
simulation environment without installing the Awomo model dependencies there.

The following path starts the model yourself and lets RPent start the bridge.
Replace the example directories with absolute local paths. Prepare the upstream
checkout for policy code and robot configuration; RPent still uses the simulator
installed above:

.. code-block:: bash

   git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git /path/to/RoboDojo
   git -C /path/to/RoboDojo checkout 2184bf8844ea9d205382c4aefa3a694311418251
   git -C /path/to/RoboDojo submodule update --init XPolicyLab
   git -C /path/to/RoboDojo/XPolicyLab fetch origin 562c14cf08b9e6e411e7d76f192740afced77160
   git -C /path/to/RoboDojo/XPolicyLab checkout 562c14cf08b9e6e411e7d76f192740afced77160
   uv venv --seed --python 3.11 /path/to/awomo-env
   source /path/to/awomo-env/bin/activate
   cd /path/to/RoboDojo/XPolicyLab
   bash policy/Awomo05/install.sh

Keep ``env_cfg/arx_x5.yml`` in the parent RoboDojo checkout and the XPolicyLab
robot configuration. Download the released
`Awomo checkpoint and matching statistics
<https://huggingface.co/Auwomo/Awomo-0.5-Robodojo>`_ together. Follow the repository's
access and license requirements. Its downloader verifies ``SHA256SUMS`` and also
fetches `FLUX.2-klein-base-4B
<https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B>`_, the
`FLUX.2-dev autoencoder <https://huggingface.co/black-forest-labs/FLUX.2-dev>`_,
and `Qwen3-VL-4B-Instruct <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>`_:

.. code-block:: bash

   # In the Awomo environment, from the XPolicyLab root.
   export AWOMO05_WEIGHTS=/path/to/awomo-weights
   hf auth login
   bash policy/Awomo05/download_weights.sh
   export AWOMO05_FLUX2_BASE="$AWOMO05_WEIGHTS/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors"
   export AWOMO05_FLUX2_AE="$AWOMO05_WEIGHTS/flux2-dev/ae.safetensors"
   export AWOMO05_QWEN_VL="$AWOMO05_WEIGHTS/Qwen3-VL-4B-Instruct"

In that same terminal, start RPent's metadata-enabled wrapper. Put both the
XPolicyLab checkout and its parent on ``PYTHONPATH`` so the official model and
server imports resolve. Reserve GPU 0 for the model in this example and GPU 1
for the simulator in :ref:`robodojo-awomo-run`; adjust both to free devices:

.. code-block:: bash

   export PYTHONPATH="/path/to/RoboDojo:/path/to/RoboDojo/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
   CUDA_VISIBLE_DEVICES=0 python /path/to/RPent/robots/robodojo/awomo05_policy_server.py \
     --config_path /path/to/RoboDojo/XPolicyLab/policy/Awomo05/deploy.yml \
     --host 127.0.0.1 --port 9005 --overrides \
     policy_name=Awomo05 bench_name=RoboDojo task_name=general_pickup \
     env_cfg_type=arx_x5 action_type=joint seed=0 ckpt=model.pt \
     input_color_order=bgr replan_steps=8

Wait for ``Awomo05 ready`` followed by ``websocket policy server listening``
in the model terminal. These indicate model initialization and listener startup.
Keep the terminal running.
The wrapper exposes ``get_model_meta`` for bridge validation. Preserve BGR
configuration, 25 Hz history, the native joint layout and the official task
instruction; RPent does not reorder actions or add image transforms.
If startup fails, use :ref:`robodojo-awomo-troubleshooting` before starting RPent.

SAM3 Configuration
------------------

SAM3 provides object segmentation. VPP2 starts it by default; Awomo only needs
it when ``--vla-agent --agent-tools ... sam3`` selects the SAM3 tool group. Request access on the
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

.. _robodojo-awomo-run:

Run with Awomo05
~~~~~~~~~~~~~~~~

After :ref:`robodojo-awomo-setup` reports the model listener ready, open a second
terminal in the RPent simulation environment created above. Install XPolicyLab's
transport package into this environment; it does not install the Awomo model:

.. code-block:: bash

   cd /path/to/RPent
   source .venv-robodojo/bin/activate
   uv pip install -e /path/to/RoboDojo/XPolicyLab
   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   export RPENT_RLINF_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000
   export OMNI_KIT_ACCEPT_EULA=YES
   rpent-check-llm --planner codex --model <planner-model> --json
   CUDA_VISIBLE_DEVICES=1 rpent --robot robodojo --vla-backend awomo05 \
     --task general_pickup --seed 0 --layout 0 \
     --policy-server-url ws://127.0.0.1:9005 \
     --xpolicylab-root /path/to/RoboDojo/XPolicyLab \
     --policy-python /path/to/RPent/.venv-robodojo/bin/python \
     --planner codex --model <planner-model> --reasoning-effort medium \
     --max-turns 40 --planner-timeout-s 1500 \
     --output-dir /path/to/awomo-run-output

Replace ``<planner-model>`` with the configured planner model. This starts the
environment and a CPU bridge to the already running Awomo model; it needs no
VPP2 or SAM3 weights. RPent waits for the bridge and validates its metadata
before execution. Inspect ``awomo05_bridge.log`` and ``robodojo_env_server.log``
in the output directory if startup stops. The planner can call ``run_policy``,
read saved state and ``finish``. For mixed control, add
``--vla-agent --agent-tools policy motion geometry``; see
:ref:`robodojo-awomo-collaboration` for tool and planner restrictions.

.. _robodojo-existing-rpc:

Existing RPC Service (Alternative)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--vla-endpoint`` denotes RPent RPC, while ``--policy-server-url`` denotes the
Awomo upstream WebSocket. They are mutually exclusive. To manage the bridge
separately, install RPent and XPolicyLab's transport package in its environment,
set the same ``PYTHONPATH`` as above, and start:

.. code-block:: bash

   python -m robots.robodojo.awomo05_bridge --task general_pickup \
     --policy-server-url ws://127.0.0.1:9005 --host 127.0.0.1 --port 9205

In the task command, replace ``--policy-server-url``, ``--xpolicylab-root`` and
``--policy-python`` with ``--vla-endpoint http://127.0.0.1:9205``. A compatible
externally managed service may also use ``socket://host:port``; the bridge
command above serves HTTP. The RPC protocol must match ``--vla-backend``:
VPP2 uses the VPP2 protocol, and Awomo uses the session and metadata contract.

RPent releases the policy episode when a task finishes and stops bridges it
started when their runtime closes. Connected model and RPC services remain under
their operator's control. Startup failure closes pending connections and stops
components already started by RPent.

Use one bridge per upstream model, and let only that bridge execute episodes.
The official server shares model history across connections. Session ownership
prevents concurrent episodes within one bridge, but cannot coordinate separate
bridges or other clients connected directly to the model. For parallel runs,
start a separate model service for each bridge.

View Results
------------

The terminal shows service startup, planner output, and tool calls. After the
run, inspect these files in the directory selected by ``--output-dir``:

- ``result.json``: the environment's task verdict and run error status.
- ``states.json``: commands, execution results and synchronized observation records.
- Per-step RGB images; available metric depth, calibration and world-xyz artifacts
  are also saved when geometry tools are enabled.
- ``videos/``: head, left-wrist, and right-wrist camera videos.
- ``robodojo_env_server.log``: simulator startup and errors. VPP2 also writes
  ``sam3_server.log`` and ``robodojo_vpp2_server.log`` for locally started services.
  An automatically started Awomo bridge writes ``awomo05_bridge.log``; its
  separately started model logs to the model terminal. SAM3 logs appear for
  Awomo only when that component is started.
- ``run.log`` and ``transcript_*.json``: main-process logs, planner conversations,
  and errors.

Check ``environment_result_available=true`` and
``agent_error_present=false`` first, then inspect ``success``. This is RoboDojo's
native verdict; a completed tool call, ended episode or planner ``finish`` is
not proof of success. For Awomo, ``finish.agent_assessment`` records the planner's
separate assessment.

``states.json`` retains complete Awomo execution results, including
``executed_steps``, ``pending_actions``, ``stop_reason`` and ``inferences`` with
same-call metadata. Planner responses show ``inference_count`` instead of the
full inference list. ``step``/``step_idx`` selects a saved observation record;
``executed_steps`` counts confirmed actions in a call; ``observation_step`` is
relative to the model session. Do not interchange these indices.

Without ``--output-dir``, results go to
``logs/<timestamp>_robodojo_<task>_s<seed>_l<layout>/``. Use different directories for
concurrent runs. For live monitoring, see :doc:`../guides/dashboard`.

Common Options
--------------

CLI and Dashboard share the same defaults: VPP2 starts the simulator, SAM3
and VPP2 locally; Awomo starts the simulator and connects to its model, adding
SAM3 only when selected by the tool groups. ``--sim-python`` selects the
simulator interpreter; ``--policy-python`` selects the Awomo bridge interpreter,
not the model process.

Use ``--task`` to select a task. For a fixed scene, ``--seed`` selects an
official layout group and ``--layout`` selects a layout within that group;
both default to ``0``. For example,
``--task stack_blocks --seed 0 --layout 2`` uses layout 2 from group 0.
Add ``--random`` to generate a random scene instead; ``--layout`` has no effect
in this mode.

- ``--cuda-device`` selects the GPU for locally started services. If omitted,
  services inherit ``CUDA_VISIBLE_DEVICES``.
- ``--vla-backend`` accepts ``vpp2`` (default) or ``awomo05``.
- ``--vla-model-path`` applies to VPP2, overrides ``VPP2_MODEL_PATH`` and points to the resource
  root containing ``policy/`` and ``checkpoints/``.
- ``--env-endpoint``, ``--sam3-endpoint`` and ``--vla-endpoint`` connect to
  existing RPent RPC services; see :ref:`robodojo-existing-rpc` for VLA endpoints.
- ``--policy-server-url`` and ``--xpolicylab-root`` configure the locally started
  Awomo bridge as shown in :ref:`robodojo-awomo-run`.
- ``--vla-agent`` enables Awomo mixed control; ``--agent-tools`` selects
  ``policy``, ``motion``, ``geometry`` or ``sam3`` tool groups.
- ``--source-root`` selects a local RoboDojo checkout instead of the packaged code.

Child services inherit the current shell environment. The CLI does not load
``config/runtime.env`` automatically. Run ``rpent --robot robodojo --help``
for the full option list.

Tools
-----

VPP2 mode offers ``vpp2_act`` and motion and perception tools. Awomo mode offers
``run_policy(max_steps)`` on the official instruction; motion and perception
tools require the matching ``--vla-agent`` groups. The planner cannot reset an
episode; exploration and Flash replay are unavailable.

``completed`` indicates completion of the tool request, not task success.
Check ``stop_reason`` for early stops; ``executed_actions`` and
``executed_steps`` both count confirmed actions. ``execution_uncertain=true``
means some actions may have run without confirmation; they are excluded from
these counts. The system does not retry automatically, and callers should
not retry directly, to avoid executing actions twice.

When working with a stateful policy, primitive takeover invalidates its
cached remaining actions; the policy then replans from fresh observations.
Lost synchronization or invalid policy history cannot be repaired by reading
observations or resuming execution. A successful reset and restored
synchronization are required; start a new episode if reset is unavailable.

``vpp2_act(instruction, chunks=1)`` runs the specified subtask. Each chunk
predicts 32 absolute dual-arm EE16 targets (xyz, wxyz and gripper per arm)
and executes the first 24 through native EE control before replanning.
VPP2 uses the initial RGB frame and eight history samples spaced by 25 actions.
Both policy and primitive actions update this history; switching subtasks
preserves it. Image reads and ``render`` do not advance the action timeline.

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
not an arrival guarantee.

``move_delta`` plans a relative end-effector target for ``arm="left"`` or
``"right"``. ``dxyz`` is a 3D translation in metres, defaulting to ``[0, 0, 0]``;
``angle_deg`` is rotation in degrees about the 3D axis vector ``axis``, with
defaults of ``0`` and ``[0, 0, 1]``, respectively. Vector components and the
angle must be finite; a nonzero angle requires a nonzero axis, which need not
be normalized. Rotation is centred on the end-effector origin at motion
start. ``frame="world"`` (the default) expresses translation and axis direction
in world coordinates; ``frame="tool"`` uses the tool coordinates at motion start.

``substeps=25`` must be a nonnegative integer, with the same path sampling
semantics as ``move_to``. The tool does not guarantee straight-line motion
or multi-turn rotation.

``gripper=None`` retains the existing gripper control target; explicit
targets use the range and opening convention of ``set_gripper``. Zero translation and rotation without a gripper target
execute no action; an explicit gripper target still triggers planning and
execution even when translation and rotation are zero.

``set_gripper(arm, val, steps=10)`` uses 0=closed and 1=open. ``release``
defaults to 1. Intermediate values are normalized opening targets rather than
physical widths. Feedback comes from measured joints. Other controller targets
are retained while the gripper moves.

``hold(steps)`` requires a positive integer and advances that many native
actions while retaining both arms' and grippers' control targets. Simulation
continues; measured poses and objects need not stay still. Episode
termination, the action budget or cancellation can stop execution early.

.. _robodojo-awomo-collaboration:

Awomo Execution Modes
---------------------

Without ``--vla-agent``, the planner observes through ``view_env_state`` and
chooses bounded ``run_policy(max_steps)`` calls before ``finish``. It cannot
insert motion primitives.

With ``--vla-agent``, the planner chooses among the enabled policy, motion and
perception tools, deciding call order and budgets from current observations.
Both modes pass the original full task instruction to Awomo; the planner does
not rewrite model subtasks.

Mixed control always provides ``view_env_state``, ``render``, ``finish`` and the
memory readers described below. ``--agent-tools`` defaults to
``policy motion geometry``:

.. list-table:: Tool groups and observation requirements
   :header-rows: 1

   * - Group
     - Tools
     - Required data or service
   * - ``policy``
     - ``run_policy``
     - Awomo service, original instruction, native joint state and three RGB views
   * - ``motion``
     - ``move_to``, ``move_delta``, ``rotate_wrist``, ``set_gripper``, ``release``, ``hold``
     - Native motion state and planning; no depth requirement
   * - ``geometry``
     - ``back_project``, ``sample_world_xyz``, ``query_world_map``
     - Saved RGB-aligned depth and valid calibration for the queried view
   * - ``sam3``
     - ``segment``
     - SAM3 service and the selected saved RGB image; no depth requirement

All Awomo sessions need the model service for history synchronization, even
without the ``policy`` tool group. Head and wrist RGB images are always required;
depth and calibration are needed only for enabled geometry queries. Missing
geometry produces a query error; malformed supplied geometry fails capture.
Saved RGB, depth and calibration belong to the same observation.

``run_policy(max_steps)`` limits native actions in one call, not model action
chunks. Budget returns retain unused actions for continuation. A motion primitive
that actually executes actions discards the old chunk while preserving model
history; rejected or zero-action calls do not take over. Read-only queries do
not advance actions or model history.

All controls share ``--max-episode-steps``. Planner limits remain
``--max-turns`` and ``--planner-timeout-s`` with the semantics of the selected
:doc:`planner <../guides/configure_planner>`; turns are not tool-call counts.
For interruption and recovery, see :ref:`robodojo-awomo-troubleshooting`.

Task Memory
-----------

Awomo mixed mode uses the existing :doc:`memory configuration <../guides/memory>`
to read prior experience. ``list_dir`` and ``read_text_file`` only allow reads
within the configured experience directory and cannot write memory. Task source, evaluators
and hidden assets are not part of the task access interface. Current instructions
and observations take precedence over prior experience. Exploration and Flash
replay are unsupported. For Codex tool sources, see the
:ref:`Planner guide <planner-codex>`.

.. _robodojo-awomo-troubleshooting:

Awomo Startup and Recovery
--------------------------

Use the model terminal for weight and inference errors, ``awomo05_bridge.log``
for WebSocket or metadata failures, and ``robodojo_env_server.log`` for scene
startup failures. Check these conditions before rerunning:

- Missing weights or import errors: check the official installation guide,
  ``AWOMO05_WEIGHTS``, the three base-weight paths, ``dataset_stats.json`` and
  both entries in ``PYTHONPATH``. Run the model in its own environment.
- Missing ``get_model_meta`` or incompatible metadata: start the RPent model
  wrapper and use ``arx_x5``, absolute ``joint`` actions, BGR input and 25 Hz
  history. A generic VPP2 or legacy policy endpoint is not an Awomo endpoint.
- Connection refused: wait for the model's listener message, verify host/port
  reachability from the bridge, and distinguish WebSocket from RPent RPC URLs.
  A listening socket alone does not validate model compatibility or inference.
- ``All scene Unstable Error`` during reset: check the existing
  ``ROBODOJO_PLACEMENT_SETTLE_STEPS=1000`` setting and scene resources.
- Known cancellation pauses control. Resume through the existing session
  controls; RPent handles cached actions without resetting model history.
- Uncertain inference, execution or observation delivery: do not retry or resume.
  Confirmed progress remains in the feedback, but further control is blocked.
  Inspect any saved record and error, then start a new episode; the planner has
  no reset tool. An explicitly reset environment starts fresh model history.


.. _robodojo-awomo-verification:

Bounded Verification
~~~~~~~~~~~~~~~~~~~~

Use the prepared RPent simulation environment with XPolicyLab transport
dependencies and the model service already running. The tests launch their
bridge with the test interpreter. Reserve simulator GPUs via
``CUDA_VISIBLE_DEVICES`` and retain the asset, RLinf and settling variables
from the first run:

.. code-block:: bash

   uv pip install -e ".[test]"
   RPENT_RUN_AWOMO05_INTEGRATION=1 \
   RPENT_AWOMO05_POLICY_SERVER_URL=ws://127.0.0.1:9005 \
   RPENT_XPOLICYLAB_ROOT=/path/to/RoboDojo/XPolicyLab \
   RPENT_E2E_OUTPUT_DIR=/path/to/new-output \
     pytest tests/e2e_tests/robodojo/test_awomo_bounded.py -v -s

These checks use real inference and a scripted planner to verify integration,
not online LLM planning quality or task success rate. Report native task success
separately. See ``tests/README.md`` for coverage and skip conditions.

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
