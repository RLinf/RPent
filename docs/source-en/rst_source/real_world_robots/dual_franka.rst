Dual-Arm Franka
===============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka-deploy.jpg
   :alt: Two Franka arms and a camera mounted around a shared workbench
   :figclass: rpent-robot-figure
   :align: center

   Dual-Franka system for real-world experiments.

RPent controls a two-node dual-Franka setup through an RLinf
``RealWorldEnv`` worker. Watch the :doc:`task demos <../real_world_demos/franka>`.

Install
-------

Install RPent on both Linux control nodes with matching code, Python, Ray and
RLinf versions. Nodes running VLA or SAM3 locally need an NVIDIA GPU.
Use the same installation workflow as :doc:`../get_started/quickstart`:

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[franka,sam3]"

If Python package downloads are slow, replace the installation command with the
`Tsinghua PyPI mirror <https://mirrors.tuna.tsinghua.edu.cn/help/pypi/>`_:

.. code-block:: bash

   uv pip install -e ".[franka,sam3]" \
     --index-url https://pypi.tuna.tsinghua.edu.cn/simple

This option changes only the Python package index; it does not proxy Git
repositories, direct download URLs or model weights.

Check libfranka compatibility with the robot firmware before connecting.
The calibration state reader also needs libfranka development files; see the
calibration steps below.

Run subsequent commands from the repository root in this environment. In a new
terminal, activate ``.venv`` and set the required model variables again.

Download SAM3 weights on the environment node if local segmentation is needed:

.. code-block:: bash

   uv pip install modelscope
   modelscope download --model facebook/sam3 sam3.pt \
     --local_dir ./checkpoints/sam3
   export SAM3_CHECKPOINT_PATH="$PWD/checkpoints/sam3/sam3.pt"

Without this variable or ``--sam3-endpoint``, segmentation is unavailable;
installing the extra alone does not download weights. VLA checkpoints are
configured in :ref:`dual-franka-vla`.

Calibration
-----------

Both ``base_camera`` and ``d455_camera`` must be calibrated against the
**right-arm base**. Use ``calibration_tools/`` or ROS
`easy_handeye <https://github.com/IFL-CAMP/easy_handeye>`_; both export YAML.
Wrist cameras are observation-only in the default configuration. To use a
wrist RGBD camera for projection, configure its calibration and projection view
as described below.

The collector uses RealSense RGB images and accepts zero distortion coefficients
or nonzero ``distortion.brown_conrady`` coefficients. Other nonzero distortion
models are rejected.

**1. Prepare the environment**

Use the installed environment from the RPent repository root on the camera node.
To install calibration dependencies separately, create a dedicated environment:

.. code-block:: bash

   uv venv .venv-calibration
   source .venv-calibration/bin/activate
   uv pip install -r calibration_tools/requirements.txt pyrealsense2

On the right-arm control node, compile the reader against the libfranka
development library matching the robot firmware:

.. code-block:: bash

   cmake -S calibration_tools -B calibration_tools/build \
     -DCMAKE_PREFIX_PATH=/absolute/path/to/libfranka/install
   cmake --build calibration_tools/build --parallel

The executable ``calibration_tools/build/read_franka_state`` uses ``readOnce()``
to output end-effector poses, joint velocities, and robot status as JSON. It
sends no motion commands. Installing Python control packages does not guarantee
the headers and CMake configuration needed to build this reader.
Replace the camera serials, SSH alias, absolute reader path, and robot IP below.
Omit ``--ssh-host`` for a local reader; add ``--library-dir /path/to/lib`` when
its shared libraries need an explicit search path.
For a remote reader, configure SSH authentication that works without a password
prompt (the collector uses ``BatchMode=yes``). Verify the reader path and library
access as that SSH user before collecting.

**2. Start the camera and collect samples**

Close programs using the cameras, then start the service:

.. code-block:: bash

   python calibration_tools/raw_camera_service.py \
     --base-serial BASE_SERIAL --d455-serial D455_SERIAL

Attach the ChArUco board rigidly to the right end effector and keep the camera
stationary. Defaults are 6×8 squares, 25 mm square length, 18 mm marker length,
DICT_4X4_100, and a non-legacy layout. Configure other boards with
``--squares-x``, ``--squares-y``, ``--square-m``, ``--marker-m``, and
``--dictionary``; lengths are in meters.
Measure the printed square size and ensure it matches the configuration.

In another terminal on the same camera node, activate the chosen environment:

.. code-block:: bash

   python calibration_tools/base_handeye_collect.py \
     --arm right --camera-serial BASE_SERIAL \
     --camera-url http://127.0.0.1:8765/raw/base \
     --ssh-host robot-right \
     --reader /absolute/path/to/read_franka_state --robot-ip ROBOT_IP \
     --output calibration_tools/sessions/base-to-right

Open ``http://127.0.0.1:8767``. Move the right arm manually, release the guidance
button, wait until stationary, and click ``Capture pose``. Sampling requires
``robot_mode=1`` (Idle), ``has_errors=false``, and all joint speeds at or below
0.015 rad/s. The state reader reports these values; the collector rejects
nonconforming samples and shows the error on the page. It does not change the
robot mode. Collect 20–30 distinct
poses with rotations about multiple axes; fitting requires at least ten.
The tool only reads state. Without ``--output``, sessions are saved under the
script directory's ``calibration_tools/sessions/``, regardless of the working
directory.

The camera service and collector bind to loopback. For a browser on another
computer, forward the collector port, for example
``ssh -L 8767:127.0.0.1:8767 CAMERA_HOST``; the raw camera service stays local to
the collector.

Use the saved-sample table to inspect each sample's corner count and
reprojection RMS in pixels. Click ``Preview`` to open its annotated image,
then ``Exclude`` to remove an unsuitable sample from the next solve. Its
files remain in the session's ``excluded/`` directory. ``Active samples``
counts only samples used by the solver; IDs remain unchanged, so gaps are
normal. Adding or excluding samples does not recalculate existing calibration
results: run the solver again before exporting.

Stop the collector with ``Ctrl+C``. To start again after an interruption or
recollect the entire session, rerun the command with a new ``--output`` directory.
Existing session files remain on disk but are not loaded into the new collection.

For D455, repeat with ``D455_SERIAL``, ``/raw/d455``, and
``calibration_tools/sessions/d455-to-right``. Check that its stream configuration
uses a supported distortion model as described above.

**3. Fit and export**

.. code-block:: bash

   python calibration_tools/solve_base_handeye.py \
     calibration_tools/sessions/base-to-right --arm right
   python calibration_tools/export_dual_franka.py \
     calibration_tools/sessions/base-to-right/base_camera_extrinsic_candidate.json \
     --output calibration_tools/exports/base_to_right.yaml

For D455, use ``d455-to-right`` and ``d455_to_right.yaml`` instead. Inspect
``quality_report.json`` and validate the candidate against independent physical
measurements. Successful fitting or YAML export does not establish accuracy.
Export does not overwrite existing files.
When exporting a recalculated result, choose a new output filename, such as
``calibration_tools/exports/base_to_right_v2.yaml``. Validate it before updating
``perception.calibration`` to use the new file.

**4. Configure RPent**

List the independently validated YAMLs in your robot configuration:

.. code-block:: yaml

   perception:
     calibration:
       base_camera: /absolute/path/to/base_to_right.yaml
       d455_camera: /absolute/path/to/d455_to_right.yaml

Load this configuration with ``--robot-config`` below. Preserve and verify
``perception.base_frames`` and localization bounds; these tools do not calibrate
the relationship between the two arm bases.

Wrist projection is optional. Uncomment the wrist entries under
``perception.calibration`` and ``perception.projection_views`` in the example
configuration, and configure RGBD cameras with aligned depth and color intrinsics.
Each wrist calibration YAML needs ``arm: left`` or ``arm: right`` at the top
level, ``parameters.eye_on_hand: true``, and
``parameters.robot_effector_frame: left_ee_O_T_EE`` or ``right_ee_O_T_EE``.
Its ``transformation`` maps camera coordinates into that end-effector frame.
RPent combines it with the selected snapshot's end-effector pose and, for poses
in ``left_base``, the configured base-frame transform, to produce ``right_base``
points. Snapshot TCP poses must represent O_T_EE and declare their base frame.
The default Lumos observation cameras are not automatically enabled for RGBD
projection; selecting a wrist anchor requires this configuration.

After calibration, stop all camera services, collectors, and robot-state readers
started for calibration before starting RPent.

Development Configuration
-------------------------

If calibration used a separate Python environment, reactivate RPent's
``.venv`` before running the following commands.

Review and edit the checked-in development defaults before enabling motion:

* ``robots/dual_franka/config/example.yaml`` contains the machine identity (both
	robot IPs, camera serials/types, gripper connections), workspace geometry
	(target poses and safety limits), the easy_handeye YAML mapping (see
	Calibration), and perception localization bounds + base-frame transform. It
	sets ``realtime_config: ignore``; use ``enforce`` on a PREEMPT_RT kernel to
	refuse non-real-time operation.

RPent translates this robot-focused schema into the internal two-node RLinf
cluster and environment objects. To use a different file, pass
``--robot-config /path/to/robot_config.yaml``.

The default observation cameras provide RGB only. The separate D455 perception
camera also supplies aligned depth for projection; registering an observation
camera under ``perception.projection_views`` does not enable depth capture.

During ``rotate_delta``, the position target stays at the starting TCP position.
On real hardware, compliance/impedance control can allow the actual pose to
deviate from the target due to load, contact, or limited tracking accuracy.

``rotation_hold.max_drift_m`` defaults to ``0.10`` m (10 cm) and bounds the
Euclidean distance from the TCP position at the start of this rotation. It is
separate from the position tolerance, ``move_tolerance_m`` (0.006 m). At or below
the drift limit, after rotation ends, ``move_delta`` returns the TCP to its
starting position only if the position error exceeds that tolerance. The call
reports ``ok=True`` only if compensation succeeds (when needed) and the final
position and angle both meet their tolerances; otherwise it reports ``False``.
Above the limit, a check during the rotation loop or at its end raises an RPC
error, ending the call without translation compensation.

The robot's own protections can still trigger independently.

Start the Two-node Ray Cluster
------------------------------

Activate the installed ``.venv`` on each node before starting Ray. The default
placement in ``robots/dual_franka/runtime_config.py`` is:

* Node ``0`` is the Ray head: it runs RPent, the environment worker, the
  **left** arm/gripper controller, the base camera and the left wrist camera.
  RPent opens the additional D455 locally here. Local VLA and SAM3 services
  also run here when configured.
* Node ``1`` is a Ray worker: it runs the **right** arm/gripper controller and
  its right wrist camera. It needs the installed environment and Ray, but no
  separate RPent runner.

RLinf creates camera handles on the configured nodes; it does not search other
nodes for a relocated USB camera. Connect each wrist camera to its arm's control
node and edit the configured type/serial for the actual device.

.. warning::

   Set ``RLINF_NODE_RANK`` before starting Ray on each controller node.

Node ``0``:

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray stop --force
   ray start --head --port=6379 --node-ip-address=HEAD_IP

Node ``1``:

.. code-block:: bash

   export RLINF_NODE_RANK=1
   ray stop --force
   ray start --address=HEAD_IP:6379 --node-ip-address=WORKER_IP

Task Selection
--------------

The main example uses task ``2``. On first connection, use task ``0`` to check
small translations, rotations and gripper actions. This is a planner-guided
check, not a fixed automated sequence, and does not require VLA.

.. list-table::
   :header-rows: 1

   * - ID
     - Task
     - VLA required
   * - 0
     - Planner-guided primitive checks
     - No
   * - 1
     - Dirty/clean sorting of bowls and plates, with cup and utensils
     - Yes
   * - 2
     - Dirty/clean sorting of bowls, plates, and two cups; no utensils
     - Yes

Task 1 was previously task 3; task 2 was previously task 6. Old logs and memory
retain their original IDs. Do not treat an old ``dual_franka_t1`` memory corpus
as the new task 1; select and review the intended corpus explicitly. A task ID
does not enable exploration: add ``--explore`` when desired.

.. _dual-franka-vla:

.. _vla-grasp-demo:

VLA Configuration
-----------------

Task ``2`` sorts bowls, plates and two cups, without utensils. Chicken nuggets
inside a bowl, or an egg-tart foil tray in a cup or on a plate, mark it as dirty.
Dirty objects go into the metal basket; clean objects go into the cardboard box.
The task provides ``vla_right_grasp`` / ``vla_handoff`` / ``vla_left_place`` skills.

``PI05_CHECKPOINT_PATH`` points to the trained Pi0.5 checkpoint, while
``DUAL_FRANKA_REPO_ID`` is the dataset ID used to locate matching normalization
statistics.

The environment variables correspond to ``--vla-model-path`` and
``--vla-repo-id``. The repo ID is the relative directory containing the matching
normalization statistics, not the model download repository name. The checkpoint
must contain:

.. code-block:: text

   actor/model_state_dict/full_weights.pt
   <DUAL_FRANKA_REPO_ID>/norm_stats.json

**Pretrained checkpoint**

A clean-desk policy checkpoint is published on ModelScope. It supplies the VLA
skills, not an automatic dirty/clean sorting decision:
`Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000
<https://modelscope.cn/models/Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000>`_.
Download it, point ``PI05_CHECKPOINT_PATH`` at the downloaded directory, and set
``DUAL_FRANKA_REPO_ID`` to the subdirectory that holds ``norm_stats.json``:

.. code-block:: bash

   uv pip install modelscope
   modelscope download \
     --model Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000 \
     --local_dir /path/to/pi05-dualfranka-clean-desk

   export PI05_CHECKPOINT_PATH=/path/to/pi05-dualfranka-clean-desk
   export DUAL_FRANKA_REPO_ID=org/dual-franka-tcp-rot6d

Replace the example repo ID with the directory actually present in your
checkpoint. Check both files above before starting the task.

.. warning::

	This checkpoint is trained only on our in-house test environment (robot
	poses, cameras, workspace layout, and objects), so it is expected to
	generalize poorly to a different setup. To deploy on your own rig, collect
	demonstrations and fine-tune your own checkpoint with RLinf by following the
	`RLinf dual-Franka guide
	<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dual_franka.html>`_
	(collect GELLO demos, convert to tcp_rot6d, run SFT, then deploy).

Run a Task
----------

Complete calibration, robot configuration and Ray startup, then configure the
planner using :doc:`../guides/configure_planner`. Before using Codex, complete
:ref:`dual-franka-codex-isolation`.

With the model variables set, run task ``2``. RPent starts the required
environment and model services automatically; do not start VLA in another terminal:

.. code-block:: bash

   rpent --robot dual_franka --task-id 2 \
     --planner codex --model gpt-6-astra --reasoning-effort xhigh \
     --robot-config robots/dual_franka/config/example.yaml \
     --memory-profile local --memory-dir ./memory/dual_franka

Connection initializes controllers and grippers without resetting robot posture.
The agent first calls ``request_scene_reset``: secure held objects, restore the
tabletop and check both reset paths, then answer ``done`` to allow reset and a
fresh observation. Motion remains blocked if confirmation or reset fails. Each
new run/session needs its own confirmation.

The example uses Codex, ``gpt-6-astra`` and ``xhigh`` reasoning. Claude Code is
also available: replace the planner/model options with
``--planner claude_code --model claude-opus-4-8``. See
:doc:`../guides/configure_planner` for model and reasoning settings.

Append options to this same command as needed:

* ``--explore`` explores and accumulates task memory, as described below.
* ``--explore --interactive`` accepts terminal feedback during exploration;
  ordinary tasks do not support ``--interactive`` on its own.
* ``--dashboard`` accepts tasks, messages and confirmations in the web input.
  Add ``--explore`` for exploration. Do not combine it with ``--interactive``.
  See :doc:`../guides/dashboard` for interface controls.

View Results
------------

Use a new ``--output-dir`` for each run. Inspect:

* ``transcript_*.json`` for planner messages, tool calls and the finish result.
* ``run.log`` and component ``*_server.log`` files for runtime/startup errors.
* ``states.json`` and its referenced image, depth and robot-state artifacts for
  the evidence after each step.
* ``operator_events.json`` for reset confirmations and human verdicts.

Exploration keeps per-session state and operator records under
``sessions/session_<NNN>/``. Physical task success comes from the operator's
verdict, not a successful RPC or the planner's own ``finish`` text. See
:ref:`run-output-files` for the common output layout.

Tools and Artifacts
-------------------

The extension exposes ``view_env_state``, ``view_camera_meta``, ``move_delta``,
``rotate_delta``, ``open_gripper``, ``close_gripper``, and ``vla_right_grasp`` / ``vla_handoff`` / ``vla_left_place``. Each
analytic motion selects exactly one arm, ``left`` or ``right``. Mutating tools
capture per-arm state and synchronized left-wrist, base, and right-wrist images
in RPent's central ``EnvState``.

Safety
------

Keep operators at both emergency stops. Validate task ``0`` with very small
single-arm motions before attempting a grasp. Stop when camera/state results
disagree, when the requested motion is not reached, or when any calibration is
uncertain.

Manual Skill Testing
--------------------

The manual entry reads the registered tool schemas. From the repository root:

.. code-block:: bash

   python robots/dual_franka/dual_franka_manual_call.py --list-primitives
   python robots/dual_franka/dual_franka_manual_call.py --schema vla_right_grasp

As with the task runner, omitted endpoints start the required services on free
ports; the manual call uses the returned clients and stops only those services
when it exits. Supply ``--env-endpoint``, ``--vla-endpoint`` or
``--sam3-endpoint`` to reuse an existing service. Attaching to an existing Env
does not reset it. With a newly started Env, a manual action first requests
scene confirmation and performs one reset. An explicit manual ``reset`` also
waits for confirmation. Listing tools and printing schemas never starts services.

For example, check the schema before running a small left-arm translation:

.. code-block:: bash

   python robots/dual_franka/dual_franka_manual_call.py --schema move_delta
   python robots/dual_franka/dual_franka_manual_call.py \
     --primitive move_delta --params '{"arm":"left","delta_xyz":[0.01,0,0]}' \
     --robot-config robots/dual_franka/config/example.yaml

Use ``--primitive NAME --params JSON`` to call a tool; manual calls do not select
a task ID. The example config already sets ``vla.instruction`` for the clean-desk
policy. Change it for a different checkpoint, or override it with
``--instruction``. This setting applies only to manual calls; task runners still
use their registered task instructions. ``prompt`` inside ``--params`` records
the skill intent and does not replace the policy instruction.

With the VLA checkpoint and normalization variables set, run one grasp skill:

.. code-block:: bash

   python robots/dual_franka/dual_franka_manual_call.py \
     --primitive vla_right_grasp \
     --params '{"prompt":"grasp the bowl","max_chunks":1}' \
     --robot-config robots/dual_franka/config/example.yaml

.. _dual-franka-codex-isolation:

Robot Codex Profile Isolation
-----------------------------

In a dedicated terminal, activate ``.venv`` and set separate Codex configuration
and session directories from the repository root. Repeat this in each new
robot-control terminal:

.. code-block:: bash

   export CODEX_HOME="$PWD/.codex-rpent-live"
   export CODEX_SQLITE_HOME="$CODEX_HOME"
   export CODEX_SERVICE_TIER=fast
   mkdir -p "$CODEX_HOME"
   unset CODEX_API_KEY CODEX_BASE_URL OPENAI_API_KEY OPENAI_BASE_URL CODEX_ACCESS_TOKEN
   export CODEX_BIN="$(python -c 'from codex_cli_bin import bundled_codex_path; print(bundled_codex_path())')"

On first use, enable file credential storage in ``$CODEX_HOME/config.toml``,
then log in with the CLI bundled with the SDK:

.. code-block:: toml

   cli_auth_credentials_store = "file"

.. code-block:: bash

   "$CODEX_BIN" login --device-auth
   "$CODEX_BIN" login status

If ``sqlite_home`` is already configured, point it at the separate directory too.
For a custom API, explicitly set this deployment's ``CODEX_API_KEY`` and
``CODEX_BASE_URL``. See `Codex authentication
<https://developers.openai.com/codex/auth/>`_ for other login methods.

RPent does not enable this isolation automatically. It neither isolates shared
workspace files nor migrates old sessions. Task memory is selected separately
with ``--memory-dir``. Never commit the private directory or credentials.

Attended Exploration
--------------------

Append ``--explore`` to the task ``2`` command. Set attempts per planner session
with ``--explore-attempts-per-session`` and session count with
``--explore-sessions``. Add ``--interactive`` for terminal input or
``--dashboard`` for web interaction.

Each planner session begins with ``request_scene_reset``. Secure held objects,
restore the scene and reply ``done``. A successful reset and camera/state capture
start an attempt; a failed reset keeps motion blocked.
``--explore-attempts-per-session`` limits attempts within each session.
External ``--env-endpoint`` servers must advertise ``explicit_reset_only=True``
for both ordinary and exploration runs.

``request_operator_verdict`` asks for ``success``, ``failure``, ``continue`` or
``abort``, with optional notes. Its success/failure reply judges the current
attempt; ``continue`` resumes it and ``abort`` stops exploration. Further motion
clears the previous verdict. Closing terminal input also stops exploration.

With Codex/Claude and ``--interactive``, use the displayed request ID, for example
``/operator <request-id> done`` or ``/operator <request-id> success object lifted``.
Other messages go to the agent. Without ``--interactive``, answer the prompt
directly. With ``--dashboard``, use the commands below in the Dashboard message input.

Records and Memory
~~~~~~~~~~~~~~~~~~

Find camera and robot state in ``sessions/session_<NNN>/``, attempt boundaries in
``exploration.json`` and human feedback in ``operator_events.json``.
Failed attempts are retained; working notes are in
``<memory-dir>/_internal/inbox/<recipe_tag>/wip/``.

Memory merging is enabled by default. Normal completion merges valid ``global``
and ``task-family`` drafts, including lessons from failed attempts. Success also
saves a complete audit/recipe pair under ``task-specific``.
Use ``--no-auto-merge-memory`` to keep records without merging. Errors, aborts,
task replacement and Dashboard Session shutdown prevent merging. See
:doc:`../guides/memory` for the memory layout.

Exploration Commands
~~~~~~~~~~~~~~~~~~~~

In Codex/Claude terminal interaction (``--explore --interactive``) or Dashboard
exploration, use:

* ``/success`` or ``/failure`` to end the task, rather than only judge an attempt.
  Success requires a confirmed reset. In the terminal, failure can end the task
  before the first reset; in Dashboard, answer a pending reset with ``/done`` or
  ``/abort`` first.
* ``/done`` to confirm a pending scene-reset request.
* ``/continue`` to answer a pending verdict and resume the attempt.
* ``/abort`` to stop the task, even if a camera is unavailable.

Enter each command on its own. Bare words such as ``success`` and ``failure``
are ordinary agent messages. Already-issued robot actions or RPCs may need to
return before the task can end; these commands are not an emergency stop.

VLA Diagnostic Console
~~~~~~~~~~~~~~~~~~~~~~

Use the standalone console for prediction recording and explicit execution.
``--task-id`` selects an existing VLA task profile; the policy instruction
comes from its ``vla_instruction`` unless overridden with ``--instruction``.
Run the following command from a source checkout. The diagnostic initializes only
the environment and VLA components, without starting SAM3.

.. code-block:: bash

   python -m tests.e2e_tests.dual_franka.dual_franka_vla --task-id 2 \
     --robot-config /path/to/robot.yaml \
     --vla-model-path /path/to/checkpoint --vla-repo-id org/dataset

Commands: ``prompt <instruction>``, ``infer`` (no execution), ``step``
(fresh prediction and execution), ``run N`` (1–20 chunks), ``reset``, ``quit``.
Initialization does not reset posture; use the explicit ``reset`` command only
after clearing both reset paths. Inputs and predictions are saved before
execution as JSON/NPZ. Invalid observations/actions are rejected; uncertain
execution blocks further motion until restart. RPC success is not task success.
Action validation expects 20 steps per prediction chunk. For a checkpoint with a
different chunk length, set ``--expected-action-steps`` explicitly to match it.
External model servers use the standard VLA prediction and health-check RPCs.

Flash replay
------------

Dual Franka uses the shared :ref:`Franka Flash workflow <franka-flash>`.
Generate with ``--robot dual_franka --task dual_franka_t0`` and replay with
``--robot dual_franka --planner flash --task-id 0``. Use a dual-arm plan and
robot configuration. Translation annotations name ``base``, ``d455``,
``left_wrist`` or ``right_wrist`` with depth and valid calibration.
Primitive moves specify ``arm: left`` or ``arm: right``; left-arm workspace
checks convert the shared right-base target into the left-base frame.

Stop the Run
------------

Press Ctrl+C in the terminal to request RPent shutdown. Already-issued robot actions or an active RPC must return before cleanup completes; use the hardware emergency stop for an emergency. Check both arms and grippers before shutting down services through the controller’s shutdown procedure.
