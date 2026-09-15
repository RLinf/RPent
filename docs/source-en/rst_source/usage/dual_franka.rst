Dual Franka
===========

RPent can control a two-node dual-Franka setup through an RLinf
``RealWorldEnv`` worker.

Install
-------

.. note::

	The following guide installs only the Python side (the custom RLinf
	Franka branch and ``rlinf-openpi``); it does **not** build the robot-node
	control stack the two arms need. Before installing RPent, follow the RLinf
	dual-Franka guide to set up both robot nodes: choose a compatible
	``LIBFRANKA_VERSION``, build the ``franka-franky`` (franky/libfranka) control
	stack, configure the PREEMPT_RT real-time kernel and permissions, and install
	the GELLO teleoperation and gripper dependencies. See the `RLinf dual-Franka
	guide
	<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dual_franka.html>`_.

From the RPent repository root:

.. code-block:: bash

	uv sync --extra franka --extra sam3

This installs the custom RLinf Franka branch and ``rlinf-openpi`` into
``.venv``.

Calibration
-----------

Hand-eye calibration is performed with ROS
`easy_handeye <https://github.com/IFL-CAMP/easy_handeye>`_. It produces one YAML
per projection camera (``base_camera`` and ``d455_camera``) and saves them under
``~/.ros/easy_handeye/`` by default.

RPent reads a JSON bundle (``hand_eye_calibration.json``) that carries each
camera's ``source_name``, ``parameters``, and ``transformation``. Generate it by
copying those fields out of each ``easy_handeye`` YAML.

The bundle location is configurable with ``--calibration-path`` (default
``~/.ros/easy_handeye/hand_eye_calibration.json``).

Development configuration
-------------------------

Review and edit the checked-in development defaults before enabling motion:

* ``robots/dual_franka/config/example.yaml`` contains the machine identity (both
	robot IPs, camera serials/types, gripper connections), workspace geometry
	(target poses and safety limits), and perception localization bounds +
	base-frame transform.

RPent translates this robot-focused schema into the internal two-node RLinf
cluster and environment objects. To use a different file, pass
``--robot-config /path/to/robot_config.yaml``.

Start the two-node Ray cluster
------------------------------

The two nodes have fixed, different roles (defined in
``robots/dual_franka/runtime_config.py``):

* Node ``0`` is the Ray head: it runs the dual-Franka environment
	worker (all cameras, perception, and arm/gripper state), and the **left**
	arm's real-time controller. For the VLA task, the
	local VLA server also runs here.
* Node ``1`` is a Ray worker: it runs only the **right** arm's real-time
	controller, with no cameras and no RPent process.

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

Run a smoke test
----------------

Task ``0`` tests conservative single-arm analytic motion and gripper primitives:

.. code-block:: bash

	uv run --extra franka rpent --robot dual_franka --task-id 0 \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

RPent starts ``robots/dual_franka/env_server.py`` with the current interpreter,
loads the RPent robot config, generates the internal RLinf adapter config,
connects to Ray, waits for ``healthz``, and records the initial state as step
``0``. Task ``0`` does not load the VLA.

VLA grasp demo
--------------

RPent provides a demo that uses a VLA to grasp objects. Task ``1`` exposes
``vla_right_grasp`` / ``vla_handoff`` / ``vla_left_place`` and can start the dual-Franka VLA server locally.
``PI05_CHECKPOINT_PATH`` points to the trained Pi-05 checkpoint, while
``DUAL_FRANKA_REPO_ID`` is the dataset ID used to locate matching normalization
statistics:

.. code-block:: bash

	export PI05_CHECKPOINT_PATH=/path/to/checkpoints/global_step_N
	export DUAL_FRANKA_REPO_ID=org/dual-franka-tcp-rot6d

	uv run --extra franka rpent --robot dual_franka --task-id 1 \
	  --cuda-device 0 \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

The checkpoint must contain:

.. code-block:: text

	actor/model_state_dict/full_weights.pt
	<DUAL_FRANKA_REPO_ID>/norm_stats.json

**Pretrained checkpoint**

A ready-made task ``1`` checkpoint is published on ModelScope:
`Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000
<https://modelscope.cn/models/Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000>`_.
Download it, point ``PI05_CHECKPOINT_PATH`` at the downloaded directory, and set
``DUAL_FRANKA_REPO_ID`` to the subdirectory that holds ``norm_stats.json``:

.. code-block:: bash

	modelscope download \
	  --model Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000 \
	  --local_dir /path/to/pi05-dualfranka-clean-desk

	export PI05_CHECKPOINT_PATH=/path/to/pi05-dualfranka-clean-desk

.. warning::

	This checkpoint is trained only on our in-house test environment (robot
	poses, cameras, workspace layout, and objects), so it is expected to
	generalize poorly to a different setup. To deploy on your own rig, collect
	demonstrations and fine-tune your own checkpoint with RLinf by following the
	`RLinf dual-Franka guide
	<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dual_franka.html>`_
	(collect GELLO demos, convert to tcp_rot6d, run SFT, then deploy).

When ``--vla-endpoint`` is absent, RPent starts
``robots/dual_franka/vla_server.py`` and loads
``pi05_dualfranka_tcp_rot6d`` once.

To run the VLA service separately:

.. code-block:: bash

	uv run --extra franka python -m robots.dual_franka.vla_server \
	  --model-path /path/to/checkpoints/global_step_N \
	  --repo-id org/dual-franka-tcp-rot6d \
	  --cuda-device 0 --transport http --host 0.0.0.0 --port 6000

Then pass ``--vla-endpoint http://VLA_HOST:6000`` to ``rpent``. An external
endpoint always takes precedence over local auto-start.

External environment server
---------------------------

To attach RPent to an already-running dual-Franka environment service:

.. code-block:: bash

	uv run --extra franka rpent --robot dual_franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

Tools and artifacts
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

Manual skill testing
--------------------

The deployment scripts live in ``robots/dual_franka/``. From the repository root:

.. code-block:: bash

   robots/dual_franka/run_manual_skill.sh --list-primitives
   robots/dual_franka/run_manual_skill.sh --schema vla_right_grasp

Use ``--primitive NAME --params JSON`` to call a tool. ``--task-id`` selects
the task's configured ``vla_instruction`` for named VLA skills; the planner's
segment prompt is recorded separately. Existing clean-desk tasks retain their
checkpoint's original training instruction. ``--robot-config`` and
``--calibration-path`` select the machine configuration and calibration.
Local SAM3 requires the ``sam3`` extra; a remote SAM3 service can be attached
with ``--sam3-endpoint``.

Robot Codex profile isolation
----------------------------------------

Operator verdict and scene-restoration tools currently require exclusive terminal
input. Run the runner in a plain TTY, without ``--interactive`` or Dashboard;
unsupported combinations are rejected before hardware connection. Evaluation
also exposes ``request_operator_verdict`` and requires a verdict before finish.
``request_scene_reset`` remains exploration-only.

The deployment wrappers select ``RPENT_CODEX_HOME`` (default:
``.codex-rpent-live`` inside the checkout), not the coding shell's
``CODEX_HOME``. Memory defaults to its ``memory`` subdirectory and the Codex
state database uses the dedicated directory too. Create a private ``config.toml``
there if needed; do not overwrite existing private settings.

For API deployments, explicitly set ``RPENT_CODEX_API_KEY`` and optionally
``RPENT_CODEX_BASE_URL``. The wrappers clear inherited ``CODEX_API_KEY``,
``CODEX_BASE_URL``, ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL``. Otherwise,
authenticate separately in the dedicated profile. File-based credential storage
can be configured; check private configurations for shared OS keychain use.
Never commit credentials, private configuration or session records.

``RPENT_CODEX_MODEL``, ``RPENT_REASONING_EFFORT`` and
``RPENT_CODEX_SERVICE_TIER`` default to ``gpt-5.5``, ``medium`` and ``fast``.
These isolation rules apply to the deployment wrappers, not the generic RPent
CLI. Prefer invoking a wrapper: sourcing its environment script directly changes
the current shell's environment.

Directory isolation is not a security sandbox or workspace-file isolation.
The planner explicitly uses no interactive approvals and full filesystem access.
Editing private configuration does not override the planner's
explicit permissions. The connectivity probe remains read-only.
