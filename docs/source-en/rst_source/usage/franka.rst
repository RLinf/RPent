Franka
======

RPent can control one physical Franka arm through an RLinf ``RealWorldEnv``
worker.

Install
-------

From the RPent repository root:

.. code-block:: bash

	uv sync --extra franka

This installs the custom RLinf Franka branch and ``rlinf-openpi`` into
``.venv``.

Calibration
-----------

Hand-eye calibration is performed with ROS
`easy_handeye <https://github.com/IFL-CAMP/easy_handeye>`_. It produces one YAML
per camera (eye-on-base for the external camera, eye-on-hand for the wrist
camera) and saves them under ``~/.ros/easy_handeye/`` by default.

RPent reads a JSON bundle (``hand_eye_calibration.json``) that carries each
camera's ``source_name``, ``parameters``, and ``transformation``. Generate it by
copying those fields out of each ``easy_handeye`` YAML.

The bundle location is configurable with ``--calibration-path`` (default
``~/.ros/easy_handeye/hand_eye_calibration.json``).

Development configuration
-------------------------

The checked-in values are development defaults and must be reviewed before
enabling motion:

* ``robots/franka/config/example.yaml`` contains the machine identity (robot IP,
	camera serials, gripper) and workspace geometry (target/reset poses and
	safety limits).

RPent translates this robot-focused schema into the internal RLinf cluster and
environment objects. To use a different file, pass
``--robot-config /path/to/robot_config.yaml``.

Start Ray
---------

Set the node rank before starting Ray, because Ray captures the environment at
startup:

.. code-block:: bash

	export RLINF_NODE_RANK=0
	ray stop --force
	ray start --head

Run a smoke test
----------------

The smoke test verifies that basic analytic motion and gripper primitives work
correctly. To run it, launch RPent with task ``0``:

.. code-block:: bash

	# replace --robot-config and --calibration-path with your own paths
	uv run --extra franka rpent --robot franka --task-id 0 \
	  --planner claude_code --model claude-opus-4-8      \
	  --robot-config robots/franka/config/example.yaml   \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

RPent starts ``robots/franka/env_server.py`` with the current interpreter,
loads the RPent robot config, generates the internal RLinf adapter config,
connects to Ray, waits for ``healthz``, and records the initial state as step
``0``.

VLA grasp demo
--------------

RPent provides a demo that uses a VLA to grasp objects. Task ``1`` exposes
``vla_grasp``. Single Franka currently requires a compatible external VLA
service whose observation layout, action layout, checkpoint, and normalization
statistics match the current Franka training configuration:

.. code-block:: bash

	uv run --extra franka rpent --robot franka --task-id 1 \
	  --vla-endpoint http://VLA_HOST:PORT \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

The VLA server must be deployed separately for now. Without
``--vla-endpoint``, analytic motion and gripper tools remain available, but
``vla_grasp`` raises a runtime error.

Tools and artifacts
-------------------

The extension exposes ``view_env_state``, ``view_camera_meta``, ``move_delta``,
``rotate_delta``, ``open_gripper``, ``close_gripper``, and ``vla_grasp``.
Mutating tools capture robot state, wrist and external RGB images, optional
aligned depth arrays, and camera metadata in RPent's central ``EnvState``.

Safety
------

Keep an operator at the emergency stop. Validate task ``0`` with very small
motions before attempting a grasp. Stop when camera/state results disagree,
when the requested motion is not reached, or when any calibration is uncertain.
