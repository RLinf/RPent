Franka
======

RPent can control one physical Franka arm through an RLinf ``RealWorldEnv``
worker.

Install
-------

.. note::

	The following guide installs only the Python side (the custom RLinf
	Franka branch and ``rlinf-openpi``); it does **not** build the robot
	controller stack the arm needs. Before installing RPent, follow the RLinf
	single-arm Franka guide to set up the controller node: check Franka firmware
	compatibility, install the real-time kernel, choose your gripper (Franka hand
	or Robotiq 2F-85/2F-140) and camera, and build the ROS control packages (ROS
	Noetic, the matching libfranka and franka_ros, and serl_franka_controllers).
	See the `RLinf single-arm Franka guide
	<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/franka.html>`_.

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

RPent loads those YAMLs directly: list them under ``perception.calibration`` in
the robot config, mapping each camera to its easy_handeye YAML (the checked-in
``robots/franka/config/example.yaml`` already does this):

.. code-block:: yaml

   perception:
     calibration:
       external: ~/.ros/easy_handeye/fr3_external_apriltag_eye_on_base.yaml
       wrist: ~/.ros/easy_handeye/fr3_wrist_apriltag_ee_eye_on_hand.yaml

Development configuration
-------------------------

The checked-in values are development defaults and must be reviewed before
enabling motion:

* ``robots/franka/config/example.yaml`` contains the machine identity (robot IP,
	camera serials, gripper), workspace geometry (target/reset poses and safety
	limits), and the easy_handeye YAML mapping (see Calibration).

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

   # replace --robot-config with your own config
   uv run --extra franka rpent --robot franka --task-id 0 \
     --planner claude_code --model claude-opus-4-8 \
     --robot-config robots/franka/config/example.yaml

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
     --robot-config robots/franka/config/example.yaml

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
