Franka
======

RPent can control one physical Franka arm through an RLinf ``RealWorldEnv``
process. RPent owns planning, tools, prompts, and state artifacts. RLinf owns
Ray, ROS, the robot controller, and RealSense cameras.

Prerequisites
-------------

Use an RLinf checkout that contains ``PhysicalAgentFrankaEnv-v1`` and the
``realworld_physical_agent_eval`` configuration. Install and verify the Franka
controller, ROS workspace, camera drivers, and RLinf environment before allowing
agent motion.

RPent and RLinf may use different Python environments. Point RPent at the RLinf
checkout and interpreter:

.. code-block:: bash

	export RPENT_RLINF_ROOT=/path/to/RLinf
	export RPENT_RLINF_PYTHON=/path/to/RLinf/requirements/.venv/bin/python

Calibrate the robot IP, camera serials, TCP reset pose, and safety bounds in the
RLinf configuration. Do not use another workspace's pose or camera calibration.

Run a smoke test
----------------

Task ``0`` is a conservative primitive smoke test. Pass temporary Hydra values
with repeated ``--rlinf-override`` arguments:

.. code-block:: bash

	rpent --env franka --task-id 0 \
	  --rlinf-override 'cluster.node_groups[0].hardware.configs[0].robot_ip=ROBOT_IP' \
	  --planner api --model anthropic:claude-sonnet-4-5

The runner starts ``robots/franka/env_server.py`` under the RLinf interpreter,
waits for its ``healthz`` endpoint, and then creates the planner toolkit. The
initial state is recorded as step ``0``.

Attach to existing services
---------------------------

Use an existing environment server instead of spawning one:

.. code-block:: bash

	rpent --env franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

Task ``1`` exposes ``vla_grasp``. It requires an external VLA service whose
observation keys, state layout, action dimension, normalization, and checkpoint
were trained for this Franka setup:

.. code-block:: bash

	rpent --env franka --task-id 1 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --vla-endpoint http://VLA_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

RPent does not automatically launch the LIBERO VLA server for Franka. Omitting
``--vla-endpoint`` leaves analytic motion and gripper tools available, while
``vla_grasp`` reports a clear runtime error.

Tools and artifacts
-------------------

The Franka extension exposes ``view_env_state``, ``view_camera_meta``,
``move_delta``, ``rotate_delta``, ``open_gripper``, ``close_gripper``, and
``vla_grasp``. Mutating tools automatically capture robot state, wrist and
external RGB images, optional aligned depth arrays, and camera metadata through
RPent's central ``EnvState`` store.

Safety
------

Keep an operator at the emergency stop. Validate task ``0`` with very small
motions before attempting a grasp. Stop when camera/state results disagree,
when the requested motion is not reached, or when any calibration is uncertain.
