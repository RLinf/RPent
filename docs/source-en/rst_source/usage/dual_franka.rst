Dual Franka
===========

RPent can control a dual-arm Franka setup through an RLinf ``RealWorldEnv``
process. RPent owns planning, tools, prompts, and state artifacts. RLinf owns
Ray, ROS, both robot controllers, and the RealSense/ZED/Lumos cameras.

Prerequisites
-------------

Use an RLinf checkout that contains ``DualFrankaTcpEnv-v1`` and the
``realworld_physical_agent_eval_dual_franka`` configuration. Install and verify
both Franka controllers, the ROS workspace, camera drivers, and RLinf
environment before allowing agent motion.

Install RPent with the Franka extra in the Python environment that will run
RPent. The environment server uses this same interpreter. Point RPent at the
RLinf checkout when it is not installed from a wheel:

.. code-block:: bash

	pip install -e '.[franka]'
	export RPENT_RLINF_ROOT=/path/to/RLinf

Calibrate the left and right robot IPs, base/left/right camera serials, per-arm
reset joints, and safety bounds in the RLinf configuration. Do not use another
workspace's pose or camera calibration.

Run a smoke test
----------------

Task ``0`` is a conservative per-arm primitive smoke test. Pass temporary Hydra
values with repeated ``--rlinf-override`` arguments:

.. code-block:: bash

	rpent --env dual_franka --task-id 0 \
	  --rlinf-override 'cluster.node_groups[0].hardware.configs[0].left_robot_ip=LEFT_ROBOT_IP' \
	  --rlinf-override 'cluster.node_groups[0].hardware.configs[0].right_robot_ip=RIGHT_ROBOT_IP' \
	  --planner api --model anthropic:claude-sonnet-4-5

The runner starts ``robots/dual_franka/env_server.py`` under the RPent
interpreter, waits for its ``healthz`` endpoint, and then creates the planner
toolkit. The initial state is recorded as step ``0``.

Attach to existing services
---------------------------

Use an existing environment server instead of spawning one:

.. code-block:: bash

	rpent --env dual_franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

Task ``1`` exposes ``vla_grasp``. It requires an external dual-arm VLA service
whose observation keys, state layout, action dimension, normalization, and
checkpoint were trained for this dual-Franka setup:

.. code-block:: bash

	rpent --env dual_franka --task-id 1 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --vla-endpoint http://VLA_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

RPent does not automatically launch a VLA server for dual Franka. Omitting
``--vla-endpoint`` leaves analytic motion and gripper tools available, while
``vla_grasp`` reports a clear runtime error.

Tools and artifacts
-------------------

The dual-Franka extension exposes ``view_env_state``, ``view_camera_meta``,
``move_delta``, ``rotate_delta``, ``open_gripper``, ``close_gripper``, and
``vla_grasp``. Each rule-based motion selects exactly one arm, ``left`` or
``right``; the other arm is left uncommanded. ``move_delta`` and ``rotate_delta``
are expressed in the fixed world (right_base) frame. Mutating tools automatically
capture per-arm robot state and the synchronized left-wrist, base, and
right-wrist RGB images through RPent's central ``EnvState`` store.

Safety
------

Keep an operator at the emergency stop for both arms. Validate task ``0`` with
very small single-arm motions before attempting a grasp. Stop when camera/state
results disagree, when the requested motion is not reached, or when any
calibration is uncertain.
