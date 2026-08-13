Franka
======

RPent can control one physical Franka arm through an RLinf ``RealWorldEnv``
worker. This development branch keeps the runtime configuration in RPent and
installs the required RLinf/OpenPI branch through the ``franka`` extra. A
separate RLinf checkout, virtual environment, or installation step is not
required.

Install
-------

From the RPent repository root:

.. code-block:: bash

	uv sync --extra franka

This installs the custom RLinf Franka branch and ``rlinf-openpi`` into
``.venv`` according to ``pyproject.toml``.

Development configuration
-------------------------

The checked-in values are development defaults and must be reviewed before
enabling motion:

* ``robots/franka/config/realworld_physical_agent_eval.yaml`` contains the
  robot IP and Ray placement.
* ``robots/franka/config/env/realworld_physical_agent_franka.yaml`` contains
  camera serials, camera names, reset/target pose, action scale, and workspace
  safety bounds.
* ``robots/franka/controller_config.yaml`` contains primitive timeouts and
  tolerances.
* ``robots/franka/calibration/hand_eye_calibration.json`` contains the hand-eye
  calibration used by perception tools.

Replace ``ROBOT_IP``, ``CAMERA_SERIAL_WRIST``, and
``CAMERA_SERIAL_EXTERNAL`` in those files. Do not reuse poses, bounds, serials,
or calibration from another workspace.

The normal RPent command uses these files directly. ``--rlinf-config-name``,
``--rlinf-override``, and ``--controller-config`` remain available as
development escape hatches, but no separate RLinf configuration workflow is
needed.

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

Task ``0`` exercises conservative analytic motion and gripper primitives:

.. code-block:: bash

	uv run --extra franka rpent --env franka --task-id 0 \
	  --planner api --model anthropic:claude-sonnet-4-5

RPent starts ``robots/franka/env_server.py`` with the current interpreter,
composes the checked-in local config, connects to Ray, waits for ``healthz``,
and records the initial state as step ``0``.

VLA task
--------

Task ``1`` exposes ``vla_grasp``. Single Franka currently requires a compatible
external VLA service trained with the same observation layout, action layout,
checkpoint, and normalization statistics:

.. code-block:: bash

	uv run --extra franka rpent --env franka --task-id 1 \
	  --vla-endpoint http://VLA_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

Without ``--vla-endpoint``, analytic motion and gripper tools remain available,
but ``vla_grasp`` raises a runtime error. The LIBERO VLA server is not
compatible with physical Franka.

External environment server
---------------------------

To attach RPent to an already-running Franka environment service:

.. code-block:: bash

	uv run --extra franka rpent --env franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

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
