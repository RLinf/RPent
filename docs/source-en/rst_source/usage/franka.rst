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

* ``robots/franka/robot_config.yaml`` contains only the machine identity (robot
	IP, camera serials, gripper) and workspace geometry (target/reset poses and
	safety limits).
* ``robots/franka/config.py`` holds the developer defaults (primitive-control
	knobs, action scale, tolerances, camera processing) applied over RLinf's
	own dataclass defaults.
* ``robots/franka/calibration/hand_eye_calibration.json`` contains the hand-eye
  calibration used by perception tools.

Machine identity (robot IP, camera serials, gripper connection) is not
committed in ``robot_config.yaml``; the tokens ``ROBOT_IP``,
``CAMERA_SERIAL_WRIST``, ``CAMERA_SERIAL_EXTERNAL``, and
``GRIPPER_CONNECTION`` there name the environment variables that supply them.
They are resolved at run time, in order: command-line flag (``--robot-ip``,
``--camera-serial-wrist``, ``--camera-serial-external``,
``--gripper-connection``) > environment variable > a literal value placed in the
config. Do not reuse poses, bounds, serials, or calibration from another
workspace.

RPent translates this robot-focused schema into the internal RLinf cluster and
environment objects. To use a different file, pass
``--robot-config /path/to/robot_config.yaml``. No Hydra or RLinf configuration
workflow is exposed to users.

Config keys are validated at load time; a typo, missing key, or wrong nesting
fails with the exact path. To inspect the RLinf config RPent derives from the
robot config without touching hardware:

.. code-block:: bash

   python robots/franka/env_server.py --print-config --task-description "grasp"

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
``~/.ros/easy_handeye/hand_eye_calibration.json``). Camera intrinsics are
captured at runtime in ``camera_meta.json``, not in the bundle.

Use a local RLinf checkout
--------------------------

To patch RLinf without reinstalling the ``franka`` extra, prepend the checkout
to ``PYTHONPATH`` before starting Ray and export ``RLINF_REPO_PATH`` so RPent's
environment subprocess imports the same checkout:

.. code-block:: bash

	export RLINF_REPO_PATH=/path/to/RLinf
	export PYTHONPATH=$RLINF_REPO_PATH:${PYTHONPATH:-}
	export RLINF_NODE_RANK=0
	ray stop --force
	ray start --head

	uv run --extra franka rpent --env franka --task-id 0 \
	  --planner api --model anthropic:claude-sonnet-4-5

``robots/franka/env_server.py`` reads ``RLINF_REPO_PATH`` and prepends it to
``sys.path`` at startup. Exporting ``PYTHONPATH`` before ``ray start`` makes Ray
workers import the same checkout. Restart Ray after changing this path; an
already-running Ray process keeps the environment it captured at startup.

Verify the selected source with:

.. code-block:: bash

	PYTHONPATH=$RLINF_REPO_PATH:$PYTHONPATH \
	  uv run --extra franka python -c \
	  'import rlinf; print(rlinf.__file__)'

Unset ``RLINF_REPO_PATH`` to return to the version installed in ``.venv``.

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
loads the RPent robot config, generates the internal RLinf adapter config,
connects to Ray, waits for ``healthz``, and records the initial state as step
``0``.

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
