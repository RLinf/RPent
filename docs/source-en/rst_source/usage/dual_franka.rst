Dual Franka
===========

RPent can control a two-node dual-Franka setup through an RLinf
``RealWorldEnv`` worker. This development branch keeps its runtime configuration
in RPent and installs the required RLinf/OpenPI branch through the ``franka``
extra. A separate RLinf checkout, virtual environment, or installation step is
not required.

Install
-------

From the RPent repository root:

.. code-block:: bash

	uv sync --extra franka

This installs the custom RLinf Franka branch and ``rlinf-openpi`` into
``.venv`` according to ``pyproject.toml``.

Development configuration
-------------------------

Review and edit the checked-in development defaults before enabling motion:

* ``robots/dual_franka/robot_config.yaml`` contains only the machine identity
	(both robot IPs, camera serials/types, gripper connections), workspace
	geometry (target poses and safety limits), and perception localization
	bounds + base-frame transform.
* ``robots/dual_franka/config.py`` holds the developer defaults (primitive
	control, perception tuning, episode length, node placement).
* ``robots/dual_franka/calibration/`` contains the ``easy_handeye`` hand-eye
  calibration (see the Calibration section below); camera intrinsics are
  captured at runtime.

The checked-in robot config intentionally preserves the current lab IPs,
serials, and gripper device paths for local testing. Review and replace them for
any other setup. Do not reuse reset poses, bounds, or calibration from another
workspace.

RPent translates this robot-focused schema into the internal two-node RLinf
cluster and environment objects. To use a different file, pass
``--robot-config /path/to/robot_config.yaml``. No Hydra or RLinf configuration
workflow is exposed to users.

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
``~/.ros/easy_handeye/hand_eye_calibration.json``). The RPent-specific
localization bounds and base-frame transform live in ``robot_config.yaml``'s
``perception`` section (``easy_handeye`` does not produce them). Camera
intrinsics are captured at runtime in ``camera_meta.json``.

Use a local RLinf checkout
--------------------------

For development, make the same local checkout available on both controller
nodes before starting Ray. The checkout may use different absolute paths on the
two nodes, but each node's ``PYTHONPATH`` must point to its local copy.

Node ``0``:

.. code-block:: bash

	export RLINF_REPO_PATH=/path/to/RLinf
	export PYTHONPATH=$RLINF_REPO_PATH:${PYTHONPATH:-}
	export RLINF_NODE_RANK=0
	ray stop --force
	ray start --head --port=6379 --node-ip-address=HEAD_IP

Node ``1``:

.. code-block:: bash

	export PYTHONPATH=/path/to/RLinf:${PYTHONPATH:-}
	export RLINF_NODE_RANK=1
	ray stop --force
	ray start --address=HEAD_IP:6379 --node-ip-address=WORKER_IP

Run RPent on node ``0`` with the local checkout:

.. code-block:: bash

	uv run --extra franka rpent --env dual_franka --task-id 0 \
	  --planner api --model anthropic:claude-sonnet-4-5

The auto-started ``env_server.py`` and ``vla_server.py`` read
``RLINF_REPO_PATH`` and prepend it to ``sys.path`` at startup. Exporting
``PYTHONPATH`` before ``ray start`` makes remote Ray workers import the matching
checkout. Restart Ray on every node whenever the source path changes.

Verify each node with:

.. code-block:: bash

	PYTHONPATH=/path/to/RLinf:$PYTHONPATH \
	  uv run --extra franka python -c \
	  'import rlinf; print(rlinf.__file__)'

Unset ``RLINF_REPO_PATH`` to return the local subprocesses to the version
installed in ``.venv``.

Start the two-node Ray cluster
------------------------------

Set ``RLINF_NODE_RANK`` before starting Ray on each controller node. Run RPent
only on node ``0``.

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

	uv run --extra franka rpent --env dual_franka --task-id 0 \
	  --planner api --model anthropic:claude-sonnet-4-5

RPent starts ``robots/dual_franka/env_server.py`` with the current interpreter,
loads the RPent robot config, generates the internal RLinf adapter config,
connects to Ray, waits for ``healthz``, and records the initial state as step
``0``.

VLA task
--------

Task ``1`` exposes ``vla_grasp`` and can start the dual-Franka VLA server
locally. ``PI05_CHECKPOINT_PATH`` points to the SFT checkpoint, while
``DUAL_FRANKA_REPO_ID`` is the dataset ID used to locate matching normalization
statistics:

.. code-block:: bash

	export PI05_CHECKPOINT_PATH=/path/to/checkpoints/global_step_N
	export DUAL_FRANKA_REPO_ID=org/dual-franka-tcp-rot6d

	uv run --extra franka rpent --env dual_franka --task-id 1 \
	  --cuda-device 0 \
	  --planner api --model anthropic:claude-sonnet-4-5

The checkpoint must contain:

.. code-block:: text

	actor/model_state_dict/full_weights.pt
	<DUAL_FRANKA_REPO_ID>/norm_stats.json

When ``--vla-endpoint`` is absent, RPent starts
``robots/dual_franka/vla_server.py`` and loads
``pi05_dualfranka_tcp_rot6d`` once. Task ``0`` does not load the VLA.

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

	uv run --extra franka rpent --env dual_franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

Tools and artifacts
-------------------

The extension exposes ``view_env_state``, ``view_camera_meta``, ``move_delta``,
``rotate_delta``, ``open_gripper``, ``close_gripper``, and ``vla_grasp``. Each
analytic motion selects exactly one arm, ``left`` or ``right``. Mutating tools
capture per-arm state and synchronized left-wrist, base, and right-wrist images
in RPent's central ``EnvState``.

Safety
------

Keep operators at both emergency stops. Validate task ``0`` with very small
single-arm motions before attempting a grasp. Stop when camera/state results
disagree, when the requested motion is not reached, or when any calibration is
uncertain.
