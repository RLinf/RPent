Single-Arm Franka
=================

.. Product image: https://store.clearpathrobotics.com/products/franka-research-3

.. figure:: https://cdn.shopify.com/s/files/1/1750/5061/products/FR3_image3_x700.png?v=1663341441
   :alt: Full view of a Franka Research 3 arm and gripper
   :figclass: rpent-robot-figure
   :align: center

   Franka arm for real-world experiments.

RPent can control one physical Franka arm through an RLinf ``RealWorldEnv``
worker.

Install
-------

Install libfranka 0.19.0 first by following the `official quick-install guide
<https://docs.ros.org/en/humble/p/libfranka/__README.html#quick-install>`_.

Clone RPent and install its Python dependencies. If you already have the
checkout, enter it and run ``uv sync``:

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   uv sync --extra franka

This installs RLinf ``release/v0.4``, ``rpent-openpi``, and Franka camera,
gripper, and teleoperation dependencies into ``.venv``. The included
``franky-control`` wheel bundles libfranka 0.19.0.

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

Paths may be absolute, ``~``-prefixed, or relative; relative paths resolve
against the working directory RPent is launched from.

Development Configuration
-------------------------

The checked-in values are development defaults and must be reviewed before
enabling motion:

* ``robots/franka/config/example.yaml`` contains the machine identity (robot IP,
	camera serials, gripper), workspace geometry (target/reset poses and safety
	limits), and the easy_handeye YAML mapping (see Calibration). It sets
	``backend: franky`` and ``realtime_config: ignore``. On a PREEMPT_RT kernel,
	change ``realtime_config`` to ``enforce`` to refuse startup when real-time
	guarantees are unavailable.

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

Run a Smoke Test
----------------

Configure the planner and model service using :doc:`../guides/configure_planner` first.

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

VLA Grasp Demo
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

Tools and Artifacts
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

.. _franka-flash:

Franka Flash plans (task cards)
--------------------------------

Single and dual Franka support ``--planner flash`` using a reviewed version-1
JSON plan. A plan records primitive order and motion intent; Molmo selects a
pixel in a fresh camera image for each translation. Existing calibrated depth
projection converts it into robot coordinates. No planning model is called.

The stored translation offset is the demonstrated **actual TCP endpoint**
minus the demonstrated anchor position. Replay adds this offset to the newly
localized anchor and subtracts the current TCP position to obtain the move.
Action order is fixed; interpolation remains the existing controller's responsibility.

First configure the robot and ``perception.calibration`` using the setup above,
start a Molmo service, and record a successful run. Review ``states.json`` and
write ``annotations.json`` keyed by the source step number::

   {
     "1": {"intent": "approach the cup rim", "phrase": "visible cup rim", "camera": "third_person"},
     "2": {"intent": "align the gripper", "rotation_mode": "relative"}
   }

Every translation needs a semantic anchor; every rotation needs an intent and
``relative`` or ``fixed`` rotation mode. Single Franka uses ``third_person`` or
``wrist``; dual Franka uses ``base``, ``d455``, ``left_wrist`` or ``right_wrist``.
The selected view must have depth and valid calibration. Keep each recorded
translation within 0.20 m and each rotation within 0.35 rad.

Generate a plan offline from the reviewed recording (Molmo must be reachable)::

   python -m robots.franka.flash.generate \
     --robot franka --task franka_t0 \
     --robot-config /path/to/robot.yaml \
     --run-dir /path/to/successful-run --annotations annotations.json \
     --molmo-endpoint http://localhost:9000 --destination plan.json

Confirm the source success when prompted. The generator rejects unsupported
commands and does not overwrite an existing destination. It stores hashes of
the robot configuration and calibration files. New recordings save
``recording_fingerprint.json``; generation rejects missing fingerprints or
configuration/calibration changes since recording before localizing anchors.
Re-record after such changes; older recordings without provenance cannot be used.
Replay also checks the plan against the current files. Known observation/verdict
records and an initial confirmed scene reset are excluded from the plan; resets
after task actions are rejected to prevent combining attempts. Dual Franka uses ``--robot dual_franka --task dual_franka_t0``.

Replay from a dedicated operator terminal::

   rpent --robot franka --planner flash --task-id 0 \
     --robot-config /path/to/robot.yaml --flash-plan plan.json \
     --molmo-endpoint http://localhost:9000

For dual Franka, use ``--robot dual_franka`` and the corresponding plan/config.
Configure Env/VLA endpoints as in the robot setup guide; single-arm plans with
``vla_grasp`` require ``--vla-endpoint``. Replay asks before driver initialization
(which may reset the robot), before execution, and for the final task verdict.
Do not use ``--interactive``, ``--dashboard`` or ``--explore`` with this mode.

Invalid pixels/depth, workspace violations, excessive motions, and primitive
errors stop replay. Left-arm workspace bounds are checked in the left-base
frame. A successful RPC does not certify task success: the final operator
verdict determines the result. ``flash_outcome.json`` and
``flash_recipe.jsonl`` record the outcome and issued actions.

Only reviewed v1 plans are supported. Experimental supervised v2 plans,
stage splicing, reference-image selection and historical-point reuse are not
part of this interface. These commands require robot-specific installation and
operator validation; offline unit tests do not establish real-robot success.

.. _optional-agent-grounding-fallback:

Large model agent assistance for point selection in Flash mode
--------------------------------------------------------------

In Flash mode, an agent can retry point selection when Molmo cannot find the
target, selects a pixel with invalid coordinates or depth, or its request fails.

When enabled, RPent first captures a fresh observation, then asks the agent to
retry once on the same object part. If that attempt also fails, replay stops
without executing that motion. The next translation starts with Molmo again.

This feature is disabled by default and only applies to live replay in Flash
mode, not offline plan generation.

To enable it, add ``--grounding-agent-model`` to the replay command:

.. code-block:: bash

   rpent --robot franka --planner flash --task-id 0 \
     --robot-config /path/to/robot.yaml --flash-plan plan.json \
     --molmo-endpoint http://localhost:9000 \
     --grounding-agent-model codex:YOUR_MODEL

You can use Astra or another large model already supported by RPent to improve
the point-selection success rate. Replace ``YOUR_MODEL`` with the model name;
the model must support images and structured output.
For dual Franka, use ``--robot dual_franka`` with its plan and robot configuration.

See :doc:`../guides/configure_planner` for model login and API setup.
Use ``codex:model`` for Codex CLI or ``provider:model`` for an API model.
Add ``--grounding-agent-base-url`` if you need to override the model endpoint.

Codex reuses your model/provider settings and file-based login to run the
fallback agent. It does not load your MCP servers or plugins. If an enabled MCP
server is detected, the request stops before sending the image. If your login
is stored only in the OS keyring, use file-based login or set ``CODEX_API_KEY``.

Each attempt is saved in the step's ``flash_grounding.json``. Agent requests
may incur model costs. The current request timeout is 90 seconds; tokens from
agent point-selection requests are not included in the Flash planner's counts.

Stop the Run
------------

Press Ctrl+C in the terminal to request RPent shutdown; use the hardware emergency stop for an emergency. Check the arm and gripper before shutting down services through the controller’s shutdown procedure.
