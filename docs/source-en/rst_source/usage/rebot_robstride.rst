reBot DevArm (RobStride)
========================

RPent can control the seven-motor reBot DevArm B601-RS directly through
``motorbridge`` and a classic SocketCAN adapter. The hardware server owns the
CAN interface exclusively and exposes guarded joint, gripper, stop, and state
operations to every RPent cerebrum.

Install
-------

Install the optional hardware dependencies in the active RPent environment:

.. code-block:: bash

   uv sync --active --inexact --extra rebot-robstride

Bring up the CAN interface before starting RPent. The server checks this state
but never invokes ``sudo`` itself:

.. code-block:: bash

   sudo ip link set can0 up type can bitrate 1000000
   ip -details link show can0

Do not run another motorbridge gateway or controller against ``can0`` while the
RPent environment server is active. The server must be the sole owner of host
ID ``0xFD``.

Passive hardware check
----------------------

Verify all seven motors before enabling torque:

.. code-block:: bash

   python scripts/check_rebot_robstride.py

The check only reads RobStride ``mechPos`` (``0x7019``) and ``mechVel``
(``0x701A``) parameters. It never clears faults, changes modes, enables motors,
or sends a target.

Configuration
-------------

Copy the example and review every raw-motor limit and gain for your arm:

.. code-block:: bash

   cp robots/rebot_robstride/config/rebot_robstride.example.yaml rebot.yaml

The default B601-RS mapping is:

.. list-table::
   :header-rows: 1

   * - Motor IDs
     - Model
     - Role
   * - 1–3
     - ``rs-06``
     - shoulder/base arm joints
   * - 4–6
     - ``rs-00``
     - wrist arm joints
   * - 7
     - ``rs-00``
     - gripper

Gripper ``open_position`` and ``closed_position`` intentionally default to
null. Calibrate the installed gripper before setting them. RPent refuses every
gripper command while either endpoint is missing.

Run
---

Start a physical-agent run with an explicit natural-language instruction:

.. code-block:: bash

   python rpent/cli/main.py \
     --env rebot_robstride \
     --env-config rebot.yaml \
     --instruction "Read the current joint state and wait" \
     --cerebrum api \
     --model anthropic:claude-opus-4-8

The arm starts disabled. An agent must call ``get_robot_state`` and then
``enable_arm`` before any motion. ``enable_arm`` clears motor errors, selects
MIT mode, enables the controller, and immediately holds the observed pose to
avoid a startup jump.

Available robot tools
---------------------

- ``get_robot_state`` — fresh positions and velocities for all motors.
- ``enable_arm`` — explicit fault-clear, mode selection, enable, and pose hold.
- ``move_joints`` — six raw-motor-radian targets through a bounded minimum-jerk
  trajectory with final read-back evidence.
- ``set_gripper`` / ``open_gripper`` / ``close_gripper`` — calibrated,
  normalized gripper control.
- ``stop_motion`` — torque-holding software stop; rejects later motion.
- ``reset_stop`` — clears only the software-stop latch; never enables motors.
- ``emergency_stop`` — disables all motors immediately.

Safety limits
-------------

``move_joints`` rejects non-finite values, targets outside the configured joint
limits, and malformed six-joint vectors. Requested durations are stretched when
needed to respect each joint's configured velocity cap. Every result includes
the target, final hardware read-back, maximum error, and ``reached`` flag.

``emergency_stop`` removes torque; an unsupported arm may fall under gravity.
Support the arm and keep the physical emergency stop accessible during initial
hardware validation.

The first implementation provides guarded joint-space control. It does not
claim collision avoidance, Cartesian planning, or perception-based grasping.
