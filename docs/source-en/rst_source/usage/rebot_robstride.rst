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

The check enables RobStride active fault reporting, reads ``mechPos``
(``0x7019``), and inspects status plus fault/warning reports. It never clears
faults, selects a control mode, enables torque, or sends a target. On exit it
still issues ``disable_all`` before releasing SocketCAN. Runtime velocity
feedback is estimated from timestamped position samples because ``mechVel``
scaling is not consistent across the tested RobStride firmware variants.

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

Safety-critical overrides have hard ceilings: control is 10–200 Hz, feedback
is at least 5 Hz, each parameter-read timeout is at most 100 ms, motion is at
most 60 seconds, and settlement is at most 5 seconds. The nominal
motion-plus-settlement budget is at most 65 seconds. All initial/final feedback
overhead counts against a separate 68-second server deadline, which leaves
seven seconds before the 75-second motion RPC timeout for fail-closed disable
and response delivery. Heartbeat timeout is 0.25–5 seconds, joint/gripper
velocity is at most 1 rad/s, and MIT gains are capped at ``kp=200`` / ``kd=20``.
Configuration that exceeds these bounds is rejected before opening SocketCAN.

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
``enable_arm`` before any motion. ``enable_arm`` first validates raw joint
limits and near-zero startup velocity, clears faults, verifies the resulting
operation-status replies and cached detailed reports, selects MIT mode, enables
only the six arm motors, and immediately holds the observed pose. Repeated
``enable_arm`` calls are idempotent and do not discard an enabled gripper's
state. The gripper remains disabled until a calibrated gripper command
explicitly enables it.

Available robot tools
---------------------

- ``get_robot_state`` — fresh positions, estimated velocities, latest available
  operation-status flags, and actively reported raw fault/warning cache for all
  motors.
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
limits, malformed six-joint vectors, and durations above the configured hard
maximum. Requested durations are stretched by the 1.875 peak derivative of the
minimum-jerk profile so the actual setpoint velocity respects each configured
cap. During motion the driver sends a hold or waypoint to every physically
enabled motor before feedback, including motors in the subsystem that is not
moving. Those commands produce operation-status replies; active reporting
updates the detailed fault/warning cache. The driver aborts and disables all
motors on missing/nonzero operation status, a nonzero raw fault/warning report,
transport error, excessive tracking error, excessive measured velocity, or the
68-second server deadline. ``motorbridge`` 0.4.9 does not expose a timestamp for
its detailed type-21 fault-report cache, so that raw cache is not described as a
synchronously queried sample. Completion requires consecutive
position-and-velocity settlement samples. Gripper results use the same
``target``/``final``/``max_error``/``reached`` evidence contract.

``robot.stop_motion``, ``robot.emergency_stop``, heartbeats, and ``shutdown``
use a priority RPC path that bypasses the serialized motion-command lock.
Every admitted operation captures a monotonic stop generation; a stop
invalidates that generation permanently, so a later ``reset_stop`` cannot
resurrect an older enable or trajectory. Feedback and multi-joint command
batches release the I/O lock between motor operations, so an emergency stop
waits behind at most one in-flight motorbridge call before disable begins. A
runtime worker sends periodic heartbeats; if the agent process disappears while
torque is enabled, the server calls emergency stop after
``heartbeat_timeout_s``.
Soft stop holds both the arm and an enabled gripper at freshly measured
positions. If ``disable_all`` itself fails, state reports ``disable_failed=true``
and ``reset_stop`` remains blocked until a later emergency-stop retry confirms
that torque was removed.

The pickle-framed hardware RPC server accepts loopback binds only. Do not expose
it through a TCP proxy or bind it to a LAN address.

``emergency_stop`` removes torque; an unsupported arm may fall under gravity.
Support the arm and keep the physical emergency stop accessible during initial
hardware validation.

Normal shutdown always calls ``disable_all``; there is no unattended ``hold``
shutdown mode. If disable fails, ``close`` does not close the controller or erase
the physically uncertain enabled/``disable_failed`` state, allowing a later
emergency-stop or close retry. The heartbeat protects loss of the agent process
while the hardware server remains alive. A hard crash or ``SIGKILL`` of the
hardware server cannot be proven fail-safe because the tested
RobStride/motorbridge API does not expose a confirmed hardware watchdog. Never
operate unattended, and keep the physical emergency stop reachable.

The first implementation provides guarded joint-space control. It does not
claim collision avoidance, Cartesian planning, or perception-based grasping.
