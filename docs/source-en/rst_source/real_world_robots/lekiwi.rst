LeKiwi
======

LeKiwi is a three-wheeled omnidirectional mobile base: three Feetech STS3215
wheel servos on one serial bus and a front USB camera. RPent drives it through
RLinf's LeKiwi real-world environment. The planner asks for one bounded body
velocity ``[x, y, theta]`` in m/s, m/s, and deg/s, and reads back the measured
wheel velocity and a synchronized camera frame after every command.

Two computers take part. The robot's onboard computer, usually a Raspberry Pi,
owns the servo bus and the camera and runs the environment server. The agent
machine, often a laptop on the same network, runs RPent and the planner. The
LeKiwi environment is external: a Dashboard Session attaches to the address
you give it and never spawns an environment of its own.

This page goes from a working robot to a running task in four steps: prepare
the robot computer, describe the robot in one YAML file, start the environment
server, and attach from the agent machine. The hardware checks that come
before that — motor IDs, wheel test, camera check — are the
`RLinf LeKiwi guide
<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/lekiwi.html>`_.

What you need
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Requirement
   * - Base
     - LeKiwi mobile base with three Feetech STS3215 wheel servos, its servo
       controller, and the kit's power supply.
   * - Camera
     - UVC webcam mounted on the front, connected to the robot computer.
   * - Robot computer
     - Linux computer with a USB port for the servo controller and enough
       memory to run the environment server. No GPU is needed.
   * - Agent machine
     - Any computer that can reach the robot computer over the network. It
       needs no hardware of its own.

How the pieces fit
------------------

The environment server owns the hardware. It builds an RLinf
``RealWorldEnv`` in its own process, opens the servo bus and the camera
directly, and serves the ``env.*`` RPC methods. There is no Ray cluster on the
robot computer: on a small board, a scheduler cluster would use more memory
than the robot has.

A ``drive`` call runs for the requested duration, stepping the wheels at the
configured control rate, and stops them before it returns, so the base moves
only while a call is executing. An interrupted run, a crashed client, or a
dropped network cannot leave the base rolling. Dashboard teleoperation uses
``hold`` instead: each button press sets a velocity that the server steps
until it expires or a stop arrives. The Dashboard refreshes it while the
button is held, so it lapses one second after the last refresh; the server
caps any hold at two seconds.

The base has no success signal of its own. The planner decides success from
the task instruction and the evidence in the step records.

Prepare the robot computer
--------------------------

Use Linux and Python 3.11. Install RPent with the ``lekiwi`` extra, which
pulls in RLinf:

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install --torch-backend cpu -e '.[lekiwi]'
   uv pip check

Released RLinf versions do not include the LeKiwi environment layer yet, so
the extra pins the revision that does. ``--torch-backend cpu`` keeps the
PyTorch wheels CPU-only, which is all the base needs.

Prepare the agent machine
-------------------------

On the agent machine, install RPent the same way:

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e '.[lekiwi]'
   uv pip check

The agent machine needs only network reachability to the robot computer, on
the port you start the server with.

Write the LeKiwi YAML
---------------------

Run the detection script on the robot computer to write a LeKiwi YAML with the
detected port and camera identifiers:

.. code-block:: bash

   rpent-lekiwi-detect --output /path/to/lekiwi.yaml

The script never commands motion. It pings the serial bus at every baud rate,
opens each camera once to confirm it produces a frame, and prints what it
found:

.. code-block:: text

   == LeKiwi detection ==
   serial ports scanned: ['/dev/ttyACM0']
     /dev/ttyACM0: motor ids [7, 8, 9]
   wheel bus: /dev/ttyACM0
   camera usb-USB_Camera_USB_Camera_2505280051-video-index0: ok at 640x480
   wrote /path/to/lekiwi.yaml
   review before the first run:
     - robot.wheel_radius / robot.base_radius: measure with a ruler (m)
     - drive.max_linear_velocity / max_angular_velocity: policy; start low
     - drive.frame_size: camera output size; raise from 128x128 if needed
     - drive.step_frequency: control rate; 10 Hz suits the STS3215 bus
     - drive.camera_max_age: frame reuse window; 0.05 tracks a 15 fps camera

Pass ``--skip-motors`` or ``--skip-cameras`` to probe only one side. The file
it writes has the shape of ``robots/lekiwi/config/example.yaml``:

.. code-block:: yaml

   robot:
     port: /dev/ttyACM0
     wheel_radius: 0.05
     base_radius: 0.125
     max_raw: 3000

   cameras:
     devices:
       front:
         serial: "<front-camera-serial>"
         type: webcam
         main: true

   drive:
     step_frequency: 10.0
     camera_max_age: 0.05
     max_linear_velocity: 0.3
     max_angular_velocity: 60.0
     frame_size: [128, 128]

Two kinds of values need a human. The geometry — ``wheel_radius`` and
``base_radius`` in metres — comes from a ruler, and ``max_raw`` (default 3000)
is the servo's raw-speed cap. The ``drive`` limits are your velocity envelope:
they are a policy choice, not something a probe can discover, so start at the
defaults and raise them gradually. The tasks below bound every command
further, and the environment clips each one to these limits.

``camera_max_age`` is how old, in seconds, the newest captured frame may be for
a step to reuse it. RLinf's default of 0.5 s yields at most two distinct frames
a second, however fast the camera runs; ``0.05`` follows the 15 fps camera
driver. The value must stay positive, so ``0`` is rejected, and raising it
trades frame freshness for fewer camera reads.

Edit the generated file where it does not fit the robot, for example to
rename the front camera entry or to raise ``frame_size`` for a resolution the
camera supports natively.

Copy the YAML to the agent machine. Both computers read it with
``--robot-config``: the robot computer needs it to open the hardware, and the
agent machine to know the velocity limits and camera layout.

Tasks
-----

Tasks are registered in ``robots/lekiwi/tasks.py`` and chosen with
``--task-id``.

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - ID
     - Task
     - What the planner does
   * - 0
     - ``drive_smoke_test``
     - One conservative forward drive, one rotation, and an inspection of the
       camera view after each, at or below 0.1 m/s and 30 deg/s for at most
       2 s per drive.
   * - 1
     - ``room_survey``
     - Drives bounded segments around the room and reports what the camera
       sees, at or below 0.2 m/s and 45 deg/s for at most 3 s per drive.

Start with task 0 on a cleared floor. It exercises the whole path — server,
serial bus, camera, planner — without moving the base far.

Start the environment server
----------------------------

On the robot computer, bind the server to the interface the agent machine can
reach:

.. code-block:: bash

   rpent-lekiwi-server \
     --robot-config /path/to/lekiwi.yaml \
     --transport http --host 0.0.0.0 --port 8120

It prints the address to attach to:

.. code-block:: text

   RPC server listening on http://0.0.0.0:8120

Check the resolved configuration before touching hardware with
``--print-config``, which prints the RLinf config as YAML and exits. Stop the
server with Ctrl+C; it stops the wheels and drops torque on shutdown.

Attach from the agent machine
-----------------------------

Run a task against the server you just started:

.. code-block:: bash

   rpent --robot lekiwi --task-id 0 \
     --env-endpoint http://ROBOT_HOST:8120 \
     --robot-config /path/to/lekiwi.yaml \
     --planner api --model MODEL_ID

``--env-endpoint`` is what makes this a remote run. Omit it and RPent spawns
an environment server on the agent machine instead, which only works when the
servo bus is attached there.

For hands-on control, start a Dashboard Session with no planner and drive
from the browser:

.. code-block:: bash

   rpent --robot lekiwi --dashboard --planner manual \
     --env-endpoint http://ROBOT_HOST:8120 \
     --robot-config /path/to/lekiwi.yaml

The Dashboard requires ``--env-endpoint`` for LeKiwi because the environment
is external. Manual mode drives the base from the Teleoperation panel: hold a
direction button to move and release it to stop, and the velocity lapses one
second after the last refresh, so a closed browser leaves the robot at rest.
``Record video`` and ``Record actions`` write each driving stretch as
``teleop_NNN.mp4`` and ``teleop_actions.jsonl``; ``End run`` finishes the
TaskRun. See :doc:`../guides/dashboard` for the rest of the Dashboard.

Tools and artifacts
-------------------

The planner has three tools: ``drive(velocity, duration_s)``, ``stop()``, and
the read-only ``view_env_state(step)``. ``duration_s`` is capped at 10 s per
call, and requested velocities are clipped to the configured limits.

Every run writes to ``logs/<timestamp>_lekiwi_t<task_id>/`` unless you pass
``--output-dir``. Each step records the synchronized front-camera frame as
``front.png`` and its metadata as ``camera_meta.json``; ``view_env_state``
returns that record together with the image. A Dashboard Session also keeps
``teleop_NNN.mp4`` and ``teleop_actions.jsonl`` for the stretches driven by
hand.
