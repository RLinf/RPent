LeKiwi
======

LeKiwi is a three-wheeled omnidirectional mobile base: three Feetech STS3215
wheel servos on one serial bus and a front USB camera. RPent drives it through
RLinf's public robot and mobile-base interfaces. The planner requests a body
velocity for a duration, or a relative rotation angle. The service stops after
each action, then records the state and front-camera view.

Two computers take part. The robot's onboard computer, usually a Raspberry Pi,
owns the servo bus and the camera and runs the environment server. The agent
machine, often a laptop on the same network, runs RPent and the planner.

Before starting, follow the `RLinf LeKiwi guide
<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/lekiwi.html>`_
to check the motor IDs, wheels and camera. Once the hardware works, follow
this page to install RPent, generate the robot configuration, start the
environment server and run a task.

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

Prepare the robot computer
--------------------------

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install --torch-backend cpu -e '.[lekiwi]'
   uv pip check

Prepare the agent machine
-------------------------

To run the agent or recording command, install the ``lekiwi`` extra:

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install --torch-backend cpu -e '.[lekiwi]'
   uv pip check

The agent machine must be able to reach the robot computer over the network.

Write the LeKiwi YAML
---------------------

Run the detection script on the robot computer to write a LeKiwi YAML with the
detected port and camera identifiers:

.. code-block:: bash

   python -m robots.lekiwi.detect --output /path/to/lekiwi.yaml

The script never commands motion. It pings the serial bus at every baud rate,
opens each camera once to confirm it produces a frame, and prints what it
found:

.. code-block:: text

   == LeKiwi detection ==
   serial ports scanned: ['/dev/ttyACM0']
     /dev/ttyACM0: motor ids [7, 8, 9]
   wheel bus: /dev/ttyACM0
   camera /dev/v4l/by-id/usb-USB_Camera_USB_Camera_2505280051-video-index0: ok at 640x480
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
     serial_port: /dev/ttyACM0
     wheel_radius: 0.05
     base_radius: 0.125
     max_raw: 3000

   cameras:
     devices:
       front:
         serial: "<front-camera-serial>"
         type: uvc
         main: true

   drive:
     step_frequency: 10.0
     camera_max_age: 0.05
     max_linear_velocity: 0.3
     max_angular_velocity: 60.0
     frame_size: [128, 128]

Measure and set ``wheel_radius`` (wheel radius) and ``base_radius``
(distance from the base centre to a wheel), both in metres.

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
       camera view after each. Drive at most 2 s at 0.1 m/s and 30 deg/s;
       rotate at most 30 degrees per call.
   * - 1
     - ``room_survey``
     - Drives bounded segments around the room and reports what the camera
       sees. Drive at most 3 s at 0.2 m/s and 45 deg/s; rotate at most
       90 degrees per call.

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

Attach from the agent machine
-----------------------------

Run a task against the server you just started:

.. code-block:: bash

   rpent --robot lekiwi --task-id 0 \
     --env-endpoint http://ROBOT_HOST:8120 \
     --robot-config /path/to/lekiwi.yaml \
     --planner api --model MODEL_ID

RLinf teleoperation and recording
---------------------------------

Use RLinf's keyboard teleoperation from the RLinf source directory.
Stop the normal RPent Env first so that only RLinf's receiver opens the motor
port. To record video at the same time, start RPent with ``--camera-only``;
this mode opens only the YAML's ``main: true`` camera and cannot move the base.

On the robot computer, run these commands in separate terminals:

.. code-block:: bash

   # Terminal 1, from the RLinf checkout: own the motor bus.
   python -m toolkits.realworld_check.test_lekiwi_env teleop \
     --listen --port /dev/ttyACM0

   # Terminal 2, from the RPent checkout: own only the main camera.
   rpent-lekiwi-server --camera-only \
     --robot-config /path/to/lekiwi.yaml \
     --transport http --host 0.0.0.0 --port 8120

On the keyboard computer, from the RLinf checkout:

.. code-block:: bash

   python -m toolkits.realworld_check.test_lekiwi_env teleop --remote ROBOT_HOST

For keys and speed options, see the `RLinf keyboard teleoperation guide
<https://github.com/RLinf/RLinf/blob/main/docs/source-en/rst_source/examples/embodied/lekiwi.rst>`_.

Independent recording
---------------------

With either Env mode running, execute this on any computer that can reach it:

.. code-block:: bash

   python -m robots.lekiwi.record \
     --env-endpoint http://ROBOT_HOST:8120 \
     --recording-dir /path/to/recordings

Recording starts immediately and saves files on the computer running the
command, including stationary frames. Ctrl-C or SIGTERM finishes writing the
video and exits. Starting or stopping recording does not affect robot control.

Defaults are 600 seconds and up to 10 frames/s. Adjust them with ``--duration``
(maximum 3600 seconds) and ``--fps`` (maximum 60).

Tools
-----

The planner has ``drive``, ``rotate``, ``stop`` and the read-only
``view_env_state(step)``. Examples of motion tool arguments:

.. code-block:: python

   drive(velocity=[0.1, 0.0, 0.0], duration_s=2.0)  # Forward for 2 seconds.
   drive(velocity=[0.0, 0.1, 0.0], duration_s=1.0)  # Strafe left for 1 second.
   rotate(angle_deg=90)                            # Turn left 90 degrees.
