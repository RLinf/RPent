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

Stereo mapping and navigation below use a ZED 2i with the deployment below. Run commands from the RPent checkout; keep configuration, recordings and maps in your own directories outside the repository.

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Location
     - Responsibilities
     - Main dependencies
   * - Robot computer
     - The Env owns the camera and base. Local workers compute stereo depth, separate ground from obstacles, and run Nav2 planning and control. Visual localization can also run here on CPU.
     - RPent, companion RLinf, OpenCV, ROS 2 Jazzy, Nav2 and ``rtabmap_util``. Local localization also needs the RTAB-Map native extension; no GPU or ZED SDK is required.
   * - Cloud
     - Generate depth and maps offline. Optionally host visual localization when the robot has insufficient CPU or memory.
     - Offline depth uses CUDA and RAFT-Stereo. Mapping and localization use RTAB-Map; localization also requires RPent's native extension. Online localization can run on CPU.
   * - Operator computer
     - Run the RPent Dashboard and planner, connected to the robot Env.
     - RPent; this machine does not compute local obstacles or relay localization images between robot and cloud.

On the robot, ``rtabmap_util/obstacles_detection`` separates ground from obstacles. A separate CPU localization process queries the existing RTAB-Map database. Install the localization extension on whichever host runs that process; RAFT-Stereo and offline mapping tools can stay on the cloud host. Local localization requires enough CPU and memory for the map alongside the camera, stereo workers and Nav2.

1. Robot installation
------------------------------------------

The companion RLinf camera interface must support ``BaseCamera.get_frame(return_timestamp=True)`` so image and wheel timestamps can be aligned.

The Env uses a separate RPent Python environment. Install RPent and the companion RLinf checkout, or reuse a compatible environment. RPent supports Python 3.10–3.12; this example uses 3.12:

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install --torch-backend cpu -e ".[lekiwi]"
   uv pip install --no-deps -e /path/to/RLinf

ROS 2 is needed for navigation, not teleoperation or image recording. Follow the `ROS 2 Jazzy installation guide <https://docs.ros.org/en/jazzy/Installation.html>`_ and install ``nav2_bringup``, ``nav2_mppi_controller``, ``launch_ros``, ``robot_localization``, ``rtabmap_util``, ``sensor_msgs_py`` and ``rmw_cyclonedds_cpp``.

Install the bridge dependencies and RPent into the Python environment used by ROS:

.. code-block:: bash

   /path/to/ros-python -m pip install numpy scipy opencv-python-headless pillow pyyaml omegaconf
   /path/to/ros-python -m pip install --no-deps -e /path/to/RPent

2. Robot configuration
--------------------------------------------

.. code-block:: bash

   cp robots/lekiwi/config/example.yaml /path/to/robot.yaml

Fill in ``robot.port``, the UVC device identifier ``cameras.devices.front.serial``, the label’s ``zed_serial``, and ``perception.calibration``. Obtain the camera’s factory calibration following the `Stereolabs calibration guide <https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration>`_. Relative paths are resolved from the robot YAML directory.

``geometry`` contains numeric body extents and height, camera position and orientation relative to the chassis center, navigation radius, velocity and acceleration. Axes are x forward, y left, z up; lengths are metres and mount angles are degrees. Include the fixed arm in the body dimensions. The example measured navigation radius is 0.17 m and is used directly without extra padding.

``eye_size`` defaults to ``[1280, 720]``; ``drive`` sets teleoperation limits and frequency. ``perception.stereo`` and ``perception.obstacles`` can be empty. Navigation loads the installed ``nav2_bringup/params/nav2_params.yaml`` with ``nav2_overrides.yaml`` for omni motion, pointclouds and topics, and injects geometry and motion limits from the robot YAML.

3. Mapping and localization installation
------------------------------------------

Offline mapping uses CUDA and RAFT-Stereo; local CPU localization needs only RTAB-Map, OpenCV and the native extension described below. RAFT-Stereo is not required on the robot.

For offline mapping, install RPent on the cloud machine using the Python commands in section 1. Clone the following projects into your chosen paths and follow their upstream dependency and model instructions; no source branch is pinned:

.. code-block:: bash

   git clone https://github.com/introlab/rtabmap.git /path/to/rtabmap
   git clone https://github.com/princeton-vl/RAFT-Stereo.git /path/to/RAFT-Stereo

- `RTAB-Map installation <https://github.com/introlab/rtabmap/wiki/Installation>`_: enable Qt image import tools and GTSAM, and add the official commands to ``PATH``.
- `RAFT-Stereo instructions <https://github.com/princeton-vl/RAFT-Stereo>`_: install a CUDA PyTorch environment, download ``raftstereo-middlebury.pth``, and install RPent in that environment. A domestic PyPI mirror may be used for Python dependencies.

In the Python environment running the localization service, install the optional build dependencies, then compile and install the extension with CMake. This requires a C++17 compiler, Python development headers, and RTAB-Map and OpenCV development libraries. Run these steps on the selected localization host; running the service on the robot requires building and installing the extension there:

.. code-block:: bash

   python -m pip install -e ".[lekiwi-localization]"
   cmake -S robots/lekiwi/rtabmap_extension -B /path/to/lekiwi-extension-build \
     -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
     -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
     -DCMAKE_PREFIX_PATH=/path/to/rtabmap-install \
     -DCMAKE_INSTALL_RPATH=/path/to/rtabmap-install/lib
   cmake --build /path/to/lekiwi-extension-build
   cmake --install /path/to/lekiwi-extension-build \
     --prefix "$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"

The extension runs inside the localization service and accepts NumPy images directly. Calls on one map instance are serialized, and RTAB-Map computation releases the GIL. ``--max-processing-ms`` discards results returned after the deadline; it cannot interrupt native calls. Restart the localization service if a native call hangs or the process crashes. The Pi continues to check localization freshness.

4. Start teleoperation
--------------------------------------------

Start the Env on the robot; it owns the base and camera:

.. code-block:: bash

   python -m robots.lekiwi.env_server --robot-config /path/to/robot.yaml \
     --host 0.0.0.0 --port 8120 --recording-dir /path/to/recordings

``0.0.0.0`` is the local listen address, not the cloud IP. Install RPent on the operator computer and start the Dashboard. Replace the example robot address ``192.168.1.10`` with its actual address, or use ``127.0.0.1`` when running on the robot.

.. code-block:: bash

   rpent --robot lekiwi --dashboard --planner manual \
     --env-endpoint http://192.168.1.10:8120 --robot-config /path/to/robot.yaml

Open the Dashboard URL printed in the terminal and submit ``/rpent-task 0`` to enter manual control. Hold a direction button to drive and release it to stop. Use the existing UI options for video and action recording.

5. Record stereo images during teleoperation
----------------------------------------------------------------------------------------

Use the existing Env interface to record stereo PNG images. In a separate interactive Python session on the operator computer, connect to the running Env and start recording. Replace the endpoint with the robot's actual address:

.. code-block:: python

   from uuid import uuid4
   from robots.lekiwi.env_client import LeKiwiEnvClient
   from rpent.utils.rpc import make_rpc_client

   env = LeKiwiEnvClient(make_rpc_client("http://192.168.1.10:8120"))
   identity = uuid4().hex
   env.recording("start", identity=identity)

Drive through the Dashboard. In the same Python session, call ``env.recording("state", identity=identity)`` to check status, frame count and the output directory. When capture is complete, run:

.. code-block:: python

   env.recording("stop", identity=identity)
   env.close()

Each recording defaults to a maximum of 600 seconds at up to 5 image pairs per second. Change these with Env options ``--capture-seconds`` and ``--recording-fps``. Ending the Dashboard manual run or closing the client does not stop stereo PNG capture; stop it explicitly, let its time limit expire, or close the Env. Existing Dashboard video and action recording is independent of stereo PNG capture.

Copy the complete robot-side ``capture-*`` directory and robot YAML to the cloud machine. The directory contains ``left/``, ``right/``, ``stereo.json`` and ``calibration.conf``. Offline processing reads files and needs no connection to the robot.

6. Offline mapping and map nodes
------------------------------------------------------------------------------

In the cloud RAFT-Stereo Python environment, export rectified images, depth and RTAB-Map configuration:

.. code-block:: bash

   python -m robots.lekiwi.mapping \
     --input /path/to/capture --output /path/to/map-output/rgbd \
     --robot-config /path/to/robot.yaml --raft-repo /path/to/RAFT-Stereo \
     --checkpoint /path/to/raftstereo-middlebury.pth

Build the map with the `official RTAB-Map tools <https://github.com/introlab/rtabmap/wiki/Tools>`_:

.. code-block:: bash

   cd /path/to/map-output
   export QT_QPA_PLATFORM=offscreen

   # Import color images and depth
   rtabmap-dataRecorder -hide rgbd/rtabmap.ini raw.db

   # Estimate camera motion and build the map
   rtabmap-reprocess -default -odom -c rgbd/rtabmap.ini raw.db slam.db

   # Apply planar ground-motion constraints
   rtabmap-reprocess -c rgbd/rtabmap.ini -noimu \
     --Reg/Force3DoF true --RGBD/ForceOdom3DoF true \
     slam.db rtabmap.db

   # Export the navigation occupancy grid
   rtabmap-export --map --opt 2 --output map rtabmap.db

If the map quality is unsatisfactory, consult the `official grid parameter descriptions <https://github.com/introlab/rtabmap/blob/master/corelib/include/rtabmap/core/Parameters.h>`_ to adjust the parameters.

Copy ``map.yaml`` and its referenced map image to the robot. Copy ``rtabmap.db`` from the same mapping run to the localization host: either the robot or an external server. Open the database with the official RTAB-Map viewer, inspect graph nodes and their images, and note the destination node ID:

.. code-block:: bash

   rtabmap-databaseViewer /path/to/map-output/rtabmap.db

Navigation takes an existing database node ID or a saved node label, such as ``door``. The arrival position and heading come directly from the node's optimized pose. No target JSON or separate picker page is required. Labels live in the RTAB-Map database; after changing labels, update the localization database and restart the localization service. When rebuilding a map, update the YAML, image and database together.

Goal resolution uses RTAB-Map's native graph API; Nav2 handles path planning and obstacle avoidance.

The Agent can use ``list_map_nodes`` to find node IDs and existing labels, and ``inspect_map_node`` to check a node's historical image. ``view_map`` shows place labels and the robot pose when navigation is ready. Unnamed nodes are hidden in the default image, but their IDs remain in the returned data. Set ``center_node_id`` to highlight a candidate and display its ID, with its label when available; add ``radius_m`` to crop the surrounding area. The image shows the static map, without live obstacles.

.. _lekiwi-navigation-start:

7. Start the localization service
------------------------------------------

Localization is a standalone CPU service, like SAM3's separately deployed service. The same command runs on the robot or another host. On the chosen host, use a Python environment with RPent and the native localization extension installed:

.. code-block:: bash

   python -m robots.lekiwi.localization_server serve \
     --database /path/to/map-output/rtabmap.db \
     --bind 127.0.0.1 --port 18125

The service loads the database specified by ``--database`` and receives rectified left-camera images, camera parameters, capture timestamps and aligned wheel odometry from the robot's navigation worker. It does not open the camera or motor bus. Its Python environment can be separate from the robot's ROS environment; the localization algorithm and request format are the same for either deployment location.

Keep this service running and start the navigation Env in another terminal as described in the next section. The robot YAML only supplies the service address to the clients:

.. code-block:: yaml

   localization:
     endpoint: http://127.0.0.1:18125

To host localization on another machine, use the same service command there with ``--bind 0.0.0.0`` and set the robot's ``localization.endpoint`` to ``http://SERVER_HOST:18125``. The service needs the database, not the robot YAML. There is no deployment mode or token setting. As with the other RPent RPC services, use a trusted network or an SSH tunnel for remote access; the default listen address is loopback.

Image requests, map queries and goal resolution all use this endpoint. The image worker waits up to 60 seconds for service readiness, regardless of where it runs. Readiness only establishes that the database loaded, not that the robot has localized. The Env manages the navigation workers; start and stop the localization service separately. Exiting the Env leaves this service running. A localization RPC failure exits the image worker and stops the navigation graph; pose-freshness and motion checks remain active.

8. Robot navigation
--------------------------------------

Stop the capture Env and start the navigation Env on the robot:

.. code-block:: bash

   source /path/to/ros-ws/install/setup.bash
   python -m robots.lekiwi.env_server --mode navigation \
     --robot-config /path/to/robot.yaml --map /path/to/map-output/map.yaml \
     --ros-python /path/to/ros-python \
     --host 0.0.0.0 --port 8120

This starts the real base; add ``--mock`` for simulated motors while retaining the real camera. Navigation goals use the mode selected by the Env, without another hardware flag.

Use the Dashboard and Agent described in the next section to choose a map node or an existing label such as ``door``. The Agent uses ``get_navigation_state`` to check readiness, ``navigate_to`` to submit the target, and ``stop`` to cancel motion. After stopping, wait until ``operation.state`` leaves ``running`` before submitting another goal.

9. Dashboard and agent
--------------------------------------------

With Codex already configured, use the Codex planner and select a model available to your account:

.. code-block:: bash

   rpent --robot lekiwi --dashboard --planner codex --model gpt-6-astra \
     --robot-config /path/to/robot.yaml --env-endpoint http://192.168.1.10:8120
