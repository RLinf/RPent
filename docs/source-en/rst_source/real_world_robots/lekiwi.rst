LeKiwi
======

Use RPent to collect ZED stereo images with LeKiwi, build a map, and let an Agent navigate to map nodes or named places.

Overview
--------

Prepare the hardware using the `RLinf LeKiwi guide <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/lekiwi.html>`_. This page uses a LeKiwi base and ZED 2i:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Location
     - Purpose
     - Dependencies
   * - Robot (Linux)
     - Env owns the camera and base; Nav2 plans and avoids obstacles.
     - RPent, companion RLinf; ROS 2 Jazzy for navigation. No GPU or ZED SDK required.
   * - Mapping machine
     - Generate depth and maps offline.
     - RPent, CUDA, RAFT-Stereo, RTAB-Map.
   * - Operator computer
     - Teleoperation, Dashboard and Agent.
     - RPent and network access to the robot.

The CPU localization service can run on the robot or another host. Supervise physical runs and keep the motor power switch within reach.

Installation and Resources
--------------------------

Use Python 3.10–3.12 and run commands from the RPent checkout. Replace example paths and ``ROBOT_HOST`` with your values; store configuration and data outside the repository.

Robot
~~~~~

Install RPent and companion RLinf source; its camera interface must support ``get_frame(return_timestamp=True)``:

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install --torch-backend cpu -e ".[lekiwi]"
   uv pip install --no-deps -e /path/to/RLinf

Navigation also needs `ROS 2 Jazzy <https://docs.ros.org/en/jazzy/Installation.html>`_ with ``nav2_bringup``, ``nav2_mppi_controller``, ``launch_ros``, ``robot_localization``, ``rtabmap_util``, ``sensor_msgs_py`` and ``rmw_cyclonedds_cpp``. Install bridge dependencies in a separate ROS Python environment:

.. code-block:: bash

   /path/to/ros-python -m pip install numpy scipy opencv-python-headless pillow pyyaml omegaconf
   /path/to/ros-python -m pip install --no-deps -e /path/to/RPent

Operator Computer
~~~~~~~~~~~~~~~~~

Install RPent in a separate Python environment:

.. code-block:: bash

   uv pip install -e /path/to/RPent

Mapping and Localization
~~~~~~~~~~~~~~~~~~~~~~~~

Install RPent on the mapping machine and prepare these resources using the upstream instructions:

- `RTAB-Map <https://github.com/introlab/rtabmap/wiki/Installation>`_: enable Qt image import tools and GTSAM, and add the commands to ``PATH``.
- `RAFT-Stereo <https://github.com/princeton-vl/RAFT-Stereo>`_: prepare a CUDA PyTorch environment and ``raftstereo-middlebury.pth``; install RPent in that environment.

.. code-block:: bash

   git clone https://github.com/introlab/rtabmap.git /path/to/rtabmap
   git clone https://github.com/princeton-vl/RAFT-Stereo.git /path/to/RAFT-Stereo

The localization host only needs RPent, RTAB-Map, OpenCV and the localization extension. With a C++17 compiler, Python headers, and RTAB-Map and OpenCV development libraries installed, run in the service's Python environment:

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

Robot Configuration
-------------------

Copy the example and fill in your hardware parameters:

.. code-block:: bash

   cp robots/lekiwi/config/example.yaml /path/to/robot.yaml

- ``robot``: motor serial port and measured wheel and base radii.
- ``cameras.devices.front``: UVC device identifier ``serial`` and the camera label's ``zed_serial``; ``eye_size`` defaults to ``[1280, 720]``.
- ``perception.calibration``: factory calibration obtained from the `Stereolabs calibration guide <https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration>`_. Relative paths resolve from the YAML directory.
- ``geometry``: measured body dimensions (including the fixed arm), camera mount pose, navigation radius and motion limits. Axes are x forward, y left, z up; lengths are metres and mount angles are degrees.
- ``drive``: teleoperation speeds and frequency. Keep defaults for ``perception.stereo`` and ``perception.obstacles`` if appropriate.

Teleoperation and Capture
-------------------------

Start the Env on the robot:

.. code-block:: bash

   python -m robots.lekiwi.env_server --robot-config /path/to/robot.yaml \
     --host 0.0.0.0 --port 8120 --recording-dir /path/to/recordings

Start the Dashboard on the operator computer:

.. code-block:: bash

   rpent --robot lekiwi --dashboard --planner manual \
     --env-endpoint http://ROBOT_HOST:8120 --robot-config /path/to/robot.yaml

Open the printed URL and submit ``/rpent-task 0``. Hold a direction button to drive; release it to stop.

Open a separate Python session on the operator computer to start stereo PNG recording:

.. code-block:: python

   from uuid import uuid4
   from robots.lekiwi.env_client import LeKiwiEnvClient
   from rpent.utils.rpc import make_rpc_client

   env = LeKiwiEnvClient(make_rpc_client("http://ROBOT_HOST:8120"))
   identity = uuid4().hex
   env.recording("start", identity=identity)

During teleoperation, call ``env.recording("state", identity=identity)`` for the pair count and output directory. Stop recording in the same Python session when finished:

.. code-block:: python

   env.recording("stop", identity=identity)
   env.close()

Each recording defaults to at most 600 seconds and 5 pairs per second, configured with Env options ``--capture-seconds`` and ``--recording-fps``. Ending the Dashboard task or closing the client does not stop PNG recording; stop it explicitly or wait for the recording limit.

Copy the complete robot-side ``capture-*`` directory (``left/``, ``right/``, ``stereo.json``, ``calibration.conf``) and robot YAML to the mapping machine.

Offline Mapping
---------------

In the mapping machine's RAFT-Stereo environment, export rectified images, depth and RTAB-Map configuration:

.. code-block:: bash

   python -m robots.lekiwi.mapping \
     --input /path/to/capture --output /path/to/map-output/rgbd \
     --robot-config /path/to/robot.yaml --raft-repo /path/to/RAFT-Stereo \
     --checkpoint /path/to/raftstereo-middlebury.pth

Build and export the navigation grid with the `RTAB-Map tools <https://github.com/introlab/rtabmap/wiki/Tools>`_:

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

Copy ``map.yaml`` and its referenced image to the robot, and ``rtabmap.db`` from the same run to the localization host. Inspect nodes and reference images with the database viewer:

.. code-block:: bash

   rtabmap-databaseViewer /path/to/map-output/rtabmap.db

Navigation targets are existing node IDs or labels such as ``door``; the arrival position and heading come from that node's pose. After editing labels, update the database and restart localization. After rebuilding, update the YAML, map image and database together.

.. _lekiwi-navigation-start:

Start Localization
------------------

Run on the host with the localization extension installed, and keep the service running:

.. code-block:: bash

   python -m robots.lekiwi.localization_server serve \
     --database /path/to/map-output/rtabmap.db \
     --bind 127.0.0.1 --port 18125

Set the endpoint in the robot YAML:

.. code-block:: yaml

   localization:
     endpoint: http://127.0.0.1:18125

For another host, change the service's ``--bind`` to ``0.0.0.0`` and the endpoint to ``http://SERVER_HOST:18125``; connect through a trusted network or SSH tunnel. Start and stop localization separately. A loaded database does not mean the robot has localized.

Run a Navigation Task
---------------------

Stop the capture Env and start the navigation Env on the robot:

.. code-block:: bash

   source /path/to/ros-ws/install/setup.bash
   python -m robots.lekiwi.env_server --mode navigation \
     --robot-config /path/to/robot.yaml --map /path/to/map-output/map.yaml \
     --ros-python /path/to/ros-python \
     --host 0.0.0.0 --port 8120

This uses real motors by default; add ``--mock`` for simulated motors with the real camera.

Configure a planner following :doc:`../guides/configure_planner`, then connect the Dashboard from the operator computer:

.. code-block:: bash

   rpent --robot lekiwi --dashboard --planner codex --model gpt-6-astra \
     --robot-config /path/to/robot.yaml --env-endpoint http://ROBOT_HOST:8120

Open the printed URL and submit an instruction to navigate to an existing node or place. See :doc:`../guides/dashboard` for interface controls and :ref:`run-output-files` for run records. Confirm arrival using the current camera view; neither Nav2 success nor ``finish`` establishes physical task acceptance.

Agent Tools
-----------

**Action tools:**

- ``drive(velocity, duration_s=1.0)``: drive at body velocity ``[x, y, theta]`` in m/s, m/s, deg/s for up to 10 seconds, then stop. Speeds are clipped to configured limits. Requires manual control mode.
- ``navigate_to(target, timeout_s=60.0)``: navigate to an existing node ID (integer) or label (string) and wait for the result. Requires navigation mode with valid localization and perception.
- ``stop()``: stop and cancel the current navigation goal. Wait until ``operation.state`` leaves ``running`` before submitting another goal.

**Query and completion tools:**

- ``list_map_nodes(label=None)``: list node IDs and labels, optionally filtered by an exact existing label.
- ``inspect_map_node(node_id)``: inspect a node's pose and historical reference image; the node is a recorded viewpoint.
- ``view_map(center_node_id=None, radius_m=None)``: view the static map, place labels and available robot pose; optionally highlight a node and crop around it. Live obstacles are excluded.
- ``get_navigation_state()``: read navigation readiness, map pose and the current or last operation.
- ``view_env_state(step=-1)``: read a recorded state and front-camera image; ``-1`` selects the latest record.
- ``finish(status, summary)``: end the Agent run and report its outcome.
