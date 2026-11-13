.. _offline-mapping:

Offline Mapping
===============

.. note::

   This mapping approach has only been validated on ZED 2i, using recordings
   from LeKiwi.

Build maps offline from aligned RGB-D images using RTAB-Map. Generate a map
database, a 2D occupancy grid and a colored 3D point cloud, with an optional
export of the optimized trajectory.

Install RTAB-Map, prepare RGB-D images and their calibration, configure data
paths and the camera mounting pose, then build and export the map. RGB-D can
come from an existing dataset, a camera SDK or RAFT-Stereo. Once prepared,
all sources use the same mapping configuration and commands.

1. Install RTAB-Map
-------------------

To install from source, clone the official standalone RTAB-Map repository:

.. code-block:: bash

   git clone https://github.com/introlab/rtabmap.git /path/to/rtabmap
   cd /path/to/rtabmap

Follow the `upstream installation instructions
<https://github.com/introlab/rtabmap/wiki/Installation#standalone-version>`_
to install dependencies and build RTAB-Map, then check that these commands
are available:

.. code-block:: bash

   command -v rtabmap-dataRecorder
   command -v rtabmap-reprocess
   command -v rtabmap-export

.. _mapping-custom-depth:

2. Prepare RGB-D Images and Calibration
---------------------------------------

Prepare a data directory containing ``rgb/``, ``depth/`` and ``camera.yaml``.
Choose the preparation method for your data:

* **Existing RGB-D:** check the input format below, then continue with
  :ref:`mapping-config`.
* **Camera SDK with depth export:** follow :ref:`mapping-sdk-depth`.
* **Only stereo images or side-by-side video:** follow :ref:`mapping-raft-depth`.
  You need matching stereo calibration and an NVIDIA GPU capable of running
  RAFT-Stereo.

RGB-D Input Format
~~~~~~~~~~~~~~~~~~~

Prepare these files:

* **RGB images:** undistorted color PNGs. Each RGB image and its corresponding
  depth image must come from the same instant.
* **Depth images:** single-channel uint16 PNGs, aligned pixel by pixel with RGB
  at the same resolution. Values represent optical-axis Z distance in
  millimetres, with zero for invalid pixels and depths beyond 65.535 metres.
  Supply numerical depth, not a colorized depth visualization.
* **Camera calibration:** RTAB-Map/OpenCV ``camera.yaml`` with dimensions and
  intrinsics matching the RGB images, used to project depth pixels into 3D.

Give each RGB/depth pair the same filename. Use increasing timestamps in
seconds, starting above zero, with consistent decimal precision. Preserve the
actual capture intervals when using relative timestamps:

.. code-block:: text

   rgbd/
     rgb/1.0.png
     rgb/1.2.png
     depth/1.0.png
     depth/1.2.png
     camera.yaml

The ``camera.yaml`` must describe the final RGB images, including their
resolution, intrinsics and undistorted state. Supply it with SDK exports;
the RAFT-Stereo preparation script generates it.

.. _mapping-sdk-depth:

Use Camera SDK Depth
~~~~~~~~~~~~~~~~~~~~

Follow the vendor's official environment and export workflow to obtain
undistorted RGB, aligned numerical depth from the same recorded frame, and
intrinsics for that output resolution. Arrange the result in the RGB-D layout
above. Use the vendor's export tools or your own exporter; this toolkit does
not include an SDK export script:

* Convert numerical depth to millimetres and save single-channel uint16 PNGs.
  Set invalid values and values beyond 65.535 metres to zero.
* Keep RGB and depth at the same resolution, aligned pixel by pixel, with
  matching filenames. Supply RTAB-Map calibration YAML for the exported RGB.

For example, the ZED SDK's `official depth sample
<https://github.com/stereolabs/zed-sdk/blob/master/tutorials/tutorial%203%20-%20depth%20sensing/python/depth_sensing.py>`_
uses ``VIEW.LEFT`` for the left image and ``MEASURE.DEPTH`` for aligned depth.
Save the sequence and its calibration in the format above. Once ``rgb/``,
``depth/`` and ``camera.yaml`` are ready, continue with :ref:`mapping-config`.

.. _mapping-raft-depth:

Generate Depth with RAFT-Stereo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section converts synchronized stereo data to RGB-D. RPent's
``prepare_stereo.py`` rectifies the images, the official RAFT-Stereo ``demo.py``
estimates pixel displacement, and ``raft_to_rgbd.py`` converts it to depth
in millimetres.

**Prepare Two Python Environments**

* **RPent:** use Python 3.10–3.12 with RPent, NumPy, SciPy, PyYAML and OpenCV.
* **RAFT-Stereo:** follow the `official instructions
  <https://github.com/princeton-vl/RAFT-Stereo#requirements>`_ to clone the
  repository and create a separate ``raftstereo`` Conda environment
  (Python 3.7.6). Inference requires an NVIDIA GPU.

Start by activating the RPent environment and run RPent commands from its source root.

**Step 1: Prepare Stereo Data and Calibration**

Choose one input:

* **Images:** ``sequence/left/`` and ``sequence/right/`` with matching PNG,
  JPEG or TIFF filenames.
* **Stereo video:** local MP4, AVI, MKV or MOV with synchronized left and right
  views side by side, left first. The decoder must support the codec.

Copy the calibration template from the repository:

.. code-block:: bash

   cd /path/to/RPent
   cp toolkits/mapping/stereo.yaml /path/to/stereo.yaml

Fill in actual calibration from your camera or dataset in ``stereo.yaml``.
This is the input calibration for ``prepare_stereo.py``; the script generates
the ``camera.yaml`` used by RTAB-Map after rectifying the images. Replace all
symbolic values in the template:

* ``image_size``: per-eye width and height. A 2560 × 720 side-by-side video
  needs ``[1280, 720]``.
* Left/right ``K`` and ``D``: OpenCV pinhole intrinsics and distortion.
* ``R`` and ``T_m``: the transform from left optical coordinates to right
  optical coordinates, ``X_right = R * X_left + T_m``, with translation in metres.

``R`` and ``T_m`` describe the relationship between the left and right cameras.
Supply the mounting pose separately through ``--camera-pose`` in the next step.

Use a horizontal pinhole stereo pair with the right camera to the right of the
left. For rectified input, supply rectified intrinsics, zero distortion,
identity rotation and ``T_m: [-baseline_m, 0, 0]``. Export fisheye images as
rectified pinhole pairs with the camera tools first.

**Step 2: Rectify and Arrange the Images**

Measure the left camera's fixed mounting pose relative to the robot frame,
then supply it through ``--camera-pose``: the left optical centre's x/y/z
position in metres, followed by mechanical roll/pitch/yaw in degrees. Both
robot and camera mechanical frames use x forward, y left and z up. The script
converts this supplied pose for the rectified images; it does not measure the
mounting pose.

Replace the example level, forward-facing pose 0.5 metres above the robot
origin with measured values. The output directory must be new and outside
the input directory.

**Image example:** this assumes zero-padded numbered filenames sampled at 5 Hz.

.. code-block:: bash

   cd /path/to/RPent
   python -m toolkits.mapping.prepare_stereo \
     --input /path/to/sequence \
     --calibration /path/to/stereo.yaml \
     --camera-pose 0 0 0.5 0 0 0 \
     --fps 5 \
     --output /path/to/prepared

Numbered filenames are sorted lexically. If filenames already contain
decimal-second timestamps, replace ``--fps 5`` with ``--timestamps-from-names``.

**Video example:** use this command instead of the image command. For a 15 fps
video, ``--step 3`` keeps one pair every three frames, giving 5 pairs per second.
Replace the camera pose with actual values here too.

.. code-block:: bash

   cd /path/to/RPent
   python -m toolkits.mapping.prepare_stereo \
     --input /path/to/stereo.mp4 \
     --calibration /path/to/stereo.yaml \
     --camera-pose 0 0 0.5 0 0 0 \
     --step 3 \
     --output /path/to/prepared

Omitting ``--fps`` reads the video's constant frame rate; omitting ``--step``
processes every frame. Convert variable-rate video to timestamped image pairs
first.

Both inputs produce the same ``prepared/`` structure: rectified
``pairs/<timestamp>/im0.png`` (left) and ``im1.png`` (right), plus
``camera.yaml`` and ``stereo.json``.

**Step 3: Run Official RAFT-Stereo**

Download the official weights from the RAFT-Stereo source directory.
The download script uses the system commands
``wget`` and ``unzip``:

.. code-block:: bash

   cd /path/to/RAFT-Stereo
   bash download_models.sh

Alternatively, download ``raftstereo-middlebury.pth`` from the
`Middlebury checkpoint link
<https://drive.google.com/file/d/1m3KoukUmKDoMv-ySOO6vBzYfWLyj9yqd/view>`_
in the official README and save it under ``/path/to/RAFT-Stereo/models/``.

The download script extracts weights under ``models/``. Replace ``--restore_ckpt`` below
with the actual path to the downloaded ``raftstereo-middlebury.pth`` and
select an available NVIDIA GPU. This example uses the standard checkpoint and
the official demo's default architecture. Choose a fresh ``flow/`` directory
for ``--output_directory``.

.. code-block:: bash

   cd /path/to/RAFT-Stereo
   conda activate raftstereo
   python demo.py \
     --restore_ckpt /path/to/raftstereo-middlebury.pth \
     --left_imgs "/path/to/prepared/pairs/*/im0.png" \
     --right_imgs "/path/to/prepared/pairs/*/im1.png" \
     --save_numpy \
     --output_directory /path/to/flow

``--left_imgs`` and ``--right_imgs`` point to the rectified pairs produced by
``prepare_stereo.py``.
Keep the quotes so the demo expands the wildcards. ``--save_numpy`` saves
each pair's pixel displacement as ``flow/<timestamp>.npy``, alongside a
visualization. The next step reads the numerical ``.npy`` files.

**Step 4: Convert to RGB-D**

Reactivate the RPent environment, then return to the RPent source directory.
This command reads the images and calibration in ``prepared/`` together with
the ``flow/`` output from RAFT-Stereo, producing RGB-D. Choose a new output
directory outside ``prepared/``:

.. code-block:: bash

   cd /path/to/RPent
   python -m toolkits.mapping.raft_to_rgbd \
     --input /path/to/prepared \
     --flow-dir /path/to/flow \
     --output /path/to/rgbd

The resulting ``rgbd/`` contains matching ``rgb/*.png`` and ``depth/*.png``,
plus ``camera.yaml``. Depth is uint16 millimetres,
aligned with RGB; invalid values and depths beyond 65.535 metres are zero.

Conversion uses ``depth = fx * baseline / disparity``, where disparity is the
negative of the official demo's saved flow. Images, RAFT output and calibration
must therefore retain the same resolution. Filenames preserve time intervals
with consistent decimal precision.

After generation, follow :ref:`mapping-rgbd-source` to configure RGB, depth,
calibration and camera pose.

.. _mapping-config:
.. _mapping-rgbd-source:

3. Configure RTAB-Map RGB-D Input
----------------------------------------

Once the data directory meets the RGB-D input requirements, copy
``rtabmap_rgbd.ini`` from the RPent source root into a new mapping directory
as ``rtabmap.ini``:

.. code-block:: bash

   cd /path/to/RPent
   mkdir /path/to/new-map
   cp toolkits/mapping/rtabmap_rgbd.ini /path/to/new-map/rtabmap.ini

The ``rgbd/`` directory holds input images and calibration. The ``new-map/``
directory holds the mapping configuration, databases and exports. Make the
following changes in the working copy of ``rtabmap.ini``.

Set Data Paths
~~~~~~~~~~~~~~

Replace the template paths with actual locations. Set ``calibrationName``
to the ``camera.yaml`` matching the RGB images:

.. code-block:: ini

   [Core]
   Rtabmap\WorkingDirectory=/path/to/new-map

   [Camera]
   calibrationName=/path/to/rgbd/camera.yaml
   RGBDImages\path_rgb=/path/to/rgbd/rgb
   RGBDImages\path_depth=/path/to/rgbd/depth

This excerpt shows only the paths to edit; retain the other template settings.
The template selects the RGB-D image driver (``type=0``, ``rgbd\driver=7``),
uses scale factor 1 for millimetre depth and reads filename timestamps.
Synchronization waits are disabled and input rate is 0 for fast offline import.

Set the Camera Mounting Pose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``localTransform`` in the ``[Camera]`` section. Its six values are x/y/z
in metres and roll/pitch/yaw in radians. They describe the camera's fixed
mounting pose relative to the robot frame, using the camera optical centre
as its origin and mechanical axes: x forward, y left and z up.

* **RAFT-generated RGB-D:** copy the ``localTransform=...`` setting logged by
  ``prepare_stereo.py`` into the configuration. This value is derived from your
  measured ``--camera-pose``, with angles converted from degrees to radians
  and the image rectification rotation taken into account.
* **SDK exports or existing RGB-D:** use the measured mounting pose with angles
  in radians. If export rotates the camera coordinate axes, also adjust the
  pose to match the final RGB image frame.

The template example describes a level, forward-facing camera 0.5 metres above
the robot origin. The z value equals height above ground only when the robot
reference origin is on the ground. Replace the example with actual values:

.. code-block:: ini

   [Camera]
   localTransform=0 0 0.5 0 0 0

The mounting pose does not require a per-frame trajectory; visual odometry
estimates motion in the next section. Continue with :ref:`mapping-build`.

.. _mapping-build:

4. Build the Map
----------------

After configuring RGB-D input, run the `official RTAB-Map tools
<https://introlab.github.io/rtabmap/api/latest/tools.html>`_ to import images,
estimate the trajectory and export the optimized map. Run these commands in
Bash; logs are displayed in the terminal and saved in the output directory:

.. code-block:: bash

   cd /path/to/new-map
   set -o pipefail
   export QT_QPA_PLATFORM=offscreen
   rtabmap-dataRecorder -hide rtabmap.ini raw.db 2>&1 | tee import.log
   rtabmap-reprocess -default -odom -c rtabmap.ini raw.db rtabmap.db 2>&1 | tee mapping.log
   rtabmap-export --map --cloud --opt 2 --output map rtabmap.db 2>&1 | tee export.log

The three commands perform these steps. Confirm that each succeeds before
running the next:

1. ``rtabmap-dataRecorder`` imports RGB-D pairs using filename timestamps and
   writes ``raw.db``.
2. ``rtabmap-reprocess -odom`` computes visual odometry from RGB-D and refines
   the trajectory during mapping, writing ``rtabmap.db``.
3. ``rtabmap-export`` exports the 2D occupancy grid and 3D point cloud.

When finished, continue to :ref:`mapping-results` to view the map. Use the
optional commands below when you need a trajectory file or planar constraints.

Optional Processing
~~~~~~~~~~~~~~~~~~~

**Export the optimized trajectory (optional)**

Export the optimized trajectory from ``rtabmap.db`` to ``trajectory_poses.txt``
in the same directory:

.. code-block:: bash

   cd /path/to/new-map
   rtabmap-export --poses --poses_format 10 --opt 2 --output trajectory rtabmap.db

Each row of ``trajectory_poses.txt`` contains ``stamp x y z qx qy qz qw``,
with positions in metres and x-forward, y-left, z-up axes.

**Apply planar motion constraints (optional)**

For a camera fixed to a platform moving on a flat floor, optionally add planar
motion constraints in a second pass and save the result as ``planar.db``:

.. code-block:: bash

   rtabmap-reprocess -c rtabmap.ini -noimu \
     --Reg/Force3DoF true --RGBD/ForceOdom3DoF true rtabmap.db planar.db
   rtabmap-export --map --cloud --opt 2 --output planar planar.db

.. _mapping-results:

5. View Mapping Results
-----------------------

After mapping, review the output files and inspect the map and trajectory.

Output Files
~~~~~~~~~~~~

After the import, mapping and export commands complete successfully,
``/path/to/new-map/`` contains the following files. The trajectory file
is generated by the optional export command in the preceding section.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Output
     - Contents and Use
   * - ``rtabmap.db``
     - Native map database with visual data and the pose graph. Keep it for
       inspection, reprocessing and subsequent RTAB-Map localization setup.
   * - ``map.pgm`` + ``map.yaml``
     - A 2D occupancy grid describing free, occupied and unknown areas, plus
       its resolution and origin. A ROS/Nav2 map server can load this pair.
   * - ``map_cloud.ply``
     - A colored 3D point cloud for viewing reconstructed geometry and
       inspecting coverage and reconstruction artifacts.
   * - ``trajectory_poses.txt`` (additional export)
     - Generated from ``rtabmap.db`` by the trajectory export command.
       Contains timestamps, positions and orientations of optimized poses
       for inspecting the reconstructed path.
   * - ``raw.db``
     - Imported RGB-D frames, retained for rerunning the mapping
       stage with different parameters.
   * - ``import.log``, ``mapping.log``, ``export.log``
     - Command output for checking progress and diagnosing failures.

If you applied planar motion constraints, the additional outputs are
``planar.db``, ``planar.pgm``, ``planar.yaml`` and ``planar_cloud.ply``.

The trajectory records estimated poses of mapping nodes after optimization.
Maps and trajectories use a local map frame. These outputs can be used for
subsequent robot localization and navigation setup.

Inspect the Map
~~~~~~~~~~~~~~~

Validate scale, obstacles, coverage and unknown areas against measurements.
Keep ``rtabmap.db`` with its visual data for subsequent RTAB-Map visual
localization.

When comparing depth sources, keep images, resolution, sampling, calibration,
camera pose, mapping and filtering settings identical.

Check common issues as follows:

* **Missing commands or Qt errors:** check the standalone RTAB-Map installation
  and ``QT_QPA_PLATFORM``.
* **Missing pairs or dimension errors:** check RGB/depth filenames, image
  dimensions, alignment and calibration. For RAFT preparation, also check the
  left/right input pairs.
* **RAFT GPU memory errors:** choose a GPU with enough memory for the per-frame
  resolution; ``--step`` controls the number of samples and total work.
* **Empty or fragmented maps:** check tracking results in the database,
  image blur, texture and calibration.
* **Preparation failures:** inspect the partial output, correct the cause and
  retry with a fresh output directory.
