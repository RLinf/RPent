.. _offline-mapping:

Offline Mapping
===============

.. note::

   This mapping approach has only been validated on ZED 2i, using recordings
   from LeKiwi.

Build maps offline from stereo images or a side-by-side video using RTAB-Map.
Generate a map database, a 2D occupancy grid and a colored 3D point cloud,
with an optional export of the optimized trajectory.

After preparing input and calibration, stereo images or video can proceed
directly to :ref:`mapping-config`. To compute depth with a camera SDK or
RAFT-Stereo, first follow :ref:`mapping-custom-depth`, then choose RGB-D input.
Both input types use the same mapping commands.

Install RTAB-Map
----------------

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

The earlier recipe used RTAB-Map 0.23.13.

Prepare Input and Calibration
-----------------------------

Prepare the following:

1. **Synchronized left/right images:** each pair must come from the same instant.
   Place them in ``left/`` and ``right/`` with matching filenames.
2. **Stereo calibration:** save the camera or dataset calibration in RTAB-Map
   format as ``camera_left.yaml`` and ``camera_right.yaml`` in the same directory.
   Per-eye dimensions, intrinsics, distortion and relative camera pose must
   match the input images. The right projection matrix must encode the actual
   baseline (the distance between cameras). RTAB-Map uses these values to
   convert pixel disparity into distance.
3. **Timestamp filenames:** use increasing capture times in seconds, starting
   above zero, with consistent decimal precision. For example, images captured
   at 5 pairs per second can be named ``1.0.png``, ``1.2.png``, ``1.4.png``.
   Relative timestamps must preserve the actual capture intervals.

Example image layout:

.. code-block:: text

   stereo/
     left/1.0.png
     left/1.2.png
     right/1.0.png
     right/1.2.png
     camera_left.yaml
     camera_right.yaml

For video, prepare a constant-rate side-by-side file supported by the installed
decoder, such as ``/path/to/stereo.mp4``. Each frame contains the left view on
the left and the right view from the same instant on the right. A 2560 × 720
video uses calibration for 1280 × 720 per-eye images.

Example: Generate Calibration for a ZED 2i Recording
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses ZED 2i data captured through UVC by LeKiwi, with
``/path/to/capture`` as the recording directory. Its ``left/`` and ``right/``
directories contain raw stereo PNGs. Convert this camera's factory
calibration into ``camera_left.yaml`` and ``camera_right.yaml``
for RTAB-Map as follows.

**1. Find the factory calibration for the recording**

Follow the
`Stereolabs calibration instructions
<https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration>`_
to obtain ``SN<serial>.conf`` using the serial number of the camera that
captured these images. Save it as ``/path/to/capture/calibration.conf``.

The ``zed.conf`` used during capture in this example is a local copy of that
factory calibration. You can also copy it into the recording directory as
``calibration.conf``.

**2. Confirm the per-eye image dimensions**

Check the width and height of an original image in ``left/`` and confirm that
the right images have the same dimensions. For a 1280 × 720 left image, use
``--image-size 1280 720``. A 2560 × 720 side-by-side video also has per-eye
dimensions of 1280 × 720.

The script supports 1280 × 720 (``HD``) and 672 × 376 (``VGA``).
Use the original recorded image dimensions.

**3. Run the conversion script**

From the RPent source root, run this command in a Python environment with
NumPy, OpenCV and PyYAML:

.. code-block:: bash

   python -m toolkits.mapping.zed_calibration \
     --calibration /path/to/capture/calibration.conf \
     --image-size 1280 720 \
     --output /path/to/capture

The command writes ``camera_left.yaml`` and ``camera_right.yaml`` into the
``--output`` directory. Choose a directory where these files have yet to be
created; use a fresh directory when repeating the conversion. The YAML files
contain raw intrinsics, distortion, stereo rectification and projection
matrices. Translation is in metres, and the right projection matrix encodes
the actual baseline.

**4. Use the generated calibration in the mapping configuration**

Copy the image configuration as described in :ref:`mapping-config`, then set
these entries in ``rtabmap.ini``:

.. code-block:: ini

   calibrationName=/path/to/capture/camera_left.yaml
   StereoImages\path_left=/path/to/capture/left
   StereoImages\path_right=/path/to/capture/right
   StereoImages\rectify=true

Setting ``true`` makes RTAB-Map rectify the raw stereo images using the
generated calibration. For raw side-by-side video, convert the factory
calibration for its actual per-eye dimensions and enable the same
rectification option in the video configuration.

.. _mapping-custom-depth:

Generate Depth Information (Optional)
-------------------------------------

Use depth exported by a camera SDK, or generate depth from stereo images with
RAFT-Stereo. Prepare RGB and depth in the format below, then supply them to
RTAB-Map.

RGB-D Output Format
~~~~~~~~~~~~~~~~~~~

Prepare these files:

* **RGB images:** undistorted PNGs.
* **Depth images:** single-channel uint16 PNGs, aligned pixel by pixel with RGB
  at the same resolution. Values represent optical-axis Z distance in
  millimetres, with zero for invalid pixels.
* **Camera calibration:** RTAB-Map/OpenCV ``camera.yaml`` with dimensions and
  intrinsics matching the RGB images, used to project depth pixels into 3D.

Give each RGB/depth pair the same filename. Use increasing timestamps in
seconds, starting above zero, with consistent decimal precision:

.. code-block:: text

   rgbd/
     rgb/1.0.png
     rgb/1.2.png
     depth/1.0.png
     depth/1.2.png
     camera.yaml

Export images and calibration with the tools for your data source, then set
``calibrationName`` to the calibration file. The RAFT-Stereo workflow generates
the matching ``camera.yaml``.

Use Camera SDK Depth
~~~~~~~~~~~~~~~~~~~~

Follow the vendor's official environment and export workflow to obtain
undistorted RGB, aligned numerical depth from the same recorded frame, and
intrinsics for that output resolution. Arrange the result in the RGB-D layout
above:

1. Convert numerical depth to millimetres and save single-channel uint16 PNGs.
   Set invalid values and values beyond 65.535 metres to zero.
2. Keep RGB and depth at the same resolution, aligned pixel by pixel, with
   matching filenames. Supply RTAB-Map calibration YAML for the exported RGB.
3. Continue with :ref:`mapping-rgbd-source`.

For example, the ZED SDK's `official depth sample
<https://github.com/stereolabs/zed-sdk/blob/master/tutorials/tutorial%203%20-%20depth%20sensing/python/depth_sensing.py>`_
uses ``VIEW.LEFT`` for the left image and ``MEASURE.DEPTH`` for aligned depth.
Save the sequence and its calibration in the format above.

Generate Depth with RAFT-Stereo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use synchronized left/right RGB images and their stereo calibration.
RAFT-Stereo estimates pixel displacement; focal length and baseline convert
this to depth in millimetres. Follow these stages:

.. code-block:: text

   Stereo images or side-by-side video + stereo.yaml
     → prepare_stereo.py: rectify and arrange pairs → prepared/
     → Official RAFT-Stereo demo: estimate pixel displacement → flow/*.npy
     → raft_to_rgbd.py: convert using focal length and baseline → rgbd/
     → RTAB-Map: build a map from RGB-D input

Step 1: Rectify Images Using Existing Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the RPent source root, use a Python environment with NumPy, SciPy, PyYAML
and OpenCV. Choose one input:

* **Images:** ``sequence/left/`` and ``sequence/right/`` with matching PNG,
  JPEG or TIFF filenames.
* **Stereo video:** local MP4, AVI, MKV or MOV with synchronized left and right
  views side by side, left first. The decoder must support the codec.

Copy the calibration template from the repository:

.. code-block:: bash

   cp toolkits/mapping/stereo.yaml /path/to/stereo.yaml

Fill in actual calibration from your camera or dataset in ``stereo.yaml``:

* ``image_size``: per-eye width and height. A 2560 × 720 side-by-side video
  needs ``[1280, 720]``.
* Left/right ``K`` and ``D``: OpenCV pinhole intrinsics and distortion.
* ``R`` and ``T_m``: the transform from left optical coordinates to right
  optical coordinates, ``X_right = R * X_left + T_m``, with translation in metres.

Use a horizontal pinhole stereo pair with the right camera to the right of the
left. For rectified input, supply rectified intrinsics, zero distortion,
identity rotation and ``T_m: [-baseline_m, 0, 0]``. Export fisheye images as
rectified pinhole pairs with the camera tools first.

**Image example:** this assumes zero-padded numbered filenames sampled at 5 Hz.
``--camera-pose`` gives the original left camera's position (x/y/z, metres)
and mechanical orientation (roll/pitch/yaw, degrees), relative to an
x-forward, y-left, z-up ground frame. Replace the example level, forward-facing
pose at 0.5 metres above ground with actual values.

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
``camera.yaml``, ``local_transform.txt`` and ``stereo.json``. All subsequent
commands use this directory. The output directory must be new and outside the
input. Timestamps start at one second and preserve sampling intervals.
The output pose accounts for left-image rectification and uses radians.

Step 2: Run Official RAFT-Stereo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the official repository:

.. code-block:: bash

   git clone https://github.com/princeton-vl/RAFT-Stereo.git /path/to/RAFT-Stereo
   cd /path/to/RAFT-Stereo

Prepare and activate the RAFT environment following the upstream README's
`Requirements <https://github.com/princeton-vl/RAFT-Stereo#requirements>`_.
From the RAFT-Stereo repository, download the official weights:

.. code-block:: bash

   bash download_models.sh

Weights are extracted under ``models/``. Replace ``--restore_ckpt`` below
with the actual path to the downloaded ``raftstereo-middlebury.pth`` and
select an available NVIDIA GPU. This example uses the standard checkpoint and
the official demo's default architecture.

.. code-block:: bash

   cd /path/to/RAFT-Stereo
   CUDA_VISIBLE_DEVICES=0 python demo.py \
     --restore_ckpt /path/to/raftstereo-middlebury.pth \
     --left_imgs "/path/to/prepared/pairs/*/im0.png" \
     --right_imgs "/path/to/prepared/pairs/*/im1.png" \
     --save_numpy \
     --output_directory /path/to/flow

``--left_imgs`` and ``--right_imgs`` point to the rectified pairs from step 1.
Keep the quotes so the demo expands the wildcards. ``--save_numpy`` saves
each pair's pixel displacement as ``flow/<timestamp>.npy``, alongside a
visualization. Use a fresh ``flow/`` directory.

Step 3: Convert to RGB-D
^^^^^^^^^^^^^^^^^^^^^^^^

Switch back to the RPent Python environment used in step 1. This command reads
the images and calibration in ``prepared/`` together with the ``flow/``
output from step 2, producing RGB-D:

.. code-block:: bash

   cd /path/to/RPent
   python -m toolkits.mapping.raft_to_rgbd \
     --input /path/to/prepared \
     --flow-dir /path/to/flow \
     --output /path/to/rgbd

The resulting ``rgbd/`` contains matching ``rgb/*.png`` and ``depth/*.png``,
plus ``camera.yaml`` and ``local_transform.txt``. Depth is uint16 millimetres,
aligned with RGB; invalid values and depths beyond 65.535 metres are zero.
The output directory must not already exist.

Conversion uses ``depth = fx * baseline / disparity``, where disparity is the
negative of the official demo's saved flow. Images, RAFT output and calibration
must therefore retain the same resolution. Filenames preserve time intervals
with consistent decimal precision.

After generation, follow :ref:`mapping-rgbd-source` to configure RGB, depth,
calibration and camera pose.

.. _mapping-config:

Configure RTAB-Map Input
------------------------

Choose one input configuration below. From the RPent source root, copy it as
``rtabmap.ini`` and fill in the following:

* **Paths:** replace every ``/path/to/...`` with the actual working directory,
  input and calibration paths. Set ``calibrationName`` to ``camera_left.yaml``
  for stereo input or ``camera.yaml`` for RGB-D input.
* **Stereo rectification:** for raw images, use raw calibration and set
  ``StereoImages\rectify=true``. For rectified images, use their rectified
  calibration and keep the default ``false``. This setting applies to both
  images and video.
* **Camera mounting pose:** set ``localTransform`` to the actual x/y/z in metres
  and roll/pitch/yaw in radians. This is the camera mechanical frame's pose in
  an x-forward, y-left, z-up reference frame with the ground at z=0. The template's
  ``0 0 0.5 0 0 0`` describes a level, forward-facing camera 0.5 metres above ground.

Stereo Images
~~~~~~~~~~~~~

Use ``rtabmap_stereo_images.ini``.
``type=1`` selects stereo and ``stereo\driver=2`` selects the image driver.
Filename timestamps are enabled, synchronization waits are disabled and
``imgRate=0`` enables fast offline import.

.. code-block:: bash

   mkdir /path/to/new-map
   cp toolkits/mapping/rtabmap_stereo_images.ini /path/to/new-map/rtabmap.ini

Stereo Video
~~~~~~~~~~~~

Use ``rtabmap_stereo_video.ini``.
``stereo\driver=3`` selects the video driver; an empty
``StereoVideo\path2`` selects a single side-by-side file. Change
``imgRate=15`` to the video's actual frame rate.

.. code-block:: bash

   mkdir /path/to/new-map
   cp toolkits/mapping/rtabmap_stereo_video.ini /path/to/new-map/rtabmap.ini

.. _mapping-rgbd-source:

RGB-D Images
~~~~~~~~~~~~

Use ``rtabmap_rgbd.ini``, which includes the common mapping parameters.
``type=0`` and ``rgbd\driver=7`` select the RGB-D image driver. The scale
factor is 1 for millimetre depth; filename timestamps are enabled,
synchronization waits are disabled and input rate is 0 for fast import:

.. code-block:: bash

   mkdir /path/to/new-map
   cp toolkits/mapping/rtabmap_rgbd.ini /path/to/new-map/rtabmap.ini

Replace the example paths and ``localTransform`` with actual values.
For RAFT, use the ``camera.yaml`` produced by ``raft_to_rgbd.py`` and copy
the six numbers from ``local_transform.txt`` into ``localTransform``.
Its angles are already in radians and account for rectification rotation.
Read the file in the terminal:

.. code-block:: bash

   cat /path/to/rgbd/local_transform.txt

For SDK depth, use calibration for the exported RGB images and set
``localTransform`` to the actual camera pose.

After configuration, continue with :ref:`mapping-build`. Here, ``raw.db``
stores RGB-D frames and ``-odom`` computes visual odometry from RGB-D.
When comparing depth sources, keep images, resolution, sampling, calibration,
camera pose, mapping and filtering settings identical.

.. _mapping-build:

Build the Map
-------------

After configuring stereo or RGB-D input, run the `official RTAB-Map tools
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

``-odom`` estimates camera motion from successive images, producing visual
odometry associated with the images. Mapping then refines the trajectory.
Images use filename timestamps; video is read at the configured frame rate.

For a camera fixed to a platform moving on a flat floor, optionally add planar
motion constraints in a second pass and save the result as ``planar.db``:

.. code-block:: bash

   rtabmap-reprocess -c rtabmap.ini -noimu \
     --Reg/Force3DoF true --RGBD/ForceOdom3DoF true rtabmap.db planar.db
   rtabmap-export --map --cloud --opt 2 --output planar planar.db

Expected Outputs
----------------

After the import, mapping and export commands complete successfully,
``/path/to/new-map/`` contains the following files. The optimized trajectory
requires the additional export command in the next section.

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
     - Timestamped positions and orientations of the mapped poses, for
       plotting and inspecting the reconstructed path.
   * - ``raw.db``
     - Imported stereo or RGB-D frames, retained for rerunning the mapping
       stage with different parameters.
   * - ``import.log``, ``mapping.log``, ``export.log``
     - Command output for checking progress and diagnosing failures.

The trajectory records estimated poses of mapping nodes after optimization.
Maps and trajectories use a local map frame. These outputs can be used for
subsequent robot localization and navigation setup.

Inspect the Map
---------------

Inspect output files, map resolution and origin, and mapping logs in the
terminal, then export the optimized trajectory:

.. code-block:: bash

   cd /path/to/new-map
   ls -lh raw.db rtabmap.db map.pgm map.yaml map_cloud.ply
   cat map.yaml
   tail -n 30 import.log mapping.log export.log
   rtabmap-export --poses --poses_format 10 --opt 2 --output trajectory rtabmap.db
   head trajectory_poses.txt

Each row of ``trajectory_poses.txt`` contains ``stamp x y z qx qy qz qw``,
with positions in metres and x-forward, y-left, z-up axes. Review tracking loss
and loop closures in the logs, and validate scale, obstacles, coverage and
unknown areas against measurements. Keep ``rtabmap.db`` with its visual data
for subsequent RTAB-Map visual localization.

Check common issues as follows:

* **Missing commands or Qt errors:** check the standalone RTAB-Map installation
  and ``QT_QPA_PLATFORM``.
* **Missing pairs or dimension errors:** check left/right filenames, image
  dimensions and calibration.
* **RAFT GPU memory errors:** choose a GPU with enough memory for the per-frame
  resolution; ``--step`` controls the number of samples and total work.
* **Empty or fragmented maps:** inspect logs and the database for tracking loss,
  motion blur, weak texture and calibration issues.
* **Preparation failures:** inspect the partial output, correct the cause and
  retry with a fresh output directory.
