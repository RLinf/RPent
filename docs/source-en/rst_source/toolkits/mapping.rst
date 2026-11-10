.. _offline-mapping:

Offline Mapping
===============

.. note::

   This mapping approach has only been validated on ZED 2i, using recordings
   from LeKiwi. Other cameras have not been tested.

Create an RTAB-Map database, occupancy grid and colored point cloud from stereo
images or a side-by-side video. The main workflow uses only official RTAB-Map
tools to load calibration, rectify images, perform stereo matching, estimate
visual odometry, optimize the map and export it. It requires neither precomputed
depth nor RAFT-Stereo or RPent's preparation tools.

To use depth from a camera SDK or RAFT-Stereo, follow
:ref:`mapping-custom-depth`: supply RGB images and aligned depth as RGB-D input,
then reuse the same mapping commands. Both workflows are independent of robot
integrations. This guide does not integrate wheel odometry or IMU data.

Install RTAB-Map
----------------

Run these steps on the machine holding the images and mapping dependencies.
To install from source, clone the official standalone RTAB-Map repository:

.. code-block:: bash

   git clone https://github.com/introlab/rtabmap.git /path/to/rtabmap
   cd /path/to/rtabmap

Then follow the `upstream installation instructions
<https://github.com/introlab/rtabmap/wiki/Installation#standalone-version>`_
for your operating system to install dependencies and build the tools. Use the
feature detector and optimizer supported by your installation. If the tools are
already installed, reuse that installation. Check the executables:

.. code-block:: bash

   command -v rtabmap-dataRecorder
   command -v rtabmap-reprocess
   command -v rtabmap-export

The earlier recipe used RTAB-Map 0.23.13.

Map Stereo Data with RTAB-Map
-----------------------------

Prepare Input and Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide synchronized left/right images and stereo calibration matching their
resolution. Calibration describes the intrinsics, distortion and relative pose
of the cameras, allowing RTAB-Map to turn pixel disparity into distance.

Use native stereo calibration files that RTAB-Map can read, such as
``camera_left.yaml`` and ``camera_right.yaml`` in the same directory. The right
projection matrix must encode the actual baseline. Obtain actual camera or
dataset calibration first, matching the image dimensions and rectification
state. This guide starts with existing calibration and does not solve camera
calibration. RPent's ``stereo.yaml`` is not required.

Stereo image filenames must match between the two directories and contain
increasing, nonzero timestamps in seconds. Relative timestamps suffice; no
calendar date is needed. Use consistent decimal precision within a sequence.

.. code-block:: text

   stereo/
     left/1.0.png
     left/1.2.png
     right/1.0.png
     right/1.2.png
     camera_left.yaml
     camera_right.yaml

Video input is ``/path/to/stereo.mp4``, with the left view on the left and the
synchronized right view on the right of each frame. A 2560 × 720 video needs
calibration for 1280 × 720 per-eye images. Use a constant-rate video supported
by the installed decoder.

ZED 2i Calibration Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

The current LeKiwi recorder reads side-by-side ZED 2i images through UVC
and splits each frame into raw left/right PNGs. It copies the configured
factory calibration to ``calibration.conf`` and writes the per-eye dimensions
to ``stereo.json``. This capture path does not use SVO2 or the ZED SDK.

Obtain the factory calibration for your camera's serial number using the
`Stereolabs calibration instructions
<https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration>`_.
The vendor file is named ``SN<serial>.conf``; the local configuration may call
it ``zed.conf``. The recording's ``calibration.conf`` is a copy of that file.
Renaming it to ``camera_left.yaml`` does not convert its format.

Run the conversion script from the RPent source root in a Python environment
with NumPy, OpenCV and PyYAML. For a recording with 1280 × 720 pixels per eye:

.. code-block:: bash

   python -m toolkits.mapping.zed_calibration \
     --calibration /path/to/capture/calibration.conf \
     --image-size 1280 720 \
     --output /path/to/capture

Use the actual per-eye dimensions recorded in ``stereo.json``. The script
supports 1280 × 720 (``HD``) and 672 × 376 (``VGA``). It leaves existing
calibration YAMLs unchanged and reports an error if either output file exists.

This writes ``camera_left.yaml`` and ``camera_right.yaml`` into the output
directory. Their ``K/D`` values describe the raw images; ``R/P`` are computed
by OpenCV for stereo rectification. Factory translation values are converted
from millimetres to metres, and the right projection matrix retains the metric
baseline. Use calibration from the same camera and the actual image mode;
do not resize or crop the raw images independently.

In ``rtabmap_stereo_images.ini``, set ``calibrationName`` to
``/path/to/capture/camera_left.yaml`` and point the left/right input paths at
the recording. Set ``StereoImages\rectify=true`` for these raw PNGs.
The configuration template defaults to ``false`` for already rectified input.

For a raw side-by-side video, use the same calibration conversion with its
actual per-eye dimensions and matching factory file, then enable rectification
in the video configuration. Video files do not supply these calibration values.
Already rectified images require their corresponding rectified calibration
and ``rectify=false``, rather than applying raw-image rectification again.

.. _mapping-config:

Write Mapping Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose either the image or video configuration below. Run its commands from
the RPent source root. Both examples assume rectified input
and a level, forward-facing camera 0.5 metres above the ground. Replace these
values with the actual configuration. ``localTransform`` is the camera's
mechanical-frame x/y/z in metres and roll/pitch/yaw in radians, relative to
an x-forward, y-left, z-up frame with ground at z=0.

**Stereo images:** use :download:`rtabmap_stereo_images.ini <../../../../toolkits/mapping/rtabmap_stereo_images.ini>`.
``type=1`` selects stereo and ``stereo\driver=2`` selects the image driver.
Filename timestamps are enabled, synchronization waits are disabled and
``imgRate=0`` enables fast offline import.

.. code-block:: bash

   mkdir /path/to/new-map
   cp toolkits/mapping/rtabmap_stereo_images.ini /path/to/new-map/rtabmap.ini

**Stereo video:** use :download:`rtabmap_stereo_video.ini <../../../../toolkits/mapping/rtabmap_stereo_video.ini>`.
``stereo\driver=3`` selects the video driver; an empty
``StereoVideo\path2`` selects a single side-by-side file. Change
``imgRate=15`` to the video's actual frame rate.

.. code-block:: bash

   mkdir /path/to/new-map
   cp toolkits/mapping/rtabmap_stereo_video.ini /path/to/new-map/rtabmap.ini

After copying, edit ``rtabmap.ini`` in the output directory. Replace every
``/path/to/...`` (working directory, input and calibration paths) and the camera
pose with actual values.

.. _mapping-build:

Build the Map
~~~~~~~~~~~~~

Run the `official RTAB-Map tools
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

``-odom`` recomputes visual odometry from images; it does not read wheel
odometry. Images can be imported at full speed; the video settings above read
at the specified frame rate.

For a camera fixed to a platform moving on a flat floor, an optional second
pass adds planar motion constraints. Keep the unconstrained database:

.. code-block:: bash

   rtabmap-reprocess -c rtabmap.ini -noimu \
     --Reg/Force3DoF true --RGBD/ForceOdom3DoF true rtabmap.db planar.db
   rtabmap-export --map --cloud --opt 2 --output planar planar.db

.. _mapping-custom-depth:

Optional: Use Your Own Depth
----------------------------

Use RGB-D input directly when depth is available from a camera SDK, depth camera
or another method. If you only have stereo images and want learned stereo
matching, the RAFT-Stereo workflow below can generate depth. Choose one depth
source; SDK and RAFT processing do not need to run in sequence.

RGB-D Input Format
~~~~~~~~~~~~~~~~~~

The following example layout can be read directly by RTAB-Map. Prepare matching
RGB and depth PNG files, named with increasing, nonzero timestamps in seconds.
Use the same number of decimal places throughout a sequence; leading zeros in
the integer part are unnecessary. Depth must be aligned to the undistorted RGB image at the same
resolution: optical-axis Z, stored as uint16 millimetres, with zero for invalid
pixels. Use the matching RTAB-Map/OpenCV camera calibration YAML.

.. code-block:: text

   rgbd/
     rgb/1.0.png
     rgb/1.2.png
     depth/1.0.png
     depth/1.2.png
     camera.yaml

Existing depth can come from any source meeting this contract. Export the images
and calibration with that source's tools. ``camera.yaml`` supplies the image
dimensions and intrinsics that RTAB-Map uses to project depth pixels into 3D.

Use your own camera or dataset's calibrated values, matching image dimensions
and rectification state; sample camera values are not a calibration for your
images. ``camera.yaml`` is only the example filename; ``calibrationName``
in the native configuration points to the actual file. Stereo preparation
below generates the matching ``camera.yaml`` automatically; use that output
directly.

Use Camera SDK Depth
~~~~~~~~~~~~~~~~~~~~

Follow the vendor's official environment and export workflow to obtain
undistorted RGB, aligned numerical depth from the same recorded frame, and
intrinsics for that output resolution. Arrange the result in the RGB-D layout
above:

1. Convert depth in metres to millimetres and save single-channel uint16 PNGs.
   Set invalid values and values beyond 65.535 metres to zero. Use measured
   depth values, not a colored depth preview.
2. Keep RGB and depth at the same resolution, aligned pixel by pixel, with
   matching filenames. Supply RTAB-Map calibration YAML for the exported RGB.
3. Continue directly with :ref:`mapping-rgbd-source`; neither RAFT nor
   the preparation tools are required.

For example, the ZED SDK's `official depth sample
<https://github.com/stereolabs/zed-sdk/blob/master/tutorials/tutorial%203%20-%20depth%20sensing/python/depth_sensing.py>`_
shows left-image retrieval with ``VIEW.LEFT`` and aligned depth retrieval with
``MEASURE.DEPTH``. Sequence saving, filenames and calibration export still need
to follow the contract above. Other SDKs can be used with the same output format.

Generate Depth with RAFT-Stereo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This route replaces RTAB-Map's internal stereo matching with RAFT-Stereo.
It still needs **synchronized left/right RGB images and actual stereo
calibration**. If aligned depth is already available, skip this section and
continue with :ref:`mapping-rgbd-source`.

RAFT-Stereo estimates pixel displacement between the views, which cannot be
passed directly to RTAB-Map as millimetre depth. Follow these stages:

.. code-block:: text

   Stereo images or side-by-side video + stereo.yaml
     → prepare_stereo.py: rectify and arrange pairs → prepared/
     → Official RAFT-Stereo demo: estimate pixel displacement → flow/*.npy
     → raft_to_rgbd.py: convert using focal length and baseline → rgbd/
     → RTAB-Map: build a map from RGB-D input

Each tool is a single file: ``prepare_stereo.py`` reads, rectifies and
arranges stereo data; ``raft_to_rgbd.py`` converts RAFT output to RGB-D.
Neither tool solves calibration, runs RAFT or builds a map.

Step 1: Rectify Images Using Existing Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the RPent source root, use a Python environment with NumPy, SciPy, PyYAML
and OpenCV. Choose one input:

* **Images:** ``sequence/left/`` and ``sequence/right/`` with matching PNG,
  JPEG or TIFF filenames.
* **Stereo video:** local MP4, AVI, MKV or MOV with synchronized left and right
  views side by side, left first. The decoder must support the codec.

Copy :download:`stereo.yaml <../../../../toolkits/mapping/stereo.yaml>` and
fill in actual calibration from your camera or dataset:

.. code-block:: bash

   cp toolkits/mapping/stereo.yaml /path/to/stereo.yaml

This YAML is input to RPent's preparation utility; the official RAFT demo does
not read it. Fill in:

* ``image_size``: per-eye width and height. A 2560 × 720 side-by-side video
  needs ``[1280, 720]``.
* Left/right ``K`` and ``D``: OpenCV pinhole intrinsics and distortion.
* ``R`` and ``T_m``: the transform from left optical coordinates to right
  optical coordinates, ``X_right = R * X_left + T_m``, with translation in metres.

Use a horizontal pinhole stereo pair with the right camera to the right of the
left. For rectified input, supply rectified intrinsics, zero distortion,
identity rotation and ``T_m: [-baseline_m, 0, 0]``. Export fisheye images as
rectified pinhole pairs with the camera's tools first. This step uses existing
calibration to rectify images; it does not estimate calibration from the input.

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
visualization. Use a fresh ``flow/`` directory. These files are not yet the
millimetre depth required by RTAB-Map.

Step 3: Convert to Depth for RTAB-Map
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Continue with :ref:`mapping-rgbd-source`: point ``rtabmap_rgbd.ini`` at this
directory's RGB images, depth and calibration, use the output camera pose, and
run the same RTAB-Map mapping commands.

.. _mapping-rgbd-source:

Connect Depth to RTAB-Map
~~~~~~~~~~~~~~~~~~~~~~~~~

For SDK or RAFT depth, use :download:`rtabmap_rgbd.ini <../../../../toolkits/mapping/rtabmap_rgbd.ini>`,
which includes the common mapping parameters. From the RPent source root, copy
it to a fresh output directory with the commands below. See
:ref:`mapping-config` for parameter descriptions.
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

For SDK depth, use calibration for the exported RGB images and the actual
camera pose; ``local_transform.txt`` is not required.

Continue with the same commands in :ref:`mapping-build`. Here, ``raw.db``
stores RGB-D frames and ``-odom`` recomputes visual odometry from RGB-D.
When comparing depth sources, keep images, resolution, sampling, calibration,
camera pose, mapping and filtering settings identical.

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

The trajectory is estimated from visual odometry and refined by map
optimization. It is not a recording of wheel odometry or ground-truth motion,
and it need not contain a pose for every input frame. Exported geometry and
optimized poses use a local map frame, not global geographic coordinates.
The files provide a map and a reconstructed path; robot localization,
navigation configuration and motion control require separate integration.

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

Each row of ``trajectory_poses.txt`` contains
``stamp x y z qx qy qz qw``, with positions in metres and x-forward,
y-left, z-up axes. Review tracking loss and loop closures in the logs.
File existence and successful trajectory export do not establish map accuracy.
Before navigation, validate scale, obstacles, coverage and unknown areas
against measurements. A standalone grid does not contain the images needed for
visual localization.

For missing executables or Qt errors, check the standalone installation and
``QT_QPA_PLATFORM``. For missing pairs or dimension errors, check the input and
its calibration. RAFT GPU memory use depends on per-frame resolution; ``--step``
reduces total work, not that memory requirement. If reconstruction is empty or
fragmented, inspect the native command output and database for tracking loss,
blur, weak texture or calibration errors. Preparation failures leave partial
files for inspection; use a new output directory when retrying.
