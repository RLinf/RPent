Map Construction
================

A walkthrough of a workspace can become a reusable 3D map. RPent builds that
map from frames a robot or camera records, annotates it with object labels,
localizes a live camera frame inside it, and reports whether the floor just
ahead is clear before a mobile base drives. The mapping service, the grounding
and localization tools, and the clearance check are robot-agnostic; a robot's
toolkit wires them in when the services are reachable.

This page covers:

1. Installing the extras and downloading the checkpoints.
2. Starting the mapping, perception, and localization services.
3. Building maps from a recorded video, in segments when it is long.
4. Grounding object labels on a built map and inspecting the result.
5. Localizing an archived or live frame, and evaluating localization.
6. Checking obstacle clearance from the terminal, over RPC, or with the agent
   tool.
7. Reading frames and units correctly, and diagnosing common failures.

The map is a reconstruction, not a floor plan: its geometry is correct up to a
similarity unless base poses anchor it, and every distance inherits the scale
the map records. The sections below state which frame and unit a result uses.

Prerequisites
-------------

Mapping, grounding, localization, and the clearance check need a CUDA GPU. One
mapping session holds 300 frames by default, about what 24 GB of GPU memory
reconstructs; ``rpent.tools.build_map`` builds a longer recording in segments
instead of asking for a larger device. Install the extras for the parts you
use:

.. list-table::
   :header-rows: 1

   * - Extra
     - Enables
   * - ``lingbot-map``
     - Reconstruction with the LingBot-Map GCT model instead of a frame archive
   * - ``grounding-sam2``
     - Grounding DINO detection and SAM 2 masks behind the perception service
   * - ``localize``
     - DISK + LightGlue localization and map-based clearance checks
   * - ``map-view``
     - The offline report of a grounded map

.. code-block:: bash

   uv pip install -e '.[lingbot-map,grounding-sam2,localize,map-view]'

Reconstruction needs one checkpoint. Download it once and export its path:

.. code-block:: bash

   # Add HF_ENDPOINT=https://hf-mirror.com for mainland China users.
   hf download robbyant/lingbot-map lingbot-map.pt --local-dir /path/to/checkpoints
   export LINGBOTMAP_CHECKPOINT_PATH=/path/to/checkpoints/lingbot-map.pt

Grounding needs two more checkpoints. Grounding DINO must use the original
release weights — the transformers-format ``model.safetensors`` on Hugging
Face does not load.

.. code-block:: bash

   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
       -P /path/to/checkpoints
   hf download facebook/sam2.1-hiera-large sam2.1_hiera_large.pt \
       --local-dir /path/to/checkpoints

   export GDINO_CHECKPOINT_PATH=/path/to/checkpoints/groundingdino_swint_ogc.pth
   export SAM2_CHECKPOINT_PATH=/path/to/checkpoints/sam2.1_hiera_large.pt

Keep the checkpoint filenames unchanged: the perception server selects the
SAM 2 configuration from the filename. Its first start also downloads the
``bert-base-uncased`` tokenizer, so that start needs network access.

Start the Services
------------------

Each capability runs as its own RPC service, so a GPU machine can host them
for a robot that has only a network connection. Start the ones you need.

The services live in the top-level ``toolkits`` package, beside ``robots``.

**Mapping.** ``toolkits.mapping.server`` owns one mapping session at a time
(start, push frames, build, stop) and writes each map below
``LINGBOTMAP_OUTPUT_DIR``.

.. code-block:: bash

   LINGBOTMAP_OUTPUT_DIR=/path/to/maps \
       python -m toolkits.mapping.server \
       --transport http --host 127.0.0.1 --port 8116

With ``LINGBOTMAP_CHECKPOINT_PATH`` exported, ``build`` returns camera poses
and per-frame depth from the GCT streaming model plus a world point cloud
unprojected from that depth. Without it, the server archives frames and poses
instead: the tool flow is identical, but the map has no depth and cannot
support localization or clearance. Export these variables in the environment
that starts the server.

**Perception.** ``toolkits.grounding.server`` wraps Grounding DINO
(open-vocabulary detection) and SAM 2 (mask refinement) behind one
``gdino_sam2.detect_segment`` call.

.. code-block:: bash

   python -m toolkits.grounding.server \
       --transport http --host 127.0.0.1 --port 8115

**Localization and clearance.** ``toolkits.localization.server`` loads one
built map, estimates the pose of a query frame, and checks a corridor from
that pose. Its ``--map-path`` is the directory a build wrote, for example
``/path/to/maps/workspace``. DISK and LightGlue download their weights on
first use.

.. code-block:: bash

   python -m toolkits.localization.server \
       --map-path /path/to/maps/workspace \
       --transport http --host 127.0.0.1 --port 8117

Pass ``--camera-height-m`` when the map has no metric scale and you want
distances in metres; see :ref:`workspace-mapping-frames`.

Service Clients
~~~~~~~~~~~~~~~

Each service has a transport-independent client in the ``toolkits`` package,
so the same call works against a local or remote server:

.. list-table::
   :header-rows: 1

   * - Call
     - Inputs
     - Returns
   * - ``LingBotMapClient.start_session(map_name=...)``
     - A map name: one path segment of letters, digits, ``_``, ``.``, and
       ``-``.
     - The session id.
   * - ``LingBotMapClient.push_frame(rgb, pose=None, timestamp_s=...)``
     - An RGB ``[H,W,3]`` ``uint8`` array or PNG bytes; optionally the base
       pose ``[x, y, yaw, roll, pitch, yaw_head]`` in the robot map frame;
       seconds on the recording clock.
     - Frames accepted so far.
   * - ``LingBotMapClient.build()``
     - Nothing.
     - ``map_path``, the frame count, and the ``map.json`` metadata; the
       service writes the map directory on its own host.
   * - ``LingBotMapClient.load_frame()``, ``load_depth()``
     - A map path and a frame index.
     - The archived PNG bytes, and the depth and confidence arrays.
   * - ``GdinoSam2Client.detect_segment(image, labels=...)``
     - An RGB frame and at most 16 labels.
     - Scored boxes and one instance mask per detection.
   * - ``LocalizeClient.localize(frame, ...)``
     - An RGB frame at the map's resolution.
     - ``status``, the camera-to-world pose, the keyframe used, and inliers.
   * - ``LocalizeClient.check_path(frame=... | pose=..., ...)``
     - Exactly one RGB frame or one 4x4 camera-to-world pose, plus corridor
       parameters.
     - The clearance status and its measurements.

A map path is always the directory a build created, never a file inside it:
``map.json`` holds the metadata contract, and the frames and ``.npy`` arrays
sit beside it. Clients read frames and depth over RPC, and
``save_annotations`` writes grounding results back, so a client does not need
filesystem access to the machine that built the map.

Build Maps from a Video
-----------------------

``rpent.tools.build_map`` samples a recording and builds one map per segment,
so a walkthrough longer than the frame cap does not need a larger GPU:

.. code-block:: bash

   python -m rpent.tools.build_map \
       --video workspace_walkthrough.mp4 \
       --map-name workspace \
       --endpoint http://127.0.0.1:8116

The command samples five frames per second (``--fps``) and closes a session
after ``--segment-frames`` frames, 300 by default, which must match the
server's ``--max-frames``. A recording that fits in one segment writes
``maps/workspace``; a longer one writes ``maps/workspace_00``,
``maps/workspace_01``, and so on, one map per segment. Segments are
reconstructed independently and share no poses, so treat each as its own map.

Add ``--labels chair desk monitor --perception-endpoint
http://127.0.0.1:8115`` to ground objects on every built map; see
:ref:`workspace-mapping-grounding`. The command prints a JSON summary:

.. code-block:: json

   {
     "video": "workspace_walkthrough.mp4",
     "map_name": "workspace",
     "fps": 5.0,
     "segment_frames": 300,
     "maps": [
       {
         "map_name": "workspace_00",
         "map_path": "/path/to/maps/workspace_00",
         "frames": 300,
         "frames_pushed": 300,
         "backend": "gct_stream",
         "has_depth": true
       }
     ]
   }

``--output summary.json`` writes the same document to a file. The map name
must be a single path segment of letters, digits, ``_``, ``.``, and ``-``,
starting with a letter or digit; numbered segments append ``_00``, ``_01``,
and so on. The service refuses to build into an existing map directory, so
stale frames and annotations cannot mix with a new build. Choose a new name,
or delete the old directory first.

A built map directory looks like this:

.. code-block:: text

   maps/workspace/
   ├── frames/                 archived source frames (000000.png, ...)
   ├── poses.json              per-frame base poses that were pushed
   ├── keyframe_index.json     which frames the reconstruction treats as keyframes
   ├── map.json                the metadata contract described below
   ├── depth.npy               per-frame depth          (GCT backend)
   ├── world_points.npy        per-frame world points   (GCT backend)
   ├── world_points_conf.npy   per-point confidence      (GCT backend)
   ├── cameras_c2w.npy         camera-to-world poses     (GCT backend)
   ├── cameras_c2w_robot_frame.npy      aligned poses    (with alignment)
   ├── world_points_robot_frame.npy     aligned points   (with alignment)
   ├── objects.json            grounded instances        (after grounding)
   ├── annotations/            per-frame mask overlays   (after grounding)
   ├── semantic_points.npz     labelled 3D points        (after grounding)
   ├── semantic_points.ply     the same cloud for external viewers (after grounding)
   ├── evaluation.json         grounding consistency     (after grounding)
   └── localization_eval.json  replay metrics            (after evaluation)

``map.json`` is the contract later stages read. The fields that change how a
result should be interpreted are:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``frame``
     - Frame of the camera poses. ``model`` means the reconstruction's own
       frame.
   * - ``backend``
     - ``gct_stream`` for a reconstruction, ``frame_archive`` for frames only.
   * - ``has_depth``, ``has_world_points``
     - Whether the reconstruction arrays exist.
   * - ``gravity``
     - Estimated up direction (``status``, ``up``) in the model frame.
   * - ``floor``
     - Floor plane in the model frame: ``normal``, ``offset``, and the
       distance from the floor to the camera (``camera_height``).
   * - ``scale``
     - ``status`` ``metric`` or ``arbitrary``; ``meters_per_unit`` when known.
   * - ``cameras``, ``camera_to_world``
     - Intrinsics and extrinsics, and the file holding camera-to-world poses.
   * - ``alignment``
     - The fit to pushed base poses, or ``unavailable`` with the reason.
   * - ``keyframes``
     - The keyframe index file and its count.

Building from a robot
~~~~~~~~~~~~~~~~~~~~~

``rpent.tools.mapping.MappingMixin`` adds three agent tools to a robot
toolkit and registers each one only when its client is attached:
``build_map`` with ``LingBotMapClient``, ``detect_objects`` with
``GdinoSam2Client``, and ``check_clearance`` with ``LocalizeClient``. The
toolkit supplies one synchronized RGB frame per step through ``_map_frame()``,
and the base pose when the robot reports odometry; ``build_map`` opens a
session, asks the operator to teleoperate, and builds the map when they reply.

No toolkit in this repository wires those tools yet, and the teleoperation
path has not been validated here. The video command above is the supported way
to build a map; the service protocol and map format are the same.

Test the Mapping Service Without a Robot
----------------------------------------

The mapping server answers plain RPC calls, so a few lines of Python confirm it
is alive before wiring it into a run. Pushing synthetic frames exercises the
session and build paths:

.. code-block:: python

   import numpy as np
   from rpent.utils.rpc.http_rpc import HttpRpcClient
   from toolkits.mapping.client import LingBotMapClient

   client = LingBotMapClient(HttpRpcClient("http://127.0.0.1:8116"))
   client.start_session(map_name="smoke_test")
   for index in range(10):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
       frame[:, :, index % 3] = 60 + index  # synthetic frames
       client.push_frame(frame, pose=None, timestamp_s=index * 0.5)
   result = client.build()
   print(result.map_path, result.frames)  # .../maps/smoke_test 10

A successful build prints the map directory and the frame count. Call
``client.stop_session()`` instead of ``build`` to discard a session.

To produce a map worth inspecting, build it from a walkthrough video with the
offline command described above:

.. code-block:: bash

   python -m rpent.tools.build_map \
       --video workspace_walkthrough.mp4 \
       --map-name from_video \
       --endpoint http://127.0.0.1:8116

Hold the camera roughly forward-facing while walking. The command pushes no
base poses, so the map stays in the model frame at arbitrary scale. Run the
offline grounder below on the resulting directory to add object annotations.

.. _workspace-mapping-grounding:

Ground Objects
--------------

Grounding can run on its own, after any build. ``rpent.tools.ground_map``
reads a built map, picks evenly spaced keyframes (or the indices passed with
``--keyframes``), and writes the annotations back into the map:

.. code-block:: bash

   python -m rpent.tools.ground_map \
       --map-path /path/to/maps/workspace \
       --labels chair desk monitor \
       --perception-endpoint http://127.0.0.1:8115 \
       --max-keyframes 12

On a reconstructed map, each mask is mapped through the server's resize and
center crop onto the depth grid, unprojected with the frame's intrinsics and
camera pose, and associated with same-label observations from other keyframes.
A map without depth keeps 2D annotations only, and ``--allow-2d`` is required
to write them.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - What it does
   * - ``--max-keyframes``
     - 12
     - How many evenly spaced keyframes to ground.
   * - ``--keyframes``
     - evenly spaced
     - Explicit frame indices, comma or space separated.
   * - ``--min-score``
     - 0.35
     - Lowest detection score to keep.
   * - ``--min-mask-pixels``
     - 32
     - Smallest mask, in source pixels.
   * - ``--min-confidence``
     - 1.0
     - Lowest depth confidence for a lifted pixel.
   * - ``--merge-factor``
     - 0.75
     - Same-label centroids within this multiple of their typical radius
       become one instance.
   * - ``--output-dir``
     - the map
     - Write the artifacts elsewhere and leave the map untouched.
   * - ``--allow-2d``
     - off
     - Keep 2D annotations on a map without depth.

``evaluation.json`` reports per-label counts and the median distance between
the nearest same-label observations in different frames. That distance
measures cross-frame consistency; precision and recall need independent
ground-truth labels and are not reported.

Two limits follow from the model geometry and the detector. Masks that lie
entirely outside the model's center crop have no depth and stay 2D detections
— a portrait frame loses rows above and below the square crop even though the
archived frame keeps them. Grounding DINO can also return several overlapping
masks for one object; association merges observations only when their 3D
centroids fall within the radius threshold, so a sparse keyframe set can leave
more than one entry for the same object. Label matching is case-insensitive
and slugs must stay distinct, so ``Chair`` and ``chair`` cannot both be
requested.

View a Grounded Map
-------------------

``rpent.tools.map_view`` renders a grounded map into a report directory that
opens without loading the reconstruction arrays:

.. code-block:: bash

   python -m rpent.tools.map_view --map-path /path/to/maps/workspace

It writes ``report/semantic_topdown.png`` (top-down view oriented by the
estimated gravity or, without one, by the camera trajectory),
``report/semantic_3d.png`` (isometric view with the camera path),
``report/detections.html`` (per-label counts, object and detection tables, and
every per-frame overlay), and ``report/objects.csv`` (one row per instance).
The top-down view centres the camera path in ``x``/``y`` and puts the
estimated floor at ``z = 0``. Axes and tables are labelled in map units; the
scale note states how many metres one unit is when the map records a metric
anchor. ``--max-points`` bounds the points drawn in each figure and
``--output-dir`` writes the report outside the map directory.

Localize a Frame
----------------

Localization preprocesses a query frame exactly as the mapping service did,
matches it against archived keyframes with DISK + LightGlue, lifts the matched
keyframe pixels with that keyframe's depth, intrinsics, and camera pose, and
solves PnP RANSAC for the query's camera-to-world pose in the frame recorded in
``map.json``. It closes the gap that LingBot-Map leaves: reconstruction does
not relocalize a later run.

Replay archived frames as held-out queries and write the metrics to
``localization_eval.json``:

.. code-block:: bash

   python -m rpent.tools.localize \
       --map-path /path/to/maps/workspace \
       --evaluate --query-stride 10

The replay hides the query frame and archived cameras within 0.1 map units and
10 degrees of it, so the query cannot match itself or an identical duplicate.
``--exclusion N`` instead (or additionally) hides archived frames within N
frame indices of the query, which on a moving camera also removes useful
overlapping views — use it as a stress test, not as the primary success rate.

Localize one archived or live frame:

.. code-block:: bash

   python -m rpent.tools.localize \
       --map-path /path/to/maps/workspace \
       --image /path/to/live_frame.png

   # Or an archived frame by index.
   python -m rpent.tools.localize \
       --map-path /path/to/maps/workspace --frame-index 240

Exhaustive search is the reliable cold-start mode and checks every model
keyframe. While tracking, pass ``--prior-frame`` with a frame from the previous
pose and only a window around the nearest keyframe is matched; the Python API
also accepts ``prior_pose`` to rank keyframes by pose distance. Without a
prior, a 32x32 global color descriptor shortlists the ``top_k`` keyframes
(``--mode retrieval``). Retrieval is only a hint: every shortlisted candidate
still has to pass PnP, and the replay below shows it missing keyframes that
exhaustive search finds.

A result is ``ok`` only with at least 25 PnP inliers, an inlier ratio of at
least 0.10, and an inlier RMS of at most 10 px. It is ``ambiguous`` when a
second, geometrically distinct pose has at least half the best pose's inliers
and differs by more than 1.0 map unit or 20 degrees; that means the frame alone
cannot choose between two places. Otherwise it is ``lost``. Only ``ok`` counts
as success. The reported pose is the camera-to-world transform in the map
frame; convert it to the base frame with the camera mount transform.

Localize over RPC
~~~~~~~~~~~~~~~~~

Start the service on the machine holding the map and GPU, then send one RGB
frame per call:

.. code-block:: bash

   python -m toolkits.localization.server \
       --map-path /path/to/maps/workspace \
       --transport http --host 0.0.0.0 --port 8117

.. code-block:: python

   import imageio
   from rpent.utils.rpc.http_rpc import HttpRpcClient
   from toolkits.localization.client import LocalizeClient

   client = LocalizeClient(HttpRpcClient("http://127.0.0.1:8117"))
   frame = imageio.imread("live_frame.png")  # RGB [H,W,3]
   result = client.localize(frame, exhaustive=True)
   print(result.status, result.keyframe, result.inliers)
   print(result.pose)  # camera-to-world 4x4 in the map frame

Measured accuracy
~~~~~~~~~~~~~~~~~

On the 366-frame workspace recording (183 model keyframes, every tenth frame
as a query, exhaustive search, 19 queries), 18 were localized and all 18 were
accurate; one was lost. Translation error had a median of 0.047 map units
(p75 0.084, max 0.099) and rotation error a median of 0.93 degrees (p75 1.20,
max 3.03). Latency had a median of 9.3 s and a maximum of 20.2 s on an RTX
4090, including the first query that populates the feature cache. Retrieval
with eight candidates localized 15 of 19 accurately, flagged 2 as ambiguous and
lost 2, at a median of 0.58 s. On a second, 143-frame courthouse walkthrough
(29 queries, exhaustive), 28 were localized and accurate, 1 was ambiguous, and
none were lost; translation error had a median of 0.021 units and rotation
error a median of 0.80 degrees (max 12.9), at a median latency of 4.5 s. Both
maps have arbitrary scale, so these distances are model units, not metres.

These numbers come from replaying the same recordings the maps were built
from, with near-identical views hidden. Cross-session localization (a new
recording of the same space) is not yet validated. Repetitive or symmetric
scenes can make two places look equally supported, which the service reports
as ``ambiguous`` rather than picking one. Low-texture frames may not yield
enough inliers; in the workspace replay the lost query had 22. Exhaustive cost
grows with the number of keyframes, so a prior or retrieval shortlist is worth
using once the robot has an approximate pose.

Check Obstacle Clearance
------------------------

Clearance answers one question before a forward drive: is the floor just
ahead clear on the map? It localizes the current camera frame, then measures a
corridor that starts at the camera centre, follows the camera's optical axis
projected onto the floor, and is turned counter-clockwise by ``heading_deg``.
The corridor is ``length`` long and ``2 * half_width`` wide, and only points
between ``min_height`` and ``max_height`` above the floor count as obstacles.
It does not plan a path or move the robot: it returns measurements, and the
agent decides what to do.

The check needs three things from the map: the world point cloud, the
estimated gravity direction, and the fitted floor plane. A map without them
reports ``unusable`` instead of guessing.

.. list-table::
   :header-rows: 1

   * - Tool argument
     - Default
     - What it does
   * - ``heading_deg``
     - 0
     - Turns the corridor counter-clockwise from the camera's view direction.
       Use it when the camera does not point along the drive direction.
   * - ``length``
     - 2.0
     - How far ahead to check.
   * - ``half_width``
     - 0.75
     - Half the corridor width; set it to the base's half-width plus clearance.
   * - ``unit``
     - ``camera_heights``
     - Unit for all three distances: ``camera_heights``, ``map_units``, or
       ``m``.

``min_height`` (0.05) and ``max_height`` (4.0) are not exposed by the tool;
the RPC and the CLI accept them. They are in the same unit as ``length``.

With the agent tool. ``check_clearance`` is read-only and registered only
when a localization service is connected. The first call localizes
exhaustively; later calls pass the previous pose and keyframe as a prior,
which narrows the search and keeps consecutive checks on one estimate.

.. code-block:: text

   check_clearance(heading_deg=0, length=2.0, half_width=0.75,
                   unit="camera_heights")

A result has one of these statuses:

.. list-table::
   :header-rows: 1

   * - Status
     - Meaning
   * - ``clear``
     - Every along-path bin has mapped geometry and fewer than three points
       reach into the corridor. Only this status permits a forward drive.
   * - ``blocked``
     - At least three mapped points lie inside the corridor. ``nearest`` gives
       the closest point's distance, lateral offset, and height above the
       floor, plus its object label when the map was grounded.
   * - ``unmapped``
     - Part of the corridor has no mapped geometry. ``observed_distance`` is
       where the mapped part ends. This is not clear: drive blind only with a
       second source of evidence.
   * - ``unlocalized``
     - The frame did not localize, so there is no pose to check. Stop, rotate
       the camera toward mapped space, and call again.
   * - ``unusable``
     - The map lacks gravity, a floor plane, a metric anchor for ``m``, or a
       usable view direction. ``reason`` says which.

The result also reports ``unit_scale`` (how many of the requested units one
map unit spans), ``camera_height`` and ``meters_per_unit`` when known,
``coverage`` (the fraction of along-path bins with mapped geometry), the pose
it used, and the keyframe it localized against.

From the command line. An archived camera pose is enough for an offline check:

.. code-block:: bash

   python -m rpent.tools.clearance \
       --map-path /path/to/maps/workspace \
       --frame-index 40 \
       --length 2.0 --half-width 0.5 --unit camera_heights

It prints the same JSON the RPC returns; ``--output`` writes it to a file, and
``--camera-height-m`` anchors metres on a map without a metric scale.

Over RPC. ``LocalizeClient.check_path`` accepts either an RGB frame or a 4x4
camera-to-world pose in the map frame, and forwards the corridor arguments:

.. code-block:: python

   from rpent.utils.rpc.http_rpc import HttpRpcClient
   from toolkits.localization.client import LocalizeClient

   client = LocalizeClient(HttpRpcClient("http://127.0.0.1:8117"))
   result = client.check_path(frame, exhaustive=True, length=2.0)
   print(result.status, result.nearest)

   # Or check from a pose that came from another source.
   result = client.check_path(pose=camera_to_world, half_width=0.5)

Pass exactly one source. ``prior_index``, ``prior_pose``, and ``exhaustive``
apply only to an image, because they steer localization; a pose needs none of
them.

On a 120-frame one-way sweep of a workspace, checking archived frame 15
through the service localized it against keyframe 15 (1467 inliers, 0.90
inlier ratio, 4.4 px RMS) and reported ``clear`` with full coverage. The first
localization took about 10 s while the models loaded; an immediate repeat took
about 6 s.

.. _workspace-mapping-frames:

Frames and Units
----------------

**Frame.** Camera poses, the point cloud, and localization results all use the
frame named in ``map.json``, which is the reconstruction's ``model`` frame.
The gravity direction and floor plane are also in that frame. Clearance
rotates points into a gravity-aligned frame internally but returns the pose it
used in the model frame, so the next check can pass it back as a prior.

When every pushed frame carries a base pose, the mapping server also fits a
similarity from the reconstructed camera trajectory to those positions and
writes ``alignment``: ``scale`` in robot-map units per model unit,
``rotation``, ``translation``, ``rms_error_m``, and how many frames were used
and how many had no pose. The robot-frame copies of the poses and cloud are
saved separately; the model-frame arrays stay authoritative for localization
and clearance. The fitted frame puts the camera trajectory on ``z = 0`` with
``+z`` up, so the floor lies below ``z = 0``.

**Scale.** A map is metric only when base poses anchored it
(``scale.status`` is ``metric``). Otherwise its distances are arbitrary model
units and a metric distance would be meaningless. Clearance therefore defaults
to ``camera_heights``, which needs no external scale:

.. list-table::
   :header-rows: 1

   * - Unit
     - Meaning
   * - ``camera_heights``
     - Multiples of the distance from the floor to the camera.
   * - ``map_units``
     - The reconstruction's own units; comparable across one map.
   * - ``m``
     - Metres. Requires a metric scale in the map, or
       ``--camera-height-m`` on the localization server, which anchors the
       scale from the real camera height.

The corridor is also reported in the unit you passed, so a 2.0-camera-height
request is about one base length for a base whose camera is at chest height.
Camera heights stay comparable between maps; model units do not.

.. _workspace-mapping-limits:

Limits and Troubleshooting
--------------------------

**The map stops at the frame cap.** A session refuses frames after
``--max-frames`` (300 by default, about what 24 GB of GPU memory reconstructs).
A 400-frame build ran out of memory on a 24 GB device. ``rpent.tools.build_map``
splits a longer recording into segments automatically. A live session must stop
at the cap instead, so lower ``--fps``, raise ``--max-frames`` on a larger
device, or rebuild from the video. After an out-of-memory failure, restart the
mapping service before the next build.

**No gravity or floor.** Clearance needs both. A short or nearly planar
recording may not give the estimator enough camera rotation to find gravity;
the map then reports ``unusable`` for clearance while localization still
works. A frame-archive map (built without the GCT checkpoint) has no geometry
at all and can only be localized against, never checked.

**An unmapped corridor is not clear.** The corridor bins are the evidence.
When the far end of the corridor has no points the map saw, the check reports
``unmapped`` with the distance where coverage ends. Expand the map or shorten
``length``.

**Noise near the camera can be reported as blocked.** Clearance counts points in a
volume, and the merged cloud carries each frame's reconstruction noise. On
both a 300-frame loop recording and a 120-frame one-way sweep, most default
checks reported ``blocked`` because a handful of points sat within a few
millimetres of the camera's ground projection, while others lay above the
0.05-camera-height obstacle floor. The defaults are small in absolute terms:
``length`` and ``half_width`` are multiples of the estimated camera height, so
a low camera also gives a short, narrow corridor. Inspect the top-down report,
and treat ``blocked`` as the conservative answer rather than a measurement of
the nearest object. On the CLI or in an RPC call, raise ``min_height`` to
ignore floor roughness before concluding that a route is blocked; the agent
tool keeps the shared defaults.

**Localization is lost.** Low-texture or motion-blurred frames, a viewpoint
outside the recorded area, or a resolution/orientation that differs from the
map frames all produce ``lost``. Rotate toward a mapped area, keep the query
frame at the map's resolution, and use exhaustive search for the first check.
Only 22 inliers were needed to lose one replay query, so a confidence margin
matters.

**The service is slow on the first call.** Disk, LightGlue, and the
perception models load lazily. The first localization can take tens of
seconds; later calls are much faster. The clearance check loads and dedupes
the point cloud once per server process.

**Distances look wrong.** Check ``unit``, ``camera_height``, and
``meters_per_unit`` in the result. A ``m`` request on an arbitrary-scale map
is refused rather than guessed; pass ``--camera-height-m`` if you know the
real camera height. The estimator's floor is a plane fit, so a tilted or
sparsely observed floor shifts every height by the same bias.
