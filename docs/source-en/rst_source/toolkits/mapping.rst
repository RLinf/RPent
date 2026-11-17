Map Construction
================

The mapping toolkit turns a walkthrough of a workspace into a persistent 3D
map. It runs as an RPC service with a transport-independent client, and ships
an offline builder that segments a recording and a report that renders a
labeled map. Object labels and localization are separate toolkits; see
:doc:`grounding` and :doc:`localization`.

The map is a reconstruction, not a floor plan: its geometry is correct up to a
similarity unless base poses anchor it, and every distance inherits the scale
the map records. :ref:`workspace-mapping-frames` states which frame and scale a
result uses.

Prerequisites
-------------

Mapping needs a CUDA GPU. One mapping session holds 300 frames by default,
about what 24 GB of GPU memory reconstructs; the video builder segments a
longer recording instead of asking for a larger device.

.. code-block:: bash

   uv pip install -e '.[lingbot-map,map-view]'

Reconstruction needs one checkpoint. Download it once and export its path:

.. code-block:: bash

   # Add HF_ENDPOINT=https://hf-mirror.com for mainland China users.
   hf download robbyant/lingbot-map lingbot-map.pt --local-dir /path/to/checkpoints
   export LINGBOTMAP_CHECKPOINT_PATH=/path/to/checkpoints/lingbot-map.pt

Start the Mapping Service
-------------------------

``toolkits.mapping.server`` owns one mapping session at a time (start, push
frames, build, stop) and writes each map below ``LINGBOTMAP_OUTPUT_DIR``.

.. code-block:: bash

   LINGBOTMAP_OUTPUT_DIR=/path/to/maps \
       python -m toolkits.mapping.server \
       --transport http --host 127.0.0.1 --port 8116

With ``LINGBOTMAP_CHECKPOINT_PATH`` exported, ``build`` returns camera poses
and per-frame depth from the GCT streaming model plus a world point cloud
unprojected from that depth. Without it, the server archives frames and poses
instead: the build path is identical, but the map has no depth and cannot
support localization or clearance. Export the variable in the environment that
starts the server.

Service Clients
---------------

``toolkits.mapping.client`` works against a local or remote service:

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

A map path is always the directory a build created, never a file inside it:
``map.json`` holds the metadata contract, and the frames and ``.npy`` arrays
sit beside it. Clients read frames and depth over RPC, and
``save_annotations`` writes grounding results back, so a client does not need
filesystem access to the machine that built the map.

Build Maps from a Video
-----------------------

``toolkits.mapping.build_map`` samples a recording and builds one map per
segment, so a walkthrough longer than the frame cap does not need a larger GPU:

.. code-block:: bash

   python -m toolkits.mapping.build_map \
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

   python -m toolkits.mapping.build_map \
       --video workspace_walkthrough.mp4 \
       --map-name from_video \
       --endpoint http://127.0.0.1:8116

Hold the camera roughly forward-facing while walking. The command pushes no
base poses, so the map stays in the model frame at arbitrary scale. Run
:mod:`toolkits.grounding.ground_map` on the resulting directory to add object
annotations.

.. _workspace-mapping-view:

View a Built Map
----------------

``toolkits.mapping.view`` renders a grounded map into a report directory that
opens without loading the reconstruction arrays:

.. code-block:: bash

   python -m toolkits.mapping.view --map-path /path/to/maps/workspace

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

.. _workspace-mapping-frames:

Frames and Scale
----------------

**Frame.** Camera poses, the point cloud, and localization results all use the
frame named in ``map.json``, which is the reconstruction's ``model`` frame.
The gravity direction and floor plane are also in that frame.

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
units and a metric distance would be meaningless. Localization and clearance
report distances in map units unless a scale is known; see :doc:`localization`
for the units a clearance check accepts.

.. _workspace-mapping-limits:

Limits
------

**The map stops at the frame cap.** A session refuses frames after
``--max-frames`` (300 by default, about what 24 GB of GPU memory reconstructs).
A 400-frame build ran out of memory on a 24 GB device.
``toolkits.mapping.build_map`` splits a longer recording into segments
automatically. A live session must stop at the cap instead, so lower ``--fps``,
raise ``--max-frames`` on a larger device, or rebuild from the video. After an
out-of-memory failure, restart the mapping service before the next build.

**Segments are independent maps.** Each segment is reconstructed on its own and
keeps its own model frame, so poses from one do not transfer to another.

**A frame-archive map has no geometry.** Built without the GCT checkpoint, the
map holds frames and poses only. It can be grounded in 2D and localized
against, but it has no depth, gravity, or floor plane, so clearance reports
``unusable``.
