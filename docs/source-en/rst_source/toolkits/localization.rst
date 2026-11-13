Localization and Clearance
==========================

The localization toolkit finds where a camera frame sits inside a built map
and reports whether the floor ahead is clear. It runs as an RPC service with a
transport-independent client, and ships offline commands for replay
evaluation and for clearance from an archived pose. The map comes from
:doc:`mapping`; grounding labels are optional and come from :doc:`grounding`.

Prerequisites
-------------

Localization and clearance need a CUDA GPU, the ``localize`` extra, and a map
with depth, gravity, and a floor plane:

.. code-block:: bash

   uv pip install -e '.[localize]'

DISK and LightGlue weights download on first use. Pass
``--camera-height-m`` when the map has no metric scale and you want distances
in metres; see :ref:`workspace-mapping-frames`.

Start the Localization Service
------------------------------

``toolkits.localization.server`` loads one built map, estimates the pose of a
query frame, and checks a corridor from that pose. Its ``--map-path`` is the
directory a build wrote, for example ``/path/to/maps/workspace``.

.. code-block:: bash

   python -m toolkits.localization.server \
       --map-path /path/to/maps/workspace \
       --transport http --host 127.0.0.1 --port 8117

Service Clients
---------------

``toolkits.localization.client`` works against a local or remote service:

.. list-table::
   :header-rows: 1

   * - Call
     - Inputs
     - Returns
   * - ``LocalizeClient.localize(frame, ...)``
     - An RGB frame at the map's resolution.
     - ``status``, the camera-to-world pose, the keyframe used, and inliers.
   * - ``LocalizeClient.check_path(frame=... | pose=..., ...)``
     - Exactly one RGB frame or one 4x4 camera-to-world pose, plus corridor
       parameters.
     - The clearance status and its measurements.

A map path is always the directory a build created, never a file inside it.

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

   python -m toolkits.localization.localize \
       --map-path /path/to/maps/workspace \
       --evaluate --query-stride 10

The replay hides the query frame and archived cameras within 0.1 map units and
10 degrees of it, so the query cannot match itself or an identical duplicate.
``--exclusion N`` instead (or additionally) hides archived frames within N
frame indices of the query, which on a moving camera also removes useful
overlapping views — use it as a stress test, not as the primary success rate.

Localize one archived or live frame:

.. code-block:: bash

   python -m toolkits.localization.localize \
       --map-path /path/to/maps/workspace \
       --image /path/to/live_frame.png

   # Or an archived frame by index.
   python -m toolkits.localization.localize \
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
It does not plan a path or move the robot: it returns measurements to the
caller.

The check needs three things from the map: the world point cloud, the
estimated gravity direction, and the fitted floor plane. A map without them
reports ``unusable`` instead of guessing.

.. list-table::
   :header-rows: 1

   * - Argument
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

``min_height`` defaults to 0.05 and ``max_height`` to 4.0. They are in the same
unit as ``length``.

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

**From the command line.** An archived camera pose is enough for an offline
check:

.. code-block:: bash

   python -m toolkits.localization.clearance \
       --map-path /path/to/maps/workspace \
       --frame-index 40 \
       --length 2.0 --half-width 0.5 --unit camera_heights

It prints the same JSON the RPC returns; ``--output`` writes it to a file, and
``--camera-height-m`` anchors metres on a map without a metric scale.

**Over RPC.** ``LocalizeClient.check_path`` accepts either an RGB frame or a
4x4 camera-to-world pose in the map frame, and forwards the corridor
arguments:

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

Units
-----

Clearance distances default to ``camera_heights``, which needs no external
scale:

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
Camera heights stay comparable between maps; model units do not. The map-side
scale contract is described in :ref:`workspace-mapping-frames`.

Limits and Troubleshooting
--------------------------

**No gravity or floor.** Clearance needs both. A short or nearly planar
recording may not give the estimator enough camera rotation to find gravity;
the map then reports ``unusable`` for clearance while localization still
works.

**An unmapped corridor is not clear.** The corridor bins are the evidence.
When the far end of the corridor has no points the map saw, the check reports
``unmapped`` with the distance where coverage ends. Expand the map or shorten
``length``.

**Noise near the camera can be reported as blocked.** Clearance counts points
in a volume, and the merged cloud carries each frame's reconstruction noise. On
both a 300-frame loop recording and a 120-frame one-way sweep, most default
checks reported ``blocked`` because a handful of points sat within a few
millimetres of the camera's ground projection, while others lay above the
0.05-camera-height obstacle floor. The defaults are small in absolute terms:
``length`` and ``half_width`` are multiples of the estimated camera height, so
a low camera also gives a short, narrow corridor. Inspect the top-down report,
and treat ``blocked`` as the conservative answer rather than a measurement of
the nearest object. On the CLI or in an RPC call, raise ``min_height`` to
ignore floor roughness before concluding that a route is blocked.

**Localization is lost.** Low-texture or motion-blurred frames, a viewpoint
outside the recorded area, or a resolution/orientation that differs from the
map frames all produce ``lost``. Rotate toward a mapped area, keep the query
frame at the map's resolution, and use exhaustive search for the first check.
Only 22 inliers were needed to lose one replay query, so a confidence margin
matters.

**The service is slow on the first call.** DISK and LightGlue load lazily. The
first localization can take tens of seconds; later calls are much faster. The
clearance check loads and dedupes the point cloud once per server process.

**Distances look wrong.** Check ``unit``, ``camera_height``, and
``meters_per_unit`` in the result. A ``m`` request on an arbitrary-scale map
is refused rather than guessed; pass ``--camera-height-m`` if you know the
real camera height. The estimator's floor is a plane fit, so a tilted or
sparsely observed floor shifts every height by the same bias.
