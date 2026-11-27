.. _generic-toolkit:

Embodied RGB-D Toolkit
======================

RPent's read-only RGB-D perception tools look up the 3D surface position at
image pixels, segment objects, and measure the geometry of specified object
parts from recorded camera observations.

For example, if the planner selects a pixel on the side of a cup,
``back_project`` can return the 3D coordinates of that surface point.
Given that pixel or the description "white cup", ``segment`` can return a
candidate mask. To locate the center of the opening, use ``geometry`` to fit
the cup's upper inner ring. It returns the center when the evidence is
sufficient. Both the surface point and the ring center use metres in the world
frame identified by ``world_frame_id``.

Choose a Tool
-------------

.. list-table::
   :header-rows: 1
   :widths: 17 40 43

   * - Tool
     - What it does
     - Result
   * - ``back_project``
     - Reads the 3D position of visible surfaces at selected pixels or inside a rectangular box. It can keep only points within a height range in the world frame.
     - For selected points, the 3D coordinates in input order, with ``null`` for points with invalid depth or points excluded by the filter. For a box, the coordinates of one measured surface point and the 3D bounds of the visible surfaces inside the box.
   * - ``segment``
     - Segments a candidate mask of the target from a text description or from one pixel on the target.
     - When the target is found, the file names of the mask and overlay image, with the overlay attached for review. If enough points inside the mask have valid depth, also ``world_xyz``, the medians of their X, Y and Z coordinates.
   * - ``geometry``
     - Experimental. Locates the whole object, then fits a specified part and returns geometric quantities such as the center of a cup opening or plate face.
     - The quantities that pass the quality checks. Quantities that do not pass are listed in ``missing_geometry``.

Supported Geometry Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``geometry`` takes the quantity to measure in ``required_geometry`` and the
object part to look at in ``feature``. The quantities implemented so far are:

.. list-table::
   :header-rows: 1
   :widths: 32 28 40

   * - Quantity
     - ``required_geometry``
     - ``feature``
   * - Center of a circular opening's inner edge
     - ``ring_center``
     - ``upper_inner_ring``
   * - Center of a circular outer edge, such as the rim of a solid disk
     - ``ring_center``
     - ``upper_outer_ring``
   * - Center of a circular face
     - ``disk_center``
     - ``upper_face``
   * - Center of a rectangular face
     - ``rectangle_center``
     - ``upper_face``
   * - Base center of an upright round object
     - ``base_center``
     - ``supported_round_base``

Geometry Test Results
~~~~~~~~~~~~~~~~~~~~~

This LIBERO fixed-frame test used RPent ``a38d777``, SAM3, and Molmo2-8B. It
covered 16 tasks with seeds 1 through 5, for 80 scenes. In each scene,
``geometry`` measured two targets in the step-0 ``agentview`` RGB-D frame,
for 160 requests in total.

For scoring, the test independently determined each feature's reference center
from the frozen simulation scene. Error is the 3D Euclidean distance between
the returned center and the reference center in world coordinates. For a wine
bottle base, the reference is where the bottle's axis meets the support plane.
For a microwave top, it is the center of the main rectangular top panel.

.. list-table::
   :header-rows: 1

   * - Requests
     - Requests with coordinates
     - Mean error
     - Median error
     - Max error
   * - 160
     - 160
     - 1.359 mm
     - 1.184 mm
     - 2.947 mm

All results had status ``usable`` and an error below 3 mm. See the
`request list and aggregate results <https://github.com/nanji17/RPent-geometry-test/tree/2f982351e23cde92ddc99cc163de1e8f0500d74a#libero-fixed-frame-tests-r1>`_.

Use the Tools in a Simulator
----------------------------

LIBERO
~~~~~~

LIBERO's adapter loads the RGB image, world XYZ map and calibration by camera
name and step. The generic ``back_project`` and ``segment`` tools support both
the ``agentview`` and ``wrist`` cameras. The experimental ``geometry`` tool
supports only the primary ``agentview`` camera.

LIBERO keeps its own ``back_project`` and ``segment`` by default. Add
``--rgbd-tools`` to replace them with the generic tools and their schemas. Add
``--geometry`` to enable ``geometry`` together with the generic
``back_project`` and ``segment``.

All three tools take a recorded ``step``. Use ``step=-1`` for the latest
record. When selecting a pixel on an image, set ``image_shape`` to that image's
height and width and give the pixel as ``[row, column]``. For a 1024×1024
``agentview_high.png``, ``image_shape`` is ``[1024, 1024]``.

To read the 3D surface position at a pixel, call ``back_project``:

.. code-block:: json

   {"camera": "agentview", "step": 0,
    "selection": {"image_shape": [1024, 1024], "points_rc": [[200, 240]]}}

``samples_xyz`` in the result lists the 3D coordinates in input order. To read
a region, replace ``points_rc`` with a rectangular box ``box_rc`` as
``[top, left, bottom, right]``; the bottom and right edges are exclusive. To keep
only points within a height range, add ``height_m`` as ``[low, high]``.

To segment an object, call ``segment`` with exactly one of ``prompt`` or
``point``. To describe the target in text:

.. code-block:: json

   {"camera": "agentview", "step": 0, "prompt": "white cup"}

Or give one pixel on the target, together with ``image_shape``:

.. code-block:: json

   {"camera": "agentview", "step": 0, "point": [200, 240],
    "image_shape": [1024, 1024]}

``--geometry`` requires ``--molmo-endpoint`` with the Molmo service address.
For Molmo startup instructions, see
`Configure Molmo <../guides/flash.html#configure-molmo>`_. LIBERO starts SAM3
locally by default. To connect an existing service, pass ``--sam3-endpoint``.

With ``--planner api --geometry``, use a model that accepts images and omit
``--no-images`` so the planner can check which object the overlay highlights.

With ``--geometry``, LIBERO checks at startup that the external SAM3 service
can return candidate masks for a text prompt. Older services that only return
the top mask still work with ``segment``. To use ``geometry`` with one of these
services, upgrade RPent on the SAM3 host and restart the service.

To measure the center of a cup opening, give SAM3 a short name for the whole
object in ``object_prompt``, such as "white cup". If two white cups are visible,
use ``query`` to identify the one on the plate. The planner checks the
highlighted cup in the overlay against that description.

.. code-block:: json

   {"camera": "agentview", "step": 0, "query": "the white cup on the plate",
    "object_prompt": "white cup", "required_geometry": ["ring_center"],
    "feature": "upper_inner_ring"}

The status is ``usable`` when every requested quantity passes, or ``warning``
when they all pass with an object-selection or fit warning. ``partial`` means
some passed, and ``insufficient_evidence`` means none passed.

.. _generic-toolkit-registration:

Integrate on a Robot
--------------------

To enable these tools on a real robot or a new simulator, implement an
observation adapter that returns each step's RGB-D frame and calibration in the
world frame (using the arm's capture-time pose for wrist cameras) and register
the tools on the robot's toolkit.

.. dropdown:: Observation adapter contract and registration example

   For a simulator or a real robot, implement ``get_observation(camera=..., step=...)``
   on the platform's observation adapter. The method returns a
   ``toolkits.observation.GeometryObservation`` containing the RGB image and its
   3D data for that camera and step. The adapter can supply an
   aligned world XYZ map giving each pixel's surface coordinates in metres, or an
   aligned optical-axis depth map in metres together with the camera intrinsics
   and the camera-to-world transform.

   To filter ``back_project`` samples by height, supply the world up direction
   (``up_direction``). When ``geometry`` measures the object described in ``query``,
   the observation adapter must supply both ``up_direction`` and the camera pose
   (``camera_to_world``).
   To restrict geometry to the platform's primary camera, pass its name as
   ``primary_camera`` when creating the geometry tool in the example below.

   Before registering tools with the platform's ``toolkit``, prepare the observation
   adapter ``observations``, the current session's state object ``state`` (an
   ``EnvState`` instance), and an initialized segmentation client ``segmenter``.

   To also register ``geometry``, replace ``grounder = None`` below with a client
   that selects points on the original image. The segmentation client must
   return scored masks from ``segment_candidates(image, text_prompt=..., min_score=...)``
   and segment the selected point with ``segment(image, point=...)``.

   .. code-block:: python

      from functools import partial

      from toolkits.back_project import back_project
      from toolkits.segment import SegmentTool

      grounder = None
      sample = back_project.with_handler(partial(back_project, observations=observations))
      toolkit.add_tool(sample)
      masks = SegmentTool(observations=observations, state=state, segmenter=segmenter)
      toolkit.add_tool(masks.segment)

      if grounder is not None:
          from toolkits.geometry.geometry_tool import GeometryTool

          geometry = GeometryTool(
              observations=observations, state=state, segmenter=segmenter,
              grounder=grounder, check_cancelled=toolkit.raise_if_cancelled,
          )
          toolkit.add_tool(geometry.geometry)

   ``geometry`` caches measurement results for the current observation step. If
   you reuse the same geometry tool instance across episodes and step numbers
   restart, call ``geometry.reset_cache()`` at the start of each new episode to
   clear the cache.
