.. _geometry-toolkit:

Geometry Toolkit
================

``GeometryTool`` handles complex geometry perception tasks. Through
``geometry``, the planner describes an object and the part it wants to
measure. The tool uses one recorded camera view to estimate a position,
direction or size and reports which visible parts support each result.

Design Purpose
--------------

Suppose the planner needs to align an object with the opening of the yellow
and white mug to the left of a gray cup. The image shows material along the
rim. The center of the opening may lie in empty space. The tool estimates
that center by fitting a circle to 3-D points along the rim.
Measuring the mug's height also requires a visible bottom boundary. For the
two ends to share an axis, both circular faces must be roughly perpendicular
to the line between their centers.

The visual model uses color, the named part and its position relative to
other objects to select a pixel. The program does the geometry using 3-D
coordinates in meters aligned with the image. The planner decides what to
measure and whether the result is enough to plan its next action.

.. _geometry-module-design:

Module Design
-------------

.. _request-and-result-contract:

Measuring the Yellow and White Mug
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The planner asks for the rim, body and bottom centers, the axis, the rim
radius and the height. This request uses the observation recorded by
``inspection_rgb`` at step 0. The required ``camera`` name must be one the
application supports. Omitting ``step`` uses its default, ``-1``, for the
latest recorded observation from that camera.

.. code-block:: json

   {
     "query": "the yellow and white mug left of the gray cup",
     "camera": "inspection_rgb",
     "step": 0,
     "required_geometry": [
       "rim_center", "body_center", "bottom_center", "axis", "radius", "height"
     ]
   }

``required_geometry`` accepts 1 to 10 quantities per call. These six use
three parts of the mug in the image (the rim, the outer wall and the bottom).
A fit for one part can supply more than one quantity. Centers below use
the observation's reference frame, and lengths are in meters.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Quantity
     - What the view must show
     - Result when the checks pass
   * - ``rim_center``
     - A long enough arc of the circular rim with valid 3-D points along
       its edge.
     - The center ``[x, y, z]`` of the fitted circle.
   * - ``radius``
     - The same visible arc used to find the rim center.
     - The radius of that circle. ``rim_radius`` requests the same quantity.
   * - ``bottom_center``
     - The circular bottom face or its edge, with enough points on the
       boundary.
     - The center of the circle fitted to the bottom edge.
   * - ``body_center``
     - A fitted circle at each end, with distinct centers and a common
       axis.
     - The midpoint of the two centers along the mug's axis.
   * - ``height``
     - The same pair of accepted end circles used to find the body center.
     - The straight-line distance between their centers.
   * - ``axis``
     - The curved outer wall when this is the only request. If the call
       also finds both ends, the tool checks whether they share an axis.
     - A ``point`` on the axis, a unit ``direction`` and a ``sign``
       convention.

A request for ``body_center`` or ``height`` also tries to fit the wall. If
that fit succeeds, its axis must agree with both ends and the ends must
enclose the visible wall. For ``axis``, an accepted pair of ends takes
precedence and gives a direction from bottom to rim (``bottom_to_rim``).
A wall fit on its own determines a line with two equivalent directions.
``canonical_undirected`` means that the program chose one by convention.

In an offline example, a synthetic ring of 3-D points represents the mug
rim and the substitute visual grounder identifies only that part. The tool
reads one observation, tries to locate three parts and segments the rim
once. This excerpt shows the returned fields with rounded values.

.. code-block:: json

   {
     "status": "partial",
     "world_frame_id": "workcell/base",
     "geometry": {
       "rim_center": {"value": [0.0, 0.8, 0.0]},
       "radius": {"value": 0.04004, "unit": "m", "evidence_status": "inferred"},
       "bottom_center": {"value": null},
       "body_center": {"value": null},
       "axis": {"value": null},
       "height": {"value": null, "reason": "both_consistent_end_boundaries_required"}
     },
     "observation_request": {"missing_parts": ["body", "bottom"]}
   }

First check the overall ``status``. A ``usable`` result supplies every
requested quantity. A ``partial`` result supplies some of the fitted fields,
and ``need_observation`` means the view supported none of the requested
results. Then read each field's ``definition`` and ``value`` under
``geometry[field]``. This view gives a rim center and a rim radius of about
4 cm. The height remains ``null`` because it needs both end centers.

A fitted value is marked ``inferred``. A field with insufficient evidence
is marked ``unknown``. Each field records its unit, assumptions, quality and
evidence references. Its ``world_frame_id`` and ``source_step`` identify the
reference frame and observation step. Failures during execution, such as
reading the observation, calling a model or saving evidence, return
``status="error"``. If a request omits the required ``camera`` argument, the
toolkit returns an argument error with the keys ``error`` and ``errors``
before calling the tool.

.. dropdown:: Other Geometry You Can Request

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Observation and result
      * - ``inner_rim_center``
        - A distinct inner upper edge in the image or depth data gives
          the center of the fitted inner circle. If only the object's outer
          outline is visible, the inner edge remains unknown.
      * - ``body_radii``
        - A cylinder or truncated cone fitted to the visible outer wall
          gives the radii at the two ends of the visible axial range,
          together with that range.
      * - ``plane``, ``plane_normal``
        - Points spread across a flat patch determine a plane. The first
          field returns its parameters, including a point and a normal.
          The second returns just the unit normal.
      * - ``plate_center``
        - The flat circular support surface inside a plate's raised rim
          and its boundary give the center in the plane of that surface.
      * - ``lid_center``
        - The flat face and boundary of a sealed circular lid give a center
          in the lid plane. The raised rolled lip is excluded.
      * - ``rectangle_center``
        - Four physical edges of the main rectangular top face give its
          center. The program uses the camera pose to distinguish an edge
          from an occluder.
      * - ``body_segment_center``
        - Radius changes or visible physical ends bound the main cylindrical
          section of a bottle. The result is the midpoint of that section
          along its axis.

Processing Flow
~~~~~~~~~~~~~~~

Each call reads one observation from the specified camera. The next steps
depend on what the planner asks about the mug.

- **A point on the wall.** With the default ``measurement="surface_point"``,
  the grounder selects one pixel on the surface named in ``query``, such as
  the visible mug wall. The program reads that pixel's aligned 3-D coordinate
  and returns it in ``world_xyz``.
- **The extent of the visible wall.** With ``measurement="visible_region"``,
  the program also segments the region. The resulting mask marks the pixels
  that belong to the wall. For the valid 3-D points at those pixels,
  ``world_xyz`` gives the median on each axis and ``visible_bounds_xyz``
  gives the minimum and maximum. The median can lie in empty space.
- **A part's center, direction or size.** Supplying ``required_geometry``
  selects fitting for the listed quantities. The program extracts 3-D points
  from each part's mask, retains connected points and filters outliers. It
  fits circles to rim or bottom edges and compares a cylinder and a truncated
  cone for the outer wall. It checks the number of supporting points,
  coverage, fit residuals and agreement between parts before filling
  ``geometry[field]``.

The first two requests use visible surface samples directly and return
``measured`` when successful. The tool saves the observation and any masks,
point clouds and fit evidence under the original step in the shared
``EnvState``. A fitted field's ``source_refs`` link it to that evidence.
The recorded step stays the same throughout the call.

Observation Contract
~~~~~~~~~~~~~~~~~~~~

The application that registers the tool supplies camera observations and
calibration.
It implements ``get_observation(*, camera, step)`` on its ``ObservationProvider``
to resolve the camera name and step and return a ``GeometryObservation``.
The resolved ``step`` must refer to an existing record in the shared
``EnvState``, and ``world_frame_id`` must consistently identify the reference
frame. The planner refers to the camera and step in its request.

Each observation needs an original-resolution RGB image (``rgb``, an
H × W × 3 ``uint8`` array) and at least one of two sources of aligned 3-D data.

- With a point map, supply ``world_xyz``. This H × W × 3 array gives each
  pixel's position in meters in the reference frame.
- With a depth map, supply optical-axis depth in meters (``depth_m``), the
  3 × 3 pinhole camera matrix (``intrinsics``) and the rigid 4 × 4 transform
  from camera to reference frame (``camera_to_world``). The tool uses these
  to compute the 3-D coordinates. Camera Z points forward. The application
  defines the orientation of the reference frame.

Observation arrays are copied into a read-only snapshot. Valid points need
finite coordinates. If optical-axis depth is supplied or can be derived from
the camera pose, it must also be finite and positive. Supplied depth is
checked even when a point map is present. An optional boolean ``valid_mask``
can exclude additional pixels. A point at the reference origin follows the
same validity rules.

The camera pose can orient a plane normal. When cleaning rim or wall points,
the program attempts to identify and remove a background plane only if the
observation supplies an explicit ``up_direction`` in the reference frame.
Use ``source_refs`` to identify the source of the observation.

Responsibilities and Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To connect the tool to a robot application, prepare the observation provider,
shared ``EnvState``, visual ``grounder`` and ``segmenter``. The segmenter is
optional for surface-point queries. An existing SAM3 client can supply
segmentation, with the application managing the connection. Passing ``model``
and an optional ``base_url`` can create the shared ``GroundingAgent``.
``fit_options`` adjusts the geometric fitting criteria.

This example constructs ``GeometryTool`` outside the toolkit, registers the
instance's ``geometry`` tool and binds cancellation checks at dispatch.
``task_instructions`` contains the planner's existing instructions.

.. code-block:: python

   from functools import partial

   from toolkits.geometry.geometry_tool import GeometryTool
   from toolkits.geometry.prompts import GEOMETRY_INSTRUCTIONS

   geometry_tool = GeometryTool(
       state=state,
       observations=observations,
       grounder=grounder,
       segmenter=segmenter,
   )
   geometry = geometry_tool.geometry
   toolkit.add_tool(
       geometry.with_handler(
           partial(geometry, check_cancelled=toolkit.raise_if_cancelled)
       )
   )
   system_prompt = task_instructions + "\n\n" + GEOMETRY_INSTRUCTIONS

See :doc:`interfaces` for toolkit registration and dispatch.

.. _geometry-prompt-design:

Prompt Design
-------------

Parent Planner Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GEOMETRY_INSTRUCTIONS`` asks the planner to identify the object in
``query`` and use ``measurement`` or ``required_geometry`` to say what it
needs. In the mug example, “yellow and white” and “left of the gray cup”
together identify the target. The rim center is one quantity to measure.
The tool determines the pixels and 3-D coordinates.

The instructions also guide the planner's use of the answer. If the task
only needs the rim radius, the ``partial`` result above already supplies it.
If it needs height as well, the planner can read the ``reason`` for ``height``
and the missing parts, then decide whether its task budget allows a view that
shows both rim and bottom. The planner arranges any movement of the camera
or robot.

.. _expanding-part-requirements:

Locating and Segmenting Each Part
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The program uses ``PART_DESCRIPTIONS`` to add each part requirement to the
original target description. For the six quantities in the example, the
grounder receives the following three descriptions, one per call with the
same RGB image.

.. code-block:: text

   the visible circular rim boundary of
   the yellow and white mug left of the gray cup

   the visible curved outer body, excluding handle, rim and interior, of
   the yellow and white mug left of the gray cup

   the visible circular bottom face or its boundary of
   the yellow and white mug left of the gray cup

.. _internal-visual-grounding-prompt:

The shared ``GroundingAgent`` includes the original image dimensions in its
prompt. It asks the visual model to select a clearly visible material point
on the named part, away from occlusions, silhouettes and image borders.
Given the RGB image and text, the model sets ``point_xy`` to
``[column, row]`` in original pixels, or to ``null`` if it cannot identify
the part. The program checks the image dimensions and pixel bounds before
looking up a coordinate by row and column. For the rim pixel ``[104, 64]`` in
the synthetic example, it reads row 64, column 104, counting from zero.

.. _segmentation-prompts-and-call-bounds:

When segmentation is needed, the segmenter first uses the part description.
The program checks whether the mask contains the independently selected
pixel. If the mask is missing or excludes the point, the program can use
that point to prompt segmentation once more at the same score threshold.
Each part uses at most one grounding call and two segmentation calls.
Across the three mug parts, that gives a limit of three grounding calls
and six segmentation calls.

For ``body_segment_center`` the grounder selects a point on the main barrel,
while the segmenter receives a description of the whole bottle so that the
program can use the shoulder, neck and base to find the barrel's limits.

Limits and Runtime Verification
-------------------------------

Occlusion, missing depth or a mask that mixes several surfaces can leave
quantities unknown. A fit residual describes how closely the points match
the chosen shape. Along with the supporting point count and segmentation
score, it helps diagnose a fit. Actual position error requires an independent
measurement.

The tool handles one view per call. The planner can compare results across
views when the target, part definition and reference frame match. It should
measure again after the object moves. Multi-view fusion and reconstruction
of a complete box or cavity are outside the scope of this tool.

With the test dependencies installed, run the offline checks from the
repository root.

.. code-block:: bash

   python -m pytest tests/unit_tests/toolkits/geometry -q

These checks use synthetic observations and substitute model services. They
exercise toolkit dispatch, coordinate conversion, segmentation retries,
cancellation and part fitting. Measuring real-scene accuracy or robot task
success requires separate experiments.
