.. _geometry-toolkit:

Geometry Toolkit
================

``GeometryTool`` is a general-purpose tool for geometric perception. Through
``geometry``, the planner describes the object and the part to measure. The
tool computes a position, direction or size from one recorded camera view
and identifies the visible parts it used for each measurement.

Design Purpose
--------------

To align an object with the opening of the yellow and white mug to the
left of a gray cup, the planner first needs to know where the center of
the opening is. A pixel can mark a point on the rim. The center itself
may lie in empty space. The tool estimates it by fitting a circle to
visible 3-D points along the rim. To measure the mug's height, the tool
also needs to see the bottom and check that the two ends share an axis.
Both circular faces should be roughly perpendicular to the line between
their centers.

The visual model uses color, the named part and its position relative to
other objects to select a pixel. The program does the geometry using 3-D
coordinates in meters aligned with the image. The planner decides what to
measure and whether the result is enough to plan its next action.

.. _geometry-module-design:

Module Design
-------------

The planner calls ``geometry`` with a description of the object or part in
``query`` and the camera name in ``camera``. The tool reads one recorded
observation from that camera. A visual model locates the target in the
color image, and the tool takes the measurement using the corresponding
3-D points from the same observation.

.. _processing-flow:

Choosing a Measurement
~~~~~~~~~~~~~~~~~~~~~~

The same object can support different measurements depending on what the
planner needs to know.

- **Where is one point on the surface?** With the default
  ``measurement="surface_point"``, the visual model selects a pixel on the
  visible surface. The tool reads the 3-D coordinate at that pixel and
  returns it in ``world_xyz``.
- **What are the 3-D bounds of the visible part?** With
  ``measurement="visible_region"``, the tool segments the region and returns
  the minimum and maximum on each axis in ``visible_bounds_xyz``. These
  bounds cover the valid 3-D points in that region. Here ``world_xyz`` is
  their median on each axis, which may lie in empty space.
- **Where is a part's center, which way does it point, or how large is it?**
  List 1 to 10 quantities in ``required_geometry``. Supplying this list
  selects shape fitting. The tool extracts 3-D points from each relevant
  part, filters outliers and checks how many points support the shape, how
  well they cover it and how closely they fit. It returns one field per
  quantity under ``geometry``.

.. _request-and-result-contract:

Measuring a Cup
~~~~~~~~~~~~~~~

Suppose there is one cup with a circular rim on the table. We want to
know how wide the opening is and how tall the cup is. We can calculate
the rim diameter from its radius, so the planner requests ``rim_radius``
and ``height``. In this example the camera is called ``table_camera``.
Omitting ``step`` uses its default of ``-1``, the latest recorded
observation from that camera.

.. code-block:: json

   {
     "query": "the cup on the table",
     "camera": "table_camera",
     "required_geometry": [
       "rim_radius",
       "height"
     ]
   }

.. list-table::
   :header-rows: 1
   :widths: 20 43 37

   * - Quantity
     - What the view must show
     - What the tool returns
   * - ``rim_radius``
     - A long enough arc of the circular rim with valid 3-D points along
       its edge.
     - The radius of the fitted circle in meters. ``radius`` is another
       name for the same quantity.
   * - ``height``
     - Enough of the rim and bottom edges to fit a circle to each and
       establish that the ends share an axis.
     - The distance between the two circle centers in meters.

Suppose a view from above shows only the rim clearly and the fit gives a
radius of about 4 cm. The following excerpt illustrates the result.

.. code-block:: json

   {
     "status": "partial",
     "geometry": {
       "rim_radius": {
         "value": 0.04,
         "unit": "m",
         "evidence_status": "inferred"
       },
       "height": {
         "value": null,
         "evidence_status": "unknown",
         "reason": "both_consistent_end_boundaries_required"
       }
     }
   }

Here ``partial`` means that one of the two quantities has a value. Doubling
the radius gives a rim diameter of about 8 cm. The height is ``null``
because this view lacks a bottom edge to pair with the rim. The status
is ``usable`` when all requested results are available and ``need_observation``
when the view supports none of them.

Surface-point and visible-region results are marked ``measured``. Fitted
values are marked ``inferred``. For each fitted field, ``definition`` and
``unit`` explain the value, and ``world_frame_id`` and ``source_step``
identify the reference frame and recorded step. ``assumptions`` and ``quality``
record the shape assumptions and fit diagnostics. ``source_parts`` lists the
parts used. Follow ``source_refs`` to find the masks, 3-D points and fit
records saved at the original step in the shared ``EnvState``.

.. dropdown:: Other Geometry You Can Request

   .. list-table::
      :header-rows: 1
      :widths: 24 40 36

      * - Quantity
        - What the view must show
        - What the tool returns
      * - ``rim_center``
        - The same visible arc that supplies ``rim_radius``.
        - The center ``[x, y, z]`` of the fitted rim circle.
      * - ``inner_rim_center``
        - A distinct upper inner edge of the opening in the image or
          depth data.
        - The center of the circle fitted to that inner edge.
      * - ``bottom_center``
        - A circular bottom or its edge, with enough points on the boundary.
        - The center of the circle fitted to the bottom edge.
      * - ``body_center``
        - The same accepted rim and bottom edges that supply ``height``.
        - The midpoint of the two circle centers.
      * - ``axis``
        - The curved outer wall. If the same request also finds both end
          centers, the tool can use them to determine the axis.
        - A ``point`` on the axis, a unit ``direction`` and a ``sign`` convention.
      * - ``body_radii``
        - Enough of the curved outer wall to fit a cylinder or truncated cone.
        - The radii at both ends of the visible axial range and the range itself.
      * - ``plane``, ``plane_normal``
        - A set of 3-D points spread across a flat patch.
        - ``plane`` gives the plane parameters, including a point and a normal.
          ``plane_normal`` gives the unit normal.
      * - ``plate_center``
        - The flat circular support surface inside a plate's raised rim
          and its boundary.
        - The circle center in the plane of the support surface.
      * - ``lid_center``
        - The flat sealed face of a circular lid and its boundary, excluding
          the raised rolled lip.
        - The circle center in the plane of the lid.
      * - ``rectangle_center``
        - The four physical edges and corners of the main rectangular top
          face, with a camera pose to distinguish edges from occlusions.
        - The center of the rectangular top face.
      * - ``body_segment_center``
        - A roughly cylindrical main barrel, with its ends visible as radius
          changes or physical ends of the object. The grounder selects a point
          on the main barrel and the segmenter marks the whole bottle so the
          tool can examine both ends.
        - The midpoint of that barrel along its axis.

   Both ``height`` and ``body_center`` require two distinct end centers. The
   two circular faces must be roughly perpendicular to the line between their
   centers. The tool also tries to fit the outer wall. If that fit succeeds,
   its axis must agree with the ends, and the ends must enclose the visible
   wall. For ``axis``, an accepted pair of ends gives a direction from bottom
   to rim (``bottom_to_rim``). A wall fit alone gives two equivalent
   directions. ``canonical_undirected`` means the tool chose one by convention.

.. _observation-contract:

Camera Observations
~~~~~~~~~~~~~~~~~~~

The robot application implements ``get_observation(*, camera, step)`` on its
``ObservationProvider`` to return a ``GeometryObservation``. It defines the
camera names. The resolved ``step`` identifies an existing record in the
shared ``EnvState``. Each ``world_frame_id`` must consistently identify the
same reference frame.

Each observation needs an original-resolution RGB image (``rgb``, an
H × W × 3 ``uint8`` array) and at least one of two sources of aligned 3-D data.

- With a point map, supply ``world_xyz``. This H × W × 3 array gives each
  pixel's position in meters in the reference frame.
- With a depth map, supply optical-axis depth in meters (``depth_m``), the
  3 × 3 pinhole camera matrix (``intrinsics``) and the rigid 4 × 4 transform
  from camera to reference frame (``camera_to_world``). The tool uses these
  to compute the 3-D coordinates. Camera Z points forward. The application
  defines the orientation of the reference frame.

.. dropdown:: Valid Points and Optional Observation Data

   Observation arrays are copied into a read-only snapshot. Valid points need
   finite coordinates. If optical-axis depth is supplied or can be derived
   from the camera pose, it must also be finite and positive. Supplied depth
   is checked even when a point map is present. An optional boolean ``valid_mask``
   can exclude additional pixels. A point at the reference origin follows the
   same validity rules.

   The camera pose can orient a plane normal. When cleaning rim or wall
   points, the tool attempts to identify and remove a background plane only
   if the observation supplies an explicit ``up_direction`` in the reference
   frame. Use ``source_refs`` to identify the source of the observation.

.. _responsibilities-and-initialization:

Registering the Tool
~~~~~~~~~~~~~~~~~~~~

Prepare the shared ``EnvState``, the observation provider and a visual
``grounder``. These are sufficient for a surface-point query. Visible-region
queries and shape fitting also need a ``segmenter``, such as a SAM3 client
already connected by the application. Passing ``model`` and an optional
``base_url`` can create the shared visual ``GroundingAgent``. Use ``fit_options``
to adjust the fitting criteria.

The following code registers the instance's ``geometry`` method with an
existing ``toolkit`` and binds its cancellation check at dispatch.

.. code-block:: python

   from functools import partial

   from toolkits.geometry.geometry_tool import GeometryTool

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

See :doc:`interfaces` for toolkit registration and dispatch.

.. _geometry-prompt-design:

Prompt Design
-------------

Parent Planner Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The application adds ``GEOMETRY_INSTRUCTIONS`` to the planner's existing
task instructions and makes the registered ``geometry`` tool available to it.

.. code-block:: python

   from toolkits.geometry.prompts import GEOMETRY_INSTRUCTIONS

   system_prompt = task_instructions + "\n\n" + GEOMETRY_INSTRUCTIONS

The instructions ask the planner to describe the target and name the
measurements it needs. With just one cup in view, ``the cup on the table``
identifies it. If several objects look alike, the planner can include color
or relative position. The tool selects pixels in the image and obtains 3-D
coordinates from the camera observation.

On ``partial``, the planner checks which field is missing and reads its
``reason``. If the task needs only the rim diameter, the radius already
answers the question. If it also needs the height, the planner can use its
task budget to obtain a new view that shows both the rim and the bottom.

.. _expanding-part-requirements:

Locating and Segmenting Each Part
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tool prefixes the original ``query`` with a part description from
``PART_DESCRIPTIONS``. The radius needs the rim. The height request also
examines the outer wall and the bottom. Both quantities use the same rim
lookup. The grounder receives this description for the rim.

.. code-block:: text

   the visible circular rim boundary of the cup on the table

.. _internal-visual-grounding-prompt:

``GroundingAgent`` supplies the visual model with this text, the original
RGB image and its dimensions. It asks for one clearly visible point on the
material of the named part, such as a point on the rim. The model returns
``point_xy`` in original ``[column, row]`` pixel coordinates, or ``null`` if
it cannot identify the part.

.. _segmentation-prompts-and-call-bounds:

The segmenter first uses the part description to mark a region of the
image. The tool checks whether that region contains the independently
selected pixel. If the region is missing or excludes the point, the tool
tries segmentation once more using that point. Each part gets at most one
grounding call and two segmentation calls.

Limits and Runtime Verification
-------------------------------

Occlusion, missing depth or a region that mixes several surfaces can
leave a measurement unknown. Before comparing two views, check that
the target, part definition and reference frame match. Measure again
after the object moves. Combining views into a model of the whole object
or its interior requires a separate reconstruction step. Fit residuals
describe how closely the points match the chosen shape. Actual measurement
error requires an independent measurement for comparison.

Errors while reading an observation, calling a model or saving evidence
return ``status="error"``. If a request omits the required ``camera``, the
toolkit returns an argument error with ``error`` and ``errors`` before
calling the tool.

With the test dependencies installed, run the offline checks from the
repository root.

.. code-block:: bash

   python -m pytest tests/unit_tests/toolkits/geometry -q

These checks use synthetic observations and substitute model services.
They exercise tool dispatch, coordinate conversion, segmentation retries,
cancellation and part fitting.
