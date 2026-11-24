.. _generic-toolkit:

Embodied RGB-D Toolkit
======================

Register the read-only tools in ``toolkits`` to let a planner look up 3D positions
at image pixels, segment objects or measure their parts using recorded RGB-D
observations.

For example, if the planner selects a pixel on the side of a cup,
``back_project`` can return the 3D coordinates of that surface point.
Given that pixel or the description "white cup", ``segment`` can return a
candidate mask for the cup's visible region. The planner can inspect the
overlay to check whether the mask covers the intended cup.
With sufficient observations, ``geometry`` can fit the rim to find its center and
radius, or estimate the cup's height from its rim and base centers. The
``world_frame_id`` in the result identifies the world frame for the surface
point and the rim center. The coordinate components, rim radius and cup height
are all in metres.

Choose a tool
-------------

All three tools take a ``camera`` and a recorded ``step``. Use ``step=-1`` for
the latest record. When selecting a pixel on a preview, use the preview's
``image_shape`` (height, width) and specify the pixel as [row, column].

.. list-table::
   :header-rows: 1
   :widths: 17 39 44

   * - Tool
     - Planner input
     - What the result means
   * - ``back_project``
     - In ``selection``, provide ``image_shape`` and either a list of points (``points_rc``) or a rectangular box (``box_rc``). Set ``height_m`` to keep only points within a given height range in the world frame.
     - For selected points, the tool returns 3D coordinates in input order, with ``null`` for points with invalid depth or points excluded by the filter. For a box, it returns the coordinates of one measured surface point and the 3D bounds of the visible surfaces inside the box.
   * - ``segment``
     - Use ``prompt`` to describe the target or ``point`` to specify a pixel on it. If you provide both, the tool generates a candidate mask from the description and returns it only after confirming that it contains the selected point.
     - The result includes the paths to the saved candidate mask and overlay image. If enough points inside the mask have valid depth, the three components of ``world_xyz`` are the medians of their X, Y and Z coordinates, respectively.
   * - ``geometry``
     - Describe the target in ``query``. You can request the 3D coordinates of a point on its visible surface or the bounds of its visible region. To measure the position or size of a specific part, list the measurements in ``required_geometry``.
     - The tool selects image points internally and checks whether they lie on the target object. If needed, it also examines images recorded by other cameras at the same step. It returns a numerical value for a fitted quantity only if the fit passes its quality checks.

Use the tools in a simulator
----------------------------

LIBERO
~~~~~~

For the requested camera (``agentview`` or ``wrist``) and step, LIBERO's
adapter loads the RGB image, world XYZ map and calibration. LIBERO runs keep
their own ``back_project`` and ``segment`` unless you add ``--rgbd-tools``,
which replaces them with the generic tools and their schemas for any planner.

``geometry`` is experimental and requires a Molmo service that supports native
pointing and target verification (``molmo.verify``). For startup instructions,
see `Configure Molmo <../guides/flash.html#configure-molmo>`_.

Set ``MOLMO_ENDPOINT`` to the service address and add
``--geometry --molmo-endpoint "$MOLMO_ENDPOINT"`` to the LIBERO run command.
``--geometry`` also enables the generic ``back_project`` and ``segment`` tools.

For example, a planner can request the rim radius and full height of the same cup
in one ``geometry`` call.

.. code-block:: json

   {"camera": "agentview", "step": 0, "query": "the white cup on the plate",
    "required_geometry": ["rim_radius", "height"]}

If both fits pass, the status is ``usable``. If the rim can be fitted but the base
is hidden in every recorded view, the planner gets ``partial`` with the rim radius
in metres and ``height`` in ``missing_geometry``. If both quantities
can be estimated but one fails a quality check, ``warning`` includes only the
accepted value. If neither passes, ``insufficient_evidence`` lets the planner
choose another action.

Integrate the tools on a real robot
-----------------------------------

Implement the robot's observation adapter
to read the recorded RGB-D frame and its camera calibration for each step. If
the camera moves with the arm, use its pose at the time of capture to transform
depth samples into world coordinates.

.. _generic-toolkit-registration:

Observation adapters and tool registration
------------------------------------------

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
For platforms with multiple calibrated cameras, the adapter can implement
``additional_observations(primary)`` to return other views from the same step
in the same world frame.

Before registering tools with the platform's ``toolkit``, prepare the observation
adapter ``observations``, the current session's state object ``state`` (an
``EnvState`` instance), and an initialized segmentation client ``segmenter``.

The example below registers ``back_project`` and ``segment`` when the platform
initializes its ``toolkit``, so the planner can call them during a task. To use
only these two tools, set ``grounder`` to ``None``. If you provide a ``grounder``,
the example also registers ``geometry``.

.. code-block:: python

   from functools import partial

   from toolkits.back_project import back_project
   from toolkits.segment import SegmentTool

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
