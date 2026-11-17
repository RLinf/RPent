.. _generic-toolkit:

Generic RGB-D Toolkit
=====================

Register the read-only tools in ``toolkits`` to let a planner look up 3D positions
at image pixels, segment objects or measure their parts using recorded RGB-D
observations.

For example, if the planner selects a pixel on the side of a cup,
``back_project`` can return the 3D coordinates of that surface point. With
sufficient observations, ``geometry`` can fit the rim to find its center and
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
     - A ``selection`` with ``image_shape`` and either ``points_rc`` or a half-open ``box_rc`` (top, left, bottom, right). The optional ``height_m`` interval filters along world up.
     - Point requests return measured surface coordinates in input order, with ``null`` for invalid or filtered pixels. A box returns a sampled surface point and visible bounds. The caller checks object identity.
   * - ``segment``
     - A text ``prompt``, one positive ``point``, or both. When given both, the text mask must contain the point.
     - The result links to the candidate mask and overlay. With enough valid depth, ``world_xyz`` is the coordinate-wise median of the masked samples. Inspect the overlay to confirm the object.
   * - ``geometry``
     - A target description in ``query``. Request a surface point, a visible region or a set of fitted quantities in ``required_geometry``.
     - The tool selects pixels, verifies the target and checks other available views from the same step when needed. Only accepted fits include a value.

Use the tools in LIBERO
-----------------------

LIBERO is the first adapter. Its API planner enables ``back_project`` and
``segment`` with SAM3 by default. For ``geometry``, set ``MOLMO_ENDPOINT`` to a
running local Molmo service and add ``--geometry --molmo-endpoint "$MOLMO_ENDPOINT"``
to the :doc:`LIBERO run command <../simulators/libero>`.

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

Register tools for a robot
--------------------------

The robot's observation adapter implements ``get_observation(camera=..., step=...)``
to return the RGB image for that camera and step along with its 3D data in a
single ``toolkits.observation.GeometryObservation``. The adapter can supply an
aligned world XYZ map giving each pixel's surface coordinates in metres, or an
aligned optical-axis depth map in metres together with the camera intrinsics
and the camera-to-world transform.

Measuring a named target also requires the camera pose and world up direction.
If the robot has multiple calibrated cameras, the adapter can implement
``additional_observations(primary)`` to return other views from the same step
in the same world frame.

Given the adapter (``observations``), the session's ``EnvState`` (``state``)
and an initialized ``segmenter``, register the tools on the robot's ``toolkit``.
Supply a ``grounder`` that selects and verifies image targets to enable named
geometry. Otherwise set it to ``None``.

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

If episode resets reuse step numbers, retain the geometry instance and call
``geometry.reset_cache()`` when a new episode starts.
