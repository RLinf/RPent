.. _geometry-toolkit:

Geometry Toolkit
================

Use ``GeometryAgent.geometry`` to request observed surface positions or fitted
object-part geometry from a recorded LIBERO RGB-D frame. The parent planner
supplies a target description, quantities and an observation reference; the
agent selects image points and computes world coordinates with its internal
perception tools. The request never moves the robot. The current integration
uses LIBERO observations; follow :doc:`../simulators/libero` for environment,
SAM3 and planner setup.

Modules
-------

The toolkit separates geometry measurement from robot execution and independent
evaluation. Modules under ``toolkits/geometry/`` have these responsibilities:

``geometry_agent.py``
   Visual selection, segmentation, coordinate lookup and part fitting.

``geometry.py``
   Point-cloud cleaning, plane, circle, cylinder and frustum fitting.

``reference_geometry.py``
   Inner opening, support face, lid, rectangular top and bottle-barrel
   references.

``candidate_geometry.py``
   Labelled approximate centers for permissive error evaluation.

``evaluation/checklist.py``
   Fixed tasks, seeds and reference definitions.

``evaluation/experiment.py``
   Initial-frame collection and geometry inference CLI.

``evaluation/metrics.py``
   Independent truth validation, position errors and coverage.

``evaluation/evaluation_scene.py``
   Evaluator-only compiled meshes and same-frame body poses.

The LIBERO environment, action primitives and tool registration live in
``robots/libero/``. Shared visual-model adaptation lives in
``rpent/robots/components/grounding_agent.py``. Import the agent directly:

.. code-block:: python

   from toolkits.geometry.geometry_agent import GeometryAgent

Request Geometry
----------------

Use ``view_env_state`` to identify task-relevant objects. Pass a target
description, the needed quantities and an observation reference to ``geometry``:

.. code-block:: json

   {
     "query": "the yellow and white mug left of the gray cup",
     "camera": "agentview",
     "step": 0,
     "required_geometry": [
       "rim_center", "body_center", "bottom_center", "axis", "radius", "height"
     ]
   }

The parent supplies no calibration. GeometryAgent obtains the matching image,
world coordinates, depth and camera metadata from the recorded observation.
``required_geometry`` selects fitting instead of the ``measurement`` surface
mode. Each part uses one visual-grounding request and at most two SAM requests,
with no synonym search or threshold reduction. Cup requests use at most three
parts; a separate ``plane``/``plane_normal`` request can add a fourth. All parts
share one pinned frame. The agent has no motion tools.

Part Cleanup and Fitting
------------------------

Each requested part uses observed candidates and a matching geometric model.
The following table describes the supported evidence and cleanup:

.. list-table::
   :header-rows: 1
   :widths: 18 38 44

   * - Part
     - Observed candidates
     - Cleaning and model
   * - Rim
     - SAM part mask, RGB/depth foreground edges near the visual rim point.
     - Target-connected cloud; 3-D circle consensus and robust refinement.
   * - Body
     - Curved outer surface excluding handle, rim and interior.
     - Background-plane removal, connected cloud, neighbor outliers; robust
       cylinder/frustum comparison.
   * - Bottom
     - Visible bottom face boundary.
     - Preserve the bottom plane; fit a circle to its observed boundary.
   * - Plane
     - Requested visible planar patch.
     - Preserve the target plane; RANSAC and SVD.
   * - Inner opening
     - Inner mask-hole edge, interior foreground depth jump, or connected
       interior RGB contour at the visually selected inner edge.
     - Exclude the outer silhouette and interior floor; clean the largest
       candidate component and fit a 3-D circle.
   * - Plate / sealed lid
     - Central support or sealed lid plane and its observed physical boundary.
     - Fit the face plane; project its observed boundary onto that plane and
       fit a circle.
   * - Rectangular top
     - Main top plane and four physical edges.
     - Reject coplanar label crops, foreground covers and image clipping;
       require edge and corner support before fitting a rectangle.
   * - Bottle barrel
     - Whole-bottle mask and a visual anchor on its main barrel.
     - Seed a dominant cylinder near the anchor; identify both barrel limits
       using shoulder radius transitions or physical terminal edges.

Voxel sampling retains actual input samples. A horizontal background plane is
removed for rim/body only when observed outside the mask and separated from the
visual anchor. Handle/interior contamination is rejected through body-model
residuals; a mixed cloud need not produce an accepted model. Small or
disconnected clouds, short arcs, collinear patches, low support and
ill-conditioned fits request more evidence. A low residual alone cannot
establish observability.

Inner-opening candidates use their largest connected component without
requiring it to pass near the visual material point. The SAM mask must still
contain that point, which also supports background-plane separation and circle
consistency checks. Voxel sampling, outlier rejection and all circle acceptance
gates apply.

The initial algorithm gates are recorded in ``quality.fit_options``: 3 mm inlier
threshold, 1 mm voxels, at least 30 points, 60% support, 180 degrees of circular
boundary coverage and 120 degrees of body coverage. Circle/body fits also
require the 95th-percentile surface residual to be within 5% of the fitted
reference radius. This relative shape gate prevents a small noncircular
boundary passing solely because the absolute threshold is coarse. Optimization
and sampling are bounded. These are experimental acceptance settings, not
measured accuracy or calibrated sensor uncertainty. float16 world coordinates
retain quantization.

Reading Fitted Results
----------------------

Read ``geometry[field]``, not the surface-mode ``world_xyz``. Every field gives
its ``value``, definition, units, ``world_frame_id``, source step, evidence
references, assumptions, quality and ``evidence_status``:

- ``rim_center``, ``rim_radius`` and ``radius`` refer to the fitted visible rim
  circle; ``radius`` is an alias for rim radius, not the body's radius at every
  height.
- ``bottom_center`` requires a visible circular bottom boundary. It is not the
  median of a bottom-plane patch or a point picked inside the cup.
- ``axis`` comes from the fitted circular body, or consistent observed end
  circles. Without both ends, its sign is conventional. It is not a handle or
  door-opening direction.
- ``body_radii`` describes only the fitted radii at visible axial sample limits.
- ``body_center`` is the midpoint of observed coaxial rim and bottom centers.
  ``height`` is their separation under the coaxial-end assumption. A partial
  body's visible length cannot establish either quantity. This center is not
  a mass or volume centroid.
- ``plane`` and ``plane_normal`` describe a visible plane, with normal sign
  toward the camera when metadata is available; they do not establish full face
  dimensions.
- ``inner_rim_center`` is the 3-D center in the inner upper opening edge plane.
  An unresolved inner edge remains unknown; the outer lip is not a substitute.
- ``plate_center`` is the circular central support-face center at that face
  height. ``lid_center`` is the sealed circular lid center and follows a tilted
  lid plane.
- ``rectangle_center`` is the main rectangular top-face center. All four
  physical edges and their corner neighborhoods must support it; a
  visible-fragment box cannot establish this center. This does not recover the
  complete solid box.
- ``body_segment_center`` is the main near-cylindrical bottle barrel's axial
  midpoint, excluding neck and mouth. Both limits need observed evidence; a
  partial cloud's min/max do not define a barrel center. The axis may be
  horizontal.

Fits are labelled ``inferred`` under their shape assumptions. Unsupported
fields remain ``unknown`` with null values. Residuals, support counts and SAM
scores are separate diagnostics; ``calibrated_error_m=null`` remains unknown.

``status="usable"`` means all requested quantities passed the implemented
gates; ``partial`` means only some did. ``need_observation`` provides reasons
and missing fields/parts. The parent decides and executes a new observation
move. ``error`` reports an input, service or evidence-persistence failure. Do
not silently use a surface median when a requested geometric fit was rejected.

``geometry_XXX.json`` stores part requests, budgets, cleaning statistics, model
comparisons and consistency checks. Mask, candidate-cloud, cleaned-cloud,
cleaned-index and fit-inlier arrays remain separate step-scoped artifacts.
``fit_points_artifact`` identifies the array indexed by the fit-inlier indices.
For circular faces it contains explicitly derived plane projections;
``observed_support_artifact`` separately preserves the original observed
boundary. Arrays are not sent as tool-result text. Complete box/cavity recovery
and multi-view fusion are not implemented.

.. _geometry-permissive-evaluation:

Permissive Error Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For error evaluation of checklist centers, construct ``GeometryAgent`` with
``permissive=True``, or add ``--permissive`` to the experiment's ``infer``
command. Rejected fits then receive labelled observation-only candidates
without asking for additional views. Check ``strict_accepted``,
``estimate_method`` and ``evaluation_only``: a returned coordinate can be an
incomplete visible-extent estimate. The default remains strict. See the
`evaluation guide <https://github.com/RLinf/RPent/blob/main/toolkits/geometry/evaluation/README.md#宽松误差评估>`_
for the relaxed gates and fallback methods.

Follow the :ref:`fixed reference checklist experiment
<libero-fixed-reference-experiment>` for the 80-case / 160-reference protocol
and its :doc:`LIBERO setup <../simulators/libero>`. It uses only the initial
agentview frame and records failures without changing targets or moving to
another view. This protocol is separate from interactive parent-planner
recovery.

Surface Measurements
--------------------

For contact points use ``measurement="surface_point"``; for visible sample
statistics use ``measurement="visible_region"``, without ``required_geometry``.
The first returns a selected material point. The second returns
coordinate-wise median and bounds of a segment containing the visual point.
Those statistics are not fitted centers; a median may lie in empty space.

Use agentview for identity, then request the same part in a new wrist
observation when closer geometry is needed. Compare only matching world
frames and point definitions. Request geometry again after target motion. Use
memory offsets only when their reference definition matches the returned
measurement.

Model Configuration and Validation
----------------------------------

API/Codex planners inherit their model and provider URL. Override with
``--geometry-model provider:model`` or ``--geometry-model codex:model`` and
``--geometry-base-url``. Bare ``codex`` uses its configured default. Claude
Code planners require an explicit ``--geometry-model``. The visual model
receives only RGB and target text, with no robot tools or object truth.
``--geometry-base-url`` overrides the inherited ``--base-url``.

Offline smoke tests in ``tests/unit_tests/toolkits/geometry/`` use analytic
point clouds and fixed perception outputs. Real-scene accuracy needs
independent labels or simulator truth available only to the evaluator. Task
success and geometry measurement error must be measured separately. The
sample-consensus approach follows standard
`PCL models <https://pointclouds.org/documentation/group__sample__consensus.html>`_;
robust refinement uses
`SciPy least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_.
