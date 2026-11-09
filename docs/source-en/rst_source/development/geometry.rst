.. _geometry-toolkit:

Geometry Toolkit
================

The geometry tool turns a description of an object or part into spatial
measurements with explicit definitions. The parent planner supplies a target,
the required quantities and an observation reference through one ``geometry``
interface. GeometryAgent locates the target, segments the relevant evidence
and computes positions, directions and dimensions for subsequent action
planning.

Design Purpose
--------------

Grasping, placing and aligning objects require geometry of specific parts. A
rim center, a body center and a visible point on a cup have different meanings.
A selected image point cannot stand in for all three. The tool defines each
measurement and returns it with its source observation and shape assumptions.

The design assigns task decisions and geometry measurement to separate roles.
The parent describes what it needs; GeometryAgent determines how to measure it
from the available observation. The visual model handles semantic localization,
while depth data and geometric algorithms supply spatial calculations. This
gives the parent one interface for different parts, allows the internal
perception process to evolve independently and ties each result to inspectable
evidence.

.. _geometry-module-design:

Module Design
-------------

GeometryAgent coordinates perception through a fixed program flow. An internal
vision-language model selects visible pixels, and SAM3 produces segmentation
masks. The program controls part expansion, call order and budgets, cloud
processing, model comparison and acceptance.

Responsibilities and Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parent planner chooses task targets, requests geometry, executes actions
and obtains new observations when needed. GeometryAgent interprets the part
requirements, uses its internal perception and fitting capabilities, and
reports available measurements or missing evidence. Segmentation and
back-projection are internal operations; the parent requests measurements
through ``geometry``.

Session setup prepares shared observation storage and perception dependencies,
then initializes GeometryAgent outside the toolkit. The toolkit receives the
prepared instance and registers its interface. The session integration owns
the preparation of observations and perception dependencies; the subagent owns
geometry request processing.

Each request pins one recorded observation. All parts within that request use
the same frame. GeometryAgent reads observations and has no motion tools. When
evidence is insufficient, it reports the missing parts and reasons. The parent
decides whether to move the camera or robot, then submits another request
against the new observation.

Inputs and Outputs
~~~~~~~~~~~~~~~~~~

The parent supplies a target description, an observation camera and step, and
the required geometry. This example uses camera names from the current
integration:

.. code-block:: json

   {
     "query": "the yellow and white mug left of the gray cup",
     "camera": "agentview",
     "step": 0,
     "required_geometry": [
       "rim_center", "body_center", "bottom_center", "axis", "radius", "height"
     ]
   }

The parent supplies no calibration parameters in the request. The environment
integration records matching RGB, depth, world coordinates and camera metadata,
which GeometryAgent reads from the selected observation. ``step=-1`` pins the
latest recorded step when the request begins.

When ``required_geometry`` is set, read ``geometry[field]``. Each field includes
its value, definition, units, world-frame identifier, source step, shape
assumptions, quality and evidence references. For example, ``radius`` means rim
radius; ``body_center`` is the midpoint of observed coaxial rim and bottom
centers; ``height`` is their separation. These definitions determine the
required observations. An arbitrary visible point or a partial cloud's extent
cannot replace them.

For visible surface measurements, omit ``required_geometry`` and use
``measurement="surface_point"`` for a selected material point, or
``measurement="visible_region"`` for coordinate-wise median and bounds of a
segmented region. Read these from surface-mode fields such as ``world_xyz``.
Region statistics describe visible samples; a median may lie in empty space
and does not establish a fitted object or part center.

Before using a result, the parent checks its status and field definitions:

- ``usable``: all requested quantities passed the checks of the current mode.
- ``partial``: only some quantities are available; missing fields retain reasons.
- ``need_observation``: current evidence is insufficient; the parent decides
  whether to obtain another observation.
- ``error``: input, service or evidence persistence failed.

Fitted values are labelled ``inferred`` under their shape assumptions. In the
default mode, unsupported fields remain ``unknown`` with null values.
Residuals, support counts and segmentation scores are diagnostics, not
calibrated position errors. Results include evidence references, while full
masks and clouds are stored separately for inspection rather than sent as tool
result text.

Processing Flow
~~~~~~~~~~~~~~~

Measurement is organized around the requested parts. It performs the following
steps as needed:

1. **Locate the target and part.** The visual model selects visible pixels from
   RGB and the target description. It provides semantic localization, not
   world coordinates.
2. **Extract observed evidence.** Internal perception tools produce an object
   or part mask and obtain 3-D points from matching depth, world-coordinate
   maps and camera metadata. The current segmentation implementation uses SAM3.
3. **Clean the part cloud.** Retain connected regions consistent with the part
   and reduce background, outliers and other surfaces. Inner edges, top faces
   and curved outer surfaces use different candidate extraction methods.
4. **Choose a geometric model.** Fit circles to circular edges and planes to
   planar regions; compare cylinder and frustum models for approximately
   axisymmetric surfaces. The requested quantities determine which parts must
   be observed.
5. **Check and return measurements.** Use support, coverage, residuals,
   degeneracy and agreement between parts to determine which quantities have
   sufficient evidence. Retain diagnostics and evidence references.

A segmentation mask bounds the candidate region. Each geometric quantity still
needs evidence from its corresponding part. The center of a whole-object mask
does not establish an inner opening center, and a partially visible body's
length does not establish full height. A low residual shows agreement between
candidate points and a model; part coverage must also support the measurement.

.. _geometry-prompt-design:

Prompt Design
-------------

Parent Planner Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``geometry`` tool description and the parent's task prompt jointly define
the calling contract. The tool description specifies inputs, measurement
meaning, result interpretation and read-only boundaries. The current hybrid
evaluation and exploration prompts share the full geometry instructions:

- Identify targets, destinations and spatial relations in the global view.
  Preserve appearance, the specific part and those relations in ``query``;
  simulator instance suffixes do not replace visual identity.
- Supply the camera, step and measurement requirements to ``geometry``.
  Do not provide parent-selected pixels or guessed XYZ. Request surface
  contact points and fitted centers according to their distinct definitions.
- Read ``status``, ``definition``, ``evidence_status``, ``quality`` and the
  source step before using coordinates. For fits, inspect ``geometry[field]``
  and ``source_refs``, retaining unsupported fields as unknown.
- On ``need_observation`` or ``error``, inspect the reason and obtain useful new
  evidence within the task budget instead of repeating unchanged requests.
  Close views must preserve the same target and part. Measure again after
  target motion.

WAM mode uses a shorter task prompt that retains the requirement to measure
scripted motion targets through ``geometry`` and inspect status and measurement
definitions. The tool interface contract remains the same.

Expanding Part Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The program maps ``required_geometry`` to observed parts and prefixes the
original target description with each part requirement. The internal visual
model does not receive the full quantity list to decide the processing flow;
each call locates one expanded part. Selected mappings are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Requested Quantity
     - Expanded Observation Requirement
   * - ``rim_center``, ``radius``
     - The visible circular rim boundary.
   * - ``axis``
     - The visible curved outer body, excluding handle, rim and interior.
   * - ``body_center``, ``height``
     - Locate rim, outer body and bottom separately, then check agreement
       between parts.
   * - ``inner_rim_center``
     - The inner upper opening edge, excluding outer lip, handle and
       interior floor.

For an inner-opening request, the target description follows this prefix,
preserving the original appearance and spatial relation:

.. code-block:: text

   the visible INNER upper opening edge,
   excluding outer lip, handle and interior floor, of {target_description}

Internal Visual Grounding Prompt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The API/Codex visual-grounding adapter uses the following prompt.
``{query}`` stands for the expanded part description for this call;
``{width}`` and ``{height}`` stand for the original image dimensions:

.. code-block:: text

   Locate {query} in this robot camera image. Preserve the named object
   part and requested spatial relation. Select a clearly visible material
   point away from silhouettes, occlusion and image borders. Do not
   substitute another object or part. Return null if it cannot be identified.
   The image has {width} columns and {height} rows. Return point_xy as
   [column, row] in ORIGINAL pixel units, not normalized coordinates.
   Only select a point; do not use tools or execute robot actions.

The visual model receives only the frame's RGB and localization text, with no
depth, object truth or robot motion tools. Its structured result contains only
``point_xy``: original-pixel ``[column, row]`` when visible, or null when the
target cannot be identified. The program checks that the pixel is finite and
inside the image, then converts it to the internal row/column order. World
coordinates are computed from the matching observation data.

Segmentation Prompts and Call Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a region cloud is needed, the program first requests SAM3 with the part
text and checks that the mask contains the independently selected visual
point. If the text candidate is absent or excludes that point, one further
segmentation request may use the point as a positive prompt. Bottle-barrel
measurement is a special case: visual grounding targets the main barrel,
while segmentation text requests the whole bottle, including shoulder, neck
and base, to retain context for determining barrel limits.

Each part uses at most one visual-grounding call and two segmentation calls,
with no synonym search or threshold reduction. Segmentation bounds the
candidate region; edge extraction, cleaning and fitting still determine the
part center. Prompts constrain the model's semantic task, while the program
enforces call budgets, cloud processing and acceptance conditions.

Integration and Limits
----------------------

The request design supports spatial measurement from observations. The current
observation integration uses LIBERO. Another environment must supply matching
images, 3-D coordinates and camera information, while preserving the result's
coordinate frames and geometric definitions. Follow :doc:`../simulators/libero`
for environment setup and :doc:`../guides/cli` for visual-model and service URL
configuration.

Occlusion, invisible boundaries, missing depth or incorrect target localization
can leave some quantities undetermined in one frame. Complete box or cavity
recovery and multi-view fusion are not implemented. Measure again after target
motion. Comparing requests or applying memory offsets requires matching world
frames and reference-point definitions.

.. _geometry-permissive-evaluation:

Independent Error Evaluation
----------------------------

Measurement and evaluation are separate. GeometryAgent produces results from
observations; an independent evaluator then compares them with annotations or
simulator truth. Reference answers do not participate in visual localization,
candidate selection or fitting. Geometry position error and robot task success
are measured separately.

The default mode returns quantities according to geometric checks. To compare
center position errors, ``permissive=True`` or the experiment's ``--permissive``
option enables labelled approximate candidates for supported part centers
without requesting additional views. Read ``strict_accepted``,
``estimate_method`` and ``evaluation_only`` to distinguish accepted fits from
candidate estimates.

See the :ref:`fixed reference experiment <libero-fixed-reference-experiment>`
for the concrete protocol, and the
`geometry evaluation guide <https://github.com/RLinf/RPent/blob/main/toolkits/geometry/evaluation/README.md>`_
for candidate methods, thresholds and error statistics.
