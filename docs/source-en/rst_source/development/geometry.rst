.. _geometry-toolkit:

Geometry Toolkit
================

The geometry tool turns a description of an object or visible part into spatial
measurements with explicit definitions. The parent planner supplies a target,
the required quantities and an observation reference through one ``geometry``
interface. GeometryAgent coordinates visual grounding, segmentation, point
cloud cleaning and geometric fitting, returning positions, directions and
dimensions with their evidence.

Design Purpose
--------------

Grasping, placing and aligning objects require geometry of specific parts. A
rim center, a body center and a visible surface point have different meanings.
A selected pixel cannot stand in for all three. The tool defines each
measurement and returns it with the source observation and shape assumptions.

The parent decides what to measure and what action to take. GeometryAgent
determines which observed parts support that measurement. A visual model
handles semantic localization; metric observations and geometric algorithms
supply spatial calculations. This keeps task decisions separate from
measurement and gives different environments the same request and result
contract.

.. _geometry-module-design:

Module Design
-------------

GeometryAgent uses a fixed program flow. A vision-language model selects
visible pixels, and an injected segmenter produces masks. The program controls
part expansion, call order and budgets, cloud processing, model comparison and
acceptance.

Responsibilities and Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The host prepares observation access, shared evidence storage and perception
clients before constructing GeometryAgent outside its toolkit. It then
registers the prepared agent's ``geometry`` declaration. The agent owns
measurement processing; the host owns sensor acquisition, calibration,
coordinate-frame definitions and movement.

The constructor accepts an ``ObservationProvider``, the shared ``EnvState``,
a visual ``grounder`` and an optional ``segmenter``. Alternatively, ``model``
and ``base_url`` construct the existing visual-grounding adapter.
``fit_options`` configures the geometric algorithm gates. Surface-point
requests need a grounder; region and fitting requests also need a segmenter.
A compatible SAM3 client can be injected without starting a service here.

The host can register the tool through the ordinary toolkit interface. In this
illustration, ``state``, ``observations``, perception clients and ``toolkit``
are supplied by the host:

.. code-block:: python

   from functools import partial

   from toolkits.geometry.geometry_agent import GeometryAgent

   agent = GeometryAgent(
       state=state,
       observations=observations,
       grounder=grounder,
       segmenter=segmenter,
   )
   geometry = agent.geometry
   toolkit.add_tool(
       geometry.with_handler(
           partial(geometry, check_cancelled=toolkit.raise_if_cancelled)
       )
   )

Cancellation remains bound at dispatch. GeometryAgent has no motion tools and
does not advance the recorded observation step. Missing evidence is reported
to the parent, which decides whether to acquire a new view or perform a
recovery action.

Observation Contract
~~~~~~~~~~~~~~~~~~~~

The provider implements
``get_observation(*, camera: str, step: int) -> GeometryObservation``.
It resolves camera names and ``step=-1`` itself and returns one aligned
snapshot. The returned step must identify an existing shared ``EnvState``
record. GeometryAgent calls the provider once per request; all requested
parts use that snapshot.

The observation supplies data and coordinate semantics explicitly:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Fields
     - Contract
   * - ``camera``, ``step``, ``world_frame_id``
     - Caller-defined camera name, resolved recorded step and stable metric
       coordinate-frame identity.
   * - ``rgb``
     - Original-resolution RGB array of shape H by W by 3 with uint8 samples.
   * - ``world_xyz``
     - Aligned H by W by 3 point map in meters in the supplied reference frame.
   * - ``depth_m``, ``intrinsics``, ``camera_to_world``
     - Alternative to a point map: optical-axis depth in meters, pinhole 3 by 3
       intrinsics and a rigid 4 by 4 camera-to-reference transform. Camera Z is
       forward; the reference frame can have any orientation.
   * - ``valid_mask``
     - Optional aligned boolean validity mask. Nonfinite coordinates and
       nonpositive supplied depths are invalid. A finite point at the reference
       origin is valid.
   * - ``up_direction``, ``source_refs``
     - Optional up vector in the supplied frame and references identifying
       the source evidence.

Arrays are copied into a read-only snapshot. If depth and a point map are both
provided, depth also constrains sample validity. Camera poses can orient plane
normals and provide optical-depth context. There are no required sensor
filenames, camera-name enumeration or generated world-frame identifiers.

Background-plane removal uses the supplied ``up_direction``; without it,
cleanup skips automatic background-plane removal. The tool does not assume
that reference-frame Z is up. Providers convert their sensor depth convention
to metric optical-axis depth before creating the observation.

Request and Result Contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parent passes a target description, a provider-supported camera name,
a recorded step and the needed quantities. It supplies no calibration in the
tool call:

.. code-block:: json

   {
     "query": "the yellow and white mug left of the gray cup",
     "camera": "inspection_rgb",
     "step": 0,
     "required_geometry": [
       "rim_center", "body_center", "bottom_center", "axis", "radius", "height"
     ]
   }

With ``required_geometry``, read ``geometry[field]``. Each field contains its
value, definition, units, ``world_frame_id``, source step, shape assumptions,
quality and evidence references. ``radius`` means rim radius;
``body_center`` is the midpoint of observed coaxial rim and bottom centers;
``height`` is their separation.

For a visible material point, omit ``required_geometry`` and use
``measurement="surface_point"``. For the coordinate-wise median and bounds
of segmented visible samples, use ``measurement="visible_region"``.
These results use ``world_xyz`` in the provider's reference frame. Region
statistics do not establish a fitted center; a median may lie in empty space.

The parent checks the result status before planning with a value:

- ``usable``: all requested quantities passed the geometric checks.
- ``partial``: some quantities are available; unsupported fields remain unknown.
- ``need_observation``: current evidence is insufficient; reasons and missing
  parts guide the parent's next observation decision.
- ``error``: input, provider, model service or evidence persistence failed.

Fit-derived values are ``inferred`` under their reported assumptions.
Unsupported fields are ``unknown`` with null values. Residuals, support counts
and segmentation scores are diagnostics, not calibrated position errors.
Results reference the saved observation, masks, cleaned clouds and fitted
support; full arrays are stored separately from tool-result text.

Processing Flow
~~~~~~~~~~~~~~~

The program performs these steps as needed by the requested measurement:

1. **Pin the observation.** Resolve one source snapshot and preserve its camera,
   step, coordinate frame and evidence metadata.
2. **Locate and segment the part.** The visual model selects a material pixel.
   Region requests use a mask consistent with that independently selected point.
3. **Extract and clean points.** Use aligned metric coordinates and validity,
   then retain connected part evidence and reject outliers. Background removal
   requires an explicit up direction.
4. **Fit the requested shape.** Fit circular edges and planar patches, compare
   cylinder and frustum surfaces, or derive supported circular-face,
   rectangular-top and barrel centers.
5. **Check and return measurements.** Inspect support, coverage, residuals,
   degeneracy and agreement between parts. Save evidence and report missing
   quantities rather than substitute unrelated visible centers.

A whole-object mask does not establish an inner opening center. A partial
body's visible length does not establish full height. A low residual shows
agreement between candidate points and a model; the observed part must also
support the measurement's definition.

.. _geometry-prompt-design:

Prompt Design
-------------

Parent Planner Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GEOMETRY_INSTRUCTIONS`` supplies reusable parent instructions. The host
includes this text in its planner prompt alongside the registered tool
description:

.. code-block:: python

   from toolkits.geometry.prompts import GEOMETRY_INSTRUCTIONS

   system_prompt = task_instructions + "\n\n" + GEOMETRY_INSTRUCTIONS

The instructions ask the parent to preserve appearance, part identity and
spatial relations; select the correct surface or fitting mode; and read status,
definitions, coordinate-frame identity, source references and quality. The
parent describes the target instead of supplying guessed pixels or XYZ.

When a request is partial or needs new evidence, the parent inspects the reason
and chooses a useful next observation within its task budget. It preserves
target identity across views, compares matching coordinate frames and measures
again after target motion. The prompt and tool schema do not grant motion
capabilities to GeometryAgent.

Expanding Part Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The program maps ``required_geometry`` to observed parts and prefixes the
original target description with each part requirement. Each visual call
receives one expanded part description, preserving the original appearance
and spatial relation. For example:

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
     - Locate rim, outer body and bottom separately, then check agreement.
   * - ``inner_rim_center``
     - The inner upper opening edge, excluding outer lip, handle and interior
       floor.

Internal Visual Grounding Prompt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The existing API/Codex visual-grounding adapter receives only RGB and target
text. Its prompt uses the expanded ``{query}`` and original image dimensions:

.. code-block:: text

   Locate {query} in this robot camera image. Preserve the named object
   part and requested spatial relation. Select a clearly visible material
   point away from silhouettes, occlusion and image borders. Do not
   substitute another object or part. Return null if it cannot be identified.
   The image has {width} columns and {height} rows. Return point_xy as
   [column, row] in ORIGINAL pixel units, not normalized coordinates.
   Only select a point; do not use tools or execute robot actions.

Its structured result contains ``point_xy``: original-pixel
``[column, row]`` when visible, or null when unidentified. The program checks
image size and pixel bounds, converts to internal row/column order, and obtains
metric coordinates from the pinned observation. The visual model receives no
depth, object truth or robot tools.

Segmentation Prompts and Call Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a region, the program first requests segmentation with part text and
requires the mask to contain the independent visual point. If that candidate
is absent or excludes the point, one further call may use the point as a
positive prompt. Bottle-barrel measurements ground the barrel but request
a whole-bottle mask to preserve shoulder, neck and base context.

Each part uses at most one visual-grounding call and two segmentation calls,
with no synonym search or threshold reduction. Segmentation bounds candidate
evidence; the program determines part centers through cleaning and fitting.
The prompt constrains the semantic task, while code enforces call budgets and
geometric acceptance.

Limits and Runtime Verification
-------------------------------

Occlusion, missing depth, mixed surfaces or incorrect target localization can
leave quantities unknown. Complete box or cavity recovery and multi-view
fusion are not implemented. Compare only matching reference frames and point
definitions, and request new measurements after target motion.

Run the offline execution checks from an installation with the test
dependencies:

.. code-block:: bash

   python -m pytest tests/unit_tests/toolkits/geometry -q

These checks exercise generic observation input, toolkit dispatch, metric
projection, bounded perception, cancellation and fitting with synthetic data
and fake model services. Real-scene position accuracy and robot task success
require separate experiments.
