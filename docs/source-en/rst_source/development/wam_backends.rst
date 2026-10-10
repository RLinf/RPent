WAM Backends
============

RPent separates platform observations and action execution from model inference.
Cosmos Policy supports LIBERO and LIBERO-Pro; Fast-WAM supports LIBERO,
LIBERO-Pro and RoboTwin. Each pair requires a matching checkpoint.
Deployment examples are in :doc:`../simulators/libero` and
:doc:`../simulators/robotwin`.

Architecture
------------

.. code-block:: text

   CLI / Dashboard
     -> robots/<platform>/robot_spec.py: configuration and service startup
     -> toolkit.py: register platform tools
     -> wam_act: observe, predict, execute, record
     -> observation.py + wam_client.py: physical observations and control identity
     -> BaseWAMClient -> existing HTTP/socket RPC -> BaseWAMFacade
     -> <model>/adapter/encode.py -> predict_native -> adapter/decode.py
     -> action chunk -> platform env_client.chunk_step -> simulator

``wam_act`` lives in LIBERO's ``tools.py`` and RoboTwin's ``primitives.py``.
Both use ``@tool`` and return ``ToolResult(data=...)``. The existing platform
toolkit owns registration, cancellation, state capture and recording.

.. list-table:: Ownership
   :header-rows: 1

   * - Location
     - Responsibility
   * - ``robots/<platform>/observation.py``
     - Extract named physical sensors and state. No model normalization.
   * - ``robots/<platform>/control.py``
     - Define native action order, semantics and execution mode.
   * - ``robots/<platform>/wam_client.py``
     - Select a compatible controller and attach execution identity to observations.
   * - ``rpent/robots/components/wam_rpc_protocol.py``
     - Declare request, capabilities and prediction dictionaries with ``TypedDict``.
   * - ``wam_control_spec.py``
     - Declare the control dictionary shared by platform clients and model adapters.
   * - ``wam_client_base.py`` / ``wam_facade_base.py``
     - Reuse RPent RPC, check boundary data, and manage reset/session hooks.
   * - ``wam_runtime.py``
     - Register backends; borrow an endpoint or launch an owned model process.
   * - ``<model>/runtime.py`` / ``server.py``
     - Own launch options, model loading and native inference.
   * - ``<model>/adapter/{__init__,encode,decode}.py``
     - Register paired transforms, build model input and restore native actions.

Physical observations are reusable by VLA and WAM integrations. Model encoders
remain distinct: Cosmos uses LIBERO's xyzw quaternion in a 9D state, whereas
Fast-WAM converts orientation to axis-angle for an 8D state. RoboTwin exposes
controller joint targets separately from measured joint positions; Fast-WAM
uses the targets from native ``joint_action.vector``.

.. _wam-protocol:

Communication and Lifecycle
---------------------------

The routes are ``wam.capabilities``, ``wam.predict`` and ``wam.reset``.
Requests contain ``images``, ``state``, ``instruction``, ``embodiment`` and
``action_space``. Capabilities contain the ``control`` dictionary, required
cameras and state dimensions, executable ``chunk_size``, backend/checkpoint
identity and session settings. Predictions contain ``actions`` and may contain
``future_observation``, ``value`` and ``metadata``.

Messages are dictionaries and NumPy arrays handled by the existing transport.
There is no object-to-wire conversion or protocol-version field. Deploy clients
and workers from the same checkout. At connection, the platform compares the
worker's control dictionary with its supported control definition and caches the
selection. Create a new client when switching services or control modes.
The service checks incoming observations; the client checks returned action
shape, chunk length and finite numeric values before execution. Decoders retain
checks needed before transformations that could hide invalid output, such as
gripper binarization or selecting an action prefix.

Platform clients expose ``predict()`` for actions and ``predict_result()`` for
the full dictionary. Episode reset clears the environment and model history.
RoboTwin reuses the environment's initial reset and clears model history when
constructing the toolkit. Connecting a client does not reset an active episode;
direct users call ``reset()`` before their first prediction and ``close()`` when
finished. Session-aware workers implement ``reset_native``; both current model
packages declare ``USES_SESSIONS = False`` once for runtime and capabilities.

Model dependencies live in separately provisioned environments. The shared
RPent installation does not install Cosmos Policy or Fast-WAM dependencies.
Owned workers use the existing daemon lifecycle; borrowed endpoints remain
running when RPent stops its owned services.

Extension Points
----------------

For another model, add a package with ``runtime.py``, ``server.py`` and paired
``adapter/encode.py`` and ``adapter/decode.py`` implementations. Register the
package in ``wam_runtime.BACKENDS`` and each supported platform in its
``ADAPTERS`` mapping. ``WAMAdapterSpec`` lives in ``wam_facade_base.py`` and
groups encode, decode and capability construction. Reuse existing platform
clients when their physical observations and control definitions fit.

For another platform, provide physical observations, native control definitions,
a platform client and tool/runtime wiring. Each supported model still needs a
matching checkpoint and paired transforms; registration does not establish
checkpoint compatibility. RoboTwin's EEF16 controller is defined, but its
current Fast-WAM adapter uses qpos14. Cosmos Policy has no RoboTwin adapter.

``future_observation`` and ``value`` reserve space for model predictions.
Separate forecasting/planning RPCs, planner consumption of predictions and WAM
memory integration are not implemented. WAM evaluation currently excludes
Flash and exploration. Passing ``memory=None`` skips memory file tools while
retaining ``finish`` and platform tools.

Offline checks cover adapters, control matching, malformed observations/actions
and real local HTTP/socket session lifecycle. Opt-in GPU checks exercise model
inference and bounded simulator/toolkit execution; see ``tests/README.md``.
Passing those checks establishes integration behavior, not benchmark success.
