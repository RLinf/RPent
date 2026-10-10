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
   * - ``rpent/robots/components/wam_contracts.py``
     - Declare control, capabilities, request and result dictionaries in separate sections.
   * - ``wam_client_base.py`` / ``wam_facade_base.py``
     - Reuse RPent RPC, check boundary data, and manage reset/session hooks.
   * - ``wam_runtime.py``
     - Register backends; borrow an endpoint or launch an owned model process.
   * - ``<model>/runtime_config.py``
     - Supply model-specific arguments, defaults and configuration requirements.
   * - ``<model>/server.py``
     - Load the model and implement native inference.
   * - ``<model>/adapter/{__init__,encode,decode}.py``
     - Register paired transforms, build model input and restore native actions.

Physical observations are reusable by VLA and WAM integrations. Model encoders
remain distinct: Cosmos uses LIBERO's xyzw quaternion in a 9D state, whereas
Fast-WAM converts orientation to axis-angle for an 8D state. RoboTwin exposes
controller joint targets separately from measured joint positions; Fast-WAM
uses the targets from native ``joint_action.vector``.

The platform ``robot_spec.py`` connects the startup layers through callbacks:

.. code-block:: text

   robot_spec._init_runtime()
     -> robots.runtime.try_spawn_server(spawn_fn)
          -> WAMConfig.start_service()
               -> <model>.runtime_config.worker_arguments()
               -> start worker and return daemon + RPC client
          -> register owned daemon
     -> robots.runtime.try_wait_server(post_fn)
          -> wait for readiness
          -> create platform WAM client and check control compatibility
          -> report ready to Dashboard

``wam_runtime.py`` owns common launch arguments such as ``--platform`` and
``--checkpoint``, Python/GPU selection and deployment-mode checks. Model
configuration modules supply only their additional arguments. The shared
``robots/runtime.py`` owns readiness, component status and group cleanup;
``start_service()`` cleans up resources if launch fails before returning them.

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

For another model, add a package with ``runtime_config.py``, ``server.py`` and paired
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

Tests
-----

WAM follows the existing VLA/SAM test layout. ``test_wam.py`` covers shared
boundaries and worker configuration; ``test_wam_adapters.py`` checks camera/state
encoding and native action restoration on CPU. Platform tool behavior stays
with each platform's unit tests. ``test_wam_loopback.py`` uses the existing
HTTP/socket fixtures for session isolation, reset and cleanup.

Real-model checks live in each platform's ``test_components.py`` and
``test_policy_chain.py``. LIBERO parametrizes the same tests over Cosmos Policy
and Fast-WAM; RoboTwin uses its Fast-WAM checkpoint. After starting a worker,
run in the matching simulator environment:

.. code-block:: bash

   RPENT_COSMOS_ENDPOINT=http://127.0.0.1:8116 \
     pytest tests/e2e_tests/libero/test_components.py \
       tests/e2e_tests/libero/test_policy_chain.py -k 'wam or horizon' -v

   RPENT_FAST_WAM_ENDPOINT=http://127.0.0.1:8117 \
     pytest tests/e2e_tests/libero/test_components.py \
       tests/e2e_tests/libero/test_policy_chain.py -k 'wam or horizon' -v

   RPENT_FAST_WAM_ROBOTWIN_ENDPOINT=http://127.0.0.1:8117 \
     pytest tests/e2e_tests/robotwin/test_components.py \
       tests/e2e_tests/robotwin/test_policy_chain.py -k wam -v

Missing endpoint variables skip the corresponding WAM cases. LIBERO defaults
to ``libero_spatial``, task 0, seed 0; set ``RPENT_WAM_SUITE=libero_spatial_task``
and use Pro assets for LIBERO-Pro. Component tests record checkpoint identity,
one inference time and native success. CLI tests use a scripted planner,
execute bounded ``wam_act`` calls and check state artifacts and ``finish``.
They use neither Qwen nor Memory and do not estimate benchmark success rates.
These opt-in tests are separate from the standard GPU provisioning runner.

Worker Deployment
-----------------

Prepare the official model environment, checkpoint and auxiliary files first.
Install the RPent runtime dependencies there as well; ``PYTHONPATH`` only makes
the checkout importable and does not install dependencies. Keep the
simulator in the RPent environment. From the provisioned Cosmos source directory:

.. code-block:: bash

   PYTHONPATH=/path/to/RPent /path/to/cosmos/.venv/bin/python \
     -m rpent.robots.components.cosmos_policy.server \
     --checkpoint /path/to/cosmos-libero --cuda-device 0 --port 8116

The checkpoint directory contains ``Cosmos-Policy-LIBERO-Predict2-2B.pt``,
``libero_dataset_statistics.json`` and ``libero_t5_embeddings.pkl``.
``--dataset-stats`` and ``--text-embeddings`` override the auxiliary paths.
Defaults are five action denoising steps, seed 1, 16 actions per prediction,
and no future-state/value generation. ``--cached-instructions-only`` rejects
uncached instructions; otherwise the official runtime handles uncached text.
``--predict-future`` enables optional outputs, which RPent does not use for planning.

For Fast-WAM, use its provisioned environment:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/RPent \
     /path/to/fast-wam/venv/bin/python \
     -m rpent.robots.components.fast_wam.server \
     --platform libero --checkpoint /path/to/fast-wam-libero.pt \
     --config /path/to/worker.yaml --dataset-stats /path/to/dataset_stats.json \
     --port 8117

RoboTwin needs ``--platform robotwin`` with its own checkpoint and configuration.
The YAML supplies resolved Hydra ``model`` and ``processor`` mappings, with the
processor taken from the official configuration's ``data.train.processor``, plus
``action_horizon``, ``execute_steps``, ``video_size`` and ``concat``.
Optional ``inference_options`` are forwarded to ``model.infer_action`` and cannot
override encoded observations or ``action_horizon``. Enable the text encoder in
the model configuration for uncached instructions. ``execute_steps`` is positive,
no larger than ``action_horizon``, and advertised as executable ``chunk_size``.
``binarize_gripper`` applies only to LIBERO. Future video generation, temporal
action ensembling and automatic Hydra composition are not implemented.

To let RPent own a worker, replace ``--wam-endpoint`` in the platform command
with ``--wam-checkpoint`` and ``--wam-python``. Cosmos also needs ``--wam-root``;
``COSMOS_POLICY_PYTHON`` and ``COSMOS_POLICY_ROOT`` provide fallbacks. Fast-WAM
needs ``--wam-config`` and ``--wam-dataset-stats``. Endpoint and checkpoint modes
are mutually exclusive. Configure borrowed workers at their own startup; use
each worker's ``--help`` to inspect its launch options.
