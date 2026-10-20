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
   * - ``rpent/robots/components/am_contracts.py``
     - Declare shared physical observations/native control and WAM-specific capability, request and result dictionaries in separate sections.
   * - ``am_client_base.py``
     - Define the shared ``BaseAMClient`` and its VLA/WAM subclasses; share prediction dispatch and timeouts, keeping each contract's arguments and checks.
   * - ``am_facade_base.py``
     - Share prediction route registration and RPC session setup through ``BaseAMFacade``; keep VLA and WAM behavior in subclasses.
   * - ``am_runtime.py``
     - Share RPent RPC service attachment, process startup and failure cleanup; keep WAM backend selection and worker configuration in a separate section.
   * - ``<model>/runtime_config.py``
     - Supply model-specific arguments, defaults and configuration requirements.
   * - ``<model>/server.py``
     - Load the model and implement native inference.
   * - ``<model>/adapter/{__init__,encode,decode}.py``
     - Register paired transforms, build model input and restore native actions.

``BaseVLAClient`` and ``BaseWAMClient`` inherit from ``BaseAMClient`` in the same
``am_client_base.py`` module. Each keeps
its existing prediction arguments and result format. WAM capability checks and
reset/close remain in ``BaseWAMClient``; VLA lifecycle methods remain with the
integrations that support them.

``BaseVLAFacade`` and ``BaseWAMFacade`` likewise inherit from ``BaseAMFacade`` in
``am_facade_base.py``. VLA models retain their native payloads and additional
routes; WAM owns capability checks, paired adapters and reset hooks.
``AMObservation`` and ``AMControlSpec`` describe physical platform data shared
across model categories. ``WAMRequest`` extends the physical observation with
execution identity; it does not change existing VLA wire formats.

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
               -> am_runtime.start_am_service(daemon, endpoint)
                    -> start worker and return daemon + RPC client
          -> register owned daemon
     -> robots.runtime.try_wait_server(post_fn)
          -> wait for readiness
          -> create platform WAM client and check control compatibility
          -> report ready to Dashboard

``am_runtime.start_am_service()`` attaches to a borrowed endpoint or starts a
prepared daemon, preserving session settings and cleaning up failed starts.
RPent RPC VLA integrations in LIBERO, RoboCasa, RoboDojo, Franka, dual-Franka and
YAM share this function with WAM. Their platform/model configuration still owns
command construction. LingBot retains its native WebSocket connection path.
The shared ``robots/runtime.py`` owns readiness, component status and group
cleanup after resource handoff.

``am_runtime.add_am_device_arg()`` registers ``--vla-cuda-device`` or
``--wam-cuda-device`` for local model services. ``am_device_env()`` supplies
the child process's GPU visibility using only its model-specific option,
otherwise preserving the inherited environment.
Platforms register ``--env-cuda-device`` for the environment and forward model settings
when launching a worker. External endpoints retain their deployment settings.

The WAM section of ``am_runtime.py`` owns ``--platform``, ``--checkpoint``,
Python/GPU selection and deployment-mode checks. Model ``runtime_config.py``
modules supply only additional configuration; VLA launch flags remain unchanged.

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
package in ``am_runtime.WAM_BACKENDS`` and each supported platform in its
``ADAPTERS`` mapping. ``WAMAdapterSpec`` lives in ``am_facade_base.py`` and
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

WAM follows the existing VLA/SAM test layout. ``test_am.py`` covers shared
startup, VLA dispatch, WAM boundaries and worker configuration;
``test_wam_adapters.py`` checks camera/state
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

For repeated tasks, start a worker once using the commands below, then connect
each RPent run with ``--wam-endpoint``. The worker stays loaded when RPent exits;
stop it separately when finished. Alternatively, use the per-run deployment
examples below to let RPent start and stop the worker automatically.

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

Run Tasks Against an Existing Worker
------------------------------------

From the RPent simulator environment, after preparing the platform assets and
planner credentials (see :doc:`../simulators/libero`), run either example:

.. code-block:: bash

   # Cosmos worker already running on port 8116.
   rpent --robot libero --suite libero_spatial --task 0 \
     --wam-backend cosmos-policy --wam-endpoint http://127.0.0.1:8116 \
     --memory-profile local --planner api --model openai:gpt-5.5

   # Fast-WAM worker already running on port 8117.
   rpent --robot libero --suite libero_spatial --task 0 \
     --wam-backend fast-wam --wam-endpoint http://127.0.0.1:8117 \
     --memory-profile local --planner api --model openai:gpt-5.5

Change the task or suite for subsequent runs against the same compatible
worker. Model paths and inference options belong to worker deployment and are
not repeated in these task commands. Choose a planner model configured for
your provider. RoboTwin task commands are in :doc:`../simulators/robotwin`.

Let RPent Start a Worker
-----------------------

These alternatives start a fresh worker for each RPent run and stop it on exit.
Use the same provisioned environments and model files as in standalone
deployment; RPent does not download weights or install model dependencies.
Replace the example paths with your deployment paths:

.. code-block:: bash

   rpent --robot libero --suite libero_spatial --task 0 \
     --wam-backend cosmos-policy --wam-checkpoint /path/to/cosmos-libero \
     --wam-python /path/to/cosmos/.venv/bin/python --wam-root /path/to/cosmos \
     --memory-profile local --planner api --model openai:gpt-5.5

   rpent --robot libero --suite libero_spatial --task 0 \
     --wam-backend fast-wam --wam-checkpoint /path/to/fast-wam-libero.pt \
     --wam-python /path/to/fast-wam/venv/bin/python \
     --fast-wam-config /path/to/worker.yaml \
     --fast-wam-dataset-stats /path/to/dataset_stats.json \
     --memory-profile local --planner api --model openai:gpt-5.5

``--wam-endpoint`` and ``--wam-checkpoint`` are mutually exclusive. Cosmos
requires ``--wam-root``; ``COSMOS_POLICY_PYTHON`` and ``COSMOS_POLICY_ROOT``
provide fallbacks for its Python executable and working directory. Fast-WAM
requires both ``--fast-wam-config`` and ``--fast-wam-dataset-stats``.
``--wam-cuda-device`` selects the worker GPU independently of
``--env-cuda-device``. Without a model GPU option, the worker inherits CUDA
visibility. Environment GPU selection never supplies a model GPU default.

Discovering Options
-------------------

``rpent --robot libero --help`` and ``rpent --robot robotwin --help`` group WAM
options by backend selection, connection, local deployment and model. Shared
options use ``--wam-*``. Cosmos-specific options use ``--cosmos-*``; Fast-WAM
options use ``--fast-wam-*``. Model-specific options on the RPent command apply
only to local deployment. Each standalone worker's ``--help`` lists its own
unprefixed options, such as ``--config`` or ``--predict-future``.

Existing commands remain supported through hidden aliases:

.. list-table:: Model option names
   :header-rows: 1

   * - Previous option
     - Preferred option
   * - ``--wam-text-embeddings``
     - ``--cosmos-text-embeddings``
   * - ``--wam-predict-future``
     - ``--cosmos-predict-future``
   * - ``--wam-cached-instructions-only``
     - ``--cosmos-cached-instructions-only``
   * - ``--wam-config``
     - ``--fast-wam-config``
   * - ``--wam-dataset-stats``
     - ``--fast-wam-dataset-stats``
