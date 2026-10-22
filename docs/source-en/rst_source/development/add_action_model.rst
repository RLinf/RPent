Add an Action Model
===================

Integrate a VLA or WAM service and connect its predictions to a platform's
native action executor. Tool declarations and state capture are described in
:doc:`add_primitive`; platform discovery and runtime wiring are in :doc:`add_robot`.

Shared Architecture
-------------------

Model packages own weights and model-specific preprocessing. Platforms own
physical observations, controllers and execution tools. The shared AM bases
reuse RPC transport and dispatch; VLA keeps model-specific payloads, while WAM
uses the physical observation and native control declarations in
``am_contracts.py``.

The WAM execution path illustrates the platform/model boundary:

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
     - Register common deployment options, resolve VLA/WAM settings into ``AMConfig``, and start workers with their Python, working directory and GPU configuration.
   * - ``rpent/robots/components/pi05_vla_client.py``
     - Adapt platform observations and model outputs for Pi0.5; other VLA integrations provide their own model clients.
   * - ``<model>/runtime_config.py``
     - Supply WAM model-specific arguments, defaults and configuration requirements.
   * - ``<model>/server.py``
     - Load a WAM model and implement native inference.
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

The platform ``robot_spec.py`` connects the startup layers through callbacks.
For a WAM worker, the call chain is:

.. code-block:: text

   robot_spec._init_runtime()
     -> am_runtime.select_am(args, platform): resolve AMConfig and model options
     -> robots.runtime.try_spawn_server(spawn_fn)
          -> am_runtime.start_wam_service(config, output_dir)
               -> AMConfig.spawn_worker() -> ProcessDaemon.start()
               -> make_rpc_client()
               -> return daemon + RPC client
          -> register owned daemon
     -> robots.runtime.try_wait_server(post_fn)
          -> wait for readiness
          -> create platform WAM client and check control compatibility
          -> report ready to Dashboard

VLA launchers also use ``AMConfig.spawn_worker()``. Platforms retain their
transport and model command arguments, including LingBot's native WebSocket
connection and RoboCasa's session-enabled RPC. ``ProcessDaemon`` closes its
log on a failed spawn. The shared
``robots/runtime.py`` owns readiness, component status and group cleanup after
resource handoff. Borrowed endpoints return no owned daemon.

``am_runtime.add_am_args(parser, platform=...)`` registers supported backends
and common options. ``AMConfig.spawn_worker()`` forwards ``--am-cuda-device``
as the child's ``CUDA_VISIBLE_DEVICES`` override;
omitting it uses ``--cuda-device``, or preserves the inherited environment
when neither is set.
Platforms register ``--env-cuda-device`` for the environment and forward model settings
when launching a worker. External endpoints retain their deployment settings.

``AMConfig`` stores the resolved backend, platform, endpoint, checkpoint,
Python, working directory, GPU and additional worker arguments. Model defaults
are resolved without modifying the parsed namespace, so Dashboard reuse does
not mistake defaults for explicit options. WAM ``runtime_config.py`` modules
provide model-specific defaults and worker arguments; VLA platform launchers
retain their native command construction.

Add a VLA
---------

Because the model runs in its own process, adding a model-based
primitive requires a few additional components:

1. **Write ``vla_server.py``.** This process owns only the model weights
   and CUDA context. Use
   :class:`rpent.robots.components.am_facade_base.BaseVLAFacade` as the base
   class, implement ``predict``, and register any additional model RPCs by
   extending ``_register_rpc``:

   - The default transport is **HTTP** (JSON over ``POST /call``),
     which works well for flat ``image + state`` payloads such as the
     LIBERO / Pi0.5 pattern.
   - Switch to **socket RPC** (``--transport socket``) if your obs is
     a nested dict of numpy arrays with history stacks (avoids the
     JSON re-encode overhead).

   ``BaseVLAFacade`` registers ``vla.predict`` and serializes model calls;
   its inherited ``RpcFacade.serve`` handles transport binding, ``healthz``,
   ``shutdown``, parent-death detection, and resource cleanup.

2. **Write a model client.** Subclass
   :class:`rpent.robots.components.am_client_base.BaseVLAClient`, which
   provides the common ``vla.predict`` call, and add only the
   environment-specific input / output adaptation. See
   ``rpent.robots.components.pi05_vla_client.Pi05VLAClient`` for the LIBERO
   implementation.

3. **Add a method to the primitives.** In the current
   robot's primitives class, call the model client, pass
   the returned action chunk to the environment, and return a
   ``ToolResult``. The model client API is
   :meth:`rpent.robots.components.pi05_vla_client.Pi05VLAClient.predict`,
   which reads the instruction from ``env_obs["task_descriptions"]`` and
   returns a ``[chunk, action_dim]`` numpy action chunk (batch dim already
   stripped):

   .. code-block:: python

      @tool
      def mymodel_pick(self, target: str) -> ToolResult:
          """Pick the requested target using the model."""
          env_obs = self._env.get_obs()
          env_obs["task_descriptions"] = f"pick {target}"
          chunk = self._model.predict(env_obs)
          self._env.chunk_step(chunk)
          return ToolResult(data={"model": "mymodel", "target": target})

4. **Register the declared tool in the toolkit.** Follow the
   same pattern as for a scripted primitive.

5. **Wire the components together in ``robot_spec.py``.** The
   robot's ``get_toolkit`` builds the toolkit with
   ``runtime_kwargs``:

   .. code-block:: python

      def get_toolkit(*, runtime_kwargs, dashboard_events, config):
          from robots.myrobot.toolkit import MyRobotToolkit
          from rpent.memory import MemoryManager

          return MyRobotToolkit(
              runtime_kwargs=runtime_kwargs,
              dashboard_events=dashboard_events,
              memory=MemoryManager(config.prompt_vars["memory_dir"]),
          )

   The robot package's ``_init_runtime`` builds
   ``runtime_kwargs``, for example
   ``{"env": MyRobotEnvClient(...), "model": MyModelClient(...)}``.
   The toolkit constructor then forwards it to the primitives.

Reuse an Existing VLA Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model servers often take a long time to start, so the runner can
connect to an instance that is already running:

.. code-block:: bash

   rpent --robot libero --am-endpoint http://vla-host:8000 ...

If the model keeps per-episode state, expose a ``vla_reset`` RPC and
call it between tasks. The same server process can then be reused safely
across sequential runs.

Per-client Policy State
~~~~~~~~~~~~~~~~~~~~~~~

Most VLA backends are stateless: ``predict`` only runs inference and keeps
no per-client state, so ``session_id`` can be ignored. Some models do carry
per-client policy state (e.g. RLDX-1's memory/RTC); when a single
``vla_server`` serves multiple clients, their policy state would
cross-contaminate, so it must be isolated per session. Wiring it up in three
parts:

- **Facade side**: construct the ``BaseVLAFacade`` subclass with
  ``enable_sessions=True`` and ``session_timeout_s``, and implement
  ``_on_session_drop`` — clean up that client's policy state when the session
  ends (the client's ``session.close`` RPC or idle expiry). If you need an
  explicit reset, expose an extra ``reset_session`` RPC (clears policy state
  only, does not destroy the session). ``serve`` must pass ``session_sweep_s``
  (> 0) so a background thread periodically reclaims expired sessions.

- **Client side**: construct the ``RpcClient`` inside the model client with
  ``enable_sessions=True``; it registers a session with the server on
  connect. ``session_id`` is derived from the connection and injected into
  the server-side handler by the facade — the client does **not** pass it,
  and must not forge ``session_ids`` inside ``predict``'s ``options``.

- **Primitives side**: call ``reset_session`` before a task starts to clear
  policy state left over from the previous episode, so consecutive runs do
  not leak state into each other.

Add a WAM
---------

Cosmos Policy supports LIBERO and LIBERO-Pro; Fast-WAM supports LIBERO,
LIBERO-Pro and RoboTwin. Each model/platform pair requires a matching checkpoint.

1. Create a model package under ``rpent/robots/components/``. Its
   ``server.py`` subclasses ``BaseWAMFacade`` and implements
   ``predict_native``; provide a required ``(encode, decode)`` adapter pair.
2. Register each supported platform in the model's ``ADAPTERS`` mapping
   using ``WAMAdapterSpec`` from ``am_facade_base.py``. The model metadata factory declares checkpoint
   identity, camera/state requirements, native control and executable chunk size.
   Encode physical observations into model inputs; decode predictions into the
   declared platform action format.
3. Reuse a platform's ``wam_client.py`` when its observation and control
   contracts fit. ``BaseWAMClient`` fetches ``wam.get_model_meta`` once at
   initialization; the platform checks control compatibility before executing
   predictions. The client retains metadata in ``model_meta``.
4. Add model deployment options in ``runtime_config.py`` and register the
   backend in ``am_runtime.WAM_BACKENDS``. Declare session usage in the
   model package, and implement ``reset_native`` if inference retains
   session history.
5. Wire the selected backend into the platform's runtime and tool execution.
   Reset model history at episode boundaries and close clients when finished.
   Every toolkit requires a ``MemoryManager``.

.. _wam-protocol:

Communication and Lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The routes are ``wam.get_model_meta``, ``wam.predict`` and ``wam.reset``.
Requests contain ``images``, ``state``, ``instruction``, ``embodiment`` and
``action_space``. Model metadata contains the ``control`` dictionary, required
cameras and state dimensions, executable ``chunk_size``, backend/checkpoint
identity and session settings. Predictions contain ``actions`` and may contain
``future_observation``, ``value`` and ``metadata``.

Messages are dictionaries and NumPy arrays handled by the existing transport.
There is no object-to-wire conversion or protocol-version field. Deploy clients
and workers from the same checkout. ``BaseWAMClient`` fetches metadata once in
its constructor and retains it in ``model_meta``; predictions use that snapshot.
At connection, the platform compares the
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

Add a Platform and Check Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For another platform, provide physical observations, native control definitions,
a platform client and tool/runtime wiring. Each supported model still needs a
matching checkpoint and paired transforms; registration does not establish
checkpoint compatibility. RoboTwin's EEF16 controller is defined, but its
current Fast-WAM adapter uses qpos14. Cosmos Policy has no RoboTwin adapter.

``future_observation`` and ``value`` reserve space for model predictions.
Separate forecasting/planning RPCs and planner consumption of predictions are
not implemented. Every toolkit requires a ``MemoryManager`` and registers memory
file tools. WAM uses local memory under the platform's memory root, in a
subdirectory named after the backend; ``--memory-dir`` overrides that directory.
The corpus may be empty. No compatible HF corpus is configured for WAM, and WAM
evaluation currently excludes HF memory, Flash and exploration.

.. _action-model-deployment:

Action-Model Deployment and Task Runs
--------------------------------------

VLA and WAM share six deployment options:

.. list-table:: Common action-model options
   :header-rows: 1

   * - Option
     - Purpose
   * - ``--am-backend``
     - Select one model supported by the platform.
   * - ``--am-endpoint``
     - Connect to an already deployed model service.
   * - ``--am-checkpoint``
     - Select weights for a locally launched worker.
   * - ``--am-python``
     - Select the model environment's Python; VLA defaults to the current interpreter.
   * - ``--am-root``
     - Set the worker's working directory; defaults to the RPent checkout, except Cosmos which requires its provisioned source directory.
   * - ``--am-cuda-device``
     - Set the local model's visible GPUs; falls back to ``--cuda-device``, then the inherited environment.

Franka and YAM connect to external services and do not expose local deployment
options. Explicit local deployment options, including model-specific ones,
are rejected with ``--am-endpoint``. Environment-variable defaults for an
unselected or external model are ignored. Other services may share the model GPU.

Start a Standalone Worker
~~~~~~~~~~~~~~~~~~~~~~~~~

For repeated tasks, start a worker once using the commands below, then connect
each RPent run with ``--am-endpoint``. The worker stays loaded when RPent exits;
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the RPent simulator environment, after preparing the platform assets and
planner credentials (see :doc:`../simulators/libero`), run either example:

.. code-block:: bash

   # Cosmos worker already running on port 8116.
   rpent --robot libero --suite libero_spatial --task 0 \
     --am-backend cosmos-policy --am-endpoint http://127.0.0.1:8116 \
     --memory-profile local --planner api --model openai:gpt-5.5

   # Fast-WAM worker already running on port 8117.
   rpent --robot libero --suite libero_spatial --task 0 \
     --am-backend fast-wam --am-endpoint http://127.0.0.1:8117 \
     --memory-profile local --planner api --model openai:gpt-5.5

Change the task or suite for subsequent runs against the same compatible
worker. Model paths and inference options belong to worker deployment and are
not repeated in these task commands. Choose a planner model configured for
your provider. RoboTwin task commands are in :doc:`../simulators/robotwin`.

Let RPent Start a Worker
~~~~~~~~~~~~~~~~~~~~~~~~

These alternatives start a fresh worker for each RPent run and stop it on exit.
Use the same provisioned environments and model files as in standalone
deployment; RPent does not download weights or install model dependencies.
Replace the example paths with your deployment paths:

.. code-block:: bash

   rpent --robot libero --suite libero_spatial --task 0 \
     --am-backend pi05 --am-checkpoint /path/to/pi05-libero \
     --am-python /path/to/pi05-env/bin/python \
     --planner api --model openai:gpt-5.5

   rpent --robot libero --suite libero_spatial --task 0 \
     --am-backend cosmos-policy --am-checkpoint /path/to/cosmos-libero \
     --am-python /path/to/cosmos/.venv/bin/python --am-root /path/to/cosmos \
     --memory-profile local --planner api --model openai:gpt-5.5

   rpent --robot libero --suite libero_spatial --task 0 \
     --am-backend fast-wam --am-checkpoint /path/to/fast-wam-libero.pt \
     --am-python /path/to/fast-wam/venv/bin/python \
     --fast-wam-config /path/to/worker.yaml \
     --fast-wam-dataset-stats /path/to/dataset_stats.json \
     --memory-profile local --planner api --model openai:gpt-5.5

``--am-endpoint`` and ``--am-checkpoint`` are mutually exclusive. Cosmos
requires ``--am-root``; ``COSMOS_POLICY_PYTHON`` and ``COSMOS_POLICY_ROOT``
provide fallbacks for its Python executable and working directory. Fast-WAM
requires both ``--fast-wam-config`` and ``--fast-wam-dataset-stats``.
``--am-cuda-device`` selects the worker GPU independently of
``--env-cuda-device``. Without a model GPU option, the worker uses the shared
``--cuda-device`` default, or inherits CUDA visibility if neither is set.
The services may share a GPU: specify the same device for each service when
GPU memory permits.

Discovering Options
~~~~~~~~~~~~~~~~~~~

Help groups options by backend selection, connection, local deployment and
selected model. Put the backend before ``--help`` to see that model's options:

.. code-block:: bash

   rpent --robot libero --am-backend cosmos-policy --help
   rpent --robot robotwin --am-backend fast-wam --help

Without ``--am-backend``, help shows the platform's default model. Common options
use ``--am-*``; model options retain ``--cosmos-*``, ``--fast-wam-*``,
``--lingbot-*`` and ``--pi05-*``. Explicit options for another model are rejected.
Each standalone worker's ``--help`` lists its own
unprefixed options, such as ``--config`` or ``--predict-future``.

Existing commands remain supported through hidden aliases:

.. list-table:: Model option names
   :header-rows: 1

   * - Previous option
     - Preferred option
   * - ``--vla-backend``, ``--wam-backend``
     - ``--am-backend``
   * - ``--policy-backend rlinf`` (RoboDojo)
     - ``--am-backend pi05``
   * - ``--vla-endpoint``, ``--wam-endpoint``
     - ``--am-endpoint``
   * - ``--vla-model-path``, ``--wam-checkpoint``
     - ``--am-checkpoint``
   * - ``--pi05-python``, ``--wam-python``
     - ``--am-python``
   * - ``--wam-root``
     - ``--am-root``
   * - ``--vla-cuda-device``, ``--wam-cuda-device``
     - ``--am-cuda-device``
   * - ``--vla-repo-id``
     - ``--pi05-repo-id``
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
They use a local memory fixture without task guidance and no Qwen planner;
they do not estimate benchmark success rates or memory benefits.
These opt-in tests are separate from the standard GPU provisioning runner.

Passing these checks establishes integration behavior, not benchmark success.
See ``tests/README.md`` for environment prerequisites and the full test workflow.
