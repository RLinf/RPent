Add an Action Model
===================

Integrate a VLA or WAM service and connect its predictions to a platform's
native action executor. Tool declarations and state capture are described in
:doc:`add_primitive`; platform discovery and runtime wiring are in :doc:`add_robot`.

Shared Architecture
-------------------

Model packages reuse RLinf's supported inference and numerical transforms;
only unsupported model/platform pairs need their own implementation. Platforms own
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
     - Load a WAM model or connect its native service, and implement the inference call.
   * - ``<model>/adapter/{__init__,encode,decode}.py``
     - Register paired transforms, build model input and restore native actions.

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
and common options. GPU selection and compatibility aliases are documented in
:doc:`../guides/cli`.

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

For existing-service deployment and endpoint examples, see
:doc:`../guides/advanced_deployment`. Stateful models also need the
per-client lifecycle hooks below.

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

Cosmos Policy and Cosmos3 support LIBERO and LIBERO-Pro; Fast-WAM supports LIBERO,
LIBERO-Pro and RoboTwin. Each model/platform pair requires a matching checkpoint.

``cosmos-policy`` uses NVIDIA's released ``Cosmos-Policy-LIBERO-Predict2-2B``
checkpoint and official LIBERO inference API. ``cosmos3`` connects to a
separately deployed SGLang ``/v1/actions/generations`` service using RLinf's
``Cosmos3SGLangAdapter`` and HTTP client. RLinf handles image composition,
request construction, action denormalization and rotation conversion; its
``prepare_actions_for_libero`` also supplies the simulator's gripper convention.
RPent reuses its existing LIBERO client and maps observations and predictions
at that boundary. These backends require different checkpoints.

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
   backend in ``am_runtime.WAM_BACKENDS``; model-specific launch logic stays
   in the model package. Declare session usage in the
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
An optional ``prediction_timeout_s`` supplies the client's request budget;
without it, the default is 300 seconds. An explicit client timeout takes precedence.
Cosmos3 advertises its native HTTP timeout plus 30 seconds for RPC overhead.

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
finished. Session-aware workers implement ``reset_native``; current model
packages declare ``USES_SESSIONS = False`` once for runtime and capabilities.

Model dependencies live in separately provisioned environments. The shared
RPent installation does not install the model workers' dependencies.
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

Common action-model options, GPU selection and compatibility aliases are described
in :doc:`../guides/cli`. Each standalone worker's ``--help`` lists its own
unprefixed options, such as ``--config`` or ``--predict-future``.

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

The LIBERO worker requires RLinf's ``release/v0.4`` Fast-WAM integration and
the Fast-WAM revision selected by RLinf's ``requirements/install.sh`` (currently
``45d8e1458921d83f8ad6cf9ce993d371208dabd0``). Prepare these together using
RLinf's ``bash requirements/install.sh embodied --model fastwam``; arbitrary
Fast-WAM revisions can have incompatible VAE interfaces.
It calls RLinf's ``get_model`` and ``predict_action_batch``;
RLinf owns image resizing/composition, state normalization, inference and
LIBERO action restoration. RPent maps physical observations to RLinf's input
fields and removes the single-environment batch dimension from the result.
RoboTwin retains its official Fast-WAM path because this RLinf policy does
not implement its three-camera/qpos14 interface.

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/RPent \
     /path/to/fast-wam/venv/bin/python \
     -m rpent.robots.components.fast_wam.server \
     --platform libero --checkpoint /path/to/fast-wam-libero.pt \
     --config /path/to/worker.yaml --dataset-stats /path/to/dataset_stats.json \
     --port 8117

LIBERO uses RLinf's native model YAML. RPent supplies the checkpoint and dataset
statistics paths from the deployment arguments; RLinf composes the official
model/processor configuration and resolves inference options. For example:

.. code-block:: yaml

   action_horizon: 32
   num_action_chunks: 10
   num_inference_steps: 20
   binarize_gripper: true
   seed: 1
   fastwam:
     config_dir: /path/to/FastWAM/configs
     config_name: sim_libero
     overrides: []

Match the official configuration and overrides to the checkpoint. Model and
processor composition belongs to ``fastwam.config_dir/config_name/overrides``.
The worker advertises the loaded policy's executable chunk size.

RoboTwin needs ``--platform robotwin`` with its own checkpoint and configuration.
Its YAML supplies resolved Hydra ``model`` and ``processor`` mappings, with the
processor taken from the official configuration's ``data.train.processor``, plus
``action_horizon``, ``execute_steps``, ``video_size`` and ``concat``.
``inference_options`` are forwarded to the official ``model.infer_action``.
Options cannot override encoded
observations or ``action_horizon``. Enable the text encoder in
the model configuration for uncached instructions. ``execute_steps`` is positive,
no larger than ``action_horizon``, and advertised as executable ``chunk_size``.
The worker checks the fixed three-camera layout, merged state/action keys and
absence of LIBERO gripper binarization at startup. Predictions still check native
output shape and finite values before selecting the execution prefix.

For Cosmos3, prepare a task-trained Cosmos3 checkpoint in Diffusers component
format and deploy SGLang using RLinf's ``Cosmos3 SGLang Evaluation`` guide.
Use an RLinf checkout containing ``Cosmos3SGLangAdapter`` and its matching
SGLang dependencies. The RPC worker runs in that environment:

.. code-block:: bash

   PYTHONPATH=/path/to/RPent:/path/to/RLinf /path/to/sglang-env/bin/python \
     -m rpent.robots.components.cosmos3.server \
     --platform libero --checkpoint nvidia/cosmos3-nano-libero10-sdg-export-S2_s43 \
     --config /path/to/cosmos3.yaml \
     --sglang-endpoint http://127.0.0.1:30000 --port 8118

``--checkpoint`` identifies the model served by SGLang; the RPC worker does not
load weights or require them on its host. It requires the local action statistics
referenced by its configuration. ``cosmos3.yaml``
contains resolved RLinf configuration, without Hydra ``defaults`` composition.
For a LIBERO rot6d checkpoint trained with the standard Cosmos3 recipe:

.. code-block:: yaml

   env:
     eval:
       env_type: libero
   rollout:
     model:
       raw_action_dim: 10
       action_dim: 7
       action_normalization: quantile_rot
       action_stats_path: /path/to/libero_native_frame_wise_relative_rot6d.json
       num_action_chunks: 16
       action_mode: policy
       action_fps: 20.0
       fps: 20
       num_inference_steps: 30
       action_view_point: ego_view
       num_frames: 17
       size: 320x192
     sglang:
       seed: 1
       http_timeout_s: 600
       server:
         batching_max_size: 1

Match these settings and normalization statistics to the checkpoint's training
recipe. SGLang owns the model process and GPU allocation; stopping the RPent
worker does not stop SGLang. RPent's ``--am-cuda-device`` configures the local
RPC worker process; it does not change that external service's devices.

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

   # Cosmos3 RPC worker already running on port 8118.
   rpent --robot libero --suite libero_10 --task 0 \
     --am-backend cosmos3 --am-endpoint http://127.0.0.1:8118 \
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
Cosmos3 requires ``--cosmos3-config`` and ``--cosmos3-sglang-endpoint``;
this starts the RPC worker against an already running SGLang service.

Tests
-----

Extend the existing adapter, RPC and platform tests for each new model/platform
pair. Test locations, GPU prerequisites, commands and the distinction between
integration checks and task-success benchmarks are documented in
``tests/README.md``.
