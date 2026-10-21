Add an Action Model
===================

Integrate a VLA or WAM service and connect its predictions to a platform's
native action executor. Tool declarations and state capture are described in
:doc:`add_primitive`; platform discovery and runtime wiring are in :doc:`add_robot`.

Shared Boundaries
-----------------

Model packages own weights and model-specific preprocessing. Platforms own
physical observations, controllers and execution tools. The shared AM bases
reuse RPC transport and dispatch; VLA keeps model-specific payloads, while WAM
uses the physical observation and native control declarations in
``am_contracts.py``.

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

Reuse an Existing vla_server Across Runs
----------------------------------------

Model servers often take a long time to start, so the runner can
connect to an instance that is already running:

.. code-block:: bash

   rpent --robot libero --am-endpoint http://vla-host:8000 ...

If the model keeps per-episode state, expose a ``vla_reset`` RPC and
call it between tasks. The same server process can then be reused safely
across sequential runs.

Session-aware VLA Backends (per-client Policy State)
----------------------------------------------------

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

1. Create a model package under ``rpent/robots/components/``. Its
   ``server.py`` subclasses ``BaseWAMFacade`` and implements
   ``predict_native``; provide a required ``(encode, decode)`` adapter pair.
2. Register each supported platform in the model's ``ADAPTERS`` mapping
   using ``WAMAdapterSpec``. The model metadata factory declares checkpoint
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

Supported model/platform pairs, protocol fields, deployment commands and
integration checks are in :doc:`wam_backends`. A new pairing requires compatible
weights and input/output transforms; registration alone does not establish
compatibility. Forecasting and planning operations are reserved for future work.
