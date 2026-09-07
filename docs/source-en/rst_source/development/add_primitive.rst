Add an Action Primitive
=======================

An *action primitive* in RPent turns a tool call into an action that
the environment can execute. It can be a learned policy (a VLA, a WAM,
a diffusion planner) or a scripted routine (``move_to``,
``open_gripper``). This page explains how to add either type.

Two types of primitives
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Family
     - Execution location
     - Examples
   * - **Model-based**
       (VLA / WAM / diffusion / …)
     - Runs in its own process (``vla_server``) and is called through
       a *model client* held by the toolkit.
     - Pi0.5 (LIBERO), RLDX-1 (RoboCasa)
   * - **Scripted**
       (kinematic / heuristic)
     - Runs in the agent process, with an optional server-side RPC for
       kinematics. It does not load model weights.
     - ``move_to``, ``rotate_wrist``, ``release``,
       ``back_project``

Both types are native tools: a typed handler receives the session's resources
through ``ToolContext`` and returns ``ToolResult``. The toolkit validates
arguments and captures the post-action observation.

Add a scripted primitive
------------------------

Define a module-level handler in ``robots/<robot>/tools.py`` and add the
resulting ``Tool`` to the robot's tool tuple. For example, this LIBERO handler
holds the current pose for a bounded number of environment steps:

.. code-block:: python

   from typing import Annotated

   from pydantic import Field

   from rpent.tools import ToolContext, ToolResult, tool

   @tool
   def hold_pose(
       steps: Annotated[int, Field(ge=1, le=100)] = 10,
       *,
       ctx: ToolContext,
   ) -> ToolResult:
       """Hold the current pose with the gripper closed.

       Args:
           steps: Number of environment steps.
       """
       runtime = ctx.robot
       for _ in range(steps):
           ctx.check_cancelled()
           obs, _, terminated, truncated, _ = runtime.env.step(
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
           )
           runtime.executed_steps += 1
           runtime.set_obs(obs)
           ctx.record_frame(obs["main_images"])
           if terminated or truncated:
               break
       return ToolResult(data={"steps_requested": steps})

   # Add hold_pose to the existing LIBERO_TOOLS tuple.

Use the concrete runtime type in ``ToolContext[LiberoRuntime]`` in robot code.
``@tool`` generates the parameter model and JSON schema from the same function
signature. Google-style docstrings provide descriptions; use ``Annotated`` /
``Field`` for constraints on model-supplied arguments. ``ctx`` is injected by
the executor and is not part of the published schema.

After adding the declaration to ``LIBERO_TOOLS``, all three planners can call
it. The toolkit handles capture through ``_capture_observation``; handlers
submit frames but do not save their own episode video or duplicate the state dump.

For a tool that reads existing observations, place ``@readonly`` below
``@tool`` to skip automatic capture. Add ``@parallel`` below ``@tool`` only
when it can safely share execution with other parallel tools. These policies
are independent: ``write_text_file`` is readonly for observation purposes but
still runs exclusively. See :doc:`interfaces` for cancellation and scheduling.

.. _add-primitive-model-based:

Add a VLA (or other model-based primitive)
------------------------------------------

Because the model runs in its own process, adding a model-based
primitive requires a few additional components:

1. **Write ``vla_server.py``.** This process owns only the model weights
   and CUDA context. Use
   :class:`rpent.robots.components.vla_facade_base.BaseVLAFacade` as the base
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
   :class:`rpent.robots.components.vla_client_base.BaseVLAClient`, which
   provides the common ``vla.predict`` call, and add only the
   environment-specific input / output adaptation. See
   ``rpent.robots.components.pi05_vla_client.Pi05VLAClient`` for the LIBERO
   implementation.

3. **Write a native tool handler.** Use ``ctx.robot`` to access the model and
   environment clients, request a prediction, execute the returned actions,
   and return ``ToolResult(data={...})``. Follow the robot's observation and
   action conventions: Pi0.5 reads the instruction from
   ``env_obs["task_descriptions"]`` and returns a ``[chunk, action_dim]``
   NumPy array. Check cancellation before inference and at safe action
   boundaries, and submit environment frames through ``ctx.record_frame``.
   See ``pi0_pick`` in ``robots/libero/tools.py`` and ``rldx_skill`` in
   ``robots/robocasa/tools.py`` for concrete implementations.

4. **Add the tool to the robot's tuple.** The toolkit receives that tuple in
   its constructor and handles observation capture after execution, just as
   for a scripted primitive.

5. **Wire the clients in ``robot_spec.py``.** ``_init_runtime`` returns
   ``(owned_daemons, runtime_kwargs)``, with entries such as ``env`` and
   ``model``. ``get_toolkit(*, runtime_kwargs, dashboard_events, config)``
   passes those inputs, ``config.output_dir``, and a ``MemoryManager`` to the
   robot toolkit. The toolkit constructs the session runtime from
   ``runtime_kwargs``. See :doc:`add_robot` for the complete factory.

Reuse an existing vla_server across runs
----------------------------------------

Model servers often take a long time to start, so the runner can
connect to an instance that is already running:

.. code-block:: bash

   rpent --robot libero --vla-endpoint http://vla-host:8000 ...

If the model keeps per-episode state, expose a ``vla_reset`` RPC and
call it between tasks. The same server process can then be reused safely
across sequential runs.

Session-aware VLA backends (per-client policy state)
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

- **Runtime / tool side**: call ``reset_session`` before a task starts to clear
  policy state left over from the previous episode, so consecutive runs do
  not leak state into each other.

Single-threaded serve (EGL-rendering backends)
----------------------------------------------

Most backends use the ``serve`` inherited from their base class, which
spawns a worker thread per request. If your server process renders with EGL
(e.g. robosuite / MuJoCo offscreen rendering, see ``render_camera``), the
EGL context must stay on one thread, and concurrent dispatch would break
context affinity.

Mix :class:`~rpent.utils.rpc.main_thread_serve.MainThreadServeMixin` into
your facade class (**before** ``BaseEnvFacade`` / ``BaseVLAFacade``) and
inherit the ``serve`` it overrides — it runs the transport server on a
daemon thread but executes every dispatch serially on the thread that
called ``serve`` (normally the process main thread), handing requests from
the transport thread over via a work queue:

.. code-block:: python

   from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin
   from rpent.robots.components.env_facade_base import BaseEnvFacade

   class MyEnvFacade(MainThreadServeMixin, BaseEnvFacade):
       ...

   facade.serve(transport="http", host=host, port=port)  # dispatch on the main thread

The overridden ``serve`` keeps the same contract as
:class:`~rpent.utils.rpc.RpcFacade`'s ``serve``: it still supports
``healthz`` / ``shutdown``, parent-watch, and sessions (when constructed
with ``enable_sessions=True``, ``serve`` still requires ``session_sweep_s``).
Subclasses do **not** need to override ``serve`` to delegate — just inherit
it (see ``RoboCasaEnvFacade`` in ``robots/robocasa/env_server.py``).
Backends that do not need EGL single-threading keep the plain inherited
``serve``.

Design principles for a new primitive
-------------------------------------

- **Tools describe intent, not motion.** A good tool name is
  ``pi0_pick``, not ``execute_action_chunk_of_length_20``.
- **Action calls include a fresh observation.** The toolkit captures it after
  the handler finishes and before returning to the planner. Readonly tools
  reuse recorded observations.
- **Return small ``ToolResult.data`` payloads.** The planner serializes them
  as text and sends ``ToolResult.images`` as PNG content. Save larger observations
  through ``EnvState.save``; ``EnvState``
  automatically records each logical base name in its owned
  ``StepRecord.artifacts`` set. Expose images through ``view_env_state`` and
  geometry through environment tools rather than returning raw paths.
- **Guardrails belong in env_server**, not in the toolkit. The LLM
  can and will call any tool with any arguments; workspace bounds
  and safety clamps must be enforced on the server side.

Beyond VLAs
-----------

The same pattern extends to non-VLA model primitives:

- **World Action Models (WAM)** — imagination-based rollouts that
  produce a plan the env then executes. Wire them exactly like a
  VLA: their own process, their own client.
- **Diffusion planners / MPC** — same shape; the "action" the tool
  returns may be a trajectory rather than a single chunk, and the
  ``env_server`` steps it out.
- **Multiple primitives sharing one server** — a single
  ``vla_server`` can host several models; the tool decides which
  head to call via a ``model`` kwarg on ``predict``.

Regardless of the implementation, the framework contract remains
unchanged: model process → model client → native ``@tool`` handler →
robot tool tuple → ``Toolkit.execute_tool``.
