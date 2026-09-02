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

From the LLM's perspective, both types expose the same interface: a
tool schema, a primitives method, and a state dump after the
call. They differ only in how the method is implemented.

Add a scripted primitive
------------------------

Adding a scripted primitive usually involves two steps:

1. **Add a method to the primitives.** Add the method to the
   current robot's primitives class, such as
   ``LiberoPrimitives`` or ``MyRobotPrimitives``. The method accepts
   the tool-call arguments, performs the work, usually through one or
   more ``self._env.step(...)`` calls, and returns a small log ``dict``.

  Primitive methods capture and re-render state (``get_env_state``)
  automatically after they run:

   .. code-block:: python

      def open_drawer(self, dx: float = 0.15) -> dict:
          # Move end-effector back by dx while gripper is closed.
          for _ in range(N):
              self._env.step(build_open_drawer_chunk(dx))
          return {"ok": True, "dx": dx}

  You can mark read-only tools (``view_env_state``, ``back_project``, ``segment``,
  ...) with :func:`~rpent.tools.toolkit.readonly` so the toolkit skips state
  capture for them, improving performance.

2. **Add the tool schema.** Add an entry to ``TOOLS_SPEC`` in
   ``robots/<robot>/tools.py``:

   .. code-block:: python

      {
          "name": "open_drawer",
          "description": "Pull the currently-grasped drawer handle "
                         "backwards by ``dx`` meters.",
          "input_schema": {
              "type": "object",
              "properties": {"dx": {"type": "number"}},
              "required": [],
          },
      }

Once both exist, the toolkit registers the tool automatically: it iterates
``TOOLS_SPEC`` and binds each spec to the matching primitive-driver method
(e.g. ``getattr(self._primitives, name)``).

After these steps, the ``api``, ``claude_code``, and ``codex`` planners
can all call the primitive without any other code changes.

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

3. **Add a method to the primitives.** In the current
   robot's primitives class, call the model client, pass
   the returned action chunk to the environment, and return a log
   ``dict``. The model client API is
   :meth:`rpent.robots.components.pi05_vla_client.Pi05VLAClient.predict`,
   which reads the instruction from ``env_obs["task_descriptions"]`` and
   returns a ``[chunk, action_dim]`` numpy action chunk (batch dim already
   stripped):

   .. code-block:: python

      def mymodel_pick(self, target: str) -> dict:
          env_obs = self._env.get_obs()
          env_obs["task_descriptions"] = f"pick {target}"
          chunk = self._model.predict(env_obs)
          self._env.chunk_step(chunk)
          return {"model": "mymodel", "target": target}

4. **Add the tool schema and register it in the toolkit.** Follow the
   same pattern as for a scripted primitive.

5. **Wire the components together in ``robot_spec.py``.** The
   robot's ``get_toolkit`` builds the toolkit with
   ``primitives_kwargs``:

   .. code-block:: python

      def get_toolkit(*, primitives_kwargs, dashboard_events):
          from robots.myrobot.toolkit import MyRobotToolkit
          return MyRobotToolkit(
              primitives_kwargs=primitives_kwargs,
              dashboard_events=dashboard_events,
          )

   The robot package's ``_init_runtime`` builds
   ``primitives_kwargs``, for example
   ``{"env": MyRobotEnvClient(...), "model": MyModelClient(...)}``.
   The toolkit constructor then forwards it to the primitives.

Reuse an existing vla_server across runs
----------------------------------------

Model servers often take a long time to start, so the runner can
connect to an instance that is already running:

.. code-block:: bash

   rpent --robot libero --vla-endpoint http://vla-host:8000 ...

If the model keeps per-episode state, expose a ``vla_reset`` RPC and
call it between tasks. The same server process can then be reused safely
across sequential runs.

Design principles for a new primitive
-------------------------------------

- **Tools describe intent, not motion.** A good tool name is
  ``pi0_pick``, not ``execute_action_chunk_of_length_20``.
- **Every tool ends with a state dump.** The next turn depends on
  the state dump reflecting the post-action world. Don't let the
  primitive return before the render finishes.
- **Return small dicts.** Tool return values are fed back to the LLM
  as text. Save larger observations through ``EnvState.save``; ``EnvState``
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
unchanged: model process → model client → primitives method →
tool schema → ``Toolkit.add_tool``.
