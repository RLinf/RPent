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
     - Runs in the agent process, with an optional driver-side RPC for
       kinematics. It does not load model weights.
     - ``move_to``, ``rotate_wrist``, ``release``,
       ``back_project``

From the LLM's perspective, both types expose the same interface: a
tool schema, a primitive-driver method, and a state dump after the
call. They differ only in how the method is implemented.

Add a scripted primitive
------------------------

Adding a scripted primitive usually involves three steps:

1. **Add a method to the primitive driver.** Add the method to the
   current environment's primitive-driver class, such as
   ``LiberoPrimitives`` or ``MyRobotPrimitives``. The method accepts
   the tool-call arguments, performs the work, usually through one or
   more ``self._env.step(...)`` calls, and returns a small log ``dict``.

   .. code-block:: python

      def open_drawer(self, dx: float = 0.15) -> dict:
          # Move end-effector back by dx while gripper is closed.
          for _ in range(N):
              self._env.step(build_open_drawer_chunk(dx))
          return {"ok": True, "dx": dx}

2. **Add the tool schema.** Add an entry to ``TOOLS_SPEC`` in
   ``toolkit.py``:

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

3. **Register the tool in the toolkit.** Route it through the toolkit's
   ``_step`` helper so that state is re-rendered after execution:

   .. code-block:: python

      self.add_tool("open_drawer", OPEN_DRAWER_SPEC,
                    lambda **kw: self._step("open_drawer", **kw))

After these steps, the ``api``, ``claude_code``, and ``codex`` planners
can all call the primitive without any other code changes.

.. _add-primitive-model-based:

Add a VLA (or other model-based primitive)
------------------------------------------

Because the model runs in its own process, adding a model-based
primitive requires a few additional components:

1. **Write ``vla_server.py``.** This process owns only the model weights
   and CUDA context. Subclass :class:`rpent.utils.rpc.RpcFacade` and
   expose your model methods (e.g. ``predict``) via ``_dispatch``:

   - The default transport is **HTTP** (JSON over ``POST /call``),
     which works well for flat ``image + state`` payloads such as the
     LIBERO / Pi0.5 pattern.
   - Switch to **socket RPC** (``--transport socket``) if your obs is
     a nested dict of numpy arrays with history stacks (avoids the
     JSON re-encode overhead).

   ``RpcFacade.serve`` handles transport binding, ``healthz``,
   ``shutdown``, parent-death detection, and resource cleanup. You
   only need to implement the model-specific methods.

2. **Write a model client.** Create a lightweight class that wraps an
   :class:`rpent.utils.rpc.RpcClient` (either :class:`HttpRpcClient` or
   :class:`SocketRpcClient`) and exposes the model's API.
   See ``rpent.utils.vla_client.VLAClient`` for the LIBERO implementation.

3. **Add a method to the primitive driver.** In the current
   environment's primitive-driver class, call the model client, pass
   the returned action chunk to the environment, and return a log
   ``dict``:

   .. code-block:: python

      def mymodel_pick(self, target: str) -> dict:
          obs = self._env.get_obs()
          chunk = self._model.predict(obs, instruction=f"pick {target}")
          self._env.chunk_step(chunk)
          return {"model": "mymodel", "target": target}

4. **Add the tool schema and register it in the toolkit.** Follow the
   same pattern as for a scripted primitive.

5. **Wire the components together in ``__init__.py``.** The
   environment's ``get_toolkit`` builds the toolkit with
   ``primitives_kwargs``:

   .. code-block:: python

      def get_toolkit(*, primitives_kwargs, video_path=None):
          from robots.myrobot.toolkit import MyRobotToolkit
          return MyRobotToolkit(
              primitives_kwargs=primitives_kwargs,
              video_path=video_path,
          )

   The environment package's ``_init_runtime`` builds
   ``primitives_kwargs``, for example
   ``{"env": MyRobotEnvClient(...), "model": MyModelClient(...)}``.
   The toolkit constructor then forwards it to the primitive driver.

Reuse an existing vla_server across runs
----------------------------------------

Model servers often take a long time to start, so the runner can
connect to an instance that is already running:

.. code-block:: bash

   rpent --env libero --vla-endpoint http://vla-host:8000 ...

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
  as text. Store larger content, such as images, depth data, and
  ``states.json``, in the state dump instead.
- **Guardrails belong in env_server**, not in the toolkit. The LLM
  can and will call any tool with any arguments; workspace bounds
  and safety clamps must be enforced on the driver side.

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
  head to call via a ``model`` kwarg on ``vla_infer``.

Regardless of the implementation, the framework contract remains
unchanged: model process → model client → primitive-driver method →
tool schema → ``Toolkit.add_tool``.
