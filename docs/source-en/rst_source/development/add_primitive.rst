Add an action primitive
=======================

An *action primitive* in RPent is anything that turns a tool call
into an executable action for the environment. It can be a learned
policy (a VLA, a WAM, a diffusion planner) or a scripted routine
(``move_to``, ``open_gripper``). This page walks through how to add
one, whichever family it falls into.

Two shapes of primitive
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Family
     - Runs where
     - Examples
   * - **Model-based**
       (VLA / WAM / diffusion / …)
     - Own process (``vla_server``). Called via a *model client* the
       toolkit holds.
     - Pi0.5 (LIBERO), RLDX-1
   * - **Scripted**
       (kinematic / heuristic)
     - Agent process, sometimes with a driver-side RPC for the
       kinematics. No model weights.
     - ``move_to``, ``rotate_wrist``, ``release``,
       ``back_project``

Both shapes surface to the LLM in the same way: a tool schema, a
primitive-driver method, and a state dump after the call. What
differs is only *what the method does*.

Add a scripted primitive
------------------------

Scripted primitives are the fastest to add. Pattern:

1. **Method on the primitive driver.** Add a method on your env's
   primitive driver class (e.g. ``LiberoPrimitives``,
   ``MyRobotPrimitives``). It takes the tool's kwargs, does the
   work — usually one or more ``self._env.step(...)`` calls — and
   returns a small ``dict`` log.

   .. code-block:: python

      def open_drawer(self, dx: float = 0.15) -> dict:
          # Move end-effector back by dx while gripper is closed.
          for _ in range(N):
              self._env.step(build_open_drawer_chunk(dx))
          return {"ok": True, "dx": dx}

2. **Tool schema.** Add an entry to ``TOOLS_SPEC`` in
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

3. **Register in the toolkit.** Route the tool through the toolkit's
   ``_step`` helper so it re-renders state after running:

   .. code-block:: python

      self.add_tool("open_drawer", OPEN_DRAWER_SPEC,
                    lambda **kw: self._step("open_drawer", **kw))

The primitive is now callable by ``api``, ``claude_code``, and
``codex`` cerebrums — no other changes needed.

Add a VLA (or other model-based primitive)
------------------------------------------

Model-based primitives require a bit more scaffolding because the
model runs in its own process. Pattern:

1. **Write a ``vla_server.py``**. Own only the model weights and the
   CUDA context. Expose ``vla_load`` / ``vla_infer`` /
   ``vla_reset``:

   - Over **HTTP** if your obs is a flat ``image + state`` payload
     (LIBERO / Pi0.5 pattern).
   - Over **socket RPC** if your obs is a nested dict of numpy
     arrays with history stacks (RLDX-1 pattern).

   Emit a ``transport_ready`` JSON event on stdout when your server
   is listening; ``rpent/cli/main.py`` blocks on it.

2. **Write a model client**. A tiny class with ``predict(obs)``
   (HTTP) or ``vla_infer(obs)`` (socket) that the toolkit will hold
   alongside its env client. See ``rpent.utils.vla_client.VLAClient``
   as the LIBERO reference.

3. **Add a primitive-driver method.** In your env's primitive-driver
   class, call the model client, forward the returned chunk to the
   env, and return a log dict:

   .. code-block:: python

      def mymodel_pick(self, target: str) -> dict:
          obs = self._env.get_obs()
          chunk = self._model.predict(obs, instruction=f"pick {target}")
          self._env.chunk_step(chunk)
          return {"model": "mymodel", "target": target}

4. **Add the tool schema** and register it in the toolkit (same
   pattern as the scripted case above).

5. **Wire it up in ``__init__.py``**. Your env's ``get_toolkit``
   builds the toolkit with the right ``primitives_kwargs``:

   .. code-block:: python

      def get_toolkit(*, primitives_kwargs, video_path=None):
          from robots.myrobot.toolkit import MyRobotToolkit
          return MyRobotToolkit(
              primitives_kwargs=primitives_kwargs,
              video_path=video_path,
          )

   And ``rpent/cli/main.py`` will pass in ``{"env": MyRobotEnvClient(...),
   "model": MyModelClient(...)}``.

Reuse an existing vla_server across runs
----------------------------------------

Model servers are expensive to start (weight-loading dominates). The
runner supports pointing at an already-running one:

.. code-block:: bash

   rpent --vla-endpoint http://vla-host:8000 ...

Design your ``vla_server`` to be **stateless across tasks** — reset
its per-episode state through an explicit ``vla_reset`` RPC — so a
single process can serve many sequential runs safely.

Design principles for a new primitive
-------------------------------------

- **Tools describe intent, not motion.** A good tool name is
  ``pi0_pick``, not ``execute_action_chunk_of_length_20``. The LLM
  picks tools by name; make the name self-explanatory.
- **Every tool ends with a state dump.** The next turn depends on
  the state dump reflecting the post-action world. Don't let the
  primitive return before the render finishes.
- **Return small dicts.** Tool return values are fed back to the
  LLM as text. Keep them under a few hundred bytes; large payloads
  go into the state dump (images, depths, ``states.json``) where
  they'll ride image content blocks instead.
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

Whatever the shape, the framework contract is unchanged: model
process → model client → primitive-driver method → tool schema →
``Toolkit.add_tool``.
