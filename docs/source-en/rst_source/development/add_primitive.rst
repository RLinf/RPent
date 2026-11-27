Add an Action Primitive
=======================

An *action primitive* in RPent turns a tool call into an action that
the environment can execute. It can be a learned policy (a VLA, a WAM,
a diffusion planner) or a scripted routine (``move_to``,
``open_gripper``). This page covers tool declaration, registration and execution.

Two Types of Primitives
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Family
     - Execution location
     - Examples
   * - **Model-based**
       (VLA / WAM / diffusion / …)
     - Runs in its own model service process and is called through
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

Add a Scripted Primitive
------------------------

Declare the method on the robot's primitives class and register its bound tool:

.. code-block:: python

   from rpent.tools import ToolResult, iter_tools, tool

   class MyRobotPrimitives:
       @tool
       def open_drawer(self, dx: float = 0.15) -> ToolResult:
           """Pull the grasped drawer handle backwards.

           Args:
               dx: Pull distance in meters.
           """
           self._env.step(build_open_drawer_chunk(dx))
           return ToolResult(data={"ok": True, "dx": dx})

   # In the toolkit, after constructing self._primitives:
   self.add_tools(iter_tools(self._primitives))

Type annotations and the docstring define the model-facing schema. Toolkit
validates arguments and captures state through ``get_env_state`` after execution.
Use ``@tool(readonly=True)`` for readers such as ``view_env_state`` and
``back_project``. For module-level tools, exclude internal resources with
``@tool(exclude=("state",))`` and bind them with ``Tool.with_handler``.
All online planners use the registered declarations.

.. _add-primitive-model-based:

Model-backed Tools
------------------

For VLA/WAM services, clients, adapters and session lifecycle, follow
:doc:`add_action_model`. The platform tool calls its model client, executes the
returned native action chunk through the environment, and returns a
``ToolResult``. Register it with the same ``@tool`` and ``iter_tools``
mechanism used above.

Check Whether a Failed Action Ran
---------------------------------

A gripper can close even if the camera misses the next frame. When observation
fails after an action, Toolkit sets ``execution_uncertain=True`` on the
``ToolResult`` returned by ``execute_tool``. The caller can then check the
gripper's position before retrying. A tool can also set this flag if the robot
cannot confirm the outcome. Only the caller sees the flag. It stays out of the
model input and the saved trajectory.

By default, an exception in an action handler sets the flag. An exception in a
readonly handler leaves it false. Writing a file does not move the robot, so
``write_text_file`` uses ``readonly=True`` to skip observation and run alongside
other readonly calls. Its decorator, ``@tool(readonly=True, uncertain_on_error=True)``,
tells the caller that a write that fails halfway can leave a partial file.

If write permission is denied before the first write, the arguments are invalid,
or the call is cancelled before the tool starts, ``execution_uncertain`` stays
false. Separately, ``tool_result_serialization_failed`` in ``ToolResult.data["code"]``
means the result could not be delivered in full. That code alone cannot tell you
whether the action ran.

Observe the Environment Between Calls
-------------------------------------

If the camera missed the gripper's movement, call ``Toolkit.refresh_observation()``
to check its position before retrying. A refresh waits for active tools to finish
and does not count toward the planner's tool-call budget. If Toolkit is paused
or closed, the method raises ``ToolCancelled``.

The trajectory gains a step only when ``get_env_state`` adds an observation.
Capture errors propagate to the caller. If cancellation arrives after a new
observation has been recorded, Toolkit publishes the step before raising
``ToolCancelled``.

Skip ``{"action": "refresh_observation"}`` during replay. It records an
observation with no robot action to repeat. For actions, keep the tool name and
arguments in ``StepRecord.command`` and record who issued the command separately.

Single-threaded Serve (EGL-rendering Backends)
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

Design Principles for a New Primitive
-------------------------------------

- **Tools describe intent, not motion.** A good tool name is
  ``pi0_pick``, not ``execute_action_chunk_of_length_20``.
- **Stateful tools capture state before returning to the planner.** Toolkit
  calls ``get_env_state`` after the handler while retaining exclusive execution.
  ``readonly`` tools skip this capture and may run alongside other readonly tools.
- **Keep result data small.** Tool return values are fed back to the LLM
  as text. Save larger observations through ``EnvState.save``; ``EnvState``
  automatically records each logical base name in its owned
  ``StepRecord.artifacts`` set. Expose images through ``view_env_state`` and
  geometry through environment tools rather than returning raw paths.
- **Guardrails belong in env_server**, not in the toolkit. The LLM
  can and will call any tool with any arguments; workspace bounds
  and safety clamps must be enforced on the server side.
