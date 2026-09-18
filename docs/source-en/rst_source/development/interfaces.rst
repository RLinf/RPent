Core interfaces
===============

When you wire a new robot or primitive into RPent, you implement the interfaces below.
Walkthroughs: :doc:`add_robot`, :doc:`add_primitive`. Repo layout: :doc:`architecture`.

Robot entry
-----------

After you add ``robots/<robot>/``, the package ``__init__.py`` re-exports two
functions implemented in ``robot_spec.py`` for ``main.py`` to call:

.. code-block:: python

   def get_robot_spec() -> RobotSpec: ...
   def get_toolkit(
       *,
       runtime_kwargs,
       dashboard_events: DashboardEventSink,
       config: RunConfig,
   ): ...

``get_robot_spec`` returns a ``RobotSpec``. You supply:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field / hook
     - What you provide
   * - ``name``
     - Robot name for ``--robot``.
   * - ``prompts``
     - A ``PromptBundle`` with ``system`` and ``user`` prompt factories (see
       ``robots/<robot>/prompt_bundle.py``).
   * - ``dashboard``
     - Optional Dashboard description. ``None`` disables Dashboard control for
       the robot. Otherwise, the spec defines its task command and
       fields, runtime components, and frame channels.
   * - ``add_cli_args``
     - Register this robot's CLI flags (e.g. ``--suite``, ``--env-endpoint``).
   * - ``parse_config``
     - Validate args and return ``RunConfig``; set at least ``recipe_tag``,
       ``output_dir``, and ``prompt_vars`` for prompt templating.
   * - ``init_runtime``
     - Start or attach to all runtime components, or to the component names in
       the optional selection, and build ``runtime_kwargs`` for them. The
       normal CLI passes ``None``; the Dashboard passes explicit shared and
       unique subsets derived from its spec. A ``DashboardEventSink``
       reports status.

``get_toolkit`` usually passes ``runtime_kwargs`` into your robot subclass;
``dashboard_events`` and ``config`` are supplied by the active runner. It must
construct a :class:`~rpent.memory.MemoryManager` (rooted at the configured
``config.prompt_vars["memory_dir"]``, falling back to
``get_memory_dir(robot_name)`` when unset) and pass it to the toolkit.
Memory access permissions are configured on the manager. Robots that need
extra toolkit arguments may declare them as keyword-only parameters; LIBERO
additionally uses ``mode``, ``attempts_per_session``, and ``state_output_dir``.

Reference: ``robots/libero/robot_spec.py``.

Planner
-------

Most users pick a built-in ``api``, ``claude_code``, or ``codex`` planner — see
:doc:`../usage/configure_planner`. Only **custom planners** need
``rpent.planner.base.Planner``:

.. code-block:: python

   def solve(
       self,
       *,
       system_prompt: str,
       user_message: str,
       toolkit: Toolkit,
       max_turns: int,
       input_queue=None,
       dashboard_interaction=None,
   ) -> PlannerResult: ...

Contract: read native declarations from ``toolkit.list_tools()`` and convert
``name``, ``description``, and ``input_schema`` to the model SDK's format.
Resolve schema placeholders with ``rpent.utils.templates.substitute`` before
sending them. Dispatch registered toolkit calls through
``toolkit.execute_tool(name, input_dict)``
and return ``PlannerResult`` when ``toolkit.finish_result`` is set or the turn
limit is reached.

Toolkit
-------

Subclass ``Toolkit`` in ``robots/<robot>/toolkit.py``. The base constructor
registers common file tools and ``finish``. Declare primitive methods with
``@tool``, then register bound methods from the instance:

.. code-block:: python

   self.add_tool(self._primitives.move_to)
   # Or collect a whole primitive object's declarations:
   self.add_tools(iter_tools(self._primitives))

A native ``Tool`` contains ``name``, ``description``, ``args_schema``, ``handler``,
and ``readonly``; ``input_schema`` exposes its generated JSON Schema. Google-style
``Args`` documentation supplies parameter descriptions. Type annotations and
``Field`` constraints define validation. Python defaults control omitted
arguments; publish a default explicitly with
``Field(json_schema_extra={"default": value})``. ``self`` is excluded from model
inputs, and the instance retains its environment and model clients.

``add_tool(declaration, replace=True)`` explicitly replaces a registered name;
otherwise duplicate names raise an error. ``declaration.with_handler(handler)``
binds internal resources or an execution guard while retaining the schema and
read-only metadata. See :doc:`add_primitive` for resource injection.

Native results and execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handlers and ``get_env_state`` return ``ToolResult`` with ``data`` (a dictionary),
``images`` (an ordered list of PNG byte strings), and ``error`` (text or ``None``).
Use ``to_dict()`` for structured output, ``to_text()`` for bounded model-facing
text, and ``is_error`` for failure. Planner adapters assemble SDK content blocks.

Registered tool calls from planners and the Dashboard pass through
``execute_tool``. It validates arguments before running the handler. Stateful
tools then capture a fresh observation through
``get_env_state(command, result, elapsed_s)``; its data is
returned with the tool's images and any execution error. ``@tool(readonly=True)``
skips this automatic capture. Primitive classes own robot runtime state and frame
buffers; ``EnvState`` owns recorded steps and artifacts.
Common file tools call ``MemoryManager.authorize_read`` / ``authorize_write``
for path access decisions.

After an accepted ``finish``, ``toolkit.finish_result`` contains the full result
without the internal ``_finish`` marker. API, Claude Code, and Codex read this
value. An error or ``_finish=False`` leaves completion unset, and robot-specific
operator metadata is preserved.

Inter-process communication
---------------------------

Relevant when attaching to existing servers or writing ``env_server`` / ``vla_server``.

Client endpoints — expose in ``add_cli_args`` and parse in the applicable
normal-CLI or Dashboard runtime hook:

.. code-block:: text

   [protocol://]host:port    # defaults to http when protocol is omitted

Common flags: ``--env-endpoint``, ``--vla-endpoint``. The default ``http`` sends
JSON over ``POST /call``, encoding NumPy arrays as
``{"__ndarray__": <base64>, "dtype": ..., "shape": ...}`` and NumPy scalars as
``{"__npscalar__": <value>, "dtype": ...}`` so dtypes survive the round trip;
switch to ``socket``
for large or history-stacked nested-NumPy observations to move length-prefixed
pickle frames and skip repeated JSON encoding. Pickle is unsafe on untrusted
input, so only point ``socket`` at trusted endpoints.

Environment and VLA clients should normally subclass ``BaseEnvClient`` and
``BaseVLAClient``; their servers should subclass ``BaseEnvFacade`` and
``BaseVLAFacade`` and register extension routes through ``_register_rpc``. The
bases provide common routing and locking on top of ``RpcFacade``. Subclass
``RpcFacade`` directly only for a service type without a specialized base. Do
not implement ``healthz`` or ``shutdown`` in application subclasses.

Details are in the env_server / vla_server sections of :doc:`add_robot`.
