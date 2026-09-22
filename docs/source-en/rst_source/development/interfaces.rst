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

Read tools from ``toolkit.list_tools()``, call them through
``toolkit.execute_tool(name, input_dict)``, and feed results back to the model.
Return ``PlannerResult`` when ``toolkit.finish_result`` is not ``None`` or the
turn limit is reached. See :doc:`../usage/configure_planner` for SDK conversion
and schema placeholder substitution.

Toolkit
-------

Subclass ``Toolkit`` in ``robots/<robot>/toolkit.py``. The base constructor
registers common file tools and ``finish``. Declare primitive methods with
``@tool``, then register bound methods from the instance:

.. code-block:: python

   self.add_tool(self._primitives.move_to)
   # Or collect a whole primitive object's declarations:
   self.add_tools(iter_tools(self._primitives))

Tool parameters need type annotations and must accept keyword arguments. Use
the docstring's ``Args`` section for descriptions and ``Annotated[..., Field(...)]``
for constraints; see :doc:`add_primitive` for an example. Signature defaults apply
at runtime. To include them in the schema, use
``Field(json_schema_extra={"default": value})``.

To replace a registered tool, use ``add_tool(declaration, replace=True)``;
otherwise duplicate names raise an error. Use ``declaration.with_handler(handler)``
to wrap a handler while retaining its schema and ``readonly`` setting. For example,
exclude an internal parameter with ``@tool(exclude=("state",))``, then register
``declaration.with_handler(partial(declaration, state=self.state))``
using ``functools.partial``.

Tool results and execution
~~~~~~~~~~~~~~~~~~~~~~~~~~

Handlers and ``get_env_state`` return ``ToolResult``: ``data`` holds a result
dictionary, ``images`` a list of PNG byte strings, and ``error`` an error message
or ``None``. Use ``to_dict()`` for a dictionary, ``to_text()`` for bounded text,
and ``is_error`` to check for errors.

``execute_tool`` applies strict Pydantic validation, rejecting unknown arguments
and non-finite numbers, then runs the tool and calls ``get_env_state`` to capture
observations. The result contains observation data, images, and any execution error.

``@tool(readonly=True)`` skips automatic observation capture without changing
file permissions or tool concurrency limits. Direct Python calls bypass this
argument validation and observation capture.

When ``finish`` succeeds without returning ``_finish=False``, Toolkit saves the
result in ``finish_result``, removing only the internal ``_finish`` marker.

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
