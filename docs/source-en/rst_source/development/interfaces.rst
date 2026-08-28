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
       primitives_kwargs,
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
       the optional selection, and build ``primitives_kwargs`` for them. The
       normal CLI passes ``None``; the Dashboard passes explicit shared and
       unique subsets derived from its spec. A ``DashboardEventSink``
       reports status.

``get_toolkit`` usually passes ``primitives_kwargs`` into your robot subclass;
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

Contract: pass ``toolkit.get_tools_spec()`` to the model; dispatch each call via
``toolkit.execute_tool(name, input_dict)``; feed results back to the model; return
``PlannerResult`` on the ``finish`` tool or when turns are exhausted.

Toolkit
-------

Subclass ``Toolkit`` in ``robots/<robot>/toolkit.py`` and register robot tools with
``add_tool``:

.. code-block:: python

   def add_tool(self, name: str, spec: dict, handler) -> None: ...

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Argument
     - Meaning
   * - ``name``
     - Tool name the LLM sees.
   * - ``spec``
     - Tool description and parameter schema (``name``, ``description``,
       ``input_schema``).
   * - ``handler``
     - Implementation; **must return a ``dict``**. Set ``_finish`` when the task
       ends; optional ``_image_bytes`` (etc.) to return camera images.

The base class already registers common file tools; call ``super().__init__()`` then
``add_tool`` for robot tools. Per-step state and ``view_env_state`` are in
:doc:`add_primitive`.

Inter-process communication
---------------------------

Relevant when attaching to existing servers or writing ``env_server`` / ``vla_server``.

Client endpoints — expose in ``add_cli_args`` and parse in the applicable
normal-CLI or Dashboard runtime hook:

.. code-block:: text

   [protocol://]host:port    # defaults to http when protocol is omitted

Common flags: ``--env-endpoint``, ``--vla-endpoint``. The default ``http`` sends
JSON over ``POST /call``, encoding NumPy arrays as
``{"__ndarray__": <base64>, "dtype": ..., "shape": ...}``; switch to ``socket``
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
