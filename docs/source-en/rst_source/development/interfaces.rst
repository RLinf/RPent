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

Contract: read ``toolkit.list_tools()`` and adapt each ``Tool``'s ``name``,
``description``, and ``input_schema`` to the model SDK. Dispatch through
``toolkit.execute_tool(name, arguments)`` and return ``PlannerResult`` when
``toolkit.finish_result`` is set or a run limit is reached. For asynchronous
adapters, use ``rpent.planner.base.execute_tool`` to run the synchronous executor
in a worker and retain it through cancellation.

Native tools and Toolkit
------------------------

Import ``Tool``, ``ToolContext``, ``ToolResult``, ``Toolkit``, ``tool``,
and ``readonly`` from ``rpent.tools``.

- ``@tool`` turns a function into a ``Tool``. Its name and Google-style
  docstring describe the tool; typed parameters and Pydantic ``Field``
  constraints generate both the validation model (``args_schema``) and
  the published JSON schema (``input_schema``).
- Every handler takes a required keyword-only ``ctx: ToolContext[RobotRuntime]``.
  The executor injects it and excludes it from the model-facing schema.
  It provides ``state``, ``memory``, ``robot``, ``output_dir``,
  ``record_frame(rgb)``, and ``check_cancelled()``.
- Handlers return ``ToolResult(data={...}, images=[png_bytes], error=None)``.
  ``to_dict()`` combines data and any error; ``to_text()`` serializes that
  payload, truncating only the model-facing text to 60,000 bytes. PNG bytes
  remain separate in ``images``; ``is_error`` indicates an error.

Construct the robot subclass with a fixed tuple of native tools:

.. code-block:: python

   super().__init__(
       state=state,
       memory=memory,
       robot=runtime,
       output_dir=output_dir,
       tools=MYROBOT_TOOLS,
       dashboard_events=dashboard_events,
   )

The base class adds ``read_text_file``, ``write_text_file``, ``list_dir``, and
``read_image`` from ``rpent.tools.common_tools``. The MCP adapters omit
``read_image`` because Claude Code and Codex use their built-in image readers.
Memory file access goes through ``MemoryManager.authorize_read`` and
``authorize_write``.

Scheduling and lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~

Tools run exclusively by default. Place ``@readonly`` below ``@tool`` to allow
execution alongside other readonly tools and skip automatic observation capture.
Pending exclusive calls take priority and keep their queue order.

Non-readonly robot tools capture a new observation after execution. Common tools
and ``finish`` are excluded from capture: ``write_text_file`` and ``finish`` run
exclusively without adding an observation. LIBERO ``segment`` runs exclusively
and captures an observation after saving its segmentation artifacts.

Override ``_capture_observation(*, command, result, elapsed_s)`` to save a
``StepRecord`` and return ``(observation_data, png_images)``. The executor
replaces action data with observation data, appends the images, and retains
any action error. Include the action log in the observation when needed.
Capture also runs after handler errors; the call remains active until capture
and Dashboard publication finish.

Long-running handlers call ``ctx.check_cancelled()`` at safe boundaries.
``cancel_active_and_wait()`` pauses admission, cancels pending and active calls,
and waits for cleanup; ``resume_calls()`` reopens admission.
``close()`` permanently closes admission, drains calls, and saves collected
frames as ``episode.mp4``. Tools submit RGB frames with ``ctx.record_frame``;
when Dashboard events are enabled, the executor also saves per-action clips.

Each robot supplies its own ``finish`` tool. A successful call stores its
``status`` and ``summary`` in ``toolkit.finish_result``; it does not close
admission. ``solved()`` reports environment success independently of the
planner's requested finish status. ``write_recipe(recipe_tag)`` exports
successful robot calls, including perception and resets, in completion order;
common file/image tools and ``finish`` are excluded. The runner decides whether
the run qualifies for memory publication.

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
