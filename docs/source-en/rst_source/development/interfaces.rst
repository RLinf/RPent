Core interfaces
===============

When you wire a new env or primitive into RPent, you implement the interfaces below.
Walkthroughs: :doc:`add_robot`, :doc:`add_primitive`. Repo layout: :doc:`architecture`.

Environment entry
-----------------

After you add ``robots/<env>/``, ``main.py`` calls two functions in ``__init__.py``:

.. code-block:: python

   def get_env_spec() -> EnvSpec: ...
   def get_toolkit(*, primitives_kwargs, video_path=None, dashboard=None): ...

``get_env_spec`` returns an ``EnvSpec``. You supply:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field / hook
     - What you provide
   * - ``name``
     - Env name for ``--env``.
   * - ``prompts``
     - A ``PromptBundle`` with ``system`` and ``user`` prompt factories (see
       ``robots/<env>/prompt_bundle.py``).
   * - ``add_cli_args``
     - Register this env's CLI flags (e.g. ``--suite``, ``--env-endpoint``).
   * - ``parse_config``
     - Validate args and return ``RunConfig``; set at least ``recipe_tag``,
       ``output_dir``, and ``prompt_vars`` for prompt templating.
   * - ``init_runtime``
     - Start or attach to env / VLA subprocesses; build ``primitives_kwargs``
       (env client, model client, etc.) for the toolkit's primitives.

``get_toolkit`` usually just passes ``primitives_kwargs`` into your env subclass;
``video_path`` and ``dashboard`` are passed by ``main.py`` — you rarely touch them.

Reference: ``robots/libero/__init__.py``.

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
   ) -> PlannerResult: ...

Contract: pass ``toolkit.get_tools_spec()`` to the model; dispatch each call via
``toolkit.execute_tool(name, input_dict)``; feed results back to the model; return
``PlannerResult`` on the ``finish`` tool or when turns are exhausted.

Toolkit
-------

Subclass ``Toolkit`` in ``robots/<env>/toolkit.py`` and register env tools with
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
``add_tool`` for env tools. Per-step state and ``view_env_state`` are in
:doc:`add_primitive`.

Inter-process communication
---------------------------

Relevant when attaching to existing servers or writing ``env_server`` / ``vla_server``.

Client endpoints — expose in ``add_cli_args`` or parse in ``init_runtime``:

.. code-block:: text

   [protocol://]host:port    # defaults to http when protocol is omitted

Common flags: ``--env-endpoint``, ``--vla-endpoint``. The default ``http`` sends
JSON over ``POST /call``, encoding NumPy arrays as
``{"__ndarray__": <base64>, "dtype": ..., "shape": ...}``; switch to ``socket``
for large or history-stacked nested-NumPy observations to move length-prefixed
pickle frames and skip repeated JSON encoding. Pickle is unsafe on untrusted
input, so only point ``socket`` at trusted endpoints.

Server: subclass ``rpent.utils.rpc.RpcFacade`` and implement ``_dispatch`` for
business RPCs (e.g. ``reset``, ``step``, ``predict``). Do not implement ``healthz`` or
``shutdown`` in the subclass.

Details are in the env_server / vla_server sections of :doc:`add_robot`.
