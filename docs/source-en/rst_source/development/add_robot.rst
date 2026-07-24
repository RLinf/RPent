Add a New Robot
===============

This guide walks through what you need to write to plug a new physical /
simulated robot into RPent's LLM-in-the-loop runner. Use
``robots/libero/`` as the worked reference.

Integration steps
-----------------

For the overall process layout, service responsibilities, and communication
model, see :doc:`System Design <architecture>`. This guide focuses on the
extension points required to add a robot. Complete them in the following
order:

1. Register the ``EnvSpec`` and toolkit factory in the
   :ref:`entry point <add-robot-entry>`.
2. Implement :ref:`env_client and env_server <add-robot-env-rpc>`. To
   integrate a VLA service and model client, see :ref:`Add a VLA (or other
   model-based primitive) <add-primitive-model-based>`.
3. :ref:`Define the prompts <add-robot-prompts>`.
4. :ref:`Implement the toolkit and primitive driver <add-robot-toolkit>`.
5. :ref:`Register environment arguments and build RunConfig
   <add-robot-config>`.
6. In :ref:`_init_runtime <add-robot-runtime>`, start or connect to
   ``env_server`` and any required supporting services.

.. _add-robot-entry:

Entry point
-----------

For a new environment named ``myenv``, use the following directory layout:

.. code-block:: text

   robots/myenv/
       __init__.py            # entry point — get_env_spec() / get_toolkit() factories
       env_client.py          # MyEnvClient — agent-side RPC stub (§1)
       prompt_bundle.py       # system()/user() prompt factories         (§2)
       toolkit.py             # MyEnvToolkit + primitives + tool definitions (§3)
       env_server.py          # driver-side facade + RPC server (§1)
       vla_server.py          # (optional) VLA model server

``__init__.py`` is the environment package's entry point. The registry in
``rpent/envs/base.py`` lazily imports ``robots.<name>`` on demand and calls its
two factory functions:

.. code-block:: python

   # robots/myenv/__init__.py
   from rpent.envs.env_spec import EnvSpec, RunConfig
   from rpent.envs.prompt_bundle import PromptBundle
   from robots.myenv.prompt_bundle import system_prompt, user_prompt

   def get_env_spec() -> EnvSpec:
       return EnvSpec(
           name="myenv",
           prompts=PromptBundle(system=system_prompt, user=user_prompt),
           add_cli_args=_add_cli_args,
           parse_config=_parse_config,
           init_runtime=_init_runtime,
       )

   def get_toolkit(*, primitives_kwargs, video_path=None):
       from robots.myenv.toolkit import MyEnvToolkit
       return MyEnvToolkit(primitives_kwargs=primitives_kwargs, video_path=video_path)

   def _add_cli_args(parser, use_dashboard) -> None:
       """Register env flags on the shared parser. See §4."""
       ...

   def _parse_config(args) -> RunConfig:
       """Validate final `args`, return a RunConfig. See §4."""
       ...

   def _init_runtime(args, output_dir):
       """Spawn env_server, vla_server, and any supporting services.

       Returns (daemons, primitives_kwargs). See §5.
       """
       ...

That's the entire registration step — ``_resolve_env(name)`` does an
``importlib.import_module(f"robots.{name}")``, so dropping the package under
``robots/`` on disk is enough. No central list to update.

The sections below describe what each referenced module must contain.
``_add_cli_args`` / ``_parse_config`` are covered in §4 and ``_init_runtime``
in §5.

.. _add-robot-env-rpc:

1. ``env_client.py`` + ``env_server.py``
-----------------------------------------

These files connect the agent process to ``env_server``. The client converts
method calls into RPC requests, and ``env_server`` handles those requests.

1.1 Env client (agent side)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base contract is two gym-style methods (``reset``, ``step``); add whatever
your env needs on top (LIBERO has ``chunk_step``, ``render_camera``,
``get_camera_meta``, ``cached_image``, …). Each method forwards through
``RpcClient.call("<rpc-name>", args=..., kwargs=...)`` with a per-method
timeout. Keep names stable — the driver-side dispatcher matches by name.

.. code-block:: python

   class MyEnvClient:
       def __init__(self, client: RpcClient, *, return_all_frames: bool = False):
           self._client = client
           self.return_all_frames = return_all_frames

       def reset(self):
           return self._client.call("env.reset", timeout_s=120.0)

       def step(self, action):
           return self._client.call("env.step", args=(action,), timeout_s=60.0)
       # ... add other env-specific methods

1.2 Env server (driver side)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mirror the client's API in a facade class on the driver side (e.g.
``MyEnvFacade``). Subclass :class:`rpent.utils.rpc.RpcFacade`, implement
``_dispatch(method, args, kwargs)`` to route ``env.*`` calls to your
methods, and delegate startup to ``self.serve(...)``. Methods take the
same positional / keyword arguments the client sends and return
pickleable values (numpy, not torch — the agent side does not import
torch).

.. code-block:: python

   from rpent.utils.rpc import RpcFacade

   class MyEnvFacade(RpcFacade):
       def __init__(self, env, meta):
           super().__init__()
           self._env = env
           self._meta = meta

       def _dispatch(self, method, args, kwargs):
           if method.startswith("env."):
               return getattr(self, method[len("env."):])(*args, **kwargs)
           raise ValueError(f"unknown RPC method: {method!r}")

       def reset(self): ...
       def step(self, action): ...

   facade = MyEnvFacade(env, meta)
   facade.serve(transport="http", host=host, port=port)

``RpcFacade.serve`` handles transport binding (HTTP or socket), the
``healthz`` / ``shutdown`` methods, parent-death detection, and clean
teardown. The subclass only needs to implement the environment methods.

.. _add-robot-prompts:

2. ``prompt_bundle.py``
-----------------------

Define two prompt factories, ``system_prompt()`` and ``user_prompt()``, and
build a ``PromptBundle(system=system_prompt, user=user_prompt)`` in the
environment's ``__init__.py`` (see the entry point above). Each factory returns an ordered
``dict[str, PromptNode]`` of titled sections; ``PromptBundle.render`` assembles
and fills them. One prompt serves every planner (API loop, Claude Code, Codex):
refer to tools by their bare names (``move_to``, ...) and note once that the
Claude Code and Codex SDKs show them as ``mcp__rpent__<name>``. Do not
maintain separate prompt copies for CLI and API planners.

.. code-block:: python

   # robots/myenv/prompt_bundle.py
   from robots.myenv.prompts import system as system_parts
   from robots.myenv.prompts import user as user_parts
   from rpent.context.prompt_utils import PromptNode

   def system_prompt() -> PromptNode:
       return {
           "INTRO": system_parts.PREAMBLE,
           "GOAL": system_parts.GOAL,
           "RULES": system_parts.RULES,
           "WORKFLOW": system_parts.WORKFLOW,
           "ENVIRONMENT": system_parts.ENVIRONMENT,
           "OUTPUT": system_parts.OUTPUT,
       }

   def user_prompt() -> PromptNode:
       return {
           "TASK": user_parts.TASK,
           "BEGIN": user_parts.BEGIN,
       }

Keep the prompt content under the env package, for example in
``robots/myenv/prompts/system.py`` and ``user.py``. Section bodies are plain
strings (or ``BulletList`` / ``Numbered``) with ``{{suite}}`` / ``{{task}}`` /
``{{seed}}`` / ``{{output_dir}}`` / ``{{recipe_tag}}`` placeholders filled at
render time.

.. _add-robot-toolkit:

3. ``toolkit.py``
------------------

This module owns everything the LLM can call: the tool schemas, the primitive
driver, the per-step state dump, and the MCP allowlist. (In the LIBERO env these
are split between ``tools.py`` and ``toolkit.py`` for historical reasons; for a
new env it is fine to keep them all in ``toolkit.py``.)

A toolkit module typically contains four pieces:

**Primitive driver class** (e.g. ``MyEnvPrimitives``) — a Python object owned
by the toolkit. It holds the ``EnvClient``, the VLA ``model`` client, and any
state needed for the current run. It exposes one method per primitive tool
(``move_to``, ``pi0_pick``, ``release``, …), with each method returning a
``dict`` log.

**Tool definitions and handlers** — a module-level ``TOOLS_SPEC`` list of
Anthropic-style tool definitions (``name``, ``description``, ``input_schema``),
plus any module-level functions referenced by the toolkit (e.g.
``view_driver_state``, ``back_project``, ``finish``).

**Per-step state dump** — ``dump_state(driver, output_dir, step_idx, log)``
serializes whatever state the agent will read back via the ``view_*`` tools
(images, depths, JSON state, camera meta) into ``output_dir``.

**Toolkit class** — subclass ``rpent.tools.toolkit.Toolkit``:

- build the primitive driver in ``__init__`` through a custom initialization
  helper (named ``init_primitives_clean`` in LIBERO; it wipes stale
  ``images/`` etc., constructs the primitives, and dumps step 0),
- register each tool with ``self.add_tool(name, spec, handler)`` — stateless
  readers (``view_driver_state``, ``finish``, …) bind directly to module-level
  functions; primitive tools route through ``_step(name, **kwargs)`` which
  calls ``getattr(self._driver, name)(**kwargs)`` and re-renders state,
- override ``close()`` to write any remaining agent-side artifacts (e.g. the
  LIBERO toolkit saves the agentview MP4 there).

``primitives_kwargs`` (forwarded from ``__init__.py:get_toolkit``) is the dict
the toolkit passes verbatim to your primitive driver's ``__init__`` — typically
``{"env": MyEnvClient(...), "model": VLAClient(...), ...}``.

Conventions worth keeping
-------------------------

- ``output_dir`` is the working directory that the runner creates for each
  run. Images, depths, ``states.json``, transcripts, ``episode.mp4``, and other
  artifacts go there.
- Tool definitions use the Anthropic format (``name`` / ``description`` /
  ``input_schema``). Every tool registered with ``self.add_tool(...)`` is
  exposed to all planners.
- Driver-side return values must be picklable and torch-free.
- Each primitive tool dumps a fresh state snapshot after running so the next
  ``view_driver_state`` call reflects the post-action world.
- Treat ``dump_state`` as the source of truth for what the agent sees — any new
  modality (e.g. tactile, force) goes through it.

.. _add-robot-config:

4. ``_add_cli_args`` + ``_parse_config`` (runner hooks)
-------------------------------------------------------

Environment-specific CLI arguments enter ``rpent/cli/main.py`` through two
hooks and participate in the final argparse pass:

**``_add_cli_args(parser, use_dashboard) -> None``.** Register the
environment's arguments on the shared parser created by main.py.
``use_dashboard`` determines whether normally required arguments remain
optional for the dashboard launcher to fill in later. main.py calls this hook
before ``parser.parse_args()``, so argparse's usage and error output includes
the environment arguments.

**``_parse_config(args) -> RunConfig``.** Called after ``parser.parse_args()``
and, if applicable, the dashboard launcher. Enforces any dashboard-only
optional flags are now populated and returns a
:class:`~rpent.envs.RunConfig`:

- ``recipe_tag`` — env's per-run tag, used in transcript filenames / recipe
  path (LIBERO: ``f"{suite.replace('libero_', '')}_t{task}_s{seed}"``).
- ``output_dir`` — path to the working directory for this run (main.py then
  calls ``init_output_dir`` to create it and configure logging).
- ``prompt_vars`` — dict passed to ``PromptBundle.render`` (typically the run
  identifiers plus anything else the prompts reference).
- ``dashboard_state`` — a :class:`~rpent.dashboard.state.State` when
  ``args.dashboard`` is set, else ``None``.
- ``task_desc`` — env-specific dict of task-identifying fields, written into
  the transcript JSON record verbatim (LIBERO:
  ``{"suite": ..., "task": ..., "seed": ...}``).

.. code-block:: python

   def _add_cli_args(parser, use_dashboard) -> None:
       required = not use_dashboard
       parser.add_argument("--suite", default=None, required=required)
       parser.add_argument("--task", type=int, default=None, required=required)
       # ... other env-specific flags ...

   def _parse_config(args) -> RunConfig:
       if not args.suite: raise ValueError("--suite is required")
       # ... derive recipe_tag, output_dir, prompt_vars, dashboard_state ...
       return RunConfig(
           recipe_tag=recipe_tag,
           output_dir=output_dir,
           prompt_vars=prompt_vars,
           dashboard_state=dashboard_state,
           task_desc={"suite": args.suite, "task": args.task, "seed": args.seed},
       )

.. _add-robot-runtime:

5. ``_init_runtime`` (runner hook)
----------------------------------

After ``parse_config`` returns, main.py calls
``env_spec.init_runtime(args, output_dir)`` to initialize the environment and
VLA services and build the toolkit inputs. The environment can spawn as many
subprocesses as it needs. The current LIBERO implementation starts
``env_server``, ``vla_server``, and ``sam3_server``. The hook returns
``(daemons, primitives_kwargs)``:

- ``daemons: list[ProcessDaemon]`` — subprocesses owned by this run; main.py
  calls ``.stop()`` on each one in its ``finally`` block.
- ``primitives_kwargs: dict`` — passed verbatim to the toolkit constructor
  (which forwards it to the primitive driver's ``__init__``). It typically
  contains ``{"env": MyEnvClient(...), "model": VLAClient(...)}``; add clients
  for any supporting services here as well, such as LIBERO's ``sam3_client``.

Endpoint parsing (``--env-endpoint``, ``--vla-endpoint``, and LIBERO's
``--sam3-endpoint``) and subprocess env composition (``CUDA_VISIBLE_DEVICES``,
``MUJOCO_GL``, ...) live here — main.py knows nothing about them. See
``robots/libero/__init__.py`` for the reference implementation.

Smoke test
----------

Once everything compiles, run this minimal smoke test:

.. code-block:: bash

   PI05_CHECKPOINT_PATH=<path> ANTHROPIC_API_KEY=<key> \
     rpent --env myenv --suite <suite> --task <id> --seed 0 \
     --output-dir /tmp/myenv_smoke --planner api --model anthropic:claude-opus-4-8

Expect the agent to complete the prompted task, and ``finish`` to be
invoked. Check ``<output_dir>/transcript_*.json`` for the post-run
summary.
