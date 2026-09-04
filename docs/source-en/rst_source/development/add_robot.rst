Add a New Robot
===============

This guide walks through what you need to write to plug a new physical /
simulated robot into RPent's LLM-in-the-loop runner. Use
``robots/libero/`` as the worked reference.

Integration guidelines
----------------------

- **Reuse RPent abstractions.** Prefer existing Env, VLA, runtime, and memory
  components such as ``BaseEnvClient``, ``BaseEnvFacade``,
  ``BaseVLAClient``, ``BaseVLAFacade``, and ``MemoryManager``.
- **Prefer RLinf Env or VLA implementations.** If RLinf already supports the
  required Env or VLA, keep RPent as a thin adapter where possible.
- **Stay consistent with existing robot integrations where possible.**
  Follow the existing ``RobotSpec``, Prompt, Toolkit, and runtime patterns
  instead of introducing a new shared mechanism for one robot.
- **Explain when reuse is not possible.** If an existing RPent or RLinf
  Env / VLA cannot be reused, explain why in the PR description or open an
  issue so the existing abstraction can be improved.

Integration steps
-----------------

For the overall process layout, service responsibilities, and communication
model, see :doc:`System Design <architecture>`. This guide focuses on the
extension points required to add a robot. Complete them in the following
order:

1. Register the ``RobotSpec`` and toolkit factory in the
   :ref:`entry point <add-robot-entry>`.
2. Implement :ref:`env_client and env_server <add-robot-env-rpc>`. To
   integrate a VLA service and model client, see :ref:`Add a VLA (or other
   model-based primitive) <add-primitive-model-based>`.
3. :ref:`Define the prompts <add-robot-prompts>`.
4. :ref:`Implement the toolkit and primitives <add-robot-toolkit>`.
5. :ref:`Register robot arguments and build RunConfig
   <add-robot-config>`.
6. Implement the :ref:`runtime hook <add-robot-runtime>`. The same hook starts
   the complete runtime for normal CLI runs or a selected component subset for
   the Dashboard.

.. _add-robot-entry:

Entry point
-----------

For a new robot named ``myrobot``, use the following directory layout:

.. code-block:: text

   robots/myrobot/
       __init__.py            # package entry point; re-exports the factories
       robot_spec.py          # RobotSpec, factories, Dashboard spec, runtime hooks
       env_client.py          # MyEnvClient — agent-side RPC stub (§1)
       prompt_bundle.py       # system()/user() prompt factories         (§2)
       toolkit.py             # MyRobotToolkit + primitives + tool definitions (§3)
       env_server.py          # server-side facade + RPC server (§1)
       vla_server.py          # (optional) VLA model server

``__init__.py`` is the robot package's entry point. Keep it small and
re-export the factories implemented in ``robot_spec.py``. The registry in
``rpent/robots/base.py`` lazily imports ``robots.<name>`` on demand and calls
these two functions:

.. code-block:: python

   # robots/myrobot/__init__.py
   from robots.myrobot.robot_spec import get_robot_spec, get_toolkit

   # robots/myrobot/robot_spec.py
   from rpent.dashboard.events import DashboardEventSink
   from rpent.memory import MemoryManager
   from rpent.robots.robot_spec import RobotSpec, RunConfig
   from rpent.robots.prompt_bundle import PromptBundle
   from rpent.utils.config import get_memory_dir
   from robots.myrobot.prompt_bundle import system_prompt, user_prompt

   MYROBOT_DASHBOARD_SPEC = {...}

   def get_robot_spec() -> RobotSpec:
       return RobotSpec(
           name="myrobot",
           prompts=PromptBundle(system=system_prompt, user=user_prompt),
           add_cli_args=_add_cli_args,
           parse_config=_parse_config,
           init_runtime=_init_runtime,
           dashboard=MYROBOT_DASHBOARD_SPEC,
       )

   def get_toolkit(
       *,
       primitives_kwargs,
       dashboard_events: DashboardEventSink,
       config: RunConfig,
   ):
       from robots.myrobot.toolkit import MyRobotToolkit
       return MyRobotToolkit(
           primitives_kwargs=primitives_kwargs,
           dashboard_events=dashboard_events,
           memory=MemoryManager(
               root=config.prompt_vars.get("memory_dir") or get_memory_dir("myrobot"),
           ),
       )

   def _add_cli_args(parser, use_dashboard) -> None:
       """Register robot flags on the shared parser. See §4."""
       ...

   def _parse_config(args) -> RunConfig:
       """Validate final `args`, return a RunConfig. See §4."""
       ...

   def _init_runtime(
       args,
       output_dir,
       dashboard_events: DashboardEventSink,
       components: set[str] | None,
   ):
       """Initialize all runtime components, or only the selected subset.

       Returns (daemons, primitives_kwargs). See §5.
       """
       ...

``dashboard`` is optional. Leave it as ``None`` if the environment does not
support Dashboard control. Otherwise, define the spec in the robot
package: its ``task`` section describes the command, validated fields, display
template, and output slug; ``runtime_components`` and ``frame_channels``
describe the robot-specific rows and camera views rendered by the
frontend. See ``robots/libero/robot_spec.py`` for the reference shape.

That's the entire registration step — ``_resolve_robot(name)`` does an
``importlib.import_module(f"robots.{name}")``, so dropping the package under
``robots/`` on disk is enough. No central list to update.

The sections below describe what each referenced module must contain.
``_add_cli_args`` / ``_parse_config`` are covered in §4 and the runtime hook
in §5. The Dashboard spec is consumed only by the Dashboard runner.

.. _add-robot-env-rpc:

1. ``env_client.py`` + ``env_server.py``
-----------------------------------------

These files connect the agent process to ``env_server``. The client converts
method calls into RPC requests, and ``env_server`` handles those requests.

1.1 Env client (agent side)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass :class:`rpent.robots.components.env_client_base.BaseEnvClient`. It already
validates ``env.get_env_meta`` at startup, performs the initial reset, caches
``last_obs``, and implements the common ``reset``, ``step``, ``chunk_step``,
``render_camera``, ``get_camera_meta``, and ``get_task_language`` RPCs.
Add only environment-specific methods, and extend the timeout table when an
extension needs its own timeout. Keep RPC names stable — the server-side
facade registers each name explicitly.

.. code-block:: python

   from rpent.robots.components.env_client_base import BaseEnvClient

   class MyEnvClient(BaseEnvClient):
       _TIMEOUT_S = {
           **BaseEnvClient._TIMEOUT_S,
           "env.custom_method": 30.0,
       }

       def custom_method(self, arg):
           return self._client.call(
               "env.custom_method",
               args=(arg,),
               timeout_s=self._TIMEOUT_S["env.custom_method"],
           )

   env = MyEnvClient(rpc_client, expected_meta=expected_meta)

1.2 Env server (server side)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mirror the client's API in a facade class on the server side (e.g.
``MyEnvFacade``). Subclass
:class:`rpent.robots.components.env_facade_base.BaseEnvFacade`; it provides the common RPC
routes and read/write dispatch locking. Implement the common environment
methods and extend ``_register_rpc`` for environment-specific routes. Methods
take the same positional / keyword arguments the client sends and return
transport-supported Python / NumPy values (not torch — the agent side does not
import torch).

.. code-block:: python

   from rpent.robots.components.env_facade_base import BaseEnvFacade

   class MyEnvFacade(BaseEnvFacade):
       def __init__(self, env, meta):
           self._env = env
           self._meta = meta
           super().__init__()

       def _register_rpc(self):
           super()._register_rpc()
           # Custom methods must be registered explicitly
           self._rpc["env.custom_method"] = self.custom_method

       # Abstract methods required by BaseEnvFacade
       def get_env_meta(self): ...
       def reset(self): ...
       def step(self, action): ...
       def chunk_step(self, actions, **kwargs): ...
       def get_camera_meta(self, camera_name, **kwargs): ...
       def render_camera(self, camera_name, **kwargs): ...
       def get_task_language(self): ...

       def custom_method(self, arg): ...

   facade = MyEnvFacade(env, meta)
   facade.serve(transport="http", host=host, port=port)

``BaseEnvFacade`` registers common routes through ``_register_rpc`` and
serializes state-changing calls with its read/write lock. Add a route to
``_readonly_methods`` only when it is genuinely safe to run concurrently with
other reads. The inherited ``RpcFacade.serve`` handles transport binding (HTTP
or socket), ``healthz`` / ``shutdown``, parent-death detection, and clean
teardown.

.. _add-robot-prompts:

2. ``prompt_bundle.py``
-----------------------

Define two prompt factories, ``system_prompt()`` and ``user_prompt()``, and
build a ``PromptBundle(system=system_prompt, user=user_prompt)`` in the
robot's ``robot_spec.py`` (see the entry point above). Each factory returns an ordered
``dict[str, PromptNode]`` of titled sections; ``PromptBundle.render`` assembles
and fills them. One prompt serves every planner (API loop, Claude Code, Codex):
refer to tools by their bare names (``move_to``, ...) and note once that the
Claude Code and Codex SDKs show them as ``mcp__rpent__<name>``. Do not
maintain separate prompt copies for CLI and API planners.

.. code-block:: python

   # robots/myrobot/prompt_bundle.py
   from robots.myrobot.prompts import system as system_parts
   from robots.myrobot.prompts import user as user_parts
   from rpent.prompt.utils import PromptNode

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

Keep the prompt content under the robot package, for example in
``robots/myrobot/prompts/system.py`` and ``user.py``. Section bodies are plain
strings (or ``BulletList`` / ``Numbered``) with ``{{suite}}`` / ``{{task}}`` /
``{{seed}}`` / ``{{output_dir}}`` / ``{{recipe_tag}}`` placeholders filled at
render time.

.. _add-robot-toolkit:

3. ``toolkit.py``
------------------

This module owns everything the LLM can call: the tool schemas, the primitives,
the per-step state dump, and the MCP allowlist. (In the LIBERO robot these
are split between ``tools.py`` and ``toolkit.py`` for historical reasons; for a
new robot it is fine to keep them all in ``toolkit.py``.)

A toolkit module typically contains four pieces:

**Primitives class** (e.g. ``MyRobotPrimitives``) — a Python object owned
by the toolkit. It holds the ``EnvClient``, the VLA ``model`` client, and any
state needed for the current run. It exposes one method per primitive tool
(``move_to``, ``pi0_pick``, ``release``, …), with each method returning a
``dict`` log.

**Tool definitions and handlers** — a module-level ``TOOLS_SPEC`` list of
Anthropic-style tool definitions (``name``, ``description``, ``input_schema``),
plus any module-level functions referenced by the toolkit (e.g.
``view_env_state``, ``back_project``, ``finish``).

**Per-step state dump** — ``dump_state(driver, env_state, log)`` opens
``env_state.record_step(...)`` and receives the allocated step index; the
``StepRecord`` is appended and committed immediately. Save large observations
through ``env_state.save(...)`` — inside a ``record_step`` block the ``step``
argument may be omitted (it defaults to the new step), pass an explicit
``step=<int>`` to target a different step, and ``step=None`` for run-level
artifacts. ``EnvState`` adds every successfully saved base name to the step's
flat ``artifacts`` set automatically. Readers use the canonical artifact
filenames rather than maintaining a parallel observation index.

**Toolkit class** — subclass ``rpent.tools.toolkit.Toolkit``:

- forward ``memory`` (a :class:`~rpent.memory.MemoryManager`) and ``state`` to
  ``super().__init__(...)``. Configure ``memory_access`` and
  ``inbox_cell_tag`` on the ``MemoryManager``; eval uses read-only access by
  default.
- build the primitives in ``__init__`` through a custom initialization
  helper (named ``init_primitives`` in LIBERO; it calls
  ``EnvState.reset()``, constructs the primitives, and dumps step 0),
- register each tool with ``self.add_tool(name, spec, handler)`` — stateless
  readers (``view_env_state``, ``finish``, …) bind directly to module-level
  functions; primitive tools route through ``_step(name, **kwargs)`` which
  calls ``getattr(self._primitives, name)(**kwargs)`` and re-renders state,
- override ``close()`` to save remaining agent-side artifacts through
  ``EnvState`` (for example ``state.save("episode.mp4", frames, step=None)``).

``primitives_kwargs`` (forwarded from ``robot_spec.py:get_toolkit``) is the dict
the toolkit passes verbatim to your primitives' ``__init__`` — typically
``{"env": MyEnvClient(...), "model": VLAClient(...), ...}``.

Conventions worth keeping
-------------------------

- ``output_dir`` is the working directory that the runner creates for each
  run. Environment observations are owned by ``EnvState``; callers use logical
  base names and never construct storage paths. Transcripts and other
  run-management outputs share the same run directory.
- Tool definitions use the Anthropic format (``name`` / ``description`` /
  ``input_schema``). Every tool registered with ``self.add_tool(...)`` is
  exposed to all planners.
- Server-side return values must be picklable and torch-free.
- Each primitive tool dumps a fresh state snapshot after running so the next
  ``view_env_state`` call reflects the post-action world.
- Treat ``dump_state`` as the source of truth for what the agent sees — any new
  modality (e.g. tactile, force) goes through it.

.. _add-robot-config:

4. ``_add_cli_args`` + ``_parse_config`` (runner hooks)
-------------------------------------------------------

Robot-specific CLI arguments enter ``rpent/cli/main.py`` through two
hooks and participate in the final argparse pass:

**``_add_cli_args(parser, use_dashboard) -> None``.** Register the
robot's arguments on the shared parser created by main.py.
``use_dashboard`` determines whether normally required arguments remain
optional. For each Dashboard TaskRun, the robot's Dashboard task command
supplies the fields declared by its spec before ``parse_config`` runs. main.py
calls this hook before ``parser.parse_args()``, so argparse's usage and error
output includes the robot arguments.

**``_parse_config(args) -> RunConfig``.** In normal CLI mode, this is called
after ``parser.parse_args()``. In Dashboard mode, it is called for each
TaskRun after the requested fields have been copied to the task arguments. It
validates those fields and returns a
:class:`~rpent.robots.RunConfig`:

- ``recipe_tag`` — robot's per-run tag, used in transcript filenames / recipe
  path (LIBERO: ``f"{suite.replace('libero_', '')}_t{task}_s{seed}"``).
- ``output_dir`` — path to the working directory for this run (main.py then
  calls ``init_output_dir`` to create it and configure logging).
- ``prompt_vars`` — dict passed to ``PromptBundle.render`` (typically the run
  identifiers plus anything else the prompts reference).
- ``task_desc`` — robot-specific dict of task-identifying fields, written into
  the transcript JSON record verbatim (LIBERO:
  ``{"suite": ..., "task": ..., "seed": ...}``).

.. code-block:: python

   def _add_cli_args(parser, use_dashboard) -> None:
       required = not use_dashboard
       parser.add_argument("--suite", default=None, required=required)
       parser.add_argument("--task", type=int, default=None, required=required)
       # ... other robot-specific flags ...

   def _parse_config(args) -> RunConfig:
       if not args.suite: raise ValueError("--suite is required")
       # ... derive recipe_tag, output_dir, and prompt_vars ...
       return RunConfig(
           recipe_tag=recipe_tag,
           output_dir=output_dir,
           prompt_vars=prompt_vars,
           task_desc={"suite": args.suite, "task": args.task, "seed": args.seed},
       )

.. _add-robot-runtime:

5. Runtime initialization hook
------------------------------

``init_runtime`` returns ``(owned_daemons, primitives_kwargs)``:

- ``owned_daemons: list[ProcessDaemon]`` contains only subprocesses started
  by this process. The active runner stops them during cleanup. A client for an
  external endpoint must not add that external service to this list.
- ``primitives_kwargs: dict`` is passed to the toolkit constructor, which
  forwards it to the primitives' ``__init__``. A complete set commonly
  contains ``{"env": MyEnvClient(...), "model": VLAClient(...)}`` plus any
  supporting clients.

The fourth argument, ``components``, selects which named services to
initialize. ``None`` means all services and is what the normal CLI passes.
The Dashboard derives two subsets from ``dashboard.runtime_components``. Every
component declares either ``scope: "shared"`` or ``scope: "unique"`` explicitly.
The Dashboard initializes shared components once, then initializes unique
components for every fresh environment instance. It calls this same hook for
both subsets and merges the returned ``primitives_kwargs`` dictionaries. For
LIBERO, the subsets are ``{"vla", "sam3"}`` and ``{"env"}``.

An implementation should reject unknown component names before starting
anything. When several selected local services are expensive to initialize,
start them all before waiting for readiness so their initialization can
overlap. See ``robots/libero/robot_spec.py`` for the ordered component registry
used by the reference implementation.

Endpoint parsing (``--env-endpoint``, ``--vla-endpoint``, and LIBERO's
``--sam3-endpoint``) and robot-specific server commands belong in the
hook that owns the corresponding service. Wrap those spawners with
``rpent.robots.runtime.try_spawn_server`` and ``try_wait_server`` so status
events, readiness failures, and owned-daemon cleanup stay consistent across
robots. The runners do not handle these environment details. See
``robots/libero/robot_spec.py`` and ``robots/robocasa/robot_spec.py`` for the
reference pattern.

Optional run-result finalizer
-----------------------------

``RobotSpec.finalize_run`` is a universal, robot-agnostic end-of-run hook.
RoboCasa is its current consumer and uses it to record per-cell evaluation
results for later statistics and aggregation. Any robot that publishes
machine-readable evaluation artifacts may register the hook. The default is
``None`` and leaves the runner unchanged. When the hook is present, the normal
terminal runner captures ``toolkit.solved()`` before closing the toolkit, then
passes a structured ``RunFinalizationContext`` to the hook after runtime
cleanup. The hook owns the artifact schema and filename; RPent only defines the
lifecycle boundary.

Use ``write_json_atomic`` when the artifact is JSON so an interrupted write
cannot leave a partial result:

.. code-block:: python

   from rpent.evaluation import RunFinalizationContext, write_json_atomic

   def _finalize_run(context: RunFinalizationContext):
       return write_json_atomic(
           context.output_dir / "result.json",
           {
               "robot": context.robot_name,
               "task": dict(context.task_desc),
               "success": context.environment_success,
           },
       )

Register the callback as ``RobotSpec(..., finalize_run=_finalize_run)``. This
hook is currently limited to normal terminal runs; the Dashboard does not call
it. Keep benchmark manifests, robot-specific runtime fields, and aggregation
logic in the robot package rather than the shared CLI.

Smoke test
----------

Once everything compiles, run this minimal smoke test:

.. code-block:: bash

   PI05_CHECKPOINT_PATH=<path> ANTHROPIC_API_KEY=<key> \
     rpent --robot myrobot --suite <suite> --task <id> --seed 0 \
     --output-dir /tmp/myrobot_smoke --planner api --model anthropic:claude-opus-4-8

Expect the agent to complete the prompted task, and ``finish`` to be
invoked. Check ``<output_dir>/transcript_*.json`` for the post-run
summary.
