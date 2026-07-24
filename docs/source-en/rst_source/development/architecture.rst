System Internals
================

This page is the implementation-level view of RPent. It walks through
what the three processes in the core control path actually own, how
they communicate, and how the pieces slot together under ``rpent/``
and ``robots/``. For a higher-level framing, see :doc:`../overview`.

.. raw:: html

   <div style="text-align: center;">
     <img src="../../architecture.svg" alt="RPent core three-process architecture"
          style="max-width: 95%; height: auto;" />
   </div>

Key features
------------

*(The following points summarize the architecture's main design goals;
later sections explain how they are implemented.)*

- **LLM-in-the-loop control.** The LLM is not fine-tuned — it drives
  the robot purely by calling tools (``pi0_pick``, ``move_to``,
  ``rotate_wrist``, ``back_project``, ``finish``, …). Each tool
  result is fed back as multimodal context (text + rendered images),
  so the model can reason from the current environment state.
- **Core three-process architecture.** The **agent process** (LLM planner
  + toolkit), the **env_server** (simulator + EGL
  rendering), and the **vla_server** (GPU policy weights) are separate
  processes that form the core control path and communicate over
  lightweight RPC. Each heavyweight server process can be restarted,
  moved to another GPU, or pointed at a remote host independently.
- **Pluggable planners.** Select a planner with
  ``--planner {api, claude_code, codex}`` without changing the tools
  or prompts:

  - ``api`` — a provider-agnostic tool-calling loop built on
    `pydantic-ai <https://ai.pydantic.dev/>`_ (Anthropic / OpenAI /
    OpenAI-compatible), with prompt caching and history-image
    pruning.
  - ``claude_code`` — the `Claude Agent SDK
    <https://docs.claude.com/en/api/agent-sdk/overview>`_, exposing
    the toolkit as an in-process MCP server.
  - ``codex`` — the OpenAI Codex SDK, bridged to the toolkit over an
    HTTP MCP server.
- **Two environments, two VLAs, one contract.** LIBERO (Pi0.5 over
  HTTP) and RoboCasa (RLDX-1 over socket-RPC) share the exact same
  env/vla process split; only the wire codec differs, selected to suit
  the structure and size of each environment's observations.
- **Live Dashboard.** With ``--dashboard``, RPent starts a local
  FastAPI monitoring page that shows the agent's reasoning, live
  camera and Pi0 views, an action timeline, and action-clip replays.
  Select the English or Simplified Chinese interface with
  ``--dashboard-language {en, zh-cn}``.
- **Package-based environment extensions.** Environment implementations
  live under ``robots/<env>/`` and RPent resolves them dynamically by
  name. See :doc:`add_robot` for the integration steps.

The LLM-in-the-loop cycle
-------------------------

A single run is an LLM-in-the-loop cycle:

1. The LLM reasons about the task and calls a tool
   (e.g. ``pi0_pick``).
2. The tool's primitive driver requests an action from the ``vla_server``
   (``predict`` / ``vla_infer``).
3. The ``env_server`` executes the action.
4. The environment returns updated observations and camera frames.
5. The results are assembled into text and image context and returned
   to the LLM for the next reasoning turn.

The loop ends when the LLM calls the ``finish`` tool
(``success`` / ``failure`` / ``stuck``) or hits ``--max-turns`` /
``--max-episode-steps``.

Repository layout
-----------------

The framework code is organized by responsibility:

.. code-block:: text

   rpent/
     planner/       # Planner backends: api_loop, claude_code, codex, base.
     cli/            # main.py entrypoint and interactive terminal support.
     context/        # Prompt utilities and shared prompt sections.
     dashboard/      # FastAPI monitor + SSE streams (optional).
     envs/           # EnvSpec, PromptBundle, and on-demand env loading.
     tools/          # Toolkit base class and shared tool helpers.
     utils/          # Config, logging, RPC, and VLA client helpers.
   robots/
     libero/         # LIBERO env_client / env_server / vla_server /
                     # toolkit / prompt_bundle. The reference env.
     (robocasa/)     # RoboCasa driver — in progress.
     (franka/)       # Franka driver — in progress.
     (so101/)        # SO-101 driver — in progress.
   scripts/          # Setup scripts (LIBERO PRO/PLUS, codex proxy).

The runner (``rpent/cli/main.py``)
----------------------------------

``rpent/cli/main.py`` connects the configuration, services, and model
components required for a run. On startup, it:

1. Parses shared CLI flags (:doc:`../quickstart` documents the ones you'll
   use day-to-day) with ``parse_known_args`` to grab ``--env`` and
   ``--dashboard`` early.
2. Resolves the env via ``get_env_spec(args.env_name)`` and calls
   ``env_spec.add_cli_args(parser, use_dashboard=args.dashboard)`` — the env
   registers its flags on the shared parser. ``use_dashboard=True`` makes
   its otherwise-required flags optional so the dashboard can supply them.
3. Runs ``parser.parse_args()`` against the complete parser to perform
   argparse-level validation and produce the final ``args``, retaining
   argparse's standard usage and error output.
4. If ``--dashboard`` is set, starts the launcher with the current arguments
   as defaults and applies the submitted configuration back to ``args``.
5. Calls ``env_spec.parse_config(args)`` to validate the run configuration
   and produce a
   :class:`~rpent.envs.RunConfig`
   (``recipe_tag`` / ``output_dir`` / ``prompt_vars`` / ``dashboard_state``
   / ``task_desc``). Under ``--dashboard``, this is where the env
   enforces that its previously-optional flags were actually filled in.
6. Calls ``init_output_dir`` to create the run's output directory and
   configure ``run.log``.
7. Builds the **planner** through ``rpent.planner.base.build_planner`` based
   on ``--planner``, then renders the system and user prompts from the env's
   prompt bundle.
8. Calls ``env_spec.init_runtime(args, output_dir)``. The env implementation
   starts ``env_server`` and ``vla_server``, or connects to existing services
   when ``--env-endpoint`` / ``--vla-endpoint`` is supplied, and returns
   ``(daemons, primitives_kwargs)``.
9. Passes ``primitives_kwargs`` to the env's ``get_toolkit`` factory to
   construct the **toolkit**.
10. Runs the tool-calling loop, streams to the dashboard if
    ``--dashboard`` is set, and then writes
    ``<output_dir>/transcript_*.json`` and flushes toolkit recordings during
    cleanup.

``main.py`` only connects these stages. Environment-specific code lives
under ``robots/<env>/``, while planner backends live under
``rpent/planner/``. As a result, ``main.py`` imports no environment-specific
class or script.

Environment loading
-------------------

``rpent/envs/base.py`` resolves environment implementations on demand.
For an environment name of ``myenv``, it imports
``robots.myenv`` with ``importlib.import_module`` and then calls the
two factories exposed by that package:

.. code-block:: python

   # robots/myenv/__init__.py
   def get_env_spec() -> EnvSpec: ...  # identity, prompt bundle, and runner hooks
   def get_toolkit(
       *, primitives_kwargs, video_path=None, dashboard=None
   ): ...

``EnvSpec`` has five fields:

- ``name`` and ``prompts`` identify the environment and provide its
  :class:`PromptBundle`.
- ``add_cli_args(parser, use_dashboard) -> None`` registers
  environment-specific arguments on the shared argparse parser. With
  ``use_dashboard=True``, arguments supplied by the launcher may remain
  optional during CLI parsing.
- ``parse_config(args) -> RunConfig`` validates the final arguments and
  produces the :class:`~rpent.envs.RunConfig` for this run.
- ``init_runtime(args, output_dir) -> (daemons, primitives_kwargs)``
  starts the environment and VLA services, or connects to existing
  services, and returns the inputs needed to construct the toolkit.

The loader itself does not maintain a list of environment names. The
current CLI, however, still restricts ``--env`` to ``libero``; adding a
new name therefore also requires updating the CLI choices. See
:doc:`add_robot` for the complete procedure.

Planner interface
-----------------

Every planner implements the ``solve`` interface defined by
``rpent.planner.base.Planner``:

- It receives the system prompt and user message already rendered by
  the runner.
- It receives a ``toolkit``, reads tool definitions through
  ``get_tools_spec()``, and executes tools through ``execute_tool()``.
- It drives the multi-turn tool-calling loop and returns tool results
  to the model. Results that contain images are passed as multimodal
  context.
- It stops after ``finish`` is called or ``max_turns`` is reached.

The three built-in planners share this interface but integrate their
model and tools differently. See :doc:`../usage/configure_planner` for
usage and ``rpent/planner/api_loop.py``, ``claude_code.py``, and
``codex.py`` for the implementations.

Toolkit interface
-----------------

``rpent.tools.toolkit.Toolkit`` is the planner-facing tool container:

- The base class registers common file and I/O tools and maps each tool
  name and input schema to a handler. Subclasses add environment-specific
  tools with ``self.add_tool(name, spec, handler)``.
- An environment-specific toolkit may own a primitive driver. In LIBERO,
  ``LiberoPrimitives`` holds the environment client, model client, and
  per-run state; primitives that advance the environment are routed
  through ``LiberoToolkit._step``.
- After each primitive, ``LiberoToolkit._step`` calls ``dump_state`` to
  save a state snapshot containing state, images, depth, and execution
  logs. A later ``view_driver_state`` call reads this post-action snapshot.

The Toolkit base class forwards tool results to the Dashboard.
``LiberoToolkit`` is responsible for the episode recording
(``episode.mp4``) and per-action clips. A new environment subclasses
Toolkit and registers the tools it exposes.

RPC transports
--------------

RPent includes HTTP and pickle-framed socket RPC transports. Server
processes select one with ``--transport {http,socket}``; HTTP is the
default. On the agent side, the protocol prefix in ``--env-endpoint``
or ``--vla-endpoint`` selects the matching client.

- **HTTP** (``rpent.utils.http_rpc``) sends JSON requests to
  ``POST /call``, which works with standard load balancers and
  cross-language clients. NumPy arrays are encoded as
  ``{"__ndarray__": <base64>, "dtype": ..., "shape": [...]}``.
- **Pickle-framed socket RPC** (``rpent.utils.socket_rpc``) uses
  length-prefixed pickle frames for requests and responses. It suits
  observations with stacked history or nested numpy dictionaries and
  avoids repeated JSON conversion. Because pickle is unsafe for
  untrusted input, use this transport only with trusted endpoints.

On the server side, subclass :class:`rpent.utils.rpc.RpcFacade` and
implement the business dispatcher
``_dispatch(method, args, kwargs)``. The base class handles
``shutdown``, ``healthz``, transport binding, parent-process exit
detection, and server cleanup. Adding a transport requires both client
and server implementations plus integration with ``RpcFacade`` and
endpoint parsing. The toolkit and planner remain unchanged as long as
the client satisfies the ``RpcClient`` interface.

Dashboard (optional)
--------------------

``rpent/dashboard/`` contains a FastAPI application and a static
frontend. With ``--dashboard``, ``rpent/cli/main.py`` starts the
Dashboard using ``--dashboard-host`` and ``--dashboard-port``. It binds
to ``127.0.0.1`` by default and lets the operating system choose a free
port. Before the run starts, the launcher lets the user review or change
the configuration.

During the run, the Dashboard shows:

- planner output and tool-call events;
- live camera and Pi0.5 views;
- the action timeline and per-action clips;
- the complete episode recording after the run, if one was generated.

The server sends state summaries over SSE, and the frontend fetches
detailed events, timeline data, and images as needed. The Dashboard
displays state produced by the planner and toolkit; it does not issue
robot actions directly.

Next steps
----------

- Integrate a robot or simulated environment: :doc:`add_robot`.
- Add a VLA or primitive: :doc:`add_primitive`.
- Learn about Memory design and extension points: :doc:`memory`.
