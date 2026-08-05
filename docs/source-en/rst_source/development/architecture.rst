System Internals
================

This page is the implementation-level view of RPent. It walks through
what the three processes in the core control path actually own, how
they communicate, and how the pieces slot together under ``rpent/``
and ``robots/``. For a higher-level framing, see :doc:`../overview`.

.. raw:: html

   <div style="text-align: center;">
     <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png" alt="RPent framework"
          style="max-width: 95%; height: auto;" />
   </div>

Key features
------------

This section highlights the design choices that set RPent apart from
other embodied-agent frameworks.

**The VLA as a retryable tool.** Rather than training an end-to-end policy
that emits actions directly, RPent has a general LLM act as the planner and
call the VLA as action primitives like ``pi0_pick`` and ``pi0_doubled``,
sitting in one tool schema alongside scripted tools such as ``move_to``,
``rotate_wrist``, and ``back_project``. Each call's text and images are fed
back to the LLM so it decides the next step from what it actually sees; with a
per-environment memory, the planner also learns when and under what conditions
the VLA is reliable. This taps the LLM's general reasoning and on-the-fly
recovery without retraining a model per task. See :doc:`add_primitive` for how
to add a new primitive.

**A swappable planner.** The planner is the LLM agent runtime that drives the
tool-calling loop. One ``--planner`` flag switches it while the tools and
prompts stay put. Three are built in: ``api`` is RPent's own tool-calling loop
(built on pydantic-ai, the default, provider-agnostic across model APIs);
``claude_code`` reuses the Claude Agent SDK runtime; ``codex`` reuses the Codex
SDK runtime. Because all three face the exact same tools, they can be compared
head-to-head on the same physical benchmark. See
:doc:`../usage/configure_planner` for configuration.

**The isolated simulation environment.** The simulator runs as a standalone
env_server that talks to the agent over lightweight RPC; the agent side imports
no simulator and is not tied to any specific environment. Swapping environments
only means implementing the same env-client interface — the env can be
restarted on its own, moved to another machine, or replaced with a different
simulator, without touching the planner or tools. Adding an environment needs
no registration code either: drop a package under ``robots/`` and the framework
discovers it. See :doc:`add_robot` for how to wire up a new environment.

The LLM-in-the-loop cycle
-------------------------

A single run is an LLM-in-the-loop cycle:

1. The LLM reasons about the task and calls a tool
   (e.g. ``pi0_pick``).
2. The tool's primitives requests an action from the ``vla_server``
   (``predict``).
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
     robocasa/       # RoboCasa env (RLDX-1 VLA, kitchen tasks).
     (franka/)       # Franka env — in progress.
     (so101/)        # SO-101 env — in progress.
   scripts/
     codex_proxy/    # LiteLLM proxy for the codex planner.
     robocasa/       # RoboCasa run / setup / sweep scripts.

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

``EnvSpec`` gathers the environment's identity, its prompt templates, and the
three runner hooks (``add_cli_args`` / ``parse_config`` / ``init_runtime``); see
:doc:`interfaces` for what each field must provide.

The loader itself does not maintain a list of environment names. The
current CLI restricts ``--env`` to ``libero`` and ``robocasa``; adding a
new name therefore also requires updating the CLI choices. See
:doc:`add_robot` for the complete procedure.

Planner, Toolkit, and RPC transports
-------------------------------------

These three layers stay decoupled, each owning one segment of the path. The
planner only pulls the tool list via ``get_tools_spec`` and invokes tools with
``execute_tool``, indifferent to whether a tool is scripted or a VLA. The
toolkit translates each tool call into a primitive call, and the primitives
issues ``reset`` / ``step`` / ``predict`` requests to ``env_server`` /
``vla_server`` over RPC. The RPC transport (HTTP or socket) only ferries those
calls and their NumPy observations across processes, transparent to the layers
above. That is why swapping the planner leaves the tools untouched, and
swapping the transport leaves the planner untouched. The concrete interface
contracts (``Planner.solve``, ``Toolkit.add_tool``, ``RpcFacade._dispatch``)
are collected in :doc:`interfaces`.

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
