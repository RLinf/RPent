.. _system-internals:

System Description
==================

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
per-robot memory, the planner also learns when and under what conditions
the VLA is reliable. This taps the LLM's general reasoning and on-the-fly
recovery without retraining a model per task. See :doc:`add_primitive` for how
to add a new primitive.

**A swappable planner.** The planner is the LLM agent runtime that drives the
tool-calling loop. One ``--planner`` flag switches it while the tools and
prompts stay put. Three are built in: ``api`` uses the native Pydantic AI loop
with Harness sliding-window history trimming (the default, provider-agnostic);
``claude_code`` reuses the Claude Agent SDK runtime; ``codex`` reuses the Codex
SDK runtime. Because all three face the exact same tools, they can be compared
head-to-head on the same physical benchmark. See
:doc:`../usage/configure_planner` for configuration.

**The isolated simulation environment.** The simulator runs as a standalone
env_server that talks to the agent over lightweight RPC; the agent side imports
no simulator and is not tied to any specific environment. Swapping environments
only means implementing the same env-client interface — the env can be
restarted on its own, moved to another machine, or replaced with a different
simulator, without touching the planner or tools. Adding a robot needs
no registration code either: drop a package under ``robots/`` and the framework
discovers it. See :doc:`add_robot` for how to wire up a new robot.

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
     planner/        # Planner backends: api_loop, claude_code, codex, base.
     cli/            # main.py entrypoint and interactive terminal support.
     prompt/         # Prompt utilities and shared prompt sections.
     dashboard/      # FastAPI monitor + SSE streams (optional).
     robots/         # RobotSpec, PromptBundle, and on-demand robot loading.
       components/   # Shared Env/policy RPC bases and model adapters.
     tools/          # Toolkit base class and shared tool helpers.
     utils/          # Config, logging, and RPC transports.
   robots/
     libero/         # Environment, policy selection, suites, tools, and prompts.
     robocasa/       # RoboCasa robot (RLDX-1 VLA, kitchen tasks).
     (franka/)       # Franka robot — in progress.
     (so101/)        # SO-101 robot — in progress.
   scripts/
     codex_proxy/    # LiteLLM proxy for the codex planner.
     robocasa/       # RoboCasa run / setup / sweep scripts.

.. _action-model-layers:

Action model layers
-------------------

VLA and WAM identify model families at the configuration boundary. They share
the action-prediction interface internally: a client accepts observations and
returns an action chunk. This does not make their observation encoders,
checkpoints, or planning capabilities interchangeable.

.. list-table:: LIBERO reference implementation
   :header-rows: 1
   :widths: 35 65

   * - Owner
     - Responsibility
   * - ``rpent/robots/components/policy_*_base.py``
     - Common prediction request and facade registration. ``RpcFacade`` owns
       locking, transports, sessions, health checks, and shutdown.
   * - ``rpent/robots/components/pi05_vla_*`` and ``cosmos_policy_*``
     - Model-specific observation/action adaptation and inference.
       Pi0.5 uses the VLA subclasses and ``vla.predict``; Cosmos directly uses
       the policy bases with ``PREDICT_METHOD = "wam.predict"`` on both ends.
   * - ``robots/libero/policy.py`` and ``suites.py``
     - ``PolicyConfig`` selects the backend, capabilities, client and service
       connection. The shared suite catalog and environment routing are
       independent of the model and the chosen evaluation subset.
   * - ``robots/libero/robot_spec.py``
     - Assemble configuration, Dashboard descriptors and runtime components.
       Both model categories enter primitives as ``runtime_kwargs["model"]``.
   * - ``robots/libero/tools.py``, ``toolkit.py`` and ``prompt_bundle.py``
     - Own action semantics, tool exposure, task prompts and state artifacts.
       ``policy_backend`` selects the matching tools and prompts.
   * - ``tests/e2e_tests/libero/benchmark_cosmos_policy.py``
     - Own the evaluation protocol, horizons, selected tasks and result
       aggregation. These settings do not belong in the model adapter.

The shared prediction bases reuse the existing RPC machinery; they do not add
a second transport stack. Model implementations stay separate: Cosmos uses
the official NVIDIA action API in a separately provisioned worker, while
Pi0.5 retains its OpenPI encoding and service. The current Cosmos observation
format and checkpoint target LIBERO. Location under ``components/`` does not
imply support for other benchmarks. Future-state/value generation and
best-of-N planning are disabled in this integration.

Memory is an optional toolkit dependency. A supplied ``MemoryManager`` binds
the common file tools; ``memory=None`` skips ``read_text_file``,
``write_text_file`` and ``list_dir`` during registration while retaining
``finish`` and robot tools. LIBERO's policy configuration currently enables
Memory for Pi0.5 only. Cosmos does not create a manager or reuse Pi0.5
experience; observations and videos remain owned by ``EnvState``. Global
HF synchronization is separate: Cosmos CLI/Dashboard runs still require
``--memory-profile local`` to skip it.

The planner-facing ``cosmos_act`` and the standalone benchmark have different
stopping protocols. The tool executes complete 16-action chunks and checks
episode-end flags between chunks. The benchmark steps predictions one action
at a time and stops on native success or the exact action limit. See
:doc:`../usage/libero` for deployment, constraints and reproduction commands.

The runner (``rpent/cli/main.py``)
----------------------------------

``rpent/cli/main.py`` connects the configuration, services, and model
components required for a run. On startup, it:

1. Parses shared CLI flags (:doc:`../quickstart` documents the ones you'll
   use day-to-day) with ``parse_known_args`` to grab ``--robot`` and
   ``--dashboard`` early.
2. Resolves the robot via ``get_robot_spec(args.robot_name)`` and calls
   ``robot_spec.add_cli_args(parser, use_dashboard=args.dashboard)`` — the robot
   registers its flags on the shared parser. ``use_dashboard=True`` makes
   task-specific flags optional because the Dashboard receives them later
   through task commands.
3. Runs ``parser.parse_args()`` against the complete parser to perform
   argparse-level validation and produce the final ``args``, retaining
   argparse's standard usage and error output.
4. If ``--dashboard`` is set, hands control to ``rpent/cli/dashboard.py`` and
   returns when that long-lived Session ends. The Dashboard-only lifecycle is
   described below; the remaining steps are the normal CLI path.
5. Calls ``robot_spec.parse_config(args)`` to validate the normal CLI run
   configuration
   and produce a :class:`~rpent.robots.RunConfig` (``recipe_tag`` /
   ``output_dir`` / ``prompt_vars`` / ``task_desc``).
6. Calls ``init_output_dir`` to create the run's output directory and
   configure ``run.log``.
7. Builds the **planner** through ``rpent.planner.base.build_planner`` based
   on ``--planner``, then renders the system and user prompts from the robot's
   prompt bundle.
8. Calls ``robot_spec.init_runtime(args, output_dir, dashboard_events, None)``. The
   environment implementation starts or connects to the runtime services
   required by that environment, such as ``env_server``, ``vla_server``, and
   optional supporting services (for example, LIBERO's ``sam3_server`` for
   segmentation), and returns ``(daemons, runtime_kwargs)``.
9. Passes ``runtime_kwargs`` and a ``dashboard_events`` sink to the robot's
   ``get_toolkit`` factory to construct the **toolkit**. The one-shot path
   uses a no-op event sink.
10. Runs the tool-calling loop, then writes
    ``<output_dir>/transcript_*.json`` and flushes toolkit recordings during
    cleanup.

``main.py`` only connects these stages. Robot-specific code lives
under ``robots/<robot>/``, while planner backends live under
``rpent/planner/``. As a result, ``main.py`` imports no robot-specific
class or script.

Robot loading
-------------

``rpent/robots/base.py`` resolves robot implementations on demand.
For a robot name of ``myrobot``, it imports
``robots.myrobot`` with ``importlib.import_module`` and then calls the
two factories exposed by that package:

.. code-block:: python

   # robots/myrobot/__init__.py
   def get_robot_spec() -> RobotSpec: ...  # identity, prompt bundle, and runner hooks
   def get_toolkit(
       *, runtime_kwargs, dashboard_events
   ): ...

``RobotSpec`` gathers the robot's identity, prompt templates, optional
Dashboard description, and three runner hooks (``add_cli_args`` /
``parse_config`` / ``init_runtime``). See :doc:`interfaces` for what each
field must provide.

The loader itself does not maintain a list of robot names. The
current CLI restricts ``--robot`` to ``libero`` and ``robocasa``; adding a
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
frontend. With ``--dashboard``, ``rpent/cli/main.py`` hands control to
``rpent/cli/dashboard.py``, which starts the Dashboard with
``--dashboard-host`` and ``--dashboard-port``. Session configuration comes from
the CLI before it calls ``robot_spec.init_runtime`` once with the shared
component names.
The robot provides a static ``robot_spec.dashboard`` or a
``resolve_dashboard(args)`` hook; the hook takes precedence when present.
The resolved descriptor defines the task
command and fields, runtime components, and frame channels exposed by the
frontend. The Session controller waits for that robot-defined command
(``/rpent-task`` for LIBERO). For every claimed TaskRun, the Dashboard calls
``parse_config`` and the same ``robot_spec.init_runtime`` hook with the unique
component names, merges the shared and unique
primitive inputs, and creates a fresh toolkit and planner conversation. Both
subsets come from explicit ``shared`` / ``unique`` scope values in the
environment's Dashboard spec. In LIBERO, the selected VLA or WAM and SAM3 are
reused while the Dashboard is running; every TaskRun gets a separate environment runtime
and executes sequentially.

LIBERO resolves the model component to ``vla`` / ``VLA`` for Pi0.5 or
``wam`` / ``WAM`` for Cosmos. The common Dashboard runner consumes that
descriptor without model-specific branches. Both categories use the same
startup, readiness and cleanup helpers; external model workers are borrowed
and are not stopped by RPent.

During a TaskRun, the Dashboard shows:

- planner output and tool-call events;
- live fixed-camera and wrist-camera views;
- environment-allowlisted primitive controls that invoke Toolkit actions;
- the action timeline and per-action clips;
- the complete episode recording after the run, if one was generated.

The page accepts ordinary planner messages, new task commands, and interrupt
requests. It also exposes the primitives listed by the environment's Dashboard
spec and validates their arguments against the Toolkit input schemas before
executing them directly. Planners, toolkits, and robot runtimes publish display
updates through a ``dashboard_events`` sink. The server sends state summaries
over SSE, and the frontend fetches detailed events, timeline data, and images
as needed.

Next steps
----------

- Integrate a robot or simulated environment: :doc:`add_robot`.
- Add a VLA or primitive: :doc:`add_primitive`.
- Learn about Memory design and extension points: :doc:`memory`.
