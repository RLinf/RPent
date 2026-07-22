System Internals
================

This page is the implementation-level view of RPent. It walks through
what the three processes actually own, how they communicate, and how
the pieces slot together under ``rpent/`` and ``robots/``. For a
higher-level framing, see :doc:`../overview`.

.. raw:: html

   <div style="text-align: center;">
     <img src="../../architecture.svg" alt="RPent three-process architecture"
          style="max-width: 95%; height: auto;" />
   </div>

Key features
------------

*(These are the framework-level guarantees the architecture is designed
around; the sections below then show how each is implemented.)*

- **LLM-in-the-loop control.** The LLM is not fine-tuned — it drives
  the robot purely by calling tools (``pi0_pick``, ``move_to``,
  ``rotate_wrist``, ``back_project``, ``finish``, …). Each tool
  result is fed back as multimodal context (text + rendered images),
  so the model reasons over what it actually sees.
- **Three-process architecture.** The **agent process** (LLM planner
  + toolkit, no ``torch``), the **env_server** (simulator + EGL
  rendering), and the **vla_server** (GPU policy weights) are
  separate processes wired by lightweight RPC. Either heavyweight
  process can be restarted, moved to another GPU, or pointed at a
  remote host independently.
- **Pluggable reasoning brains (planners).** Swap the decision brain
  with one flag — ``--planner {api, claude_code, codex}`` —
  without touching the tools or prompts:

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
  env/vla process split; only the wire codec differs, chosen to fit
  each env's observation shape.
- **Live dashboard.** An optional ``--dashboard`` starts a local
  FastAPI monitor that streams the agent's reasoning, real-time
  camera / Pi0 views, an action timeline, and clip replays — with a
  **bilingual UI** (``--dashboard-language {en, zh-cn}``).
- **Add an environment by dropping a package on disk.** No central
  registry to edit — see :doc:`add_robot`.

How a single turn happens
-------------------------

A single run is an LLM-in-the-loop cycle:

1. The LLM reasons about the task and calls a tool
   (e.g. ``pi0_pick``).
2. The tool's **primitive driver** asks the ``vla_server`` for an
   action chunk (``predict`` / ``vla_infer``).
3. The ``env_server`` executes that chunk (``chunk_step`` for LIBERO,
   stepwise ``step`` for RoboCasa).
4. The env renders the resulting observation and camera frames.
5. Results are turned into text + image content blocks and fed back
   to the LLM for the next turn.

The loop ends when the LLM calls the ``finish`` tool
(``success`` / ``failure`` / ``stuck``) or hits ``--max-turns`` /
``--max-episode-steps``.

Repository layout
-----------------

The code that implements the framework is split cleanly by concern:

.. code-block:: text

   rpent/
     planner/       # Reasoning brains: api_loop, claude_code, codex, base.
     cli/            # main.py entrypoint (no __init__.py — not a subpackage).
     context/        # Prompt bundles, prompt utils, shared prompt sections.
     dashboard/      # FastAPI monitor + SSE streams (optional).
     envs/           # EnvSpec, PromptBundle, and the lazy env registry.
     tools/          # Toolkit base class and shared tool helpers.
     utils/          # Config, logging, RPC client/server, VLA HTTP shim.
   robots/
     libero/         # LIBERO env_client / env_server / vla_server /
                     # toolkit / prompt_bundle. The reference env.
     (robocasa/)     # RoboCasa driver — in progress.
     (franka/)       # Franka driver — in progress.
     (so101/)        # SO-101 driver — in progress.
   scripts/          # Setup scripts (LIBERO PRO/PLUS, codex proxy).

The runner (``rpent/cli/main.py``)
----------------------------------

``rpent/cli/main.py`` is the choreographer. On each invocation it:

1. Parses the CLI flags (:doc:`../quickstart` documents the ones you'll
   use day-to-day).
2. Creates the per-run scratch directory (``--output-dir`` or an
   auto-generated one under ``runs/``).
3. Pre-allocates a free port on the loopback for the **env_server**,
   spawns it as a subprocess (with the port passed on the CLI), and
   polls ``healthz`` via :func:`rpent.utils.rpc.wait_for_ready` until
   the child is up.
4. Does the same for the **vla_server**, or attaches to one via
   ``--vla-endpoint`` when reusing a running instance.
5. Builds the **toolkit** for the chosen env via the env's
   ``get_toolkit(primitives_kwargs=...)`` factory, wiring in the env
   client and the VLA client.
6. Builds the **planner** via ``rpent.planner.base.build_planner``,
   selecting one of ``api_loop.py`` / ``claude_code.py`` /
   ``codex.py`` based on ``--planner``.
7. Runs the tool-calling loop, streams to the dashboard if
   ``--dashboard`` is set, and on exit writes
   ``<output_dir>/transcript_*.json`` plus ``<output_dir>/episode.mp4``.

The runner is intentionally thin: everything env-specific lives under
``robots/<env>/``, and everything brain-specific lives under
``rpent/planner/``.

Env-side registry
-----------------

``rpent/envs/base.py`` maintains a **lazy** registry keyed on the env
name. When you pass ``--env myenv``, it does an
``importlib.import_module("robots.myenv")`` and calls the two
factories the package exposes:

.. code-block:: python

   # robots/myenv/__init__.py
   def get_env_spec() -> EnvSpec: ...
   def get_toolkit(*, primitives_kwargs, video_path=None): ...

There is **no central list** of envs. Dropping a package under
``robots/`` is enough. This is the mechanism you use to add a new
robot (see :doc:`add_robot`).

Planner interface
-----------------

Every planner implements the same tiny interface (see
``rpent.planner.base``):

- Take the rendered ``prompt_bundle`` (system + user sections).
- Take a ``toolkit`` (which exposes tool schemas + a ``dispatch``
  method).
- Drive the tool-calling loop.
- Feed each tool result back as multimodal context.
- Terminate on ``finish`` or when caps are hit.

That is the entire abstraction. The three built-in planners differ
only in *how* they meet the contract — see
:doc:`../usage/configure_planner` for the user-facing view and
``rpent/planner/api_loop.py`` / ``claude_code.py`` / ``codex.py``
for the code.

Toolkit interface
-----------------

A toolkit (``rpent.tools.toolkit.Toolkit``) owns:

- A **primitive driver** — a plain Python object that holds the env
  client, the VLA client, and any per-run state. Each tool the LLM
  can call corresponds to a method on this object.
- A set of **tool schemas** in Anthropic shape (``name``,
  ``description``, ``input_schema``), registered via
  ``self.add_tool(name, spec, handler)``.
- A per-step **state dump** — every primitive tool re-renders the
  world after it runs, so the next ``view_driver_state`` call sees
  the post-action state.

The base class also handles video capture (``episode.mp4``) and the
dashboard event stream. Any new env's ``toolkit.py`` subclasses this
class and registers whatever tools that env exposes.

Transport substrate
-------------------

Two codecs are supported natively, selected via the server's
``--transport {http,socket}`` flag (default ``http``) and mirrored on
the client side by ``--env-endpoint`` / ``--vla-endpoint`` protocol
prefix:

- **HTTP** (``rpent.utils.http_rpc``) — JSON body over ``POST /call``.
  Convenient for standard load balancing and cross-language clients.
  Numpy arrays cross the wire tagged as
  ``{"__ndarray__": <base64>, "dtype": ..., "shape": [...]}``.
- **Pickle-framed socket RPC** (``rpent.utils.socket_rpc``) — for
  history-stacked nested numpy dicts and other wide, variable-shape
  payloads where JSON re-encoding is wasteful.

Server-side, subclass :class:`rpent.utils.rpc.RpcFacade` and implement
``_dispatch(method, args, kwargs)``; the base provides shutdown, healthz,
transport binding, parent-death watch, and clean teardown. Adding a new
transport is a matter of implementing the two-method ``RpcClient``
interface (``call(method, args, kwargs, timeout_s)``); the toolkit and
planner stay unchanged.

Dashboard (optional)
--------------------

``rpent/dashboard/`` is a FastAPI app plus a static frontend. When
``--dashboard`` is set, ``rpent/cli/main.py`` binds it on
``--dashboard-host:--dashboard-port`` (default localhost, random
port), boots a launcher page for picking config, and then streams:

- The agent's reasoning tokens (SSE).
- Live camera / Pi0.5 overlay frames.
- An action timeline.
- On-completion clip replays.

The dashboard is *observational* — it never affects the loop — so a
failure inside the dashboard cannot break a run.

From here
---------

- Adding a new robot? — :doc:`add_robot`.
- Adding a new VLA / action primitive? — :doc:`add_primitive`.
- Curious how memory is designed and where to hook it? —
  :doc:`memory`.
- Need the full-detail extension checklist? — :doc:`add_robot`.
