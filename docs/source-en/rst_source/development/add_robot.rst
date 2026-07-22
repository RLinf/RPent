Add a new robot
===============

This guide walks through what you need to write to plug a new physical /
simulated robot into RPent's LLM-in-the-loop runner. Use
``robots/libero/`` as the worked reference.

RPent splits an env into two processes:

- **Agent side** (``robots/<env>/``) — runs in the agent process; contributes
  the tool schemas, primitive-driver logic, and prompts.
- **Driver side** (``robots/<env>/env_server.py``) — owns the heavyweight
  simulator / robot; exposes its env via a
  :class:`rpent.utils.rpc.RpcFacade` served over HTTP by default
  (``--transport socket`` switches to the pickle-framed TCP transport
  when the obs shape is a better fit).

The two are connected by an ``EnvClient`` class that turns each agent-side
method call into one RPC against the driver.

VLA model runs in its OWN process (env / vla split)
---------------------------------------------------

When an env uses a VLA policy (a learned model that consumes camera obs and
emits actions), that model runs in a **third, separate process** — never inside
the env_server:

- **VLA side** (``robots/<env>/vla_server.py``) — owns ONLY the VLA policy
  (the GPU model). It exposes ``vla_load`` / ``vla_infer`` / ``vla_reset`` over
  its own RPC/HTTP endpoint. It imports NO simulator.
- The toolkit receives a **model client** (e.g. ``VLAClient`` for LIBERO/Pi0.5,
  ``RLDXVLAClient`` for RoboCasa/RLDX-1) as its ``model`` argument, alongside
  the ``EnvClient``. The two clients point at two different server processes.

**Why the split is mandatory (not optional):** the model (large GPU weights,
its own CUDA context, its own heavy deps like ``transformers``/``openpi``) and
the simulator (MuJoCo/robosuite, EGL rendering bound to the main thread) have
conflicting process-level requirements. Co-locating them in one process couples
their lifecycles, forces one interpreter to satisfy both dependency trees, and
lets a model OOM take down the sim. Keeping them separate means either can be
restarted, scaled, or pointed at a remote host independently
(``--vla-endpoint host:port`` reuses an already-running model server). Every env
MUST follow this: env_server owns the sim, vla_server owns the model.

**Transport may differ per env; the architecture may not.** LIBERO uses
HTTP by default for both env_server and vla_server; a robot with
history-stacked nested numpy dict obs may prefer pickle-framed sockets
(``--transport socket``) to avoid JSON re-encoding overhead. Both
transports share the same ``predict`` / ``env.*`` method surface via
:class:`RpcFacade`. Choose the codec that fits the obs; keep the
env/vla process split identical.

**Anything that needs the sim env object stays in env_server.** For RoboCasa,
``check_grasp`` and ``assemble_action`` (the eval ``unmap_action`` +
composite-controller split-index assembly) require the live robosuite env, so
they are env_server RPCs — NOT part of the VLA server. The agent-side skill
(``RLDXSkill``) therefore holds BOTH clients: the env client for
render/step/grasp/assemble, the model client for inference.

Entry point
-----------

For a new env ``myenv``, the file layout is:

.. code-block:: text

   robots/myenv/
       __init__.py            # entry point — get_env_spec() / get_toolkit() factories
       env_client.py          # MyEnvClient — agent-side RPC stub (§1)
       prompt_bundle.py       # system()/user() prompt factories         (§2)
       toolkit.py             # MyEnvToolkit + primitives + tool schemas (§3)
       env_server.py          # driver-side facade + RPC server (§1)
       vla_server.py          # (optional) VLA model server (§1)

``__init__.py`` is the package's entry point. The registry in
``rpent/envs/base.py`` lazily imports ``robots.<name>`` on demand and calls its
two factories:

.. code-block:: python

   # robots/myenv/__init__.py
   from rpent.envs.env_spec import EnvSpec
   from rpent.envs.prompt_bundle import PromptBundle
   from robots.myenv.prompt_bundle import system_prompt, user_prompt

   def get_env_spec() -> EnvSpec:
       return EnvSpec(name="myenv", prompts=PromptBundle(system=system_prompt, user=user_prompt))

   def get_toolkit(*, primitives_kwargs: dict[str, Any], video_path: str | None = None):
       from robots.myenv.toolkit import MyEnvToolkit
       return MyEnvToolkit(primitives_kwargs=primitives_kwargs, video_path=video_path)

That's the entire registration step — ``_resolve_env(name)`` does an
``importlib.import_module(f"robots.{name}")``, so dropping the package under
``robots/`` on disk is enough. No central list to update.

The three sections below describe what each of the three referenced modules
must contain.

1. ``env_client.py`` + ``env_server.py``
-----------------------------------------

These two files form the agent ↔ driver bridge. The client lives in the agent
process and turns method calls into RPCs; the env_server lives in the driver
process and answers them.

1.1 Env client (agent side)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base contract is two gym-style methods (``reset``, ``step``); add whatever
your env needs on top (LIBERO has ``chunk_step``, ``render_agentview``,
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
teardown — you only write the business methods.

``rpent/cli/main.py`` currently imports ``LiberoEnvClient`` and the LIBERO env_server
script path directly. Adding a new env means either branching on
``args.env_name`` to pick the client class + driver script, or factoring those
two callsites out behind a per-env helper.

2. ``prompt_bundle.py``
-----------------------

Define two prompt factories — ``system_prompt()`` and ``user_prompt()`` — and
build a ``PromptBundle(system=system_prompt, user=user_prompt)`` in the env's
``__init__.py`` (see entry point above). Each factory returns an ordered
``dict[str, PromptNode]`` of titled sections; ``PromptBundle.render`` assembles
and fills them. One prompt serves every planner (API loop, Claude Code, Codex):
refer to tools by their bare names (``move_to``, ...) and note once that the
Claude Code / Codex SDK shows them namespaced as ``mcp__rpent__<name>`` — do not
maintain separate CLI/API copies.

.. code-block:: python

   # robots/myenv/prompt_bundle.py
   from rpent.context.prompt_utils import PromptNode
   from rpent.context.prompts import prompt as base_prompt
   from robots.myenv import prompts as myenv_prompt

   def system_prompt() -> dict[str, PromptNode]:
       return {
           "Intro": myenv_prompt.PREAMBLE,
           "Goal": myenv_prompt.GOAL,
           "Rules": myenv_prompt.RULES,
           "Workflow": myenv_prompt.WORKFLOW,
           "Environment": myenv_prompt.ENVIRONMENT,
           "Output": base_prompt.OUTPUT,
       }

   def user_prompt() -> dict[str, PromptNode]:
       return dict(base_prompt.USER)

Reuse the shared sections in ``rpent.context.prompts.prompt`` (``OUTPUT``,
``USER``) or write your own. Section bodies are plain strings (or ``BulletList``
/ ``Numbered``) with ``{{suite}}`` / ``{{task}}`` / ``{{seed}}`` /
``{{output_dir}}`` / ``{{recipe_tag}}`` placeholders filled at render time.

3. ``toolkit.py``
------------------

This module owns everything the LLM can call: the tool schemas, the primitive
driver, the per-step state dump, and the MCP allowlist. (In the LIBERO env these
are split between ``tools.py`` and ``toolkit.py`` for historical reasons; for a
new env it is fine to keep them all in ``toolkit.py``.)

A toolkit module typically contains four pieces:

**Primitive driver class** (e.g. ``MyEnvPrimitives``) — a Python object the
toolkit owns. It holds the ``EnvClient``, the VLA ``model`` client, and any
per-run state. It exposes one method per primitive tool (``move_to``,
``pi0_pick``, ``release``, …) returning a ``dict`` log.

**Tool schemas + handler helpers** — a module-level ``TOOLS_SPEC`` list of
Anthropic-shaped schema dicts (``name``, ``description``, ``input_schema``),
plus any free functions referenced by the toolkit (e.g. ``view_driver_state``,
``back_project``, ``finish``).

**Per-step state dump** — ``dump_state(driver, output_dir, step_idx, log)``
serializes whatever state the agent will read back via the ``view_*`` tools
(images, depths, JSON state, camera meta) into ``output_dir``.

**Toolkit class** — subclass ``rpent.tools.toolkit.Toolkit``:

- build the primitive driver in ``__init__`` via ``init_driver_clean`` (wipes
  stale ``images/`` etc., constructs the primitives, dumps step 0),
- register each tool with ``self.add_tool(name, spec, handler)`` — stateless
  readers (``view_driver_state``, ``finish``, …) bind directly to module-level
  functions; primitive tools route through ``_step(name, **kwargs)`` which
  calls ``getattr(self._driver, name)(**kwargs)`` and re-renders state,
- override ``close()`` to flush any agent-side artifacts (e.g. the LIBERO
  toolkit saves the agentview MP4 there).

``primitives_kwargs`` (forwarded from ``__init__.py:get_toolkit``) is the dict
the toolkit passes verbatim to your primitive driver's ``__init__`` — typically
``{"env": MyEnvClient(...), "model": VLAClient(...), ...}``.

Conventions worth keeping
-------------------------

- ``output_dir`` is the per-run scratch directory and is created by the runner;
  every artifact (images, depths, ``states.json``, transcripts, ``episode.mp4``)
  goes there.
- Tool schemas are Anthropic-shaped (``name`` / ``description`` /
  ``input_schema``). Every tool registered with ``self.add_tool(...)`` is
  exposed to all planners.
- Driver-side return values must be picklable and torch-free.
- Each primitive tool dumps a fresh state snapshot after running so the next
  ``view_driver_state`` call reflects the post-action world.
- Treat ``dump_state`` as the source of truth for what the agent sees — any new
  modality (e.g. tactile, force) goes through it.

Smoke test
----------

Once everything compiles, the minimal smoke loop is:

.. code-block:: bash

   PI05_CHECKPOINT_PATH=<path> ANTHROPIC_API_KEY=<key> \
     rpent --env myenv --suite <suite> --task <id> --seed 0 \
     --output-dir /tmp/myenv_smoke --planner api --model anthropic:claude-opus-4-8

Expect the agent to complete the prompted task, and ``finish`` to be
invoked. Check ``<output_dir>/transcript_*.json`` for the post-run
summary.
