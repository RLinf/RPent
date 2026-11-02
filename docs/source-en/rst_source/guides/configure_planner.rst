.. _planners-and-model-services:

Planner Configuration
=====================

Select the Agentic Planner backend with one CLI flag:

.. code-block:: text

   --planner {api,claude_code,codex,deepseek_harness,flash}

All online planners receive the same rendered system and user prompts
and use the RPent tool schemas from the same toolkit. They differ in
how those schemas are connected to the model, how the tool-calling
loop is orchestrated, and which model SDK is used.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ``--planner``
     - What it is
     - When to pick it
   * - ``api``
     - A tool-calling loop built on `Pydantic AI <https://ai.pydantic.dev/>`_
       that supports multiple model APIs. For long conversations, it sends
       fewer older messages to the model.
     - Call Anthropic, OpenAI, or a compatible model service directly.
   * - ``claude_code``
     - The `Claude Agent SDK
       <https://code.claude.com/docs/en/agent-sdk/overview>`_. Exposes
       RPent's toolkit as an in-process MCP server; the Claude Agent
       SDK drives the loop.
     - Run the tool-calling loop through the Claude Agent SDK.
   * - ``codex``
     - The OpenAI **Codex Python SDK**. RPent starts an in-process
       Streamable HTTP MCP server that connects the toolkit to Codex.
     - Run tasks through the Codex SDK, using existing authentication or a configured API.
   * - ``deepseek_harness``
     - The DeepSeek Harness Python SDK and bundled runtime. RPent exposes its
       toolkit through a local Streamable HTTP MCP server.
     - Run CLI, TUI, Dashboard, and exploration tasks with the dsh agent loop
       and Codex/GPT.
   * - ``flash``
     - **Flash Mode**, for evaluation only. Replays a plan from memory,
       recorded from an earlier
       run, re-localizing each waypoint's anchor so the plan follows
       objects that moved. See :doc:`flash`.
     - You want to re-run a known-good plan on new layouts, without online
       LLM planning. Perception and VLA services are still required.

.. _planner-direct-actions:

Direct actions
--------------------

For LIBERO, RoboCasa, and RoboTwin, add ``--enable-direct-action`` to the launch
command to let the planner call ``execute_action(values=[...])``. The planner
supplies a list of control values, which the environment executes as one action.
RPent then records the resulting state.

The required values depend on the robot and controller. The tool description
provided to the planner lists the expected number of values and their allowed
ranges. VLA and scripted tools remain available when direct actions are enabled.

The ``api`` Planner (direct Model API)
---------------------------------------

``--planner api`` is the default. It uses the native Pydantic AI tool-calling
runtime and requires a provider prefix in ``--model``. The
project currently installs the Anthropic and OpenAI integrations, so it
can directly use the Anthropic Messages API, the OpenAI Responses API,
and OpenAI-compatible Chat Completions APIs.

Pick the provider by prefixing ``--model``:

.. code-block:: bash

   # Anthropic Claude
   rpent --planner api --model anthropic:claude-opus-4-8 ...

   # OpenAI Responses (e.g. GPT-5.5)
   rpent --planner api --model openai:gpt-5.5 ...

   # OpenAI-compatible chat (e.g. GLM 5.2, text-only)
   rpent --planner api --model openai-chat:glm-5.2 --no-images ...

Environment variables it reads (override with ``--base-url`` if
needed):

- ``anthropic:*`` → ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY``
- ``openai:*`` / ``openai-chat:*`` → ``OPENAI_BASE_URL`` /
  ``OPENAI_API_KEY``

Relevant ``api`` planner knobs:

- ``--max-tokens`` — cap each LLM reply (default ``8192``).
- ``--no-images`` — never send image bytes; this is required for
  text-only models. The agent then reasons from textual state alone,
  so task performance may not be satisfactory.

For ``--tui`` usage, see :ref:`Terminal interaction <quickstart-interactive>`.

.. _planner-claude-code:

The ``claude_code`` Planner
----------------------------

``--planner claude_code`` delegates the loop to the Claude Agent SDK.
RPent creates an in-process MCP server through the SDK and registers
the toolkit's tools under the ``mcp__rpent__<name>`` namespace.

RPent disables filesystem settings sources for Claude planner sessions, so
project ``CLAUDE.md`` instructions and development skills are not loaded
automatically. The working directory remains the repository root.

.. code-block:: bash

   rpent --robot libero --planner claude_code \
     --model claude-opus-4-8 \
     --suite libero_object_swap --task 2 --seed 0

Notes:

- Do **not** add a provider prefix to ``--model``. If it is omitted,
  RPent uses ``sonnet``.
- ``--planner-timeout-s`` limits non-interactive runs. It defaults to
  ``CELL_TIMEOUT_S``, or ``1200`` seconds when that variable is unset.
  The limit is not applied in ``--tui`` mode.
- A dollar budget can be set via ``--claude-code-max-budget-usd``
  (defaults to ``MAX_BUDGET_USD`` env or ``10``).
- RPent already depends on the Claude Agent SDK, which bundles the
  Claude Code binary; no separate CLI installation is required.
  Authentication normally uses ``ANTHROPIC_API_KEY``. See the
  `Claude Agent SDK docs
  <https://code.claude.com/docs/en/agent-sdk/overview>`_.

Local Models with Claude Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Claude Code can use a local model server that implements the Anthropic
Messages API. For a server exposing Qwen3.6-27B as
``Qwen/Qwen3.6-27B``, configure:

.. code-block:: bash

   export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
   export ANTHROPIC_API_KEY=EMPTY

   rpent --robot libero --planner claude_code \
     --model Qwen/Qwen3.6-27B \
     --suite libero_goal_task --task 1 --seed 0

Claude Code assumes a 200,000-token context window for unrecognized model IDs.
If the local server uses a different limit, see the `Claude Code environment
variables <https://code.claude.com/docs/en/env-vars>`_ for its context and
auto-compaction settings.

.. _planner-codex:

The ``codex`` Planner
----------------------

``--planner codex`` uses the OpenAI Codex Python SDK. For each run,
RPent starts a local Streamable HTTP MCP server on a background thread
in the current process, and Codex calls the same toolkit through that
server. You do not need to start ``scripts/codex_proxy/`` first.

RPent excludes repository ``AGENTS.md`` instructions and development skills
in ``.agents/skills/`` from the Codex planner's automatic context loading.
The working directory remains the repository root; robot guides and memory
remain available through the existing tools.

.. code-block:: bash

   rpent --robot libero --planner codex \
     --model gpt-5.5 \
     --suite libero_goal_task --task 1 --seed 0

Notes:

- Set ``CODEX_SERVICE_TIER=fast`` to pass the fast service tier to the Codex
  backend. This does not change ``--reasoning-effort``. When unset, RPent
  does not override the service tier.
- ``--model`` overrides ``CODEX_MODEL``. If neither is set, RPent uses
  the model configured as the Codex SDK default.
- ``--planner-timeout-s`` limits the Codex run. Its default is
  ``CODEX_TIMEOUT_S``, then ``CELL_TIMEOUT_S``, then ``1200`` seconds.
- By default, the Codex SDK reuses existing Codex authentication. For
  a custom Responses-compatible endpoint, set ``CODEX_BASE_URL`` and
  ``CODEX_API_KEY``. This backend does not read ``OPENAI_BASE_URL`` or
  ``OPENAI_API_KEY``.

.. _planner-codex-isolation:

Separate Codex Configuration and Login
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use separate directories to avoid mixing RPent's configuration and session history
with your everyday Codex development setup.

In the terminal that runs RPent, activate ``.venv`` and set separate Codex
configuration and session directories from the repository root. Repeat this setup
in each new terminal used to run RPent:

.. code-block:: bash

   cd /path/to/RPent
   source .venv/bin/activate
   export CODEX_HOME="$PWD/.codex-rpent-live"
   export CODEX_SQLITE_HOME="$CODEX_HOME"
   export CODEX_SERVICE_TIER=fast
   mkdir -p "$CODEX_HOME"
   unset CODEX_API_KEY CODEX_BASE_URL OPENAI_API_KEY OPENAI_BASE_URL CODEX_ACCESS_TOKEN
   export CODEX_BIN="$(python -c 'from codex_cli_bin import bundled_codex_path; print(bundled_codex_path())')"

Before first use, set file-based credential storage in ``$CODEX_HOME/config.toml``, then
log in with the CLI bundled with the SDK:

.. code-block:: toml

   cli_auth_credentials_store = "file"

.. code-block:: bash

   "$CODEX_BIN" login --device-auth
   "$CODEX_BIN" login status

If ``sqlite_home`` is already configured, point it to the separate directory as well.
For a custom API, set ``CODEX_API_KEY`` and ``CODEX_BASE_URL`` for this deployment.
Other login methods are described in
`Codex authentication <https://developers.openai.com/codex/auth/>`_.

RPent does not enable this isolation automatically. It does not isolate the shared
workspace or migrate old sessions. Task memory is selected separately with
``--memory-dir``. Do not commit this directory or credentials.

Local Models with Codex
~~~~~~~~~~~~~~~~~~~~~~~

Codex can use a local model server that implements the OpenAI Responses API.
For example, start Qwen3.6-27B with vLLM:

.. code-block:: bash

   vllm serve /path/to/Qwen3.6-27B \
     --served-model-name Qwen/Qwen3.6-27B \
     --max-model-len 262144 \
     --reasoning-parser qwen3 \
     --enable-auto-tool-choice \
     --tool-call-parser qwen3_coder

Point Codex at the local endpoint and configure the context limits exposed by
the server:

.. code-block:: bash

   export CODEX_BASE_URL=http://127.0.0.1:8000
   export CODEX_API_KEY=EMPTY
   export CODEX_MODEL_CONTEXT_WINDOW=262144
   export CODEX_AUTO_COMPACT_TOKEN_LIMIT=230000

   rpent --robot libero --planner codex \
     --model Qwen/Qwen3.6-27B \
     --suite libero_goal_task --task 1 --seed 0

vLLM reports this limit as ``max_model_len`` in its OpenAI-compatible
``/v1/models`` response, while Codex expects ``context_window`` in its own
model-catalog format. Codex therefore uses fallback metadata for an
unrecognized vLLM model ID. Set ``CODEX_MODEL_CONTEXT_WINDOW`` to the
``--max-model-len`` value accepted by the running server. This runtime limit
may be lower than the checkpoint's advertised maximum to fit the available
GPU memory.

``CODEX_AUTO_COMPACT_TOKEN_LIMIT`` controls when Codex compacts the conversation
history. Keep it below ``CODEX_MODEL_CONTEXT_WINDOW`` to leave room for the
next response; ``230000`` is an example for a ``262144``-token server. Both
variables are optional; when they are unset, Codex uses its defaults.

The value passed to RPent with ``--model`` must match vLLM's
``--served-model-name``. For another model, use its recommended vLLM parser
settings.

.. _planner-deepseek-harness:

The ``deepseek_harness`` Planner
--------------------------------

DeepSeek Harness (dsh) runs the agent loop using RPent's task prompts, memory,
and tools. It currently supports Codex/GPT.

Install and Sign In
~~~~~~~~~~~~~~~~~~~

Install the optional extra in your RPent environment. It pins the Python SDK
and bundled runtime to ``0.2.1rc2``; no separate Node.js installation is needed.

.. code-block:: bash

   uv pip install -e ".[deepseek-harness]"
   export DSH_HOME="$PWD/.dsh-rpent"
   rpent-dsh-login --dsh-home "$DSH_HOME"

Check and Run
~~~~~~~~~~~~~

Check the exact model and reasoning level before starting robot services:

.. code-block:: bash

   rpent-check-llm --planner deepseek_harness \
     --dsh-home "$DSH_HOME" \
     --model gpt-6.1-sol --reasoning-effort xhigh

After :doc:`LIBERO setup <../simulators/libero>`, run a LIBERO-PRO Long Task:

.. code-block:: bash

   rpent --robot libero --planner deepseek_harness \
     --dsh-home "$DSH_HOME" \
     --model gpt-6.1-sol --reasoning-effort xhigh \
     --libero-type pro --suite libero_10_task --task 0 --seed 0 \
     --max-turns 100 --planner-timeout-s 1200

The model ID and reasoning level are separate options. ``xhigh`` means Extra
High; the default is ``high``, and ``none`` is unsupported. The selected model
must support the requested reasoning level and image input.

Interaction and Exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add ``--tui`` or ``--dashboard`` for interactive use; see :doc:`dashboard` for
message queues, interruption, manual actions, and exit commands. These modes
use the same planner configuration as ordinary runs.

For LIBERO exploration, prepare a local seed corpus:

.. code-block:: bash

   python -m robots.libero.memory sync \
     --planner deepseek_harness --memory-version GPT_5.5_xhigh \
     --revision 47f301ef33bf8d0e8c770342f16b3c39c6a0e8c1 \
     --output-dir "$PWD/my-libero-memory"
   rpent --robot libero --planner deepseek_harness \
     --dsh-home "$DSH_HOME" --model gpt-6.1-sol --reasoning-effort xhigh \
     --libero-type pro --suite libero_10_task --task 0 --seed 0 \
     --explore --memory-profile local --memory-dir "$PWD/my-libero-memory" \
     --explore-sessions 3 --explore-attempts-per-session 5 \
     --max-turns 100 --planner-timeout-s 1200

A safe unsolved finish or exhausted step/time budget can start a fresh session
while the configured session quota remains. Each session has its own budget.
The original task and consumed operator constraints carry forward as context;
pending or withdrawn messages do not. Later explicit corrections take precedence.
Actions are not replayed, and complete chat history is not restored. See :doc:`../simulators/libero` for reset,
memory collection, and evaluation rules.

Current Limits and Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``--base-url`` and ``--no-images`` are unsupported. Only RPent tools are
  exposed, one call at a time; dsh's default coding tools, repository
  instructions, skills, subagents, goals, and
  scheduled work are disabled.
- ``--max-turns`` limits root model steps. ``--planner-timeout-s`` covers
  planner startup and active execution, excluding simulator initialization and
  interactive waits for the user. Cleanup can extend beyond the deadline.
  ``--max-tokens`` is a runtime budget, not a hard per-reply transport limit.
- Continuation requires a live host process and a confirmed safe pause.
  Compaction uses the selected model and reasoning level; failed or cancelled
  compaction ends the run. Saved session IDs do not enable cross-process recovery.
- Uncertain execution, incomplete result delivery, or failed cleanup stops
  continuation and queued replacement tasks. Invalid outputs and tool errors
  carrying images also stop the run. Robot actions are not automatically retried.
- Accepted ``finish`` ends the agent loop; it does not establish simulator
  success. A non-interactive model stopping without ``finish`` is an error.
  Use LIBERO's ``result.json`` for native success, finish status, and the
  separate operator verdict; unavailable native evidence is ``null``.
- Reported input tokens include cache reads/writes. Totals cover accepted root
  responses; missing usage remains unknown. Model steps, observed requests, and
  billable requests are different measures.

Images are archived under ``dsh-*/attachments`` in the run directory. The
runtime also needs writable temporary storage supporting cross-directory hard
links; RPent falls back to ``/tmp`` when needed. An archive failure reports an
error and retains the temporary path. See :doc:`cli` for defaults and outputs.

.. _planner-check:

Verify Your Configuration
-------------------------

Before starting a full task, use ``rpent-check-llm`` to check the model
service's connection and authentication settings. It sends the smallest
real request the selected backend supports, without tools, images, or a
robot runtime:

.. code-block:: bash

   rpent-check-llm --planner api --model anthropic:claude-opus-4-8
   rpent-check-llm --planner claude_code
   rpent-check-llm --planner codex --json

It exits ``0`` on success and ``1`` on any failure, and classifies the
failure as one of ``missing_config``, ``unsupported_provider``,
``missing_api_key``, ``auth_failed``, ``network_error``,
``provider_error``, or ``sdk_error``. Use ``--json`` for scripting and
CI. ``--base-url`` overrides supported API endpoints, and ``--timeout-s``
overrides the diagnostic timeout (30 s for ``api``, 90 s for SDK
backends; the ``1200`` s run default is never reused).

Before using the Dashboard, run the same check in a terminal with the
planner and model settings you intend to use for the task. The Dashboard
receives its configuration from the command line and opens directly to the
live monitor. See :doc:`dashboard` for startup instructions.

A passing check proves authentication and reachability only. It does not
prove the model will accept image blocks (see ``--no-images``), your tool
schemas, or your context length.

.. _add-a-custom-planner:

.. _planner-custom:

Add a Planner
-------------

See :doc:`../development/add_planner` for the interface, integration steps, and validation requirements.

Configure Planner Limits
------------------------

``--max-turns N`` sets the planner's turn limit; the default is ``100``.
A turn is not a robot action: one model response can request several tools.
The counting rule depends on the backend:

- **API:** Each model request counts once across the whole conversation,
  including retries and follow-up user inputs. Pydantic AI enforces the limit;
  RPent reports the request count as ``turns_used``. Reaching the limit is a
  normal stop, not a planner error or a claim of task success. Exploration can
  continue with the next session and merge memory when otherwise eligible.
- **Codex:** Each model response counts once, including responses with
  only reasoning or tool calls. Several tools requested in the same response
  still count as one turn. RPent enforces this limit.
- **DeepSeek Harness:** Each root-agent model step counts once. The limit is
  checked before the next step starts. Provider retries and conversation
  compaction requests do not consume separate steps, and multiple tools in
  one response still count as one step. Reaching the limit is a normal stop.
- **Claude Code:** A turn means the model requests tools, those tools run,
  and their results return to the model. A final answer without tool calls
  does not use this budget. RPent passes the limit to Claude Code, which
  returns ``error_max_turns`` if the limit is reached.

For example, with no retries, the model requests two file reads in one
response, then summarizes their results in another. API sends two requests,
Codex counts two responses, and Claude uses one tool-use turn. All three
report ``turns_used=2``; Claude's reported response count differs from its
tool-use budget.

In interactive Claude sessions, each new user input gets a fresh turn
budget, while ``turns_used`` keeps accumulating. See
`Claude's turn-limit documentation
<https://code.claude.com/docs/en/agent-sdk/agent-loop#turns-and-budget>`_.

``flash`` replays a plan without an LLM loop, so this budget does not apply;
it reports ``turns_used=0``.

Other limits have different scopes:

- ``--max-tokens`` caps each reply's tokens for ``api`` only (default ``8192``).
  LIBERO-style tasks usually finish comfortably under this default;
  longer-horizon RoboCasa episodes benefit from raising it if your model
  supports it.
- ``--planner-timeout-s`` limits elapsed planner time, with backend-specific
  defaults and interactive-mode behavior described above.

When the model calls ``finish``, the planner records the finish state.
Reaching a turn limit stops the current loop; an interactive Claude session
can still accept another query. The main program saves the transcript when
the run ends. Timeouts or SDK exceptions are stored in the planner result
and written to the log.
