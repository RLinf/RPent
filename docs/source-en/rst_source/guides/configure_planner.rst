.. _planners-and-model-services:

Planner Configuration
=====================

Choose a planner, configure its model and credentials, then check the connection
before running a task. Install RPent first; the task examples below assume LIBERO
is ready as described in :doc:`../get_started/quickstart`.

Select a backend with ``--planner`` and follow its setup section:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Planner
     - When to use it
     - Authentication and endpoint
   * - ``api`` (default)
     - Connect directly to a model API, choosing a prefix for its protocol.
     - Environment variables; ``--base-url`` overrides the endpoint.
   * - ``claude_code``
     - Use Claude Code with an Anthropic Messages-compatible service.
     - Claude login, or ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``.
   * - ``codex``
     - Use Codex with an OpenAI Responses-compatible service.
     - Codex login, or ``CODEX_API_KEY`` / ``CODEX_BASE_URL``.
   * - ``gemini_cli``
     - Use an existing Gemini CLI setup.
     - Configure in Gemini CLI.
   * - ``opencode``
     - Use an existing OpenCode model provider.
     - Configure in OpenCode.
   * - ``flash``
     - Replay a stored plan without online model planning.
     - No model API needed; perception and VLA services are still required.
       See :doc:`flash`.

``--base-url`` is only for ``api``; other online planners reject it. For the CLI
backends, RPent starts and manages the selected tool automatically.

The ``api`` Planner (direct Model API)
---------------------------------------

Choose the ``--model`` prefix for your service's API protocol. The part after
the colon must be a model ID accepted by that service:

.. list-table::
   :header-rows: 1
   :widths: 20 25 30 25

   * - API protocol
     - Example ``--model``
     - Key variable
     - Custom endpoint variable
   * - Anthropic Messages
     - ``anthropic:claude-opus-4-8``
     - ``ANTHROPIC_API_KEY``
     - ``ANTHROPIC_BASE_URL``
   * - Gemini
     - ``google:Argon``
     - ``GOOGLE_API_KEY`` first, then ``GEMINI_API_KEY``
     - ``GEMINI_BASE_URL``
   * - OpenAI Responses
     - ``openai:gpt-5.5``
     - ``OPENAI_API_KEY``
     - ``OPENAI_BASE_URL``
   * - OpenAI Chat Completions
     - ``openai-chat:glm-5.2``
     - ``OPENAI_API_KEY``
     - ``OPENAI_BASE_URL``

For example, set an Anthropic key, check the connection, and run a task:

.. code-block:: bash

   export ANTHROPIC_API_KEY="your-api-key"
   rpent-check-llm --planner api --model anthropic:claude-opus-4-8
   rpent --robot libero --planner api --model anthropic:claude-opus-4-8 \
     --suite libero_goal_task --task 1 --seed 0

For a custom service, set the endpoint variable above or add ``--base-url`` to
both the check and task commands; the flag takes precedence. Text-only models
require ``--no-images``. This leaves the planner with textual state only and
may reduce task performance.

.. _planner-claude-code:

The ``claude_code`` Planner
----------------------------

RPent includes the Claude Code dependencies; no separate CLI installation is
needed. Configure a Claude login or ``ANTHROPIC_API_KEY``, then run:

.. code-block:: bash

   rpent-check-llm --planner claude_code --model claude-opus-4-8
   rpent --robot libero --planner claude_code --model claude-opus-4-8 \
     --suite libero_goal_task --task 1 --seed 0

Use ``--model`` without a provider prefix; omitting it selects ``sonnet``.
Set ``--claude-code-max-budget-usd`` to limit spending. Its default comes from
``MAX_BUDGET_USD``, or 10 dollars when unset. See the `Claude Agent SDK docs
<https://code.claude.com/docs/en/agent-sdk/overview>`_ for authentication options.

Local Models with Claude Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The local server must implement the Anthropic Messages API. For example, if
port 8000 already serves ``Qwen/Qwen3.6-27B``:

.. code-block:: bash

   export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
   export ANTHROPIC_API_KEY=EMPTY
   rpent --robot libero --planner claude_code --model Qwen/Qwen3.6-27B \
     --suite libero_goal_task --task 1 --seed 0

Use ``EMPTY`` only for a local service that needs no authentication.
Claude Code assumes a 200,000-token context window for unknown models. If the
server has a different limit, configure context and auto-compaction through the
`Claude Code environment variables <https://code.claude.com/docs/en/env-vars>`_.

.. _planner-gemini-cli:

The ``gemini_cli`` Planner
---------------------------

Install a `Gemini CLI <https://github.com/google-gemini/gemini-cli>`_ version
supporting ``--acp``, then configure its authentication and endpoint:

.. code-block:: bash

   npm install -g @google/gemini-cli
   gemini

Confirm that Gemini CLI can answer a message, exit it, then check and run RPent:

.. code-block:: bash

   rpent-check-llm --planner gemini_cli
   rpent --robot libero --planner gemini_cli \
     --suite libero_goal_task --task 1 --seed 0

Omit ``--model`` to use Gemini CLI's default. To override it, use the service's
model ID, such as ``--model Argon``, without ``google:``. Use the same model
for the check and task. Set ``GEMINI_BIN`` if the executable is not named
``gemini`` or is not on ``PATH``.

.. _planner-opencode:

The ``opencode`` Planner
--------------------------------

Follow the `OpenCode installation guide <https://opencode.ai/docs/>`_, configure
its provider, credentials, and endpoint, and confirm that it can answer a message.
The installed version must support ``acp --pure``; explicit model selection also
requires ACP model configuration options.

.. code-block:: bash

   rpent-check-llm --planner opencode
   rpent --robot libero --planner opencode \
     --suite libero_goal_task --task 1 --seed 0

Omit ``--model`` to use OpenCode's default. To switch models, pass a complete
``provider/model`` ID from ``opencode models`` to both the check and task commands.
Set ``OPENCODE_BIN`` if the executable is not on ``PATH``.

Gemini CLI and OpenCode sessions expose only RPent tools and disable extensions
or external plugins, without modifying global configuration. Personal tool
settings are not all carried into robot tasks. These limits are not an OS sandbox.

.. _planner-codex:

The ``codex`` Planner
----------------------

Codex reuses an existing login by default. For a custom OpenAI Responses-compatible
service, set ``CODEX_BASE_URL`` and ``CODEX_API_KEY``. This planner does not read
``OPENAI_BASE_URL`` or ``OPENAI_API_KEY``.

.. code-block:: bash

   rpent-check-llm --planner codex --model gpt-5.5
   rpent --robot libero --planner codex --model gpt-5.5 \
     --suite libero_goal_task --task 1 --seed 0

Model selection follows ``--model``, then ``CODEX_MODEL``, then Codex's default.
Set ``CODEX_SERVICE_TIER=fast`` to request the fast service tier; it does not
change reasoning effort.

.. _planner-codex-isolation:

Separate Codex Configuration and Login
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a separate directory to keep RPent's Codex configuration, credentials, and
session history apart from your development setup. Use the Codex executable
bundled with RPent's SDK dependency in the terminal used for RPent:

.. code-block:: bash

   cd /path/to/RPent
   source .venv/bin/activate
   export CODEX_HOME="$PWD/.codex-rpent-live"
   export CODEX_SQLITE_HOME="$CODEX_HOME"
   export CODEX_BIN="$(python -c 'from codex_cli_bin import bundled_codex_path; print(bundled_codex_path())')"
   mkdir -p "$CODEX_HOME"
   unset CODEX_API_KEY CODEX_BASE_URL OPENAI_API_KEY OPENAI_BASE_URL CODEX_ACCESS_TOKEN

Before the first login, set this in ``$CODEX_HOME/config.toml``:

.. code-block:: toml

   cli_auth_credentials_store = "file"

Then log in and check the status:

.. code-block:: bash

   "$CODEX_BIN" login --device-auth
   "$CODEX_BIN" login status

Set these environment variables in each new terminal. If ``sqlite_home`` is
already configured, point it to the separate directory too. Custom API users
should instead set ``CODEX_API_KEY`` and ``CODEX_BASE_URL`` for this deployment.
See `Codex authentication <https://developers.openai.com/codex/auth/>`_ for other
login methods.

This does not isolate the workspace or migrate old sessions. Task memory is
selected separately with ``--memory-dir``. Do not commit credentials or this
directory.

Local Models with Codex
~~~~~~~~~~~~~~~~~~~~~~~

For example, serve Qwen3.6-27B through vLLM's Responses API:

.. code-block:: bash

   vllm serve /path/to/Qwen3.6-27B \
     --served-model-name Qwen/Qwen3.6-27B \
     --max-model-len 262144 \
     --reasoning-parser qwen3 \
     --enable-auto-tool-choice \
     --tool-call-parser qwen3_coder

In the RPent terminal, set the endpoint and the context limit the server accepts:

.. code-block:: bash

   export CODEX_BASE_URL=http://127.0.0.1:8000
   export CODEX_API_KEY=EMPTY
   export CODEX_MODEL_CONTEXT_WINDOW=262144
   export CODEX_AUTO_COMPACT_TOKEN_LIMIT=230000
   rpent --robot libero --planner codex --model Qwen/Qwen3.6-27B \
     --suite libero_goal_task --task 1 --seed 0

``--model`` must match ``--served-model-name``. Set ``CODEX_MODEL_CONTEXT_WINDOW``
to the server's ``--max-model-len`` and keep the auto-compaction threshold below
it to leave room for subsequent responses. Both variables are optional; omitting
them uses Codex's defaults. For other models, use their recommended parser settings.

.. _planner-check:

Verify Your Configuration
-------------------------

Run ``rpent-check-llm`` with the same planner and model before starting a task.
It sends a real model request without starting a robot, calling tools, or sending
images. Increase the diagnostic timeout for a slow service:

.. code-block:: bash

   rpent-check-llm --planner gemini_cli --timeout-s 120 --json

The exit code is ``0`` on success and ``1`` on failure; ``--json`` provides output
for scripts. Follow failure hints to check authentication, the model ID, endpoint,
or network. Passing this check does not validate tool use, image input, or a
robot task.

Diagnostic defaults are 30 seconds for ``api`` and 90 seconds for ``opencode``.
Claude Code, Codex, and Gemini CLI use 90 seconds when the corresponding key is in
the launch environment, otherwise 25 seconds. Stored CLI logins can still be used;
Claude Code/Codex report missing authentication immediately if no key or login
configuration is detected. ``--timeout-s`` overrides these limits independently
of the task timeout.

.. _planner-limits:

Configure Planner Limits
------------------------

Common settings are listed below; see :doc:`cli` for all options:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Purpose
   * - ``--max-turns``
     - Planner turn limit, default ``100``. This is not a robot action count;
       see the table below.
   * - ``--planner-timeout-s``
     - Task time limit. When omitted, use ``CELL_TIMEOUT_S``, then 1200 seconds.
       Codex checks ``CODEX_TIMEOUT_S`` first; Gemini CLI checks
       ``GEMINI_CLI_TIMEOUT_S`` first. Terminal interactive API/Claude sessions
       are exempt.
   * - ``--max-tokens``
     - ``api`` only: token limit per response, default ``8192``.
   * - ``--reasoning-effort``
     - API/Claude Code/Codex accept ``none``, ``low``, ``medium``, ``high``, or
       ``xhigh``, subject to model support. API defaults to ``high``; others to
       ``none``. Gemini CLI/OpenCode accept only ``none``.

A model response can request multiple tools, so turn counts do not measure
robot actions:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Backend
     - Meaning of one turn
   * - API
     - One model request, including retries and follow-ups; the budget covers
       the whole conversation.
   * - Claude Code
     - Request tools, execute them, and return their results. A final answer
       without tools does not consume the budget. Each follow-up gets a new budget.
   * - Codex / Gemini CLI
     - One model request or response; multiple tools in one response do not
       count separately.
   * - OpenCode
     - One model step; multiple tools in one step do not count separately.
       Each follow-up gets a new budget.

Reaching a limit does not mean the task succeeded. OpenCode produces a final
response without further tool calls when its turn budget is reached.
``flash`` does not run a model loop or use this budget.

Claude Code, Codex, Gemini CLI, and OpenCode support terminal ``--tui`` and
``--dashboard`` modes. Follow-ups queue by default; see :doc:`dashboard` for
manual interruption. Planner tasks do not automatically load Claude/Codex/Gemini
project development instructions or skills. OpenCode disables project and
Claude configuration while retaining its own global configuration.

.. _planner-direct-actions:

Direct actions
--------------------

For LIBERO, RoboCasa, or RoboTwin, add ``--enable-direct-action`` to let the
planner submit control values directly. VLA and scripted tools remain available.
The environment defines the values and ranges; see :doc:`configure_primitives`.

.. _add-a-custom-planner:

.. _planner-custom:

Add a Planner
-------------

See :doc:`../development/add_planner` for the developer interface to add a backend.
