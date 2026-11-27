.. _planners-and-model-services:

Planner Configuration
=====================

RPent can call model APIs directly or plan tasks through tools such as Claude Code
and Codex. After choosing a planner, configure its model and credentials, then
check the connection. The task examples below use LIBERO; complete the installation
and environment setup in :doc:`../get_started/quickstart` before running them.

Select a backend with ``--planner`` and follow its setup section:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Planner
     - When to use it
     - Authentication and endpoint
   * - :ref:`api <planner-api>` (default)
     - Connect directly to a model API, choosing a prefix for its protocol.
     - Environment variables; ``--base-url`` overrides the endpoint.
   * - :ref:`claude_code <planner-claude-code>`
     - Use Claude Code with an Anthropic Messages-compatible service.
     - Claude login, or ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``.
   * - :ref:`codex <planner-codex>`
     - Use Codex with an OpenAI Responses-compatible service.
     - Codex login, or ``CODEX_API_KEY`` / ``CODEX_BASE_URL``.
   * - :ref:`gemini_cli <planner-gemini-cli>`
     - Use an existing Gemini CLI setup.
     - Configure in Gemini CLI.
   * - :ref:`opencode <planner-opencode>`
     - Use an existing OpenCode model provider.
     - Configure in OpenCode.
   * - :doc:`flash`
     - Replay a stored plan without online model planning.
     - No model API needed; perception and VLA services are still required.
       See :doc:`flash`.

Configuration depends on the backend:

- ``--base-url`` is only for ``api``; passing it to another online planner is an error.
- RPent starts and manages CLI backends automatically, so you do not need to
  start a background process yourself.

.. _planner-interaction:

To add instructions during a run, use terminal ``--tui`` or ``--dashboard`` mode;
all online planners support both. Claude Code, Codex, Gemini CLI, and OpenCode queue
follow-ups by default, while ``api`` has a different terminal input flow. See
:doc:`dashboard` for each mode's behavior and manual interruption.

.. _planner-api:

The API Planner (Direct Model API)
---------------------------------------

The ``api`` planner calls a model service directly, reading its key and endpoint
from environment variables. Choose the ``--model`` prefix for the service's API
protocol; the part after the colon must be a model ID the service accepts:

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
both the check and task commands. If both are set, ``--base-url`` takes precedence.

If the model supports only text, add ``--no-images``. The planner must then rely
on textual state, which may reduce task performance.

.. _planner-claude-code:

The Claude Code Planner
----------------------------

RPent includes the Claude Code dependencies, so no separate CLI installation is
needed. Configure a Claude login or ``ANTHROPIC_API_KEY``, then run:

.. code-block:: bash

   rpent-check-llm --planner claude_code --model claude-opus-4-8
   rpent --robot libero --planner claude_code --model claude-opus-4-8 \
     --suite libero_goal_task --task 1 --seed 0

Use ``--model`` without a provider prefix; omitting it selects ``sonnet``.

To limit spending, set ``--claude-code-max-budget-usd``. If omitted, RPent uses
``MAX_BUDGET_USD``, or 10 dollars if that variable is also unset.

For other authentication options, see the `Claude Agent SDK docs
<https://code.claude.com/docs/en/agent-sdk/overview>`_.

.. dropdown:: Local Models with Claude Code

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

.. _planner-codex:

The Codex Planner
----------------------

The Codex planner reuses an existing login by default. For a custom service,
first confirm that it supports the OpenAI Responses API, then set
``CODEX_BASE_URL`` and ``CODEX_API_KEY``. This planner does not read
``OPENAI_BASE_URL`` or ``OPENAI_API_KEY``.

.. code-block:: bash

   rpent-check-llm --planner codex --model gpt-5.5
   rpent --robot libero --planner codex --model gpt-5.5 \
     --suite libero_goal_task --task 1 --seed 0

Model selection follows ``--model``, then ``CODEX_MODEL``, then Codex's default.

To request the fast service tier, set ``CODEX_SERVICE_TIER=fast``; this does not
change reasoning effort.

.. dropdown:: Local Models with Codex

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

   Set the context length manually here because vLLM reports ``max_model_len``,
   while Codex reads ``context_window``. Without the expected field, Codex falls
   back to default metadata for unrecognized model IDs.

   ``--model`` must match ``--served-model-name``. Set ``CODEX_MODEL_CONTEXT_WINDOW``
   to the server's ``--max-model-len`` and keep ``CODEX_AUTO_COMPACT_TOKEN_LIMIT``
   below it to leave room for subsequent responses. These two context environment
   variables are optional; omitting them uses Codex's defaults.

   If you use another model, follow its deployment instructions for parser settings.

.. _planner-codex-isolation:

Separate Codex Configuration and Login
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to keep RPent's Codex configuration, credentials, and session history
apart from your development setup, use a separate directory. In the RPent terminal,
first set that directory and locate the Codex executable bundled with the SDK:

.. code-block:: bash

   cd /path/to/RPent
   source .venv-libero/bin/activate
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

A separate configuration directory does not isolate the workspace or migrate old
sessions; task memory is still selected with ``--memory-dir``. Also keep credentials
and this directory out of version control.

.. _planner-gemini-cli:

The Gemini CLI Planner
---------------------------

RPent reuses Gemini CLI's authentication and endpoint configuration. If Gemini CLI
is not installed, install a version supporting ``--acp`` from
`Gemini CLI <https://github.com/google-gemini/gemini-cli>`_, then start it to
complete the setup:

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
for the check and task.

Set ``GEMINI_BIN`` if the executable is not named ``gemini`` or is not on ``PATH``.

.. _planner-opencode:

The OpenCode Planner
--------------------------------

RPent reuses OpenCode's provider, credentials, and endpoint configuration. If you
have not configured it, follow the `OpenCode installation guide
<https://opencode.ai/docs/>`_ and confirm that it can answer a message.

RPent requires an OpenCode version whose ``acp`` subcommand supports ``--pure``;
v1.18.18 has been tested. Before running RPent, check the installed version and options:

.. code-block:: bash

   opencode --version
   opencode acp --help

Once the help output confirms ``--pure`` support, check the connection and run a task:

.. code-block:: bash

   rpent-check-llm --planner opencode
   rpent --robot libero --planner opencode \
     --suite libero_goal_task --task 1 --seed 0

Omit ``--model`` to use OpenCode's default. To switch models, pass a complete
``provider/model`` ID from ``opencode models`` to both commands; the connection
check verifies model selection.

Set ``OPENCODE_BIN`` if the executable is not on ``PATH``.

.. _planner-check:

Verify Your Configuration
-------------------------

Run ``rpent-check-llm`` in a terminal with the same online planner and model before
starting a task or opening ``--dashboard``.
The check sends a real model request without starting a robot, calling tools, or
sending images.

If the service responds slowly, increase the wait with ``--timeout-s``. This command
also uses ``--json`` to produce diagnostic output for scripts:

.. code-block:: bash

   rpent-check-llm --planner gemini_cli --timeout-s 120 --json

The exit code is ``0`` on success and ``1`` on failure. If it fails, follow the hints
to check authentication, the model ID, endpoint, or network. A successful check
only confirms basic connectivity; tool use, image input, and robot tasks still
need separate verification.

.. dropdown:: Diagnostic Timeouts and JSON Status Values

   Diagnostic defaults are 30 seconds for ``api`` and 90 seconds for ``opencode``.
   Claude Code, Codex, and Gemini CLI use 90 seconds when the corresponding key is
   in the launch environment. Without that key, the wait is shorter at 25 seconds,
   but stored CLI logins still work.

   Claude Code and Codex report missing authentication immediately if neither a
   key nor a login configuration is detected.
   ``--timeout-s`` overrides these limits independently of the task timeout.

   Scripts can branch on the ``status`` field and read ``detail`` for diagnostics:

   - ``ok``: the model replied.
   - ``missing_config``: model configuration is missing or incomplete.
   - ``unsupported_provider``: the API provider prefix is not supported.
   - ``missing_api_key``: credentials are missing or login is required.
   - ``invalid_model``: the service rejected the model ID.
   - ``auth_failed``: authentication was rejected.
   - ``network_error``: connection, DNS, TLS, or timeout failure.
   - ``provider_error``: the service refused the request, such as a quota or rate limit.
   - ``sdk_error``: an SDK, CLI process, or unclassified error.

Configuration and Tool Isolation
--------------------------------

To keep local development settings from interfering with robot tasks, RPent limits
the instructions and tools loaded automatically in planner sessions:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Planner
     - Instructions and skills
     - Available tools
   * - Claude Code
     - Does not load user or project Claude Code configuration.
     - RPent tools plus Bash, Read, Write, Glob, and Grep by default.
   * - Codex
     - Does not load project development instructions; disables skills in the RPent repository.
     - RPent tools and native Codex tools; native tool permissions depend on ``--codex-sandbox``.
   * - Gemini CLI
     - Does not load project development instructions; disables skills.
     - RPent tools only; extensions are disabled.
   * - OpenCode
     - Disables project configuration, Claude Code configuration, and external skills; retains its own global configuration.
     - RPent tools only; external plugins are disabled.

To add tools, see :ref:`external-mcp-servers`; to let the planner submit direct
control values, see :ref:`planner-direct-actions`.

These session settings do not modify global configuration.

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
     - RPent tool-call budget for the whole task, default ``100``. Follow-up messages and interrupts do not reset it.
   * - ``--planner-timeout-s``
     - Wall-clock timeout in seconds for online planner runs (interactive API/Claude terminal sessions are exempt). Defaults to ``CODEX_TIMEOUT_S`` (``codex`` only), ``GEMINI_CLI_TIMEOUT_S`` (``gemini_cli`` only), ``CELL_TIMEOUT_S``, or ``1200``.
   * - ``--max-tokens``
     - ``api`` only: token limit per response, default ``8192``.
   * - ``--reasoning-effort``
     - ``none``, ``low``, ``medium``, ``high``, or ``xhigh``, subject to backend and model support. API defaults to ``high``; others to ``none``.

Gemini CLI supports reasoning levels up to ``high``; OpenCode uses the selected
model's advertised levels. For these two backends, ``none`` keeps the CLI configuration.

Each RPent tool call accepted for execution counts once, including read-only tools,
``finish``, and calls that fail during execution. Multiple calls in one response
count separately. Invalid calls rejected before execution and the planner CLI's built-in
tools do not count. ``turns_used`` records this count for every online planner.

At the limit, RPent waits for accepted calls to finish and ends planning without
an error. Reaching the limit does not mean the task succeeded. Each planner session
starts a fresh budget; ``flash`` does not use this budget.

.. _add-a-custom-planner:

.. _planner-custom:

Add a Planner
-------------

See :doc:`../development/add_planner` for the developer interface to add a backend.
