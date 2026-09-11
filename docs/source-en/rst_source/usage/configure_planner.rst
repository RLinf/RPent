Agentic Planner
===============

Select the Agentic Planner backend with one CLI flag:

.. code-block:: bash

   --planner {api, claude_code, codex}

All three planners receive the same rendered system and user prompts
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
     - Provider-agnostic tool-calling loop built on
       `pydantic-ai <https://ai.pydantic.dev/>`_. It currently supports
       the Anthropic Messages API, the OpenAI Responses API, and
       OpenAI-compatible Chat Completions APIs. It handles prompt caching
       and history-image pruning.
     - You want the tightest control over model calls, the widest
       provider coverage, or the cheapest per-turn spend.
   * - ``claude_code``
     - The `Claude Agent SDK
       <https://code.claude.com/docs/en/agent-sdk/overview>`_. Exposes
       RPent's toolkit as an in-process MCP server; the Claude Agent
       SDK drives the loop.
     - You want the agent capabilities built into Claude Code (memory,
       thinking-mode budgets, robust tool retries).
   * - ``codex``
     - The OpenAI **Codex Python SDK**. RPent starts an in-process
       Streamable HTTP MCP server that connects the toolkit to Codex.
     - You want the agent capabilities built into Codex or already have
       OpenAI or Codex quota available.
   * - ``task_card``
     - No LLM at all. Replays a **task card** recorded from an earlier
       run, re-localizing each waypoint's anchor so the plan follows
       objects that moved. See :doc:`task_card`.
     - You want to re-run a known-good plan on new layouts, cheaply and
       without model spend.

The ``api`` planner (direct model API)
---------------------------------------

``--planner api`` is the default. It uses Pydantic AI to implement the
tool-calling loop and requires a provider prefix in ``--model``. The
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
- ``--max-turns`` — cap the number of tool-calling turns (default
  ``100``).
- ``--no-images`` — never send image bytes; this is required for
  text-only models. The agent then reasons from textual state alone,
  so task performance may not be satisfactory.

.. _planner-claude-code:

The ``claude_code`` planner
----------------------------

``--planner claude_code`` delegates the loop to the Claude Agent SDK.
RPent creates an in-process MCP server through the SDK and registers
the toolkit's tools under the ``mcp__rpent__<name>`` namespace.

.. code-block:: bash

   rpent --robot libero --planner claude_code \
     --model claude-opus-4-8 \
     --suite libero_object_swap --task 2 --seed 0

Notes:

- Do **not** add a provider prefix to ``--model``. If it is omitted,
  RPent uses ``sonnet``.
- ``--max-turns`` is passed to the Claude Agent SDK and defaults to
  ``100``.
- ``--planner-timeout-s`` limits non-interactive runs. It defaults to
  ``CELL_TIMEOUT_S``, or ``1200`` seconds when that variable is unset.
  The limit is not applied in ``--interactive`` mode.
- A dollar budget can be set via ``--claude-code-max-budget-usd``
  (defaults to ``MAX_BUDGET_USD`` env or ``10``).
- RPent already depends on the Claude Agent SDK, which bundles the
  Claude Code binary; no separate CLI installation is required.
  Authentication normally uses ``ANTHROPIC_API_KEY``. See the
  `Claude Agent SDK docs
  <https://code.claude.com/docs/en/agent-sdk/overview>`_.

Local models with Claude Code
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

The ``codex`` planner
----------------------

``--planner codex`` uses the OpenAI Codex Python SDK. For each run,
RPent starts a local Streamable HTTP MCP server on a background thread
in the current process, and Codex calls the same toolkit through that
server. You do not need to start ``scripts/codex_proxy/`` first.

.. code-block:: bash

   rpent --robot libero --planner codex \
     --model gpt-5.5 \
     --suite libero_goal_task --task 1 --seed 0

Notes:

- ``--model`` overrides ``CODEX_MODEL``. If neither is set, RPent uses
  the model configured as the Codex SDK default.
- ``--planner-timeout-s`` limits the Codex run. Its default is
  ``CODEX_TIMEOUT_S``, then ``CELL_TIMEOUT_S``, then ``1200`` seconds.
- By default, the Codex SDK reuses existing Codex authentication. For
  a custom Responses-compatible endpoint, set ``CODEX_BASE_URL`` and
  ``CODEX_API_KEY``. This backend does not read ``OPENAI_BASE_URL`` or
  ``OPENAI_API_KEY``.

Local models with Codex
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

.. _planner-check:

Verify your configuration
-------------------------

A full run boots the env server, the VLA server, and the memory corpus
before it ever reaches the model, so a wrong API key can cost minutes of
startup before it surfaces. ``rpent-check-llm`` sends the smallest real
request the selected backend supports — no tools, no images, no robot
runtime — and reports the outcome:

.. code-block:: bash

   rpent-check-llm --planner api --model anthropic:claude-opus-4-8
   rpent-check-llm --planner claude_code
   rpent-check-llm --planner codex --json

It exits ``0`` on success and ``1`` on any failure, and classifies the
failure as one of ``missing_config``, ``unsupported_provider``,
``missing_api_key``, ``auth_failed``, ``network_error``,
``provider_error``, or ``sdk_error``. Use ``--json`` for scripting and
CI. ``--base-url`` overrides the backend's endpoint, and ``--timeout-s``
overrides the diagnostic timeout (30 s for ``api``, 90 s for the two SDK
backends; the ``1200`` s run default is never reused).

The Dashboard exposes the same check: the launcher's **Test connection**
button runs it against the planner and model currently selected in the
form, so what you test is exactly what **Start Session** will use. Both
front ends call one implementation in ``rpent.planner.check``.

A passing check proves authentication and reachability only. It does not
prove the model will accept image blocks (see ``--no-images``), your tool
schemas, or your context length.

.. _planner-custom:

Add a custom planner
--------------------

If none of the three planners fit — say you want to plug in an
in-house planner, a research prototype, or a different agent SDK —
implement the ``rpent.planner.base.Planner`` protocol and add a
construction branch to ``rpent.planner.base.build_planner``:

.. code-block:: python

   # rpent/planner/my_planner.py
   from rpent.planner.base import PlannerResult

   class MyPlanner:
       def solve(
           self,
           *,
           system_prompt,
           user_message,
           toolkit,
           max_turns,
           input_queue=None,
       ):
           tool_specs = toolkit.get_tools_spec()
           # Call the model with system_prompt, user_message, and tool_specs.
           # Execute each tool call through this interface:
           tool_result = toolkit.execute_tool(tool_name, arguments)
           ...
           return PlannerResult(
               finish_result=finish_result,
               messages=messages,
               stats=stats,
               error=error,
           )

Any planner must:

1. Accept the rendered ``system_prompt`` and ``user_message``.
2. Read the tool schemas from ``toolkit.get_tools_spec()`` and execute
   tools with ``toolkit.execute_tool(name, arguments)``.
3. Convert the text and images in ``ToolResult.content_blocks`` to the
   format expected by the model SDK.
4. Detect ``ToolResult.is_finish`` and stop according to
   ``max_turns`` and any other limits.
5. Return a ``PlannerResult`` containing the finish state, messages,
   statistics, and an optional error.

Because the RPent tool schemas and prompt-rendering path stay the same,
adding a planner does not require changes to tools or environment
servers. See :doc:`../development/architecture` for the interface, and
:doc:`../development/add_primitive` if you want to expose new tools to
your custom planner.

Configure planner limits
------------------------

The limiting options apply to different planners:

- ``--max-tokens`` caps *per-reply* tokens only for the ``api``
  planner. LIBERO-style tasks usually
  finish comfortably under ``8192``; longer-horizon RoboCasa episodes
  benefit from raising it if your model supports it.
- ``--max-turns`` caps the *total number of tool-calling turns*. A
  single LIBERO task rarely needs more than ~30 turns; RoboCasa
  long-horizon tasks can approach the default ``100``.
- ``--planner-timeout-s`` limits the planner's running time.

When the model calls the ``finish`` tool, the planner records the
corresponding finish state. Reaching a turn limit or timeout ends the
run, and the main program still saves the transcript. Timeouts or SDK
exceptions are stored in the planner result and written to the log.
