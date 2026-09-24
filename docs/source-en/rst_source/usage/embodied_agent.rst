EmbodiedAgent: external MCP benchmarks
======================================

``EmbodiedAgent`` exposes RPent's existing planner loop to a benchmark without
requiring a robot package under ``robots/``. The benchmark owns episode reset,
robot actions, observation capture, and success scoring. It exposes actions and
observations through an MCP server, supplies a system prompt and optional skill
files, then calls ``run`` once per episode.

.. code-block:: python

   from rpent.embodied_agent import EmbodiedAgent, McpServer
   from rpent.llm import LLMConfig

   agent = EmbodiedAgent(
       mcp_servers=[McpServer(name="robot", url="http://127.0.0.1:8000/mcp")],
       output_dir="runs/episode-001",
       planner="api",
       llm=LLMConfig(provider="openai", model="gpt-5.5"),
   )
   result = agent.run(
       "Place the red block in the bowl.",
       system_prompt=(
           "You control the robot through MCP tools. The move_eef tool accepts "
           "a target pose and returns a camera observation. Check each result "
           "before choosing the next action."
       ),
       skills=["benchmark/SKILL.md"],
   )
   # Read the benchmark's own success flag to score this episode.
   usage = result.stats["llm_usage"]
   print(usage["input_tokens"], usage["output_tokens"])
   print(usage["cache_read_tokens"], usage["cache_write_tokens"])

The same entry point works when motion and camera capture are separate tools:
describe their intended sequence in ``system_prompt`` or a skill. Tool names,
input schemas, and descriptions come directly from MCP discovery. The planner
sees each as ``<server_name>__<tool_name>``. MCP text and image content are
passed back to the model. Return camera frames as MCP image content when the
model must see pixels; a text path or URL remains text. Include the task,
coordinate frame, action limits, and observation rules in the supplied context;
RPent does not assume a fixed ``move_eef`` or ``snapshot`` schema.

For a local stdio server, pass ``command`` and optional ``args``, ``env``, and
``cwd`` instead of ``url``. Exactly one of ``url`` or ``command`` is required
per server. Multiple servers may be supplied, with unique names. A stdio
server is launched and closed for every call to ``run``. Use a distinct
``output_dir`` for each episode.

Supported planners are ``api``, ``claude_code``, and ``codex``. ``LLMConfig``
works with the ``api`` planner and supports ``provider="openai"`` or
``provider="anthropic"``. It reads ``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``
by default; ``api_key`` and ``base_url`` may also be passed explicitly. OpenAI
uses the Responses API by default; set ``openai_format="chat"`` for a Chat
Completions-compatible endpoint. The previous ``model`` and ``base_url``
arguments remain available as described in :doc:`configure_planner`.

For a Responses endpoint that supports explicit prompt caching, set
``prompt_cache_key`` to a stable value and ``prompt_cache_mode="explicit"`` in
``LLMConfig``. RPent marks the initial task and multimodal tool feedback as
cache breakpoints. Set ``image_history_groups=2`` to retain the last two
observation groups; older camera images become text placeholders. These
options are opt-in because compatible endpoints may not support them.
Calculate the provider-reported cache hit rate as
``cache_read_tokens / input_tokens`` over completed requests. The input count
already includes cached tokens.

``result.stats["llm_usage"]`` contains ``input_tokens``, ``output_tokens``,
``cache_read_tokens``, ``cache_write_tokens``, ``reasoning_output_tokens``,
``requests``, ``cost_usd``, and ``total_tokens``. Cached input is included in
``input_tokens``; reasoning output is included in ``output_tokens``. ``None``
for requests or cost means it was not reported. A zero cache count may mean the
provider did not report cache usage.

For standalone model calls, ``LLMClient(LLMConfig(...))`` offers ``generate``
and ``generate_sync``. Each response has ``text`` and ``usage``; the client also
exposes cumulative ``total_usage``.

LLM requests retry HTTP 408, 409, 425, 429, and 5xx responses, plus provider
API or connection errors without an HTTP status. The default is two retries
with capped exponential backoff;
configure it with ``LLMConfig(retry=RetryPolicy(max_retries=...))``. HTTP 400,
401, 403, and other permanent failures are logged and returned immediately.
Each failed provider attempt writes a sanitized record to
``<output_dir>/llm_errors.jsonl`` for the ``api`` planner. The record includes
the status, attempt number, and retry decision, without prompts, response
bodies, or API keys. For direct calls, pass ``log_path`` to ``LLMClient`` to
save the same records to a file. Errors still propagate from direct calls;
``EmbodiedAgent.run`` returns them in ``PlannerResult.error``.

Skill files are read and inserted into the prompt on every run; they are not
automatically discovered from the working directory. The local ``finish`` tool records the
agent's own conclusion in ``PlannerResult.finish_result``. That conclusion is
not a benchmark success signal; always use the environment's score or success
predicate for evaluation.

Selected memory and initial observations
----------------------------------------------------

Pass already selected memory excerpts with ``memory``. Each ``ContextDocument``
retains a title, text, and optional source identifier:

.. code-block:: python

   from rpent.context import ContextDocument

   result = agent.run(
       "Place the red block in the bowl.",
       system_prompt="Use the registered robot tools and inspect each result.",
       skills=["benchmark/SKILL.md"],
       memory=[ContextDocument(
           title="Top grasp",
           text="Approach this object from above; recheck its current pose.",
           source="memory/grasp.md",
       )],
   )

Memory is appended to the task as labeled reference text, including its source
when provided. It is not added to ``system_prompt``. The caller owns selection
and authorization of these excerpts; ``source`` is provenance and is never
opened by the assembler. Existing robot memory access and merge policies are
unchanged.

``initial_context`` accepts a sequence of text and PydanticAI ``BinaryContent``
objects for the ``api`` planner. These parts follow the task and selected memory
in their supplied order, preserving image bytes and metadata. Other planners do
not accept this argument. Skills, memory, and observations use the same
``rpent.context.assemble_context`` function as CLI and Dashboard runs; see
:ref:`planner-context` for direct use of the structured bundle.

RoboDojo example
----------------

``examples/robodojo`` contains an XPolicyLab bridge that uses ``EmbodiedAgent``
for RoboDojo decisions. Its ``snapshot`` MCP tool supplies camera frames and
state; ``submit_action`` returns short discrete commands. RoboDawn's motion
controller executes those commands, and RoboDojo supplies the native score.
The example pins the RoboDawn controller revision, keeps credentials outside
Git, and includes a launcher for parallel GPU workers. See the example's
``README.md`` for the installation-specific setup and result paths.
