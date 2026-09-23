EmbodiedAgent: external MCP benchmarks
======================================

``EmbodiedAgent`` exposes RPent's existing planner loop to a benchmark without
requiring a robot package under ``robots/``. The benchmark owns episode reset,
robot actions, observation capture, and success scoring. It exposes actions and
observations through an MCP server, supplies a system prompt and optional skill
files, then calls ``run`` once per episode.

.. code-block:: python

   from rpent.embodied_agent import EmbodiedAgent, McpServer

   agent = EmbodiedAgent(
       mcp_servers=[McpServer(name="robot", url="http://127.0.0.1:8000/mcp")],
       output_dir="runs/episode-001",
       planner="api",
       model="openai:gpt-5.5",
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

Supported planners are ``api``, ``claude_code``, and ``codex``. Set ``model``
and provider credentials as described in :doc:`configure_planner`. Skill files
are read and inserted into the prompt on every run; they are not automatically
discovered from the working directory. The local ``finish`` tool records the
agent's own conclusion in ``PlannerResult.finish_result``. That conclusion is
not a benchmark success signal; always use the environment's score or success
predicate for evaluation.
