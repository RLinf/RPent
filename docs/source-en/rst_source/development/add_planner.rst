Add a Planner
=============

Add a model backend by implementing ``Planner``, which receives prompts, calls
tools, and returns a task result. For a CLI supporting ACP (Agent Client
Protocol), subclass ``AcpPlanner`` to reuse session execution and interaction.
Both paths use the same toolkit and registration steps below.

Custom Planners
----------------

Subclass ``Planner`` and implement ``solve`` for your model API or SDK.

The following interface sketch omits model requests and the tool loop; it is not a runnable implementation. See ``rpent/planner/api_loop.py`` for a complete backend.

.. code-block:: python

   # rpent/planner/my_planner.py
   from rpent.planner.base import Planner, PlannerResult

   class MyPlanner(Planner):
       def solve(
           self,
           *,
           system_prompt,
           user_message,
           toolkit,
           max_turns,
           input_queue=None,
           dashboard_interaction=None,
       ):
           tools = toolkit.list_tools()
           # Call the model with system_prompt, user_message, and tools.
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
2. Read each tool's ``name``, ``description``, and ``input_schema`` from
   ``toolkit.list_tools()`` and execute
   tools with ``toolkit.execute_tool(name, arguments)``.
3. Convert ``ToolResult.to_text()`` and PNG bytes in ``ToolResult.images`` to the
   format expected by the model SDK.
4. Detect ``ToolResult.data.get("_finish")`` and stop according to
   ``max_turns`` and any other limits.
5. Return a ``PlannerResult`` containing the finish state, messages,
   statistics, and an optional error.

Because the RPent tool schemas and prompt-rendering path stay the same,
adding a planner does not require changes to tools or environment
servers. See :doc:`architecture` for the interface, and
:doc:`add_primitive` if you want to expose new tools to
your custom planner.

ACP CLI Adapters
----------------

For an ACP CLI, subclass ``AcpPlanner`` from ``rpent/planner/utils/acp.py`` and
set its ``backend`` and display ``label``. Implement ``_create_session`` to
prepare the command, environment, and tool policy and return an ``AcpSession``.
See ``gemini_cli.py`` and ``opencode.py``: both use ACP for sessions and HTTP MCP
for RPent tools. Register the adapter under the CLI's name.

The base handles sessions, event recording, connection probes, queued input,
cancellation, and cleanup. OpenCode passes its MCP URL through ``AcpSession``;
Gemini CLI uses workspace settings. Pass a model ID to ``AcpSession`` only when
using the CLI's ACP model selector. The adapter owns authentication and native
turn limits; ACP has no standard model-turn budget.

Wire and Validate the Planner
-----------------------------

Add a construction branch in ``rpent/planner/base.py:build_planner``. Register
an online backend in ``rpent/planner/check.py`` with ``CHECK_PLANNERS``, its
diagnostic timeout, and a connection probe; the CLI derives its online planner
choices from that list. For CLI/SDK planners that own their provider configuration,
also update ``EXTERNALLY_CONFIGURED_PLANNERS``. Dashboard sessions use the
planner selected on the command line. For an ACP adapter, also update the
``--reasoning-effort`` validation in ``rpent/cli/main.py``: its explicit
``gemini_cli`` / ``opencode`` group accepts only ``none``.

Verify tool dispatch, text and image results, actual ``finish`` calls, turn limits, errors, queued input, cancellation, and process cleanup. Reuse test patterns in ``tests/unit_tests/rpent/planner/``. Validate real model connectivity separately.
