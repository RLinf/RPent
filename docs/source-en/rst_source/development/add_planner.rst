Add a Planner
=============

A planner receives prompts, chooses tools, and consumes their results.

ACP Harnesses
--------------

ACP is a shared driver, not a selectable planner. For an ACP harness, subclass
``AcpPlanner`` from ``rpent/planner/utils/acp.py`` and set its ``backend`` and
display ``label``. Implement ``_create_session`` to prepare the command,
environment, and tool policy and return an ``AcpSession``; see ``opencode.py``
for standard HTTP MCP setup or ``gemini_cli.py`` for workspace settings.
Register the adapter under the harness's name using the steps below.

The base shares session handling, event recording, probes, and
``utils/agent_session.py`` interaction, HTTP MCP, artifacts, and cleanup.
Pass the MCP URL to ``AcpSession`` for agents advertising HTTP MCP support;
Gemini supplies it through workspace settings. Pass a model ID only when using
the agent's ACP model selector. Authentication and native turn limits belong
to the harness adapter; ACP has no standard model-turn budget.
Verify text and image tool results, actual ``finish`` calls, queued follow-ups,
cancellation, and process cleanup separately from the connection probe.

Custom Planners
----------------

For a different execution protocol, subclass ``Planner``, implement ``solve``,
and expose it through the construction path and CLI choices.

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

Wire and Validate the Planner
-----------------------------

Add a construction branch in ``rpent/planner/base.py:build_planner``. Register
an online backend in ``rpent/planner/check.py`` with ``CHECK_PLANNERS``, its
diagnostic timeout, and a connection probe; the CLI derives its online planner
choices from that list. For harnesses that own their provider configuration,
also update ``EXTERNALLY_CONFIGURED_PLANNERS``. Dashboard sessions use the
planner selected on the command line.

Verify tool dispatch, text and image results, ``finish``, turn limits, errors, and the interruption and interaction paths you support. Reuse test patterns in ``tests/unit_tests/rpent/planner/``. Validate real model connectivity separately.
