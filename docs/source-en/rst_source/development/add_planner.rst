Add a Planner
=============

Planners call models, execute robot tools, and return task results. To add a
backend, subclass ``Planner`` for a custom API or SDK, or subclass ``AcpPlanner``
for a CLI that supports ACP (Agent Client Protocol) to reuse session execution
and interaction. Both paths use the RPent toolkit and require registration and
verification after implementation.

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

If the CLI supports ACP, you can reuse ``AcpPlanner`` from
``rpent/planner/utils/acp.py``. Subclass it, set its ``backend`` and display
``label``, and implement ``_create_session`` to prepare the command, environment,
and tool policy and return an ``AcpSession``.

See ``rpent/planner/gemini_cli.py`` and ``rpent/planner/opencode.py`` for complete
examples; both use ACP for sessions and HTTP MCP for RPent tools.

``AcpPlanner`` handles sessions, event recording, connection probes, queued input,
cancellation, and cleanup. Each adapter configures the following for its CLI:

- MCP registration: pass ``mcp_url`` to ``AcpSession``, as OpenCode does, or write
  workspace settings, as Gemini CLI does.
- Model selection: pass a model ID to ``AcpSession`` only when using the CLI's
  ACP model selector.
- Authentication: reuse the CLI's authentication configuration.
- Reasoning effort: pass ``reasoning_effort`` to ``AcpSession`` for an ACP
  ``thought_level`` selector, or configure the CLI directly.

The shared driver enforces ``max_turns`` as a task-wide RPent tool-call budget.
Disable native model-turn limits so they do not stop the task early.

Wire and Validate the Planner
-----------------------------

Once implemented, register the backend so RPent can construct it and check its
connection. Dashboard uses the planner selected on the command line, so it needs
no separate integration. Registration depends on the base class:

- Custom ``Planner``: add a construction branch in ``build_planner`` in
  ``rpent/planner/base.py``. For an online backend, register it in ``CHECK_PLANNERS``
  in ``rpent/planner/check.py`` with a diagnostic timeout and connection probe;
  the CLI derives its online planner choices from that list. If the CLI/SDK owns
  its provider configuration, also add it to ``EXTERNALLY_CONFIGURED_PLANNERS``.
- ``AcpPlanner`` adapter: add its name to ``ACP_PLANNERS`` in
  ``rpent/planner/check.py``, set its diagnostic timeout, and select its class in
  ``build_planner`` and ``_check_acp``. ``ACP_PLANNERS`` supplies online planner
  registration and external configuration handling.

After registration, use the test patterns in ``tests/unit_tests/rpent/planner/``
to verify tool dispatch, text and image results, actual ``finish`` calls, tool-call
budgets, errors, queued input, cancellation, and process cleanup offline. Once
those tests pass, validate real model connectivity separately.
