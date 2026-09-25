Add a Planner
=============

A planner receives prompts, chooses tools, and consumes their results. To add one, implement ``Planner.solve`` and expose it through the construction path and CLI choices.

The following interface sketch omits model requests and the tool loop; it is not a runnable implementation. See ``rpent/planner/api_loop.py`` for a complete backend.

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
           dashboard_interaction=None,
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
servers. See :doc:`architecture` for the interface, and
:doc:`add_primitive` if you want to expose new tools to
your custom planner.

Wire and Validate the Planner
-----------------------------

Add a construction branch in ``rpent/planner/base.py:build_planner`` and a CLI choice in ``rpent/cli/main.py``. If the planner supports Dashboard selection or connection diagnostics, wire those entry points and ``rpent/planner/check.py`` as well.

Verify tool dispatch, text and image results, ``finish``, turn limits, errors, and the interruption and interaction paths you support. Reuse test patterns in ``tests/unit_tests/rpent/planner/``. Validate real model connectivity separately.
