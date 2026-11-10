Add a Planner
=============

A planner receives prompts, chooses tools, and consumes their results. To add one, subclass ``Planner``, implement ``solve``, and expose it through the construction path and CLI choices.

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
   statistics, and optional ``error`` and ``stop_reason`` fields.

Planners may supply ``stop_info`` with their raw ``reason`` and a common
``category``: ``finish``, ``budget``, ``timeout``, ``cancelled``, ``error``, or
``unknown``. ``normalize_stop_info`` preserves backend reasons and fills the
category for common reasons; unknown reasons remain unknown. For result artifacts,
prefer ``stop_info.reason``, then ``stop_reason``, then the legacy reason in
``stats``. Keep execution
certainty, result delivery, cleanup, and permission to start a new session as
separate evidence. Starting a new session requires ``execution_certain``,
``result_delivery_complete``, ``cleanup_complete``, and ``allow_new_session``
to all be explicitly ``True``; missing evidence or a category alone does not
authorize continuation. Existing planners may omit ``stop_info`` and retain
their current recovery behavior.

``get_planner_capabilities(name)`` exposes static runner requirements without
importing an optional SDK. ``PlannerCapabilities`` defaults all flags to false:
``requires_memory_copy`` for exploration, ``refresh_manual_observations`` before
resuming after manual actions, and ``withdraw_queued`` for queued message
withdrawal. When adding a backend, register its requirements and wire the relevant
entry points to enforce these flags; a declaration alone does not enable behavior.

Because the RPent tool schemas and prompt-rendering path stay the same,
adding a planner does not require changes to tools or environment
servers. See :doc:`architecture` for the interface, and
:doc:`add_primitive` if you want to expose new tools to
your custom planner.

Wire and Validate the Planner
-----------------------------

Add a construction branch in ``rpent/planner/base.py:build_planner`` and a CLI choice in ``rpent/cli/main.py``. If the planner supports Dashboard selection or connection diagnostics, wire those entry points and ``rpent/planner/check.py`` as well.

Verify tool dispatch, text and image results, ``finish``, turn limits, errors, and the interruption and interaction paths you support. Reuse test patterns in ``tests/unit_tests/rpent/planner/``. Validate real model connectivity separately.
