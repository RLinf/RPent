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

.. _deepseek-runtime-bridge:

DeepSeek Harness Runtime Bridge
-------------------------------

The pinned Harness SDK can consume RPent tools through
``HttpMcpServer(..., include_structured=True)``. The opt-in structured result
preserves fields beyond the text limit, normalizes finite negative zero for
Harness's JSON contract, and rejects non-finite numbers. Existing MCP consumers
keep their default text/image representation. ``ToolResult.execution_uncertain``
is internal provenance: a handler or observation failure may follow a physical
effect even when the returned payload is an error.

``deepseek_tools._ToolBoundary`` serializes native calls, retains accepted
``finish``, records delivery provenance, and seals admission after uncertain
execution. ``deepseek_bridge.mjs`` applies the same constraints inside the
runtime: only allowlisted RPent tools, serial execution, the literal complete
system prompt, an exact root-step budget, and the configured model/provider/effort
for both root and summary requests. It validates final tool delivery and image
count/pixel dimensions before admitting another action. These checks do not
establish task success or safe physical motion by themselves.

A caller must disable the default non-RPent tool providers and extra agents,
set ``agent-loop.maxParallelToolCalls`` to ``1``, and require
``rpentBridgeReady`` in ``sdk-jsonrpc-server`` startup dependencies. The caller
also owns cancellation, Toolkit drain, SDK/process cleanup, and interpretation
of the final control record. The bridge has no host control channel or persisted
session recovery. ``tests/e2e_tests/planner/test_deepseek_foundation.py`` shows
this direct SDK/MCP composition with a deterministic local model, including
startup, delivery, images, compaction, and cleanup.
