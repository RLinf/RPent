.. _external-mcp-servers:

External MCP Tools
==================

Attach any Streamable HTTP MCP server to an RPent run with one ``--mcp-config`` YAML file—no Python code required. RPent registers the server's tools on the robot's toolkit, so ``api``, ``claude_code``, ``codex``, and ``gemini_cli`` all use the same tool filtering, timeouts, cancellation, and trajectory capture.

Prerequisites:

- Your MCP server exposes a Streamable HTTP endpoint (``http://`` or ``https://``). RPent does not spawn stdio servers through a ``command`` field.
- Your robot and planner environment is ready (see :doc:`../get_started/quickstart`).

Quick Start
-----------

1. Start a local test server. Save the script below as ``power_meter.py``:

   .. dropdown:: Sample server (``power_meter.py``)

      .. code-block:: python

         from mcp.server.fastmcp import FastMCP

         mcp = FastMCP("power-meter", host="127.0.0.1", port=8765)


         @mcp.tool()
         def read_power_meter() -> str:
             """Return the current power meter reading."""
             return "battery 82%, outlet idle"


         if __name__ == "__main__":
             mcp.run(transport="streamable-http")

   ``mcp`` is included with RPent. Start the server (it listens at ``http://127.0.0.1:8765/mcp``):

   .. code-block:: bash

      python power_meter.py

2. Create ``power.yaml`` pointing at the server. Every tool on the server is exposed by default:

   .. code-block:: yaml

      mcp_servers:
        power:
          url: http://127.0.0.1:8765/mcp

3. Pass ``--mcp-config`` to any robot run:

   .. code-block:: bash

      rpent --robot libero --mcp-config power.yaml \
        --suite libero_goal_task --task 1 --seed 0 --planner claude_code

When the run starts, RPent connects to ``power`` and registers ``power__read_power_meter`` on the toolkit. Calling it returns ``battery 82%, outlet idle``.

Configuration and Authentication
--------------------------------

Each entry under ``mcp_servers`` uses the following fields:

.. list-table::
   :header-rows: 1
   :widths: 22 10 68

   * - Field
     - Required
     - Meaning
   * - ``url``
     - Yes
     - Streamable HTTP endpoint (``http://`` or ``https://``). URLs containing a user name or password are rejected.
   * - ``tools``
     - No
     - Tools to expose: ``"*"`` (default, every tool on the server) or a list of tool names. Only listed tools are registered on the toolkit across all planners.
   * - ``headers``
     - No
     - Request headers sent with every MCP call. Supports ``${oc.env:VAR_NAME}`` interpolation when the file is loaded.
   * - ``sync_with_robot``
     - No
     - ``false`` (default) or ``true``. Controls whether calls serialize with robot actions and capture a fresh robot observation (see below).

Header values stay inside the RPent process—they are never passed on child-process command lines—and task records redact header values as well as URL query strings and fragments.

For an authenticated server exposing a subset of its tools:

.. code-block:: yaml

   mcp_servers:
     power:
       url: https://power.example.com/mcp
       tools: [read_power_meter]
       headers:
         Authorization: "Bearer ${oc.env:POWER_API_TOKEN}"

Export ``POWER_API_TOKEN`` before launching RPent. If the variable is unset, RPent exits with a configuration error before starting the robot runtime.

Synchronizing with Robot Actions
--------------------------------

By default (``sync_with_robot: false``), external MCP tools run independently of robot motion and return their own text, structured data, and PNG images directly to the planner.

Set ``sync_with_robot: true`` when an external device interacts with the physical cell—such as a charging dock, safety interlock, or lighting controller:

.. code-block:: yaml

   mcp_servers:
     dock:
       url: http://127.0.0.1:8766/mcp
       tools: [set_charging_dock]
       sync_with_robot: true

With ``sync_with_robot: true``:

- Calls run exclusively alongside robot actions so the planner cannot trigger the device while the arm is moving.
- After each call, RPent captures a fresh robot observation and includes the remote response under ``tool_result`` alongside the updated state and camera images.
- Any ``image/png`` blocks returned by the server are forwarded alongside the robot's camera images.

Robot integrations can also declare robot-owned servers in their own configuration or register selected tools programmatically; see :ref:`add-robot-toolkit`.

Tool Naming, Timeouts, and Limits
---------------------------------

- **Naming:** Each remote tool is registered on the toolkit as ``<server>__<tool>`` (for example, ``power__read_power_meter``). Planner adapters apply their normal toolkit namespace on top (``api`` uses ``power__read_power_meter``, ``claude_code`` and ``codex`` use ``mcp__rpent__power__read_power_meter``, and ``gemini_cli`` uses ``mcp_rpent_power__read_power_meter``). Server names must start with a letter or digit, contain only letters, digits, ``_``, or ``-``, omit ``__`` and trailing ``_``, and not be ``rpent``.
- **Startup validation:** During toolkit initialization, RPent contacts each server and checks that every tool named in ``tools`` exists. Unreachable servers or unknown tool names fail the run with the list of tools the server does expose.
- **Timeouts and cancellation:** Every MCP request times out after 60 seconds by default; set ``MCP_CALL_TIMEOUT_S`` (in seconds) to override it. Cancelling a run or Dashboard task interrupts in-flight MCP calls immediately without waiting for the remote server to finish.
- **Content types:** Text, structured JSON, and ``image/png`` blocks are forwarded to the model and bounded to RPent's tool-text limit. Other media or resource blocks include metadata only, and ``--no-images`` omits external PNG blocks as well.
