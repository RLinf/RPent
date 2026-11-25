.. _external-mcp-servers:

External MCP Tools
==================

Connect a Streamable HTTP MCP server to an RPent run by passing a YAML file path to ``--mcp-config`` so the planner can call external sensors, lab instruments, or other devices alongside robot primitives. The same configuration works across ``api``, ``claude_code``, ``codex``, and ``gemini_cli``.

Prerequisites:

- Your MCP server exposes a Streamable HTTP endpoint (``http://`` or ``https://``). RPent does not spawn stdio servers through a ``command`` field; if a server only supports stdio, bridge it through a Streamable HTTP proxy (such as ``mcp-proxy`` or ``supergateway``) and point ``url`` at that endpoint.
- Your robot and planner environment is ready (see :doc:`../get_started/quickstart`).

Quick Start
-----------

This walkthrough starts a local mock power-meter MCP server exposing ``read_power_meter``, attaches it via ``--mcp-config power.yaml``, and verifies in interactive terminal mode (``--tui``) that the planner can call it during a run.

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

   Start the server (it listens at ``http://127.0.0.1:8765/mcp``). Keep this terminal running and open a new terminal for step 3:

   .. code-block:: bash

      python power_meter.py

2. Create ``power.yaml`` pointing at the server. Every tool on the server is exposed by default:

   .. code-block:: yaml

      mcp_servers:
        power:
          url: http://127.0.0.1:8765/mcp

3. Pass the YAML file path via ``--mcp-config`` to an interactive run (replace ``--robot libero`` and ``--planner claude_code`` with your target robot and online planner):

   .. code-block:: bash

      rpent --robot libero --mcp-config power.yaml \
        --suite libero_goal_task --task 1 --seed 0 --planner claude_code --tui

When the run starts, RPent connects to ``power`` and registers ``power__read_power_meter``. Under ``claude_code`` and other CLI planners, the ``--tui`` input box is pre-filled with the LIBERO task prompt—clear it first, then enter ``Read the power meter`` (under ``--planner api``, ``--tui`` starts running the LIBERO task immediately and accepts input after the first planner turn). The planner calls ``power__read_power_meter`` and receives ``battery 82%, outlet idle``, and the call is recorded in ``transcript_*.json`` under the run output directory (see :ref:`run-output-files`). If you use Dashboard (see :ref:`dashboard-usage`), you can also inspect the tool call and its result after submitting a task in the UI.

If the server is unreachable or a configured tool is missing, RPent exits before planning starts and prints the error to the terminal and ``run.log``; when a tool name does not match, the error also lists the tools exposed by the server.

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
     - Tools to expose: ``"*"`` (default, every tool on the server) or a list of tool names. Only listed tools are registered across all planners.
   * - ``headers``
     - No
     - Request headers sent with every MCP call. Supports ``${oc.env:VAR_NAME}`` interpolation when the file is loaded.
   * - ``sync_with_robot``
     - No
     - ``false`` (default) or ``true``. Controls whether calls serialize with robot actions and capture a fresh robot observation (see below).
   * - ``timeout_s``
     - No
     - Positive per-request timeout in seconds for discovery and tool calls. Overrides ``MCP_CALL_TIMEOUT_S`` and the 60-second default.

.. warning::

   Omitting ``tools`` or setting ``tools: "*"`` exposes every tool on the server so planners can invoke them without per-tool confirmation. If the server includes write, hardware-control, or administrative tools, list only the allowed tool names in ``tools``.

The example below configures authentication headers, restricts exposed tools, and sets a custom timeout:

.. code-block:: yaml

   mcp_servers:
     power:
       url: https://power.example.com/mcp
       tools: [read_power_meter]
       timeout_s: 30
       headers:
         Authorization: "Bearer ${oc.env:POWER_API_TOKEN}"

Export ``POWER_API_TOKEN`` before launching RPent. If the variable is unset, RPent exits with a configuration error before starting the robot runtime.

Synchronizing with Robot Actions
--------------------------------

By default (``sync_with_robot: false``), external MCP tools run in read-only mode: they can run concurrently with other read-only tools, skip robot state capture, and return their text, structured data, and PNG images directly to the planner. They still do not run simultaneously with robot actions; if a remote call is slow, queued robot actions wait for it to finish.

Set ``sync_with_robot: true`` when an external device interacts with the physical cell (such as a charging dock, safety interlock, or lighting controller):

.. code-block:: yaml

   mcp_servers:
     dock:
       url: http://127.0.0.1:8766/mcp
       tools: [set_charging_dock]
       sync_with_robot: true

With ``sync_with_robot: true``:

- Calls run mutually exclusively with robot actions so the planner cannot trigger the device while the arm is moving.
- After each call, RPent captures a fresh robot observation and includes the remote response alongside the updated state and camera images.
- Any ``image/png`` blocks returned by the server are forwarded alongside the robot's camera images.

If one server exposes both read-only tools and synchronized control tools, configure two differently named server entries under ``mcp_servers`` with the same ``url`` and separate ``tools`` and ``sync_with_robot`` settings.

Reference Details
-----------------

External tools are exposed as ``<server>__<tool>``, time out according to YAML ``timeout_s`` → ``MCP_CALL_TIMEOUT_S`` → 60 seconds, and inherit RPent's cancellation and trajectory capture.

.. dropdown:: Planner prefixes and server naming rules

   - Planner adapters apply their normal toolkit prefix on top of ``<server>__<tool>``: ``api`` uses ``power__read_power_meter``, ``claude_code`` and ``codex`` use ``mcp__rpent__power__read_power_meter``, and ``gemini_cli`` uses ``mcp_rpent_power__read_power_meter``.
   - Server names must start with a letter or digit, contain only letters, digits, ``_``, or ``-``, omit ``__`` and trailing ``_``, and not be ``rpent``.

.. dropdown:: Config merging and sessions

   - **Config merging:** If the selected robot defines default MCP servers, they are merged with the servers in ``--mcp-config``; server names must be unique across both.
   - **Stateless sessions and static discovery:** Each call opens a fresh stateless MCP session, so session-scoped server state is not preserved across calls. Only MCP tools discovered at startup are exposed; MCP Resources, Prompts, and dynamic ``tools/list_changed`` updates are not forwarded.

.. dropdown:: Cancellation, payload limits, and redaction

   - **Cancellation:** Cancelling a run or Dashboard task interrupts in-flight MCP calls immediately without waiting for the remote server to finish.
   - **Content types and size limit:** Text, structured JSON, and ``image/png`` blocks are forwarded to the model, with text and structured JSON automatically truncated beyond 60,000 bytes. Other media or resource blocks include metadata only, and ``--no-images`` (``api`` only) omits external PNG blocks as well.
   - **Credential handling and redaction:** Header values stay inside the RPent process and are never passed on child-process command lines. When written to ``task.json`` and Dashboard task records, header values as well as URL query strings and fragments are redacted.

For programmatic MCP registration inside a robot integration, see :ref:`add-robot-toolkit`.
