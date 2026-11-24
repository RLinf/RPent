.. _external-mcp-servers:

External MCP Tools
==================

Connect a Streamable HTTP MCP server to an RPent run by passing a YAML file path to ``--mcp-config`` so the planner can call external sensors, lab instruments, or facility services alongside robot primitives. The same configuration works across ``api``, ``claude_code``, ``codex``, and ``gemini_cli``.

This guide first walks through a minimal local power-meter example, then covers YAML fields and authentication, robot-action synchronization, and naming, timeout, and session limits.

Prerequisites:

- Your MCP server exposes a Streamable HTTP endpoint (``http://`` or ``https://``). RPent does not spawn stdio servers through a ``command`` field.
- Your robot and planner environment is ready (see :doc:`../get_started/quickstart`).

Quick Start
-----------

This walkthrough starts a local mock power-meter MCP server exposing one ``read_power_meter`` tool, connects it to RPent through ``--mcp-config power.yaml``, and shows how to confirm the planner can call it during a run.

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

   Start the server (it listens at ``http://127.0.0.1:8765/mcp``):

   .. code-block:: bash

      python power_meter.py

2. Create ``power.yaml`` pointing at the server. Every tool on the server is exposed by default:

   .. code-block:: yaml

      mcp_servers:
        power:
          url: http://127.0.0.1:8765/mcp

3. Pass the YAML file path via ``--mcp-config`` to any robot run (replace ``--robot libero`` and ``--planner claude_code`` with your target robot and online planner):

   .. code-block:: bash

      rpent --robot libero --mcp-config power.yaml \
        --suite libero_goal_task --task 1 --seed 0 --planner claude_code

When the run starts, RPent connects to ``power`` and registers ``power__read_power_meter``. You can confirm the registered tool and its response (``battery 82%, outlet idle``) in Dashboard or in ``transcript_*.json`` under the run output directory (see :ref:`run-output-files`). If the server is unreachable or a configured tool is missing, startup stops before planning begins and prints the error—including the tools exposed by the server—to stderr and ``run.log``.

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

For an authenticated server exposing a subset of its tools with a custom timeout:

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

By default (``sync_with_robot: false``), external MCP tools run as read-only toolkit calls: they can run concurrently with other read-only tools, skip robot state capture, and return their own text, structured data, and PNG images directly to the planner (they still wait for active robot actions under the toolkit lock, and a slow external call delays queued robot actions).

Set ``sync_with_robot: true`` when an external device interacts with the physical cell (such as a charging dock, safety interlock, or lighting controller):

.. code-block:: yaml

   mcp_servers:
     dock:
       url: http://127.0.0.1:8766/mcp
       tools: [set_charging_dock]
       sync_with_robot: true

With ``sync_with_robot: true``:

- Calls run exclusively alongside robot actions so the planner cannot trigger the device while the arm is moving.
- After each call, RPent captures a fresh robot observation and includes the remote response alongside the updated state and camera images.
- Any ``image/png`` blocks returned by the server are forwarded alongside the robot's camera images.

If one server exposes both read-only tools and synchronized device actions, declare two differently named entries under ``mcp_servers`` pointing to the same ``url`` with separate ``tools`` lists and ``sync_with_robot`` settings.

Reference Details
-----------------

External tools are exposed as ``<server>__<tool>``, time out after ``timeout_s`` (or ``MCP_CALL_TIMEOUT_S``, defaulting to 60 seconds), and inherit RPent's cancellation and trajectory capture. Expand the sections below for naming rules, config merging, session behavior, allowlist guidance, and credential redaction.

.. dropdown:: Tool naming and planner prefixes

   - Each remote tool is registered on the toolkit as ``<server>__<tool>`` (for example, ``power__read_power_meter``).
   - Planner adapters apply their normal toolkit namespace on top: ``api`` uses ``power__read_power_meter``, ``claude_code`` and ``codex`` use ``mcp__rpent__power__read_power_meter``, and ``gemini_cli`` uses ``mcp_rpent_power__read_power_meter``.
   - Server names must start with a letter or digit, contain only letters, digits, ``_``, or ``-``, omit ``__`` and trailing ``_``, and not be ``rpent``.

.. dropdown:: Config merging, validation, sessions, timeouts, content limits, and redaction

   - **Merging shared and robot configs:** The CLI merges servers from the ``--mcp-config`` YAML file with any robot-owned servers declared on ``RunConfig.mcp_servers`` (parsed from a robot's own config via :func:`~rpent.utils.mcp_servers.parse_mcp_servers`). Server names must be unique across both sources; duplicate names fail startup.
   - **Startup validation and tool allowlisting:** During toolkit initialization, RPent contacts each server and checks that every tool named in ``tools`` exists. Unreachable servers or unknown tool names fail startup with the list of tools the server does expose. Omitting ``tools`` (or setting ``tools: "*"``) exposes and auto-approves every tool on the server across all planners; list explicit tool names in ``tools`` when the server includes side-effecting or administrative tools.
   - **Stateless sessions and static discovery:** Each call opens a fresh stateless MCP session, so session-scoped server state is not preserved across calls. Only MCP tools discovered at startup are exposed; MCP Resources, Prompts, and dynamic ``tools/list_changed`` updates are not forwarded.
   - **Timeouts and cancellation:** Per-server ``timeout_s`` in YAML takes precedence, followed by the ``MCP_CALL_TIMEOUT_S`` environment variable (in seconds), then the 60-second default. Cancelling a run or Dashboard task interrupts in-flight MCP calls immediately without waiting for the remote server to finish.
   - **Content types:** Text, structured JSON, and ``image/png`` blocks are forwarded to the model and bounded to RPent's tool-text limit. Other media or resource blocks include metadata only, and ``--no-images`` (``api`` only) omits external PNG blocks as well.
   - **Credential handling and redaction:** Header values stay inside the RPent process and are never passed on child-process command lines. Task records redact header values as well as URL query strings and fragments.

For programmatic MCP registration inside a robot integration, see :ref:`add-robot-toolkit`.
