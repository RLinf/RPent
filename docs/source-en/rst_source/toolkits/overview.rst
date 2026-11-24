.. _toolkits-overview:

Action Primitives and Tools
===========================

The planner controls a robot through tool calls registered on the robot's toolkit. Every online planner (``api``, ``claude_code``, ``codex``, and ``gemini_cli``) reads schemas from the same toolkit and dispatches through the same execution layer.

Tool Families
-------------

RPent organizes tools into four families:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Family
     - Purpose
     - Examples
   * - **Action primitives**
     - Execute robot motions using a VLA/WAM policy, a scripted routine, or direct control values.
     - ``pi0_pick``, ``move_to``, ``rotate_wrist``, ``release``, ``execute_action``
   * - **Embodied perception and state tools**
     - Read camera observations, locate 3D surface points, segment objects, and measure object parts.
     - ``view_env_state``, ``back_project``, ``segment``, ``geometry`` (see :doc:`rgbd`)
   * - **Common and memory tools**
     - Inspect run artifacts and task memory, and signal when planning ends.
     - ``read_text_file``, ``write_text_file``, ``list_dir``, ``finish``
   * - **External MCP tools**
     - Call third-party Streamable HTTP MCP services alongside robot tools with zero Python code.
     - Power meters, charging docks, facility sensors (see :doc:`mcp`)

Action primitives advance the environment and trigger observation capture after execution. Perception, memory, and default external MCP tools return their own payload directly without overwriting robot state. ``finish`` ends the planner loop; task success is determined by the environment's native ``terminated`` signal or a real-robot operator.

.. _planner-direct-actions:

Direct Actions
--------------

For LIBERO, RoboCasa365, and RoboTwin, add ``--enable-direct-action`` to let the planner call ``execute_action(values=[...])``. The planner supplies one list of control values, the environment executes it as one action, and RPent records the resulting state.

The required values depend on the robot and controller. The tool description sent to the planner lists the expected number of values and their allowed ranges. VLA and scripted primitives remain available when direct actions are enabled.

Action Models by Platform
-------------------------

.. list-table::
   :header-rows: 1

   * - Environment / robot
     - Action model and configuration
   * - :doc:`../simulators/libero`
     - Pi0.5
   * - :doc:`../simulators/robocasa`
     - RLDX-1
   * - :doc:`../simulators/robotwin`
     - LingBot-VLA
   * - :doc:`../simulators/robodojo`
     - VPP2, Awomo05
   * - :doc:`../real_world_robots/franka`
     - Pi0.5 / openpi
   * - :doc:`../real_world_robots/dual_franka`
     - Pi0.5 / openpi

Each platform guide lists checkpoint paths, available tools, and task requirements. For SO-101 status, see :doc:`../real_world_robots/yam` and :doc:`../real_world_robots/so101`.

Services and Extensions
-----------------------

- **Standalone services:** VLA services expose action prediction through ``predict`` and health checks through ``healthz`` over HTTP or socket transports. See :doc:`../guides/advanced_deployment`.
- **RGB-D perception tools:** Enable generic ``back_project``, ``segment``, and ``geometry`` tools or adapt them to a real robot in :doc:`rgbd`.
- **External MCP tools:** Attach third-party MCP servers with ``--mcp-config`` in :doc:`mcp`.
- **Custom Python primitives:** Define arguments, execution, and state handling in :doc:`../development/add_primitive`.
