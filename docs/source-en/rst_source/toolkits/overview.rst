.. _toolkits-overview:

Action Primitives and Tools
===========================

Use this page to see which tool families RPent exposes to a planner—action primitives, RGB-D perception tools, memory tools, and external MCP tools—and where to configure each one. All online planners (``api``, ``claude_code``, ``codex``, and ``gemini_cli``) read schemas from the same robot toolkit and dispatch through the same execution layer.

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
     - Read and write run files and task memory, and signal when planning ends.
     - ``read_text_file``, ``write_text_file``, ``list_dir``, ``finish``
   * - **External MCP tools**
     - Call third-party Streamable HTTP MCP services alongside robot tools via ``--mcp-config``.
     - Power meters, charging docks, facility sensors (see :doc:`mcp`)

Action primitives advance the environment and trigger observation capture after execution. Perception, memory, and default external MCP tools return their own payload directly without triggering environment state capture. ``finish`` ends the planner loop; task success is determined by the environment's native success criterion or a real-robot operator.

.. _planner-direct-actions:

Direct Actions
--------------

For LIBERO, RoboCasa365, and RoboTwin, add ``--enable-direct-action`` to let the planner call ``execute_action(values=[...])``. The planner supplies one list of control values, the environment executes it as one action, and RPent records the resulting state.

The required values depend on the robot and controller. The tool description sent to the planner lists the expected number of values and their allowed ranges. VLA and scripted primitives remain available when direct actions are enabled.

Services and Extensions
-----------------------

Each simulator and real-robot guide lists its own action models, checkpoint paths, available tools, and task requirements.

.. dropdown:: Action models by platform

   .. list-table::
      :header-rows: 1

      * - Environment / robot
        - Action models
      * - :doc:`../simulators/libero`
        - Pi0.5 (default), Cosmos Policy, Cosmos3, Fast-WAM
      * - :doc:`../simulators/robocasa`
        - RLDX-1
      * - :doc:`../simulators/robotwin`
        - LingBot-VLA (default), UniWAM, Fast-WAM
      * - :doc:`../simulators/robodojo`
        - VPP2, Awomo05
      * - :doc:`../simulators/robolab`
        - Pi0.5, or native actions without a model
      * - :doc:`../simulators/efe`
        - None (analytical IK ``Skills`` or native ``Raw`` actions)
      * - :doc:`../real_world_robots/franka`
        - Pi0.5 / openpi
      * - :doc:`../real_world_robots/dual_franka`
        - Pi0.5 / openpi
      * - :doc:`../real_world_robots/yam`
        - Pi0.5
      * - :doc:`../real_world_robots/so101`
        - Coming soon

- **Standalone services:** VLA and WAM services expose action prediction through ``predict`` and health checks through ``healthz`` over HTTP or socket transports. See :doc:`../guides/advanced_deployment`.
- **Custom Python primitives:** Define arguments, execution, and state handling in :doc:`../development/add_primitive`.
