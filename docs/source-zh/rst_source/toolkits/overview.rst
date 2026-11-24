.. _toolkits-overview:

动作原语与工具
==============

规划器通过机器人 ``toolkit`` 中注册的工具控制机器人。所有在线规划器（``api``、``claude_code``、``codex`` 和 ``gemini_cli``）都从同一个 ``toolkit`` 读取工具定义，并通过同一层调度执行。

工具分类
--------

RPent 将工具分为四类：

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 类别
     - 用途
     - 示例
   * - **动作原语**
     - 通过 VLA/WAM 模型、程序化运动或直接控制数值执行机器人动作。
     - ``pi0_pick``、``move_to``、``rotate_wrist``、``release``、``execute_action``
   * - **具身感知与状态工具**
     - 查看相机观测、查询三维表面坐标、分割目标物体以及测量物体部位尺寸。
     - ``view_env_state``、``back_project``、``segment``、``geometry``\ （见 :doc:`rgbd`\ ）
   * - **通用与记忆工具**
     - 读取运行工件和任务经验，并在规划结束时退出循环。
     - ``read_text_file``、``write_text_file``、``list_dir``、``finish``
   * - **外部 MCP 工具**
     - 通过 ``--mcp-config`` 将第三方 Streamable HTTP MCP 服务接入任务运行。
     - 功率计、充电底座、外部传感器等（见 :doc:`mcp`\ ）

动作原语会推进环境状态，并在执行后采集新的观测。感知工具、记忆工具以及默认模式下的外部 MCP 工具直接返回自身结果，不触发环境状态采集。``finish`` 用于结束规划器循环，任务是否成功仍以环境原生的成功判定或真机操作员判定为准。

.. _planner-direct-actions:

直接执行动作
------------

使用 LIBERO、RoboCasa365 或 RoboTwin 时，在启动命令中加入 ``--enable-direct-action``，即可开放 ``execute_action(values=[...])`` 工具。规划器传入一组控制数值，环境将其作为单步动作执行，RPent 随后记录执行后的状态。

所需数值的维度与范围取决于机器人和控制器，规划器收到的工具描述会列出具体要求。启用后，VLA 与程序化动作原语仍可继续使用。

各平台使用的动作模型
--------------------

.. list-table::
   :header-rows: 1

   * - 环境 / 机器人
     - 动作模型与配置
   * - :doc:`../simulators/libero`
     - Pi0.5
   * - :doc:`../simulators/robocasa`
     - RLDX-1
   * - :doc:`../simulators/robotwin`
     - LingBot-VLA
   * - :doc:`../simulators/robodojo`
     - VPP2、Awomo05
   * - :doc:`../real_world_robots/franka`
     - Pi0.5 / openpi
   * - :doc:`../real_world_robots/dual_franka`
     - Pi0.5 / openpi

各平台页面提供模型权重路径、可用工具和任务要求。SO-101 当前状态见 :doc:`../real_world_robots/yam` 和 :doc:`../real_world_robots/so101`。

服务与扩展
----------

- **独立服务部署：** VLA 服务通过 ``predict`` 提供动作预测，通过 ``healthz`` 提供健康检查，支持 HTTP 和 socket 传输，详见 :doc:`../guides/advanced_deployment`。
- **RGB-D 感知工具：** 启用通用 ``back_project``、``segment`` 和 ``geometry`` 工具或为真机编写观测适配器，见 :doc:`rgbd`。
- **外部 MCP 工具：** 通过 ``--mcp-config`` 接入第三方 MCP 服务，见 :doc:`mcp`。
- **自定义 Python 原语：** 定义工具参数、执行逻辑和状态采集，见 :doc:`../development/add_primitive`。
