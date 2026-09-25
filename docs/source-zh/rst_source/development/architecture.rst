.. _system-internals:

系统架构与执行流程
===========================

RPent 将任务规划、工具执行和环境运行分开：规划器选择工具，toolkit 将调用交给感知或动作组件，环境返回执行后的状态。各环境通过 ``RobotSpec`` 接入共同的运行器。

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rpent_framework.png
   :alt: RPent 系统架构
   :width: 100%

一次任务的执行过程
---------------------------

运行器先准备任务与服务器，再由规划器接管执行。每次工具调用都会返回观测，供规划器决定下一步操作。下面按顺序说明一次运行从 CLI 初始化到结束清理的过程。

1. CLI 读取共享参数，按 ``--robot`` 加载环境，再注册该环境的参数并完成校验。
2. ``parse_config`` 生成 ``RunConfig``，确定任务、输出目录和提示词变量。
3. 运行器按记忆配置同步或使用本地数据，构造规划器并渲染提示词。
4. ``init_runtime`` 启动或连接服务，返回运行参数和本次运行拥有的进程；``get_toolkit`` 创建工具集与记忆管理器。
5. 规划器通过 ``get_tools_spec`` 获取工具定义，通过 ``execute_tool`` 执行调用，再读取工具返回的文本和图像，继续规划。
6. 动作工具调用 VLA 或程序化动作，环境执行后记录状态和观测，供下一轮规划使用。
7. 规划器调用 ``finish``、达到运行限制或发生错误时结束。运行器保存对话，完成录像及清理；需要评测产物的环境通过结束钩子输出结果。

``finish`` 表示规划器结束，成功标准由具体环境或真机操作员提供。对应关系见各环境页。主入口为 ``rpent/cli/main.py``。

组件职责
------------

工具集负责把规划器的决策转成具体环境中的动作。模型服务器和环境各自维护状态，运行器负责管理它们的生命周期。规划器因此可以通过相同的工具接口使用不同机器人。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 组件
     - 职责
   * - Planner
     - 选择工具、维护模型交互并返回 ``PlannerResult``。在线后端包括 ``api``、``claude_code``、``codex``；LIBERO Flash 执行保存的计划。
   * - Toolkit
     - 定义工具参数、分发调用、保存状态，并将工具返回值转换为供规划器读取的 ``ToolResult``。
   * - Environment service
     - 拥有仿真器或真机连接，执行动作并提供原生状态与成功判定。
   * - VLA / perception services
     - 加载模型并提供动作预测、分割等能力，具体服务由环境配置。
   * - MemoryManager
     - 控制记忆读取与探索草稿写入，管理经验合并和索引。

以 LIBERO 为例，环境、Pi0.5 和 SAM3 各自运行在服务进程中。动作工具会推进环境；``view_env_state`` 等读取工具查看已记录状态。RPC 支持 HTTP 和 socket，接口详见 :doc:`interfaces`，独立部署方法见 :doc:`../usage/advanced_deployment`。

代码目录
------------

共享执行逻辑和接口位于 ``rpent/`` 中，各环境的具体实现位于 ``robots/`` 下的独立包中。

.. code-block:: text

   rpent/
     cli/          # CLI, Dashboard launcher, memory commands
     planner/      # Model backends and Planner protocol
     prompt/       # Shared prompt construction
     session/      # Session and task lifecycle
     dashboard/    # Web UI and event delivery
     robots/       # Discovery, descriptors, runtime components
     tools/        # Toolkit and shared tool behavior
     memory/       # Memory permissions, merge, and indexing
     evaluation/   # Evaluation result support
   robots/
     libero/
     robocasa/
     robotwin/
     franka/
     dual_franka/

环境发现与接入
---------------------

``rpent/robots/base.py`` 通过 ``enumerate_robots`` 发现 ``robots/`` 中的包；CLI 使用发现结果生成 ``--robot`` 的可选值。选择 ``myrobot`` 后，加载器按需导入 ``robots.myrobot``。包入口提供两个工厂：

.. code-block:: python

   def get_robot_spec() -> RobotSpec: ...

   def get_toolkit(*, runtime_kwargs, dashboard_events, config): ...

``RobotSpec`` 声明提示词、CLI 参数、运行配置、服务启动逻辑及可选的 Dashboard 与探索能力。``get_toolkit`` 接收运行配置，创建对应工具集；部分环境还接收模式或状态目录参数。完整契约见 :doc:`interfaces`，接入步骤见 :doc:`add_robot`。

Dashboard 会话
----------------

Dashboard 使用长期运行的 Session 管理服务，每次任务创建新的 TaskRun。``DashboardSpec`` 定义任务命令、可见相机、可用控件，以及标记为 ``shared`` 或 ``unique`` 的运行组件。

LIBERO 会话复用 Pi0.5 和 SAM3，每个任务拥有独立环境并按顺序执行。任务创建新的工具集和规划器对话；组件由创建它的会话或任务清理。

浏览器通过 ``/api/session/stream`` 接收服务器发送的事件（SSE），并在新的对话事件产生后读取内容，更新对话、相机画面和动作时间线。直接操作动作原语的控件使用机器人声明的参数定义，后端会在调用工具前校验参数。使用方法见 :doc:`../usage/dashboard`，Dashboard 接入方式见 :doc:`add_robot`。

扩展入口
------------

根据要扩展的组件选择对应指南。

.. list-table::
   :header-rows: 1

   * - 目标
     - 指南
   * - 接入环境、机器人或 Dashboard 控件。
     - :doc:`add_robot`
   * - 将动作或感知能力提供为工具。
     - :doc:`add_primitive`
   * - 添加规划器后端。
     - :doc:`add_planner`
   * - 理解记忆的访问和持久化机制。
     - :doc:`memory`
