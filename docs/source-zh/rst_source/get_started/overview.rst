RPent 简介
============

RPent（Recursive Physical Agent）是一个开源框架，用于构建能够在与物理世界的递归交互中持续演进的具身智能体。RPent 不限定基础模型的选择，而是提供一套递归的智能体框架，将感知、推理、记忆、执行和自我演进等不同类型的智能能力整合到统一的物理智能体中。物理智能体在持续交互中不断反思和调整，从而获得新能力，逐步突破初始设计的能力边界。

Pent 这个名字源自五芒星（Pentagram），其五个顶点象征多模态智能融合为一个统一的具身智能体。中心的无穷符号（∞）代表感知、推理、执行和自我演进的递归循环，让智能持续向物理世界扩展。

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rpent_framework.png
   :alt: RPent 规划、感知、记忆和执行架构
   :width: 100%

RPent 的三条核心设计原则是服务化、标准化和可组合（service-oriented, standardized, and composable）。各种能力以可复用服务的形式部署，通过统一接口连接，并灵活组合成不同的物理智能体。这些原则使 RPent 能够超越传统机器人控制框架，成为面向物理世界的智能体基础设施，让智能在部署后继续被构建、扩展和演进。

排行榜
------

对比 LIBERO、LIBERO-PRO、RoboCasa365 Target50 和 RoboTwin C2R 上的成功率。
排名仅限图中方法及评测范围；完整结果、模型配置和来源见 :doc:`../leaderboard/index`。

.. image:: https://cdn.jsdelivr.net/gh/RLinf/misc@705bd44bfc8ad7586b76239de13167db35abcce7/rpent/benchmarks/leaderboard-zh-light.png
   :alt: RPent 排行榜
   :class: only-light
   :width: 100%
   :target: ../leaderboard/index.html

.. image:: https://cdn.jsdelivr.net/gh/RLinf/misc@705bd44bfc8ad7586b76239de13167db35abcce7/rpent/benchmarks/leaderboard-zh-dark.png
   :alt: RPent 排行榜
   :class: only-dark
   :width: 100%
   :target: ../leaderboard/index.html

选择平台
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 平台
     - 文档内容
   * - :doc:`LIBERO <../simulators/libero>`
     - Pi0.5、SAM3、LIBERO / LIBERO-PRO 运行与复现。
   * - :doc:`RoboCasa365 <../simulators/robocasa>`
     - RLDX-1、厨房任务和 Target50 复现。
   * - :doc:`RoboTwin <../simulators/robotwin>`
     - LingBot-VLA、双臂仿真任务和 C2R 复现。
   * - :doc:`单臂 Franka <../real_world_robots/franka>`
     - 硬件准备、标定、自检和运行。
   * - :doc:`双臂 Franka <../real_world_robots/dual_franka>`
     - 双节点部署、操作与人工参与的探索。
   * - :doc:`YAM <../real_world_robots/yam>`
     - 已有任务演示；安装使用文档即将推出。
   * - :doc:`SO-101 <../real_world_robots/so101>`
     - 内容即将推出。

规划器可选择 ``api``、``claude_code`` 或 ``codex``；LIBERO 还提供用于执行已有计划的 :doc:`Flash Mode <../guides/flash>`。模型服务配置见 :doc:`../guides/configure_planner`。

从 :doc:`quickstart` 开始运行第一个任务。评测成绩和资源开销见 :doc:`../leaderboard/index`，研究背景见 :doc:`../resources/harnessvla`。

DreamZero、Cosmos Policy 和 RoboDojo 也列在项目的规划中，相关使用文档待补充。
