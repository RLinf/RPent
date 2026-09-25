RPent 简介
============

RPent（Recursive Physical Agent）是一个具身智能体框架。它让语言模型根据任务和视觉观测选择工具，再通过动作模型或程序化动作控制机器人。每次执行的结果会返回给规划器，用于判断下一步操作。

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rpent_framework.png
   :alt: RPent 规划、感知、记忆和执行架构
   :width: 100%

如何完成任务
------------------

1. 规划器读取任务描述、当前观测和可用的任务经验。
2. 通过视觉工具定位目标，选择 VLA 或程序化动作工具。
3. 环境执行动作，返回状态和相机画面。
4. 规划器根据结果继续、调整策略或结束任务。

探索模式允许多次尝试并整理本地记忆。评测模式读取已有记忆，按环境规定判断任务结果。实现细节见 :doc:`development/architecture`。

排行榜
------

对比 LIBERO、LIBERO-PRO、RoboCasa365 Target50 和 RoboTwin C2R 上的成功率。
排名仅限图中方法及评测范围；完整结果、模型配置和来源见 :doc:`leaderboard`。

.. image:: https://cdn.jsdelivr.net/gh/RLinf/misc@a6657fc43a6b3874a20ee1695a480090a4737c35/rpent/benchmarks/leaderboard-zh-light.png
   :alt: RPent 排行榜
   :class: only-light
   :width: 100%
   :target: leaderboard.html

.. image:: https://cdn.jsdelivr.net/gh/RLinf/misc@a6657fc43a6b3874a20ee1695a480090a4737c35/rpent/benchmarks/leaderboard-zh-dark.png
   :alt: RPent 排行榜
   :class: only-dark
   :width: 100%
   :target: leaderboard.html

选择平台
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 平台
     - 文档内容
   * - :doc:`LIBERO <usage/libero>`
     - Pi0.5、SAM3、LIBERO / LIBERO-PRO 运行与复现。
   * - :doc:`RoboCasa365 <usage/robocasa>`
     - RLDX-1、厨房任务和 Target50 复现。
   * - :doc:`RoboTwin <usage/robotwin>`
     - LingBot-VLA、双臂仿真任务和 C2R 复现。
   * - :doc:`单臂 Franka <usage/franka>`
     - 硬件准备、标定、自检和运行。
   * - :doc:`双臂 Franka <usage/dual_franka>`
     - 双节点部署、操作与人工参与的探索。
   * - :doc:`YAM <usage/yam>`
     - 已有任务演示；安装使用文档即将推出。
   * - :doc:`SO-101 <usage/so101>`
     - 内容即将推出。

规划器可选择 ``api``、``claude_code`` 或 ``codex``；LIBERO 还提供用于执行已有计划的 :doc:`Flash Mode <usage/flash>`。模型服务配置见 :doc:`usage/configure_planner`。

从 :doc:`quickstart` 开始运行第一个任务。评测成绩和资源开销见 :doc:`leaderboard`，研究背景见 :doc:`awesome_works/harnessvla`。

DreamZero、Cosmos Policy 和 RoboDojo 也列在项目的规划中，相关使用文档待补充。
