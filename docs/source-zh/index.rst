.. _home:

欢迎使用 RPent
==================

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rpent_logo.png
   :alt: RPent
   :class: rpent-home-logo

RPent（Recursive Physical Agent）是一个开源框架，用于构建在与物理世界的递归交互中持续演进的具身智能体。它不限定基础模型，将感知（perception）、推理（reasoning）、记忆（memory）、执行（execution）和自我演进（self-evolution）整合到统一的智能体框架中。智能体通过持续交互反思和调整行为，积累经验、获得新能力，并逐步拓展初始设计的能力边界。

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 快速开始
      :link: rst_source/quickstart
      :link-type: doc

      安装 RPent，运行一个 LIBERO-PRO 任务，查看执行结果。

   .. grid-item-card:: RPent 简介
      :link: rst_source/overview
      :link-type: doc

      了解规划、感知、动作和记忆如何协同完成任务。

   .. grid-item-card:: 使用记忆与探索模式
      :link: rst_source/usage/memory
      :link-type: doc

      使用已有任务经验，或通过探索积累本地记忆。

   .. grid-item-card:: 排行榜
      :link: rst_source/leaderboard
      :link-type: doc

      查看评测成绩、运行耗时和 Token 开销。

   .. grid-item-card:: 真实世界演示
      :link: rst_source/usage/real_world_demos_franka
      :link-type: doc

      观看双臂 Franka 任务演示，并进入对应的部署文档。

   .. grid-item-card:: 扩展 RPent
      :link: rst_source/development/architecture
      :link-type: doc

      了解执行流程，添加机器人、动作原语或规划器。

选择环境：:doc:`LIBERO <rst_source/usage/libero>` · :doc:`RoboCasa365 <rst_source/usage/robocasa>` · :doc:`RoboTwin <rst_source/usage/robotwin>`。真机部署见 :doc:`单臂 Franka <rst_source/usage/franka>` 和 :doc:`双臂 Franka <rst_source/usage/dual_franka>`；:doc:`YAM <rst_source/usage/real_world_demos_yam>` 提供任务演示。

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: 开始使用

   RPent 简介 <rst_source/overview>
   快速开始 <rst_source/quickstart>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:

   排行榜 <rst_source/leaderboard>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: 真实世界演示

   双臂 Franka <rst_source/usage/real_world_demos_franka>
   YAM <rst_source/usage/real_world_demos_yam>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: 使用指南

   使用记忆与探索模式 <rst_source/usage/memory>
   动作原语与工具 <rst_source/usage/configure_primitives>
   配置规划器与模型服务 <rst_source/usage/configure_planner>
   命令行与配置参考 <rst_source/usage/cli>
   使用 Dashboard <rst_source/usage/dashboard>
   使用 Flash Mode <rst_source/usage/flash>
   采集轨迹与数据飞轮 <rst_source/usage/flywheel>
   远程服务与并行运行 <rst_source/usage/advanced_deployment>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: 仿真环境

   LIBERO <rst_source/usage/libero>
   RoboCasa365 <rst_source/usage/robocasa>
   RoboTwin <rst_source/usage/robotwin>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: 真实机器人

   单臂 Franka <rst_source/usage/franka>
   双臂 Franka <rst_source/usage/dual_franka>
   YAM <rst_source/usage/yam>
   SO-101 <rst_source/usage/so101>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: 原理与开发

   系统架构与执行流程 <rst_source/development/architecture>
   核心接口 <rst_source/development/interfaces>
   记忆机制 <rst_source/development/memory>
   添加机器人或仿真环境 <rst_source/development/add_robot>
   添加动作原语 <rst_source/development/add_primitive>
   添加规划器 <rst_source/development/add_planner>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: 项目资源

   Harness VLA <rst_source/awesome_works/harnessvla>
   贡献指南 <rst_source/resources/contributing>
   版本说明 <rst_source/resources/release_notes>
