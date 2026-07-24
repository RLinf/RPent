.. _home:

欢迎使用 RPent
==============

.. raw:: html

   <div class="rpent-hero">
     <h1 class="rpent-hero-title">欢迎使用 RPent</h1>
     <img class="rpent-hero-architecture"
          src="https://github.com/RLinf/misc/raw/main/pic/rpent_logo.png"
          alt="RPent logo" />
     <p class="rpent-hero-subtitle">
       RPent（Recursive Physical Agent）是一个开源框架，用于构建能够在与物理世界的递归交互中持续演进的具身智能体。RPent 不限定基础模型的选择，而是提供一套递归的智能体框架，将感知（perception）、推理（reasoning）、记忆（memory）、执行（execution）和自我演进（self-evolution）等不同类型的智能能力整合到统一的物理智能体中。物理智能体在持续交互中不断反思和调整，从而获得新能力，逐步突破初始设计的能力边界。
     </p>
   </div>

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 概览
      :link: rst_source/overview
      :link-type: doc
      :text-align: center

      介绍 RPent 的基本概念、五芒星与 ∞ 标志的含义，
      以及整体架构。

   .. grid-item-card:: 安装
      :link: rst_source/installation
      :link-type: doc
      :text-align: center

      克隆 RPent，并通过一条 ``pip install``
      命令安装完整依赖。

   .. grid-item-card:: 快速开始
      :link: rst_source/quickstart
      :link-type: doc
      :text-align: center

      配置 LLM API key 和 checkpoint，端到端运行一个 LIBERO 任务。

   .. grid-item-card:: 使用教程
      :link: rst_source/usage/configure_planner
      :link-type: doc
      :text-align: center

      使用 LIBERO / RoboCasa 仿真环境或 Franka / SO-101 机械臂，
      切换 planner 并选择动作原语。

   .. grid-item-card:: 开发教程
      :link: rst_source/development/architecture
      :link-type: doc
      :text-align: center

      了解 RPent 的实现架构，以及如何添加机器人、
      动作原语或扩展 memory。

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:

   概览 <rst_source/overview>
   安装 <rst_source/installation>
   快速开始 <rst_source/quickstart>

.. toctree::
   :maxdepth: 1
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: 使用教程

   Agentic Planner <rst_source/usage/configure_planner>
   动作原语 <rst_source/usage/configure_primitives>
   LIBERO <rst_source/usage/libero>
   RoboCasa <rst_source/usage/robocasa>
   Franka <rst_source/usage/franka>
   SO-101 <rst_source/usage/so101>
   高级部署 <rst_source/usage/advanced_deployment>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: 开发教程

   系统设计 <rst_source/development/architecture>
   添加新机器人 <rst_source/development/add_robot>
   添加动作原语 <rst_source/development/add_primitive>
   Memory 管理 <rst_source/development/memory>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: 论文

   Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents <rst_source/awesome_works/harnessvla>
