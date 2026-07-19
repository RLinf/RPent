.. _home:

欢迎使用 RPent
==============

.. raw:: html

   <div class="rpent-hero">
     <h1 class="rpent-hero-title">欢迎使用 RPent</h1>
     <p class="rpent-hero-subtitle">
       面向物理世界的智能体基础设施 ——
       通过与环境的递归交互, 让具身智能体持续演进的开放框架。
     </p>
     <img class="rpent-hero-architecture" src="architecture.svg" alt="RPent 架构" />
   </div>

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 概览
      :link: rst_source/overview
      :link-type: doc
      :text-align: center

      RPent 是什么, 五芒星 + ∞ logo 的含义,
      以及一览的高层架构。

   .. grid-item-card:: 安装
      :link: rst_source/installation
      :link-type: doc
      :text-align: center

      并排克隆 RLinf + RPent, 创建虚拟环境, 安装
      LIBERO PRO/PLUS 相关依赖。

   .. grid-item-card:: 快速开始
      :link: rst_source/quickstart
      :link-type: doc
      :text-align: center

      配置 API key, 指向 checkpoint, 端到端跑通一个 LIBERO 任务。

   .. grid-item-card:: 使用教程
      :link: rst_source/usage/configure_planner
      :link-type: doc
      :text-align: center

      驱动 LIBERO / RoboCasa 仿真器或 Franka / SO-101 机械臂,
      切换 planner, 选择 action primitive。

   .. grid-item-card:: 开发教程
      :link: rst_source/development/architecture
      :link-type: doc
      :text-align: center

      RPent 的实现级架构, 以及如何添加新机器人、
      新 action primitive, 或扩展 memory。

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

   配置 planner <rst_source/usage/configure_planner>
   配置 action primitive <rst_source/usage/configure_primitives>
   LIBERO <rst_source/usage/libero>
   RoboCasa <rst_source/usage/robocasa>
   Franka <rst_source/usage/franka>
   SO-101 <rst_source/usage/so101>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: 开发教程

   RPent 架构 <rst_source/development/architecture>
   添加新机器人 <rst_source/development/add_robot>
   添加 action primitive <rst_source/development/add_primitive>
   Memory 管理 <rst_source/development/memory>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: 优秀工作

   HarnessVLA <rst_source/awesome_works/harnessvla>
