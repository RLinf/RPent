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
       RPent (Recursive Physical Agent) 是一个用于构建具身智能体的开放
       框架, 让智能体通过与物理世界的递归交互持续演进。RPent 不预设某个
       具体的基础模型, 而是提供一个递归智能体框架, 将异构智能能力 ——
       感知 (perception)、推理 (reasoning)、记忆 (memory)、执行
       (execution)、自我演进 (self-evolution) —— 统一到一个物理智能体中。
       通过持续的交互、反思与适应, RPent 让物理智能体获得超出其初始
       设计的新能力。
     </p>
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

   Agentic planner <rst_source/usage/configure_planner>
   Action primitives <rst_source/usage/configure_primitives>
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

   系统内部 <rst_source/development/architecture>
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
