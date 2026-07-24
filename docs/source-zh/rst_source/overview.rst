概览
====

**RPent（Recursive Physical Agent）** 是一个开源框架，用于构建能够在与物理世界的递归交互中持续演进的具身智能体。RPent 不限定基础模型的选择，而是提供一套递归的智能体框架，将感知（perception）、推理（reasoning）、记忆（memory）、执行（execution）和自我演进（self-evolution）等不同类型的智能能力整合到统一的物理智能体中。物理智能体在持续交互中不断反思和调整，从而获得新能力，逐步突破初始设计的能力边界。

Pent 这个名字源自五芒星 (Pentagram), 其五个顶点象征多模态智能融合为一个
统一的具身智能体。五芒星的中心是无穷符号 (∞), 代表感知、推理、执行、
自我演进永无止境的递归循环, 让智能持续向物理世界扩展。

.. image:: https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png
   :alt: RPent 框架图
   :align: center
   :width: 90%

RPent 建立在三条核心设计原则之上: **服务化、标准化、可组合
(service-oriented, standardized, and composable)**。RPent 把各种能力以
可复用服务的形式部署, 通过统一接口连接, 并灵活组合成多样的物理智能体。
这三条原则让 RPent 超越了传统的机器人控制框架, 成为面向物理世界的
智能体基础设施 (agentic infrastructure for the physical world) —— 在这里,
智能不只是被部署, 而是被持续构建、扩展与演进。

功能矩阵
--------

.. list-table::
   :header-rows: 1
   :widths: 26 28 26 20

   * - Agentic Planner
     - Action Primitive
     - 仿真环境
     - 真实机器人
   * - - Claude Code ✅
       - Codex ✅
       - Custom planner ✅
     - - **VLA**

         - Pi0.5 ✅
         - RLDX-1

       - **WAM**

         - DreamZero
     - - LIBERO-PRO ✅
       - RoboCasa
     - - Franka
       - SO-101

接下来
------

- 初次使用 RPent？先完成 :doc:`installation`，再按照 :doc:`quickstart`
  端到端运行一个 LIBERO 任务。
- 想使用某个具体的机器人环境？查看对应的使用教程，例如 :doc:`usage/libero`。
- 想切换 planner？查看 :doc:`usage/configure_planner`。
- 打算基于 RPent 扩展？看 :doc:`development/architecture`。
