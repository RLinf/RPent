概览
====

**RPent (Recursive Physical Agent)** 是一个用于构建具身智能体的开放框架,
让智能体通过与物理世界的递归交互持续演进。RPent 不预设某个具体的基础模型,
而是提供一个递归智能体框架, 将异构智能能力 —— 感知 (perception)、推理
(reasoning)、记忆 (memory)、执行 (execution)、自我演进 (self-evolution)
—— 统一到一个物理智能体中。通过持续的交互、反思与适应, RPent 让物理
智能体获得超出其初始设计的新能力。

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
     - - **VLA 操作**

         - Pi0.5 ✅
         - RLDX-1

       - **WAM 操作**

         - DreamZero
     - - LIBERO-PRO ✅
       - RoboCasa
     - - Franka
       - SO-101

下一步去哪里
------------

- 第一次接触 RPent? 先看 :doc:`installation`, 再看 :doc:`quickstart`,
  端到端跑通一个 LIBERO 任务。
- 想驱动一个具体的机器人或切换 planner? 直接看 :doc:`usage/configure_planner`。
- 打算基于 RPent 扩展? 看 :doc:`development/architecture`。
