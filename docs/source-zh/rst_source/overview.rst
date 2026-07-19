概览
====

RPent —— 全称 **Recursive Physical Agent** —— 是一个用于构建具身智能体的
开放框架, 让智能体通过与物理世界的递归交互持续演进。RPent 不预设某个具体的
基础模型, 而是提供一个 **递归智能体框架**, 将异构智能能力 —— 感知 (perception)、
推理 (reasoning)、记忆 (memory)、执行 (execution)、自我演进 (self-evolution)
—— 统一到一个物理智能体中。通过持续的交互、反思与适应, RPent 让物理智能体
获得超出其初始设计的新能力。

.. image:: https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png
   :alt: RPent 框架图
   :align: center
   :width: 90%

名字的由来
----------

**Pent** 源自 **五芒星 (Pentagram)**, 其五个顶点象征多模态智能融合为一个
统一的具身智能体的五个方面:

- **感知 (Perception)** —— 看见、听见、触摸世界。
- **推理 (Reasoning)** —— 规划并决定下一步做什么。
- **记忆 (Memory)** —— 保留有效与无效的经验。
- **执行 (Execution)** —— 把意图转化为动作。
- **自我演进 (Self-evolution)** —— 从每一次交互中学习。

五芒星的中心是 **无穷符号 (∞)**, 代表 *感知 → 推理 → 执行 → 自我演进*
永无止境的递归循环, 让智能持续向物理世界扩展。五芒星加上其中心的无穷符号,
既是 RPent 的 logo, 也是这个框架的形状: 五个一等公民子系统, 通过闭环连接。

*RPent* 中的 **R** 强调框架的核心承诺: 智能不是一次性部署, 而是一个
*递归 (recursive)* 的过程 —— 每一段 episode 都会反馈到影响下一段的记忆中。

设计原则
--------

RPent 建立在三条原则之上:

- **服务化 (Service-oriented)。** 每一项能力 —— VLA 策略、planner brain、
  memory store、simulator —— 都作为一个独立服务部署, 有自己的进程边界和
  生命周期。GPU 大模型永远不会和物理引擎共享一个 Python 解释器。
- **标准化 (Standardized)。** 各服务通过统一接口通信 (RPC、HTTP、MCP),
  这样接入一个新 brain 或新机械臂时不需要改 runner。
- **可组合 (Composable)。** 任意 planner 都能驱动任意 primitive 组合, 作用
  于任意 environment。用一个 flag 切换 brain; 只要把包放进 ``robots/``
  就能接入新机械臂。

这三条原则让 RPent 超越了传统的机器人控制框架, 成为 **面向物理世界的
智能体基础设施 (Agentic Infrastructure for the Physical World)** ——
在这里, 智能不只是被部署, 而是被持续构建、扩展与演进。

支持的 environment 与 brain
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Agentic planner
     - Action primitive
     - Simulator
     - 真实机器人
   * - ``api`` —— pydantic-ai ✅
     - **VLA 操作** ✅
     - **LIBERO** ✅
     - **Franka** (研发中)
   * - Anthropic (Claude) ✅
     - Pi0.5 (LIBERO, HTTP) ✅
     - libero_object / _goal / _spatial / _10
     - **SO-101** (研发中)
   * - OpenAI (responses) ✅
     - RLDX-1 (RoboCasa, socket-RPC) ✅
     - **RoboCasa** ✅
     -
   * - OpenAI 兼容 (chat) ✅
     -
     - PickPlace, Open/Close, TurnOn/Off …
     -
   * - ``claude_code`` —— Claude Agent SDK ✅
     -
     -
     -
   * - ``codex`` —— OpenAI Codex SDK ✅
     -
     -
     -

下一步去哪里
------------

- 第一次接触 RPent? 先看 :doc:`installation`, 再看 :doc:`quickstart`,
  端到端跑通一个 LIBERO 任务。
- 想驱动一个具体的机器人或切换 planner? 直接看 :doc:`usage/configure_planner`。
- 打算基于 RPent 扩展? 看 :doc:`development/architecture`。
