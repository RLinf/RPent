Harness VLA
===========

*Steering Frozen VLAs into Reliable Manipulation Primitives via
Memory-Guided Agents*

**资源：** `论文 <https://arxiv.org/abs/2607.08448>`_ | `项目主页
<https://harnessvla.github.io/>`_ | `代码 <https://github.com/RLinf/RPent>`_

概述
----

当前的视觉—语言—动作（Vision-Language-Action，VLA）模型在标准机器人基准上表现出色，
但当任务指令、目标绑定或空间布局发生变化时，性能可能明显下降。π\ :sub:`RLinf`
在标准 LIBERO 上的成功率为 95.3%，面对 LIBERO-PRO 扰动时则降至 50.0%。许多失败并不是因为
VLA 不会抓取或放置，而是因为它在错误的目标、不合适的状态或长时序任务的错误阶段，
执行了局部上看似合理的动作。

Harness VLA 将问题的重点从“如何训练更大的 VLA”转向“应该如何组织和调用已有的
VLA”。它把冻结的 VLA 封装为可重试的接触密集型 Action Primitive，并由 Agentic
Planner 将其与一组规模较小且固定的 Analytic Primitives 组合。Planner 负责重新理解当前任务、
为 VLA 创造合适的局部接管条件、检查实际执行结果，并在失败后重新组织后续操作；
整个过程中，VLA 权重始终保持冻结。

Harness VLA 是 RPent 的首篇论文。在 RPent 中，Claude Code 与 Opus-4.8 在 LIBERO-PRO
和 RoboTwin C2R 上分别取得 **82.4%** 和 **58.4%** 的成功率；Codex 与 GPT-5.5 在
RoboCasa365 上取得 **57.1%**。部署期间 VLA 保持冻结。模型配置、各基准分项和
评测说明见 :doc:`../benchmarks`。

.. figure:: https://github.com/RLinf/misc/raw/main/pic/harnessvla_scheme.png
   :alt: Harness VLA 框架概览
   :align: center
   :width: 100%

   Harness VLA 框架概览

框架
----

Harness VLA 将 Agentic Planner、Action Primitives 与两类记忆组织在同一框架中：

* **Agentic Planner。** 编程智能体结合任务描述和当前 RGB-D 观测，重新绑定
  目标物体与目标区域，检查执行反馈，并选择、组合或重试可用的
  Action Primitives。
* **Action Primitives。** RPent 将不同类型的机器人能力封装为可由 Planner
  调用的 Action Primitives。Harness VLA 主要组合以下两类：

  * **Analytic Primitives。** 固定的 Analytic Primitives 负责预置位、空间搬运、
    姿态调整、夹爪控制、导航和释放等非接触操作，与 VLA 负责的接触密集型操作
    形成明确分工。
  * **VLA Primitive。** Harness VLA 将冻结的 VLA 转化为可由 Agentic Planner
    灵活调用的 Action Primitive，负责不规则物体抓取、受约束放置、按钮按压，
    以及抽屉和门等铰接机构交互中的接触密集型操作。每次执行后，Agentic Planner
    会结合最新的 RGB-D 观测检查结果，并在必要时调整机器人状态，发起更有针对性
    的尝试。
* **Memory。** Task-Specific Memory 将经过验证的执行策略沉淀为可参数化的
  Action Primitive 组合，使 Agentic Planner 能够结合当前观测重新绑定目标与空间
  参数；Global Memory 则提炼可跨任务复用的成功经验、失败模式和恢复策略，
  为后续规划提供指导。

在探索阶段，Agentic Planner 从一个 seed 任务实例出发，探索 Analytic Primitives
与 VLA 的合理分工，并将有效的执行策略和恢复经验沉淀到 Task-Specific Memory 与
Global Memory 中。部署时，Agentic Planner 将这些记忆与实时观测结合，动态重新绑定目标与空间参数，
使经过验证的策略能够适应目标绑定和空间布局的变化。

实验结果
--------

:doc:`基准测试结果页 <../benchmarks>` 按套件或划分横向对比 RPent 的不同规划模型。
页面覆盖标准 LIBERO、LIBERO-PRO Task/Swap、RoboCasa365 的 Atomic-Seen、
Composite-Seen、Composite-Unseen、RoboTwin C2R 和 Goal 零样本消融，并说明模型配置与指标定义。
GPT-5.5 与 Opus-4.8 的已报告结果与 Harness VLA
`论文 v4 <https://arxiv.org/html/2607.08448v4#S3.SS3>`_ 对齐；GPT-6 Astra 为新增模型评测结果。
外部基线方法另列参考。

快速开始
--------

各环境的安装配置、运行命令与历史复现记录见下列教程及对应分支。

* **LIBERO：** :doc:`教程 <../usage/libero>` —
  `reproduce/libero <https://github.com/RLinf/RPent/tree/reproduce/libero>`_
* **RoboCasa：** :doc:`教程 <../usage/robocasa>` — 使用 ``main``
* **RoboTwin：** :doc:`教程 <../usage/robotwin>` —
  `reproduce/robotwin <https://github.com/RLinf/RPent/tree/reproduce/robotwin>`_

引用
----

.. code-block:: bibtex

   @article{zhang2026harnessvla,
     title={Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents},
     author={Zhang, Yixian and Zhang, Huanming and Gao, Feng and Li, Xiao and
             Liu, Zhihao and Zhu, Chunyang and Qiu, Jiaxing and Yan, Yuchen and
             Liu, Jiyuan and Tang, Wenhao and Fang, Zhengru and Nie, Yi and
             Wei, Changxu and Wang, Yu and Ding, Wenbo and Yu, Chao},
     journal={arXiv preprint arXiv:2607.08448},
     year={2026},
     url={https://arxiv.org/abs/2607.08448}
   }
