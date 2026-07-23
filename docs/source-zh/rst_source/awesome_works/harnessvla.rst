Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents
================================================================================================

**资源：** `论文 <https://arxiv.org/abs/2607.08448>`_ | `项目主页
<https://harnessvla.github.io/>`_ | `代码 <https://github.com/RLinf/RPent>`_

概述
----

当前的视觉—语言—动作（Vision-Language-Action，VLA）模型在标准机器人
基准上表现出色，但当任务指令、目标绑定或空间布局发生变化时，性能可能明显
下降。π\ :sub:`RLinf` 在标准 LIBERO 上的成功率为 95.3%，面对 LIBERO-PRO
扰动时则降至 50.0%。许多失败并不是因为 VLA 不会抓取或放置，而是因为它在
错误的目标、不合适的状态或长时序任务的错误阶段，执行了局部上看似合理的
动作。

Harness VLA 将问题的重点从“如何训练更大的 VLA”转向“应该如何组织和调用
已有的 VLA”。它把冻结的 VLA 封装为可重试的接触密集型 Action Primitive，
并由 Agentic Planner 将其与一组规模较小且固定的 Analytic Primitives 组合。Planner
负责重新理解当前任务、为 VLA 创造合适的局部接管条件、检查实际执行结果，
并在失败后重新组织后续操作；整个过程中，VLA 权重始终保持冻结。

Harness VLA 是 RPent 的首篇论文。在部署阶段不更新 VLA、也不扩展
Action Primitive library 的条件下，它在 LIBERO-PRO、RoboCasa365 和 RoboTwin C2R
上分别取得 **82.4%**、**55.4%** 和 **58.4%** 的成功率。

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
与 VLA 的合理分工，并将有效的执行策略和恢复经验沉淀到 Task-Specific Memory
与 Global Memory 中。部署时，Agentic Planner 将这些记忆与实时观测结合，动态
重新绑定目标与空间参数，使经过验证的策略能够适应目标绑定和空间布局的变化。

实验结果
--------

Harness VLA 的评估覆盖标准和扰动后的桌面操作、家庭厨房长时序任务，以及
RoboTwin C2R 双臂操作。代表性成功率如下。

.. list-table:: Harness VLA 代表性实验结果
   :header-rows: 1
   :widths: 24 34 42

   * - 基准
     - 评估设置
     - 报告的成功率
   * - LIBERO
     - 标准任务套件
     - Harness VLA：**96.0%**；π\ :sub:`RLinf`：95.3%
   * - LIBERO-PRO
     - 扰动后的桌面操作
     - Harness VLA：**82.4%**；π\ :sub:`RLinf`：50.0%；RATS：43.8%；Cap-X：18.2%
   * - RoboCasa365
     - 家庭厨房操作
     - Harness VLA：**55.4%**；RLDX-1：30.0%
   * - RoboTwin C2R
     - 双臂操作（Clean to Random）
     - Harness VLA：**58.4%**；LingBot-VLA：50.4%

Harness VLA 在标准 LIBERO 上取得 96.0% 的成功率，与 π\ :sub:`RLinf` 的
95.3% 相当；在更具挑战性的 LIBERO-PRO 上，Harness VLA 达到 82.4%，超过
π\ :sub:`RLinf` 的 50.0%、RATS 的 43.8% 和 Cap-X 的 18.2%。在 RoboCasa365
上，Harness VLA 将任务加权总体成功率从 RLDX-1 的 30.0% 提升至 55.4%。
在 RoboTwin C2R 上，Harness VLA 达到 58.4%，超过 LingBot-VLA 的 50.4%。
这些提升来自三种相互配合的机制：Planner 完成语义重新定位，在重新预置位后
对 VLA 进行稀疏且有针对性的重试，以及使用 Analytic Primitives 隔离非接触执行。

快速开始
--------

* **教程：** :doc:`LIBERO <../usage/libero>`

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
