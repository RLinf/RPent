.. _benchmark-results:

基准测试结果
============

本页汇集 RPent 评测与 Harness VLA 论文参考结果中的规划模型、原生推理设置和成功率。
结果按基准和评测设置分组，每个来源对应一份独立记录。概览列出各来源组内的最高记录，
不表示整个基准当前的 SOTA。

结果核对日期为 2026-09-09；论文参考使用 arXiv v4（2026-09-02）。

**表格说明。** “模型”指规划模型；``Reasoning`` 表示其原生推理模式，``Effort`` 表示配置的推理强度。
``xhigh`` 和 ``max`` 属于不同提供方的设置，不代表相同的计算预算。
“未报告”表示来源未提供相应结果或设置；“待收录”表示预留给后续更新的结果；``N/A`` 表示
该规划器设置不适用。以 **R** 开头的来源编号表示 RPent 评测，**P** 表示带版本的论文表格。

规划模型身份和推理设置已经实验贡献者确认：历史 Codex 记录使用 GPT-5.5 与 ``xhigh``，
Claude Code 使用 Opus-4.8 与 ``max``，Codex 另有使用 GPT-6 Astra 与 ``low`` 的记录。
三种配置均开启原生推理。论文将后端标为 Codex 和 CC（Claude Code）；本页列出的
对应模型与推理强度来自贡献者提供的元数据。

概览
--------

以下概览在各来源组内，为相同基准和设置选取已报告的最高成功率。
记忆库、运行时版本和评测协议的差异都可能影响成绩，因此这些分数差异不构成
仅改变规划模型的受控比较。

.. list-table:: RPent 评测中的最高记录
   :header-rows: 1
   :widths: 20 11 16 11 9 23 10

   * - 基准
     - 后端
     - 模型
     - Reasoning
     - Effort
     - 成功率
     - 来源
   * - PRO Long Task
     - Codex
     - GPT-6 Astra
     - 开启
     - ``low``
     - 85% (85/100)
     - :ref:`评测 R1 <benchmark-source-r1>`
   * - PRO Long Swap
     - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
     - 55% (55/100)
     - :ref:`复现 R2 <benchmark-source-r2>`
   * - RoboCasa Target50
     - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
     - 57.00%（按任务加权）
     - :ref:`复现 R3 <benchmark-source-r3>`
   * - RoboTwin C2R
     - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
     - 58.0% (145/250)
     - :ref:`复现 R4 <benchmark-source-r4>`

.. list-table:: 论文参考中的最高记录
   :header-rows: 1
   :widths: 17 12 11 13 10 8 20 9

   * - 基准
     - 方法
     - 后端
     - 模型
     - Reasoning
     - Effort
     - 成功率
     - 来源
   * - Standard LIBERO
     - AtomVLA
     - N/A
     - N/A
     - N/A
     - N/A
     - 97.0%
     - :ref:`v4 P2 <benchmark-source-p2>`
   * - LIBERO-PRO 总体
     - Harness VLA
     - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
     - 82.4%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - PRO Long Task
     - Harness VLA
     - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
     - 71.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - PRO Long Swap
     - Harness VLA
     - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
     - 62.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - RoboCasa Target50
     - Harness VLA
     - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
     - 57.1%（按任务加权）
     - :ref:`v4 P4 <benchmark-source-p4>`
   * - RoboTwin C2R
     - Harness VLA
     - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
     - 58.4%
     - :ref:`v4 P6 <benchmark-source-p6>`

本页没有 Standard LIBERO 或涵盖八个单元的 LIBERO-PRO 总体指标的独立 RPent 评测记录。
即使模型名称或分数相同，论文参考结果与 RPent 评测仍保留各自的来源。

LIBERO-PRO Long
------------------------------

Long Task 对应 ``libero_10_task``，Long Swap 对应 ``libero_10_swap``。
每个已报告套件包含 100 个评测回合。两者是 LIBERO-PRO 的两个子集，
不能视为标准 LIBERO-10 套件或完整 LIBERO-PRO 的总体指标。

.. list-table:: Long Task 与 Long Swap
   :header-rows: 1
   :widths: 13 18 12 10 19 18 10

   * - 后端
     - 模型
     - Reasoning
     - Effort
     - Task
     - Swap
     - 来源
   * - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
     - 70% (70/100)
     - 55% (55/100)
     - :ref:`复现 R2 <benchmark-source-r2>`
   * - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
     - 52.0%
     - 49.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - Codex
     - GPT-6 Astra
     - 开启
     - ``low``
     - 85% (85/100)
     - 待收录
     - :ref:`评测 R1 <benchmark-source-r1>`
   * - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
     - 71.0%
     - 62.0%
     - :ref:`v4 P3 <benchmark-source-p3>`

R2 使用 LIBERO 复现分支及其发布的记忆；R1 使用独立构建并冻结的本地记忆库。
两份结果各自保留评测上下文，详见来源说明。

LIBERO-PRO 各任务族
-------------------------------

论文分别在 Task（指令重定向）和 Swap（位置交换）扰动下评测 Spatial、Object、Goal 和 Long。
每个单元包含 10 个任务，每个任务使用 10 个评测 seed，共 100 个回合。
seed 0 专用于构建记忆。以下列出论文完整的少样本结果。

.. list-table:: Task 扰动
   :header-rows: 1
   :widths: 12 15 10 8 11 11 11 12 10

   * - 后端
     - 模型
     - Reasoning
     - Effort
     - Spatial
     - Object
     - Goal
     - Long
     - 来源
   * - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
     - 81.0%
     - 94.0%
     - 75.0%
     - 52.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
     - 94.0%
     - 88.0%
     - 87.0%
     - 71.0%
     - :ref:`v4 P3 <benchmark-source-p3>`

.. list-table:: Swap 扰动
   :header-rows: 1
   :widths: 12 15 10 8 11 11 11 12 10

   * - 后端
     - 模型
     - Reasoning
     - Effort
     - Spatial
     - Object
     - Goal
     - Long
     - 来源
   * - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
     - 69.0%
     - 91.0%
     - 66.0%
     - 49.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
     - 80.0%
     - 90.0%
     - 87.0%
     - 62.0%
     - :ref:`v4 P3 <benchmark-source-p3>`

.. list-table:: 全部八个 PRO 单元的总体指标
   :header-rows: 1
   :widths: 15 20 12 10 15 18 10

   * - 后端
     - 模型
     - Reasoning
     - Effort
     - 总体
     - 评测回合数
     - 来源
   * - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
     - 72.1%
     - 800
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
     - 82.4%
     - 800
     - :ref:`v4 P3 <benchmark-source-p3>`

总体指标涵盖全部八个单元，各单元的评测样本数相同，不能用 Long Task 和 Long Swap
两项的均值替代。RATS 和 Cap-X 仅报告六个非 Long 单元，其覆盖范围见基线参考表。

Standard LIBERO
------------------------------

Standard LIBERO 评测原始 Spatial、Object、Goal 和 Long（LIBERO-10）套件，
不施加 PRO 扰动。每个套件包含 100 个回合，总计 400 个回合。
论文仅报告 Claude Code 的结果，未报告 Codex 在四个标准套件及其总体指标上的结果。

.. list-table:: Standard LIBERO 论文结果
   :header-rows: 1
   :widths: 12 14 10 8 10 10 10 10 16 10

   * - 后端
     - 模型
     - Reasoning
     - Effort
     - Spatial
     - Object
     - Goal
     - Long
     - 总体
     - 来源
   * - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
     - 97.0%
     - 100.0%
     - 94.0%
     - 93.0%
     - 96.0% (384/400)
     - :ref:`v4 P2 <benchmark-source-p2>`

RoboCasa365 Target50
----------------------------------------

Target50 包含 18 个 Atomic-Seen 任务，每个任务使用 10 个 seed；16 个 Composite-Seen
任务与 16 个 Composite-Unseen 任务，每个任务各使用五个 seed。
三个划分分别包含 180、80 和 80 个回合。``Seen`` 和 ``Unseen`` 表示预训练中的
任务模板覆盖情况。总体指标包含全部 50 个任务。

.. list-table:: RoboCasa 各划分结果
   :header-rows: 1
   :widths: 12 15 10 8 16 16 17 12 10

   * - 后端
     - 模型
     - Reasoning
     - Effort
     - Atomic-Seen
     - Composite-Seen
     - Composite-Unseen
     - 总体
     - 来源
   * - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
     - 90.56% (163/180)
     - 61.25% (49/80)
     - 15.00% (12/80)
     - 57.00%
     - :ref:`复现 R3 <benchmark-source-r3>`
   * - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
     - 92.0%
     - 61.0%
     - 13.8%
     - 57.1%
     - :ref:`v4 P4 <benchmark-source-p4>`
   * - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
     - 79.4%
     - 47.5%
     - 15.0%
     - 48.6%
     - :ref:`v4 P4 <benchmark-source-p4>`

**汇总口径。** 总体指标按任务加权，每个任务的权重相同，尽管 Atomic-Seen
每个任务的 seed 数是其他划分的两倍。RPent 记录的计算方式为：

.. math::

   \mathrm{Overall} = \frac{18(163/180) + 16(49/80) + 16(12/80)}{50}
   \times 100\% = 57.00\%.

合并所有回合后得到的比例 ``224/340`` 不是此处的总体分数。论文百分比保留其原始精度，
不根据四舍五入后的百分比反推精确成功次数。R3 的 43 个任务使用冻结的同任务记忆，
另有七个任务在不使用任务记忆的情况下评测。完整 Target50 协议见 :doc:`usage/robocasa`。

RoboTwin C2R
------------------------

从干净设置迁移到随机设置的评测涵盖 50 个任务，每个任务使用五个经官方专家验证的
随机化 seed，共 250 个回合。任务记忆来自已验证的 ``demo_clean`` 实例，
直接迁移到 ``demo_randomized``，不在随机设置中进行探索。

.. list-table:: RoboTwin 从干净设置到随机设置的结果
   :header-rows: 1
   :widths: 16 22 14 12 24 12

   * - 后端
     - 模型
     - Reasoning
     - Effort
     - 成功率
     - 来源
   * - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
     - 58.0% (145/250)
     - :ref:`复现 R4 <benchmark-source-r4>`
   * - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
     - 58.0%
     - :ref:`v4 P6 <benchmark-source-p6>`
   * - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
     - 58.4%
     - :ref:`v4 P6 <benchmark-source-p6>`

R4 是 ``reproduce/robotwin`` 分支的结果。论文中相同的 Codex 百分比保留为独立来源记录，
不应作为额外的独立评测样本重复计数。

LIBERO-PRO Goal：零样本
--------------------------------------

该消融设置不使用目标设置的任务特定记忆和全局记忆。
每种扰动包含 10 个任务，每个任务使用 10 个 seed。这些结果不与使用记忆的 PRO 表格合并。
Codex 的 Goal 零样本结果未报告。

.. list-table:: Goal 零样本论文结果
   :header-rows: 1
   :widths: 15 15 20 12 10 18 10

   * - 扰动
     - 后端
     - 模型
     - Reasoning
     - Effort
     - 成功率
     - 来源
   * - Task
     - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
     - 79.0%
     - :ref:`v4 P5 <benchmark-source-p5>`
   * - Swap
     - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
     - 31.0%
     - :ref:`v4 P5 <benchmark-source-p5>`

基线参考
-------------------

以下是所链接论文表格中的参考方法，不是额外的 RPent 规划器评测。
“模型”和推理相关列描述独立规划器：直接使用 VLA 的方法标为 ``N/A``；
论文表格未说明规划器配置的智能体基线标为“未报告”。不能仅根据方法名称确定其原生推理设置。

.. list-table:: Standard LIBERO 基线
   :header-rows: 1
   :widths: 23 13 13 13 10 16 12

   * - 方法
     - 后端
     - 模型
     - Reasoning
     - Effort
     - 总体
     - 来源
   * - OpenVLA
     - N/A
     - N/A
     - N/A
     - N/A
     - 76.5%
     - :ref:`v4 P2 <benchmark-source-p2>`
   * - NORA
     - N/A
     - N/A
     - N/A
     - N/A
     - 79.5%
     - :ref:`v4 P2 <benchmark-source-p2>`
   * - π0
     - N/A
     - N/A
     - N/A
     - N/A
     - 94.2%
     - :ref:`v4 P2 <benchmark-source-p2>`
   * - π_RLinf
     - N/A
     - N/A
     - N/A
     - N/A
     - 95.3%
     - :ref:`v4 P2 <benchmark-source-p2>`
   * - AtomVLA
     - N/A
     - N/A
     - N/A
     - N/A
     - 97.0%
     - :ref:`v4 P2 <benchmark-source-p2>`

所有标准 LIBERO 行都涵盖 Spatial、Object、Goal 和 Long。
论文中每个套件 100 个回合的样本数适用于 π_RLinf 和 Harness VLA，
不套用于外部基线报告。AtomVLA 的总体成绩为 97.0%，高于 Harness VLA
标准 LIBERO 参考结果的 96.0%。

.. list-table:: LIBERO-PRO 基线
   :header-rows: 1
   :widths: 17 12 12 12 10 15 12 10

   * - 方法
     - 后端
     - 模型
     - Reasoning
     - Effort
     - 覆盖范围
     - 总体
     - 来源
   * - OpenVLA
     - N/A
     - N/A
     - N/A
     - N/A
     - 八个单元
     - 0.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - π0
     - N/A
     - N/A
     - N/A
     - N/A
     - 八个单元
     - 0.3%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - π0.5
     - N/A
     - N/A
     - N/A
     - N/A
     - 八个单元
     - 11.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - MolmoAct
     - N/A
     - N/A
     - N/A
     - N/A
     - 八个单元
     - 1.5%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - NORA
     - N/A
     - N/A
     - N/A
     - N/A
     - 八个单元
     - 0.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - X-VLA
     - N/A
     - N/A
     - N/A
     - N/A
     - 八个单元
     - 3.8%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - AtomVLA
     - N/A
     - N/A
     - N/A
     - N/A
     - 八个单元
     - 6.3%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - π_RLinf
     - N/A
     - N/A
     - N/A
     - N/A
     - 八个单元
     - 50.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - Cap-X
     - 未报告
     - 未报告
     - 未报告
     - 未报告
     - 六个非 Long 单元
     - 18.2%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - RATS
     - 未报告
     - 未报告
     - 未报告
     - 未报告
     - 六个非 Long 单元
     - 43.8%
     - :ref:`v4 P3 <benchmark-source-p3>`

Cap-X 和 RATS 的总体指标是 Spatial、Object 和 Goal 在 Task 与 Swap 两种扰动下的均值。
此表中两者均未报告 Long Task 或 Long Swap。比较时应保留其六个单元的覆盖范围，
不能将其均值视为全部八个 PRO 单元的结果。

.. list-table:: RoboCasa365 基线
   :header-rows: 1
   :widths: 23 13 13 13 10 16 12

   * - 方法
     - 后端
     - 模型
     - Reasoning
     - Effort
     - 总体
     - 来源
   * - RLDX-1
     - N/A
     - N/A
     - N/A
     - N/A
     - 30.0%
     - :ref:`v4 P4 <benchmark-source-p4>`
   * - WorldDreamer
     - N/A
     - N/A
     - N/A
     - N/A
     - 35.3%
     - :ref:`v4 P4 <benchmark-source-p4>`
   * - π0.5
     - N/A
     - N/A
     - N/A
     - N/A
     - 16.9%
     - :ref:`v4 P4 <benchmark-source-p4>`
   * - π0
     - N/A
     - N/A
     - N/A
     - N/A
     - 14.8%
     - :ref:`v4 P4 <benchmark-source-p4>`

RoboCasa 总体分数沿用论文对 Atomic-Seen、Composite-Seen 和 Composite-Unseen
的汇总结果。RLDX-1 是直接运行冻结 VLA 的基线，其余各行来自外部论文。

.. list-table:: RoboTwin C2R 基线
   :header-rows: 1
   :widths: 23 13 13 13 10 16 12

   * - 方法
     - 后端
     - 模型
     - Reasoning
     - Effort
     - 总体
     - 来源
   * - GR00T-N1.7
     - N/A
     - N/A
     - N/A
     - N/A
     - 20.7%
     - :ref:`v4 P6 <benchmark-source-p6>`
   * - π0.5
     - N/A
     - N/A
     - N/A
     - N/A
     - 47.9%
     - :ref:`v4 P6 <benchmark-source-p6>`
   * - StarVLA
     - N/A
     - N/A
     - N/A
     - N/A
     - 10.6%
     - :ref:`v4 P6 <benchmark-source-p6>`
   * - LingBot-VLA
     - N/A
     - N/A
     - N/A
     - N/A
     - 50.4%
     - :ref:`v4 P6 <benchmark-source-p6>`

LingBot-VLA 既是 Harness VLA 使用的冻结接触策略后端，也是直接评测基线。
其他行来自外部报告，不将 Harness VLA 的 250 个回合样本数套用于这些行。

.. list-table:: Goal 零样本智能体基线
   :header-rows: 1
   :widths: 12 12 14 14 14 12 14 8

   * - 方法
     - 扰动
     - 后端
     - 模型
     - Reasoning
     - Effort
     - 成功率
     - 来源
   * - Cap-X
     - Task
     - 未报告
     - 未报告
     - 未报告
     - 未报告
     - 16.8%
     - :ref:`v4 P5 <benchmark-source-p5>`
   * - Cap-X
     - Swap
     - 未报告
     - 未报告
     - 未报告
     - 未报告
     - 25.6%
     - :ref:`v4 P5 <benchmark-source-p5>`

协议与来源
---------------------

任务成功依据基准判定条件：LIBERO 环境轨迹中的 ``terminated``、RoboCasa 的
``state.success``，或 RoboTwin 的 ``TASK_ENV.eval_success``。规划器调用 ``finish`` 或动作原语
返回局部成功，本身不构成任务成功标签。用于构建记忆的探索回合不计入评测成绩。

LIBERO 系列的 VLA 后端是冻结的 RLinf π0.5 full-shot LIBERO 检查点；
RoboCasa 使用冻结的 RLDX-1；RoboTwin 使用后训练后冻结的 LingBot-VLA 检查点。
规划模型、VLA 后端、记忆来源和基准设置分别描述被评测系统的不同组成部分。

.. _benchmark-source-r1:
.. _benchmark-protocol-r1:

**R1 — RPent LIBERO-PRO 评测。** 实验
``libero_long_gpt6_astra_20260907`` 于 2026-09-07 开始，结果核对日期为
2026-09-09。运行时版本为
`014a0fa <https://github.com/RLinf/RPent/commit/014a0fa97f69c991ee5e0f62f14e1d6f89c3dcd7>`_。
使用 Codex 与 GPT-6 Astra，开启原生推理，推理强度为 ``low``。
任务记忆在 seed 0 上构建并冻结，随后对 10 个任务分别使用 seed 1–10 评测。
Long Task 的 100 个回合中有 85 个达到环境成功条件（85%）；本页没有 Long Swap 的已发布结果。
该评测使用自己的本地记忆库，规划器时间上限为 5000 秒，环境步数上限为 10000 步。

.. _benchmark-source-r2:

**R2 — RPent LIBERO 复现。**
`已发布结果与命令 <https://github.com/RLinf/RPent/blob/371ac90ece4f95676b7ea5ca05473986f134d5a4/docs/source-en/rst_source/usage/libero.rst#L278-L303>`_
记录了 ``reproduce/libero`` 分支的 Long Task 70/100 和 Long Swap 55/100。
贡献者确认其配置为 Codex、GPT-5.5、开启原生推理、``xhigh`` 推理强度。
该记录使用对应分支发布的记忆与运行时；所引用结果没有注明确切的历史 seed 列表。

.. _benchmark-source-r3:

**R3 — RPent RoboCasa Target50 复现。**
`逐任务结果 <https://github.com/RLinf/RPent/blob/371ac90ece4f95676b7ea5ca05473986f134d5a4/robots/robocasa/eval/target50_codex_results.md>`_
和
`Target50 清单 <https://github.com/RLinf/RPent/blob/371ac90ece4f95676b7ea5ca05473986f134d5a4/robots/robocasa/eval/target50.json>`_
记录了 ``robocasa-harness-vla-v1`` 的模型配置、固定资源和 340 个评测单元的矩阵。
公开数据提供逐任务成功次数，不包含各 seed 的轨迹。其中较早的 Harness VLA 55.40%
对照值对应
`v3 论文结果 <https://arxiv.org/html/2607.08448v3#S3.T4>`_。
本页论文对照采用 v4 的 57.1%，同时保留 RPent 的 57.00% 成绩。

.. _benchmark-source-r4:

**R4 — RPent RoboTwin 复现。**
`结果与协议 <https://github.com/RLinf/RPent/blob/371ac90ece4f95676b7ea5ca05473986f134d5a4/docs/source-en/rst_source/usage/robotwin.rst#L191-L228>`_
记录了 ``reproduce/robotwin`` 分支的 145/250，配置为 Codex、GPT-5.5 与 ``xhigh`` 推理强度。
`评测清单 <https://github.com/RLinf/RPent/blob/371ac90ece4f95676b7ea5ca05473986f134d5a4/robots/robotwin/eval/demo_randomized.json>`_
列出每个任务的五个已验证 seed，各任务的 seed 列表不同。
复现分支的分数不代表其效果与 ``main`` 完全一致。

.. _benchmark-source-p2:

**P2 — Standard LIBERO。** Harness VLA，
`arXiv v4，表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_
（2026-09-02）。四个标准套件，Harness VLA 每个套件包含 100 个回合。

.. _benchmark-source-p3:

**P3 — LIBERO-PRO。** Harness VLA，
`arXiv v4，表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_
（2026-09-02）。八个 Task/Swap 单元，Harness VLA 每个单元包含 100 个回合。

.. _benchmark-source-p4:

**P4 — RoboCasa365。** Harness VLA，
`arXiv v4，表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_
（2026-09-02）。Atomic-Seen、Composite-Seen、Composite-Unseen 及已报告的总体指标。

.. _benchmark-source-p5:

**P5 — LIBERO-PRO Goal 零样本。** Harness VLA，
`arXiv v4，表 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_
（2026-09-02）。不使用目标设置记忆的 Goal Task 和 Swap。

.. _benchmark-source-p6:

**P6 — RoboTwin C2R。** Harness VLA，
`arXiv v4，表 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_
（2026-09-02）。从干净设置到随机设置的迁移，Harness VLA 包含 250 个回合。

添加结果时，请同时记录后端、精确规划模型、原生推理设置、推理强度、基准划分、
评测样本数、记忆协议和带版本的来源。不可用的单元标为“未报告”；协议或实现发生变化时，
添加独立的来源记录。
