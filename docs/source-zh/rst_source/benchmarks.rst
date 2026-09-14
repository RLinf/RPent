:html_theme.sidebar_secondary.remove:

.. _benchmark-results:
.. _benchmark-leaderboard:
.. _leaderboard:

RPent 排行榜
============

.. raw:: html

   <div id="rpent-interactive-leaderboard" data-language="zh"
        data-results-url="https://raw.githubusercontent.com/RLinf/misc/e858f627dbcc1b35440c3cb5ecaaefd016eb8680/rpent/benchmarks/results.json">
     <div class="rpent-static-leaderboard">
       <p>完整结果表位于下方。</p>
     </div>
   </div>

**GPT-6 Astra：六个完整套件、600 回合。**
:doc:`逐任务与 seed 结果 <results/libero_pro_astra>`；Goal 与完整 Overall 完成后再补充。

.. _id2:

模型配置
------------


.. list-table:: RPent 规划模型配置
   :header-rows: 1
   :widths: 25 30 25 20

   * - 后端
     - 模型
     - Reasoning
     - Effort
   * - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
   * - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
   * - Codex
     - GPT-6 Astra
     - 开启
     - ``low``


``Reasoning`` 指模型的原生推理模式，``Effort`` 为实际配置的推理强度。
``xhigh`` 和 ``max`` 属于不同提供方的设置，不表示相同计算预算。
模型身份、后端对应关系及推理设置已经实验贡献者确认。

.. _libero-series:

LIBERO 系列
----------------

Standard LIBERO
~~~~~~~~~~~~~~~

标准 LIBERO 使用未施加 PRO 扰动的 Spatial、Object、Goal、Long 套件。
这里的 Long 为标准 LIBERO-10，与下表的 PRO Long Task/Swap 分开评测。


.. list-table:: RPent 成功率
   :header-rows: 1
   :widths: 31 23 23 23

   * - 评测项
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - Spatial
     - 未报告
     - 97.0%
     - 未报告
   * - Object
     - 未报告
     - 100.0%
     - 未报告
   * - Goal
     - 未报告
     - 94.0%
     - 未报告
   * - Long
     - 未报告
     - 93.0%
     - 未报告
   * - 总体
     - 未报告
     - 96.0% (384/400)
     - 未报告


.. _libero-pro-long:

.. _libero-pro-across-task-families:

LIBERO-PRO
~~~~~~~~~~

Task 为指令重定向，Swap 为位置交换。Overall 覆盖 Spatial、Object、Goal、Long 的全部
八个 Task/Swap 单元；只完成 Long 的结果不构成该总体指标。


.. list-table:: RPent 成功率
   :header-rows: 1
   :widths: 31 23 23 23

   * - 评测项
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - Spatial Task
     - 81.0%
     - 94.0%
     - 100% (100/100)
   * - Spatial Swap
     - 69.0%
     - 80.0%
     - 98% (98/100)
   * - Object Task
     - 94.0%
     - 88.0%
     - 100% (100/100)
   * - Object Swap
     - 91.0%
     - 90.0%
     - 99% (99/100)
   * - Goal Task
     - 75.0%
     - 87.0%
     - 未报告
   * - Goal Swap
     - 66.0%
     - 87.0%
     - 未报告
   * - Long Task
     - 52.0%
     - 71.0%
     - 85% (85/100)
   * - Long Swap
     - 49.0%
     - 62.0%
     - 72% (72/100)
   * - 总体
     - 72.1%
     - 82.4%
     - 未报告


.. _libero-pro-goal:


GPT-6 Astra 本轮只公布 **六个完整套件、600 回合** （554 成功、46 失败）。
Goal Task、Goal Swap 与八项 Overall 保持未报告，完成后再补充。

LIBERO-PRO Goal：零样本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

此消融不使用目标设置的 Task Specific Memory 和 Global Memory，
与上方使用记忆的 PRO 主结果分别统计。


.. list-table:: RPent 成功率
   :header-rows: 1
   :widths: 31 23 23 23

   * - 评测项
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - Goal Task
     - 未报告
     - 79.0%
     - 未报告
   * - Goal Swap
     - 未报告
     - 31.0%
     - 未报告


RoboCasa365 Target50
--------------------

列出全部三个划分及 Overall。Overall 按 50 个任务均权；各划分的回合数不同，
不能直接把所有成功回合合并求比率。


.. list-table:: RPent 成功率
   :header-rows: 1
   :widths: 31 23 23 23

   * - 评测项
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - Atomic-Seen
     - 92.0%
     - 79.4%
     - 未报告
   * - Composite-Seen
     - 61.0%
     - 47.5%
     - 未报告
   * - Composite-Unseen
     - 13.8%
     - 15.0%
     - 未报告
   * - 总体（任务均权）
     - 57.1%
     - 48.6%
     - 未报告


RoboTwin C2R
------------

C2R 表示从干净设置到随机设置的迁移评测。


.. list-table:: RPent 成功率
   :header-rows: 1
   :widths: 31 23 23 23

   * - 评测项
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - C2R
     - 58.0%
     - 58.4%
     - 未报告


.. _id3:

论文完整结果表
----------------------------

以下补齐 Harness VLA v4 表 2–6 的全部成绩，保留来源精度与缺失值。
RPent 对应论文的 Harness VLA；Codex 与 CC 分别对应 GPT-5.5 与 Opus-4.8。
外部方法的样本数沿用原始报告，不从舍入后的百分比反推成功次数。

Standard LIBERO
~~~~~~~~~~~~~~~~

:ref:`表 2 <benchmark-source-p2>`

.. csv-table::
   :name: paper-table-2
   :header: "方法", "Spatial", "Object", "Goal", "Long", "Overall"
   :class: table-sm

   "OpenVLA", "84.7", "88.4", "79.2", "53.7", "76.5"
   "NORA", "85.6", "89.4", "80.0", "63.0", "79.5"
   "π0", "96.8", "98.8", "95.8", "85.2", "94.2"
   "π_RLinf", "99.0", "96.0", "97.0", "89.0", "95.3"
   "AtomVLA", "96.4", "99.6", "97.6", "94.4", "97.0"
   "RPent / Opus-4.8", "97.0", "100.0", "94.0", "93.0", "96.0"

LIBERO-PRO
~~~~~~~~~~~~~~~~

:ref:`表 3 <benchmark-source-p3>`

.. csv-table::
   :name: paper-table-3
   :header: "方法", "Spatial Task", "Spatial Swap", "Object Task", "Object Swap", "Goal Task", "Goal Swap", "Long Task", "Long Swap", "Overall"
   :class: table-sm

   "OpenVLA", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"
   "π0", "0.0", "0.0", "0.0", "2.0", "0.0", "0.0", "0.0", "0.0", "0.3"
   "π0.5", "1.0", "20.0", "1.0", "17.0", "2.0", "38.0", "1.0", "8.0", "11.0"
   "MolmoAct", "0.0", "0.0", "0.0", "6.0", "0.0", "0.0", "6.0", "0.0", "1.5"
   "NORA", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"
   "X-VLA", "0.0", "0.0", "8.0", "2.0", "9.0", "1.0", "10.0", "0.0", "3.8"
   "AtomVLA", "1.0", "16.0", "0.0", "10.0", "11.0", "2.0", "9.0", "1.0", "6.3"
   "Cap-X", "14.0", "12.0", "18.0", "22.0", "17.0", "26.0", "``-``", "``-``", "18.2 (6)"
   "RATS", "31.0", "29.0", "63.0", "61.0", "36.0", "43.0", "``-``", "``-``", "43.8 (6)"
   "π_RLinf", "42.0", "59.0", "71.0", "78.0", "45.0", "42.0", "49.0", "14.0", "50.0"
   "RPent / GPT-5.5", "81.0", "69.0", "94.0", "91.0", "75.0", "66.0", "52.0", "49.0", "72.1"
   "RPent / Opus-4.8", "94.0", "80.0", "88.0", "90.0", "87.0", "87.0", "71.0", "62.0", "82.4"

``(6)``：Cap-X 与 RATS 只报告 Spatial/Object/Goal 六项，其 Overall 不参与八项总体排名。
``-`` 为未报告，不能作为 0。Astra 六套结果是另外提供的实验，不属于本论文表。

RoboCasa365 Target50
~~~~~~~~~~~~~~~~~~~~

:ref:`表 4 <benchmark-source-p4>`

.. csv-table::
   :name: paper-table-4
   :header: "方法", "Atomic-Seen", "Composite-Seen", "Composite-Unseen", "Overall (task-weighted)"
   :class: table-sm

   "RLDX-1", "60.0", "21.3", "5.0", "30.0"
   "WorldDreamer", "66.3", "26.7", "9.0", "35.3"
   "π0.5", "39.6", "7.1", "1.2", "16.9"
   "π0", "34.6", "6.1", "1.1", "14.8"
   "RPent / GPT-5.5", "92.0", "61.0", "13.8", "57.1"
   "RPent / Opus-4.8", "79.4", "47.5", "15.0", "48.6"

LIBERO-PRO Goal 零样本：逐任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`表 5 <benchmark-source-p5>`

.. csv-table:: Task (T)
   :name: paper-table-5-task
   :header: "方法", "Task 0", "Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8", "Task 9", "平均"
   :class: table-sm

   "Cap-X", "0.0", "0.0", "10.0", "38.0", "12.0", "4.0", "34.0", "12.0", "40.0", "18.0", "16.8"
   "RPent / Opus-4.8", "10.0", "100.0", "90.0", "100.0", "20.0", "80.0", "90.0", "100.0", "100.0", "100.0", "79.0"

.. csv-table:: Swap (S)
   :name: paper-table-5-swap
   :header: "方法", "Task 0", "Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8", "Task 9", "平均"
   :class: table-sm

   "Cap-X", "0.0", "4.0", "0.0", "36.0", "22.0", "60.0", "4.0", "2.0", "62.0", "66.0", "25.6"
   "RPent / Opus-4.8", "0.0", "10.0", "0.0", "20.0", "90.0", "0.0", "10.0", "80.0", "100.0", "0.0", "31.0"

RoboTwin C2R
~~~~~~~~~~~~~~~~

:ref:`表 6 <benchmark-source-p6>`

.. csv-table::
   :name: paper-table-6
   :header: "方法", "成功率 (%)"
   :class: table-sm

   "GR00T-N1.7", "20.7"
   "π0.5", "47.9"
   "StarVLA", "10.6"
   "LingBot-VLA", "50.4"
   "RPent / GPT-5.5", "58.0"
   "RPent / Opus-4.8", "58.4"

.. _id4:

指标、协议与来源
------------------------

结果快照更新于 2026-09-14。RPent 的 GPT-5.5 与 Opus-4.8 已报告成绩与
Harness VLA 论文 v4（2026-09-02）对齐；GPT-6 Astra 为新增模型评测结果。
论文使用 Codex 和 CC（Claude Code）标记后端，精确模型与推理设置由贡献者提供。

任务成功依据基准判定条件：LIBERO 环境轨迹中的 ``terminated``、RoboCasa 的
``state.success``，或 RoboTwin 的 ``TASK_ENV.eval_success``。规划器调用 ``finish`` 或原语
返回局部成功，本身不构成任务成功标签。用于构建记忆的探索回合不计入评测成绩。

LIBERO 系列使用冻结的 RLinf π0.5 full-shot LIBERO 检查点；RoboCasa 使用冻结的
RLDX-1；RoboTwin 使用后训练后冻结的 LingBot-VLA 检查点。规划模型、VLA 后端、
记忆库与评测协议分别描述系统的不同组成部分，成绩差异不等同于只改变规划模型的受控实验。

.. _benchmark-source-p2:

**Standard LIBERO。** `Harness VLA，表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_。
四个标准套件，每个套件 100 回合，Overall 共 400 回合。

.. _benchmark-source-r2:
.. _benchmark-source-p3:

**LIBERO-PRO。** `Harness VLA，表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_。
八个 Task/Swap 单元，每个单元 10 个任务、每个任务 10 个评测 seed，共 100 回合；Overall 共 800 回合。seed 0 用于构建记忆。Long Task 对应 ``libero_10_task``，Long Swap 对应 ``libero_10_swap``。

.. _benchmark-source-r3:
.. _benchmark-source-p4:

**RoboCasa365 Target50。** `Harness VLA，表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_。
Atomic-Seen 有 18 个任务、每任务 10 个 seed；Composite-Seen 和 Composite-Unseen 各有 16 个任务、每任务 5 个 seed，分别为 180、80、80 回合。Seen/Unseen 表示预训练中的任务模板覆盖情况。Overall 对 50 个任务等权平均。百分比保留来源精度，不从四舍五入后的比例反推成功次数。

.. _benchmark-source-p5:

**LIBERO-PRO Goal 零样本。** `Harness VLA，表 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_。
Goal Task 和 Goal Swap 各有 10 个任务、每任务 10 个 seed，共 100 回合；不使用目标设置的 Task Specific Memory 和 Global Memory。

.. _benchmark-source-r4:
.. _benchmark-source-p6:

**RoboTwin C2R。** `Harness VLA，表 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_。
50 个任务，每个任务 5 个经专家验证的官方随机 seed，共 250 回合。任务记忆来自已验证的 ``demo_clean`` 实例，并迁移到 ``demo_randomized``，不在随机设置中探索。

.. _benchmark-source-r1:
.. _benchmark-protocol-r1:

**GPT-6 Astra 评测。** 实验 ``libero_long_gpt6_astra_20260907`` 于 2026-09-07 开始，
使用运行时版本 `014a0fa <https://github.com/RLinf/RPent/commit/014a0fa97f69c991ee5e0f62f14e1d6f89c3dcd7>`_。
Long Task 与 Long Swap 各有 10 个任务；任务记忆在 seed 0 上独立构建并冻结，
随后对每个任务使用 seed 1–10 评测。使用本地记忆库，规划器时间上限为 5000 秒，
环境步数上限为 10000 步；表中记录环境成功率。

新增模型或成绩时，保持同一评测项的行与模型列顺序，记录后端、模型、推理设置、
样本规模及协议；更新来源说明，并将没有结果的单元保持为“未报告”。

.. astra-supplementary-source-begin

.. _benchmark-source-astra-pro:

**GPT-6 Astra 补充 LIBERO-PRO 评测。** 实验
``libero_pro_remaining_gpt6_astra_20260913`` 使用运行版本
`014a0fa <https://github.com/RLinf/RPent/commit/014a0fa97f69c991ee5e0f62f14e1d6f89c3dcd7>`_。Spatial、Object、Goal 分别评测 Task 和 Swap；
每个已报告分项覆盖 10 个任务，各使用 seed 1–10，共 100 回合。
各任务在 seed 0 独立构建记忆，经合并并冻结后开始评测。

GPT-6 Astra 的完整 800 回合 Overall 仅在八个 Task/Swap 分项全部完成后报告。
完整八项范围由 Long 200 回合与补充评测 600 回合组成；各已报告批次使用
对应的预先冻结记忆库，这些批次并非使用同一份记忆快照。

**2026-09-14 14:00:55 UTC** 核验报告中，本次只发布六个完整套件、600 回合。
:doc:`逐任务与 seed 结果表 <results/libero_pro_astra>` 覆盖这 600 个已完成位置，
共 554 成功、46 失败；不发布未完成套件的部分数据。

.. astra-supplementary-source-end


.. _benchmark-demo:

演示视频
--------

**RPent 仿真演示：GPT-6 Astra 与 GPT-5.6 xhigh，4 倍速播放。**
此视频为独立演示，与 GPT-5.5 统计记录分开说明。

.. image:: https://raw.githubusercontent.com/RLinf/misc/e858f627dbcc1b35440c3cb5ecaaefd016eb8680/rpent/demo/demo-poster.jpg
   :alt: RPent simulation demo
   :width: 100%
   :target: https://raw.githubusercontent.com/RLinf/misc/e858f627dbcc1b35440c3cb5ecaaefd016eb8680/rpent/demo/demo.mp4

查看或下载`完整 MP4（约 24 秒，5.4 MiB） <https://raw.githubusercontent.com/RLinf/misc/e858f627dbcc1b35440c3cb5ecaaefd016eb8680/rpent/demo/demo.mp4>`_。

“未报告”表示没有对应成绩，不能视为零。

结果维护
--------

图片和演示媒体统一存放在 `RLinf/misc <https://github.com/RLinf/misc>`_ 的
``rpent/`` 目录。本页引用不可变媒体 commit
``e858f627dbcc1b35440c3cb5ecaaefd016eb8680``；配套的
`结果快照 <https://raw.githubusercontent.com/RLinf/misc/e858f627dbcc1b35440c3cb5ecaaefd016eb8680/rpent/benchmarks/results.json>`_ 记录数据来源。
交互展示代码 ``leaderboard.js`` 与 ``leaderboard.css`` 也由 misc 维护；
RPent 只接入固定版本的展示资源。图表直接读取同一 JSON 快照。
JavaScript 或网络不可用时，原生结果表仍然可读。构建文档无需绘图工具。

更新成绩时同步修改中英文表格，保留评测范围与来源精度，并将对应图片提交到 misc。
确认图片与表格一致后再更新媒体 commit。不要从四舍五入的百分比推算成功次数，
也不要把缺失结果当作零。保持模型列顺序、RoboCasa 任务加权口径，以及完整
LIBERO-PRO 与六项或零样本比较的区别。维护说明集中在本文档页面，不再增加
代码目录中的独立 README。

.. toctree::
   :hidden:

   results/libero_pro_astra
