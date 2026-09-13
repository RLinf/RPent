:html_theme.sidebar_secondary.remove:

.. _benchmark-results:

基准测试结果
============

本页汇总 RPent 在不同仿真基准上的成功率，并与已收录的代表性方法比较。

.. raw:: html

   <details class="rpent-page-toc"><summary>本页目录</summary>

.. contents::
   :local:
   :depth: 2
   :backlinks: none

.. raw:: html

   </details>

.. _benchmark-leaderboard:

Leaderboard
-----------

每个面板展示一个评测范围，按成功率排序；紫色表示 RPent，灰色表示外部参考方法。

可按基准、任务套件和模型配置筛选。悬停或用键盘聚焦柱子可查看详情，
也可下载当前筛选结果的 CSV。

.. raw:: html

   <div id="rpent-interactive-leaderboard" data-language="zh"
        data-results-url="../_static/benchmarks/results.json" data-asset-base="../_static/benchmarks/">
   <noscript><p>交互筛选需要 JavaScript，当前仍可查看静态图表与完整结果表。</p></noscript>
   <div class="rpent-leaderboard-legend" aria-label="图例">
     <span><i class="rpent-legend-swatch rpent-legend-rpent" aria-hidden="true"></i>RPent 模型配置</span>
     <span><i class="rpent-legend-swatch rpent-legend-reference" aria-hidden="true"></i>代表性外部方法</span>
   </div>
   <div class="rpent-leaderboard-grid rpent-static-leaderboard">
     <figure class="rpent-leaderboard-panel" data-benchmark="standard-libero">
       <img class="rpent-chart-light" src="../_static/benchmarks/standard-libero-zh-light.svg"
            alt="Standard LIBERO 总体成功率对比，完整数值见下方结果表。" />
       <img class="rpent-chart-dark" src="../_static/benchmarks/standard-libero-zh-dark.svg"
            alt="Standard LIBERO 总体成功率对比，完整数值见下方结果表。" />
       <figcaption><a href="#benchmark-source-p2">Standard LIBERO 总体：结果与评测说明</a></figcaption>
     </figure>
     <figure class="rpent-leaderboard-panel" data-benchmark="libero-pro">
       <img class="rpent-chart-light" src="../_static/benchmarks/libero-pro-zh-light.svg"
            alt="LIBERO-PRO 总体成功率对比，覆盖八项 Task／Swap，完整数值见下方结果表。" />
       <img class="rpent-chart-dark" src="../_static/benchmarks/libero-pro-zh-dark.svg"
            alt="LIBERO-PRO 总体成功率对比，覆盖八项 Task／Swap，完整数值见下方结果表。" />
       <figcaption><a href="#benchmark-source-p3">LIBERO-PRO 总体：结果与评测说明</a></figcaption>
     </figure>
     <figure class="rpent-leaderboard-panel" data-benchmark="robocasa">
       <img class="rpent-chart-light" src="../_static/benchmarks/robocasa-zh-light.svg"
            alt="RoboCasa365 Target50 任务均权总体成功率对比，完整数值见下方结果表。" />
       <img class="rpent-chart-dark" src="../_static/benchmarks/robocasa-zh-dark.svg"
            alt="RoboCasa365 Target50 任务均权总体成功率对比，完整数值见下方结果表。" />
       <figcaption><a href="#benchmark-source-p4">RoboCasa365 Target50 总体：结果与评测说明</a></figcaption>
     </figure>
     <figure class="rpent-leaderboard-panel" data-benchmark="robotwin">
       <img class="rpent-chart-light" src="../_static/benchmarks/robotwin-zh-light.svg"
            alt="RoboTwin C2R 成功率对比，完整数值见下方结果表。" />
       <img class="rpent-chart-dark" src="../_static/benchmarks/robotwin-zh-dark.svg"
            alt="RoboTwin C2R 成功率对比，完整数值见下方结果表。" />
       <figcaption><a href="#benchmark-source-p6">RoboTwin C2R：结果与评测说明</a></figcaption>
     </figure>
     <figure class="rpent-leaderboard-panel" data-benchmark="long-task">
       <img class="rpent-chart-light" src="../_static/benchmarks/long-task-zh-light.svg"
            alt="LIBERO-PRO Long Task 的 RPent 模型成功率对比，完整数值见下方结果表。" />
       <img class="rpent-chart-dark" src="../_static/benchmarks/long-task-zh-dark.svg"
            alt="LIBERO-PRO Long Task 的 RPent 模型成功率对比，完整数值见下方结果表。" />
       <figcaption><a href="#libero-pro-across-task-families">LIBERO-PRO Long Task：结果与评测说明</a></figcaption>
     </figure>
     <figure class="rpent-leaderboard-panel" data-benchmark="long-swap">
       <img class="rpent-chart-light" src="../_static/benchmarks/long-swap-zh-light.svg"
            alt="LIBERO-PRO Long Swap 的 RPent 模型成功率对比，完整数值见下方结果表。" />
       <img class="rpent-chart-dark" src="../_static/benchmarks/long-swap-zh-dark.svg"
            alt="LIBERO-PRO Long Swap 的 RPent 模型成功率对比，完整数值见下方结果表。" />
       <figcaption><a href="#libero-pro-across-task-families">LIBERO-PRO Long Swap：结果与评测说明</a></figcaption>
     </figure>
   </div>
   </div>

图中排名仅适用于各面板列出的评测范围。各图纵轴从零开始，上限按该设置已报告的最高成绩
向上取整到 10 个百分点的整数倍，隐藏模型后保持不变；跨面板比较时请以标注的百分比为准。
完整成绩、模型配置及协议见下方。
未报告的配置不绘制柱子。Long Task 与 Long Swap 只比较 RPent 模型，不能替代完整 PRO Overall。

.. _benchmark-demo:

演示视频
--------

**RPent 仿真演示：GPT-6 Astra 与 GPT-5.6 xhigh，4× 播放。**
此视频为独立演示，与上方 GPT-5.5 统计记录分开说明。

.. raw:: html

   <video class="rpent-benchmark-video" controls playsinline preload="metadata"
          poster="../_static/videos/demo-poster.jpg" width="2048" height="1024"
          aria-label="RPent 仿真演示：GPT-6 Astra 与 GPT-5.6 xhigh，4倍速">
     <source src="../_static/videos/demo.mp4" type="video/mp4" />
     <a href="../_static/videos/demo.mp4">下载完整演示视频</a>
   </video>
   <p class="rpent-demo-download"><a href="../_static/videos/demo.mp4" download>下载完整 MP4（约 24 秒，5.4 MiB）</a></p>

“未报告”表示当前没有该模型在相应评测项上的成绩，不能视为零。

.. _id2:

模型配置
------------

.. benchmark-data-begin: configurations

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

.. benchmark-data-end: configurations

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

.. benchmark-data-begin: standard-libero

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

.. benchmark-data-end: standard-libero

.. _libero-pro-long:

.. _libero-pro-across-task-families:

LIBERO-PRO
~~~~~~~~~~

Task 为指令重定向，Swap 为位置交换。Overall 覆盖 Spatial、Object、Goal、Long 的全部
八个 Task/Swap 单元；只完成 Long 的结果不构成该总体指标。

.. benchmark-data-begin: libero-pro

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
     - 未报告
   * - Spatial Swap
     - 69.0%
     - 80.0%
     - 未报告
   * - Object Task
     - 94.0%
     - 88.0%
     - 未报告
   * - Object Swap
     - 91.0%
     - 90.0%
     - 未报告
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

.. benchmark-data-end: libero-pro

.. _libero-pro-goal:

LIBERO-PRO Goal：零样本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

此消融不使用目标设置的 Task Specific Memory 和 Global Memory，
与上方使用记忆的 PRO 主结果分别统计。

.. benchmark-data-begin: libero-pro-zero-shot

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

.. benchmark-data-end: libero-pro-zero-shot

RoboCasa365 Target50
--------------------

列出全部三个划分及 Overall。Overall 按 50 个任务均权；各划分的回合数不同，
不能直接把所有成功回合合并求比率。

.. benchmark-data-begin: robocasa

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

.. benchmark-data-end: robocasa

RoboTwin C2R
------------

C2R 表示从干净设置到随机设置的迁移评测。

.. benchmark-data-begin: robotwin

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

.. benchmark-data-end: robotwin

.. _id3:

外部基线参考
------------------

以下附表保留来源中的其他方法，供查看相关评测覆盖范围。它们不作为 RPent 的规划模型列，
也不与主表合并计分。直接执行动作的 VLA 方法没有单独规划器的 Reasoning/Effort 设置；
Cap-X 和 RATS 的具体规划模型与推理配置在这些来源中未报告。

.. dropdown:: Standard LIBERO

   .. benchmark-data-begin: baseline-standard-libero

   .. list-table:: 参考方法
      :header-rows: 1
      :widths: 24 36 20 20

      * - 方法
        - 覆盖范围
        - 成功率
        - 来源
      * - OpenVLA
        - 四个标准套件
        - 76.5%
        - :ref:`表 2 <benchmark-source-p2>`
      * - NORA
        - 四个标准套件
        - 79.5%
        - :ref:`表 2 <benchmark-source-p2>`
      * - π0
        - 四个标准套件
        - 94.2%
        - :ref:`表 2 <benchmark-source-p2>`
      * - π_RLinf
        - 四个标准套件
        - 95.3%
        - :ref:`表 2 <benchmark-source-p2>`
      * - AtomVLA
        - 四个标准套件
        - 97.0%
        - :ref:`表 2 <benchmark-source-p2>`

   .. benchmark-data-end: baseline-standard-libero

   外部基线的样本量沿用各自来源，不统一套用 RPent 的每套件 100 回合。

.. dropdown:: LIBERO-PRO

   .. benchmark-data-begin: baseline-libero-pro

   .. list-table:: 参考方法
      :header-rows: 1
      :widths: 24 36 20 20

      * - 方法
        - 覆盖范围
        - 成功率
        - 来源
      * - OpenVLA
        - 八个 Task/Swap 单元
        - 0.0%
        - :ref:`表 3 <benchmark-source-p3>`
      * - π0
        - 八个 Task/Swap 单元
        - 0.3%
        - :ref:`表 3 <benchmark-source-p3>`
      * - π0.5
        - 八个 Task/Swap 单元
        - 11.0%
        - :ref:`表 3 <benchmark-source-p3>`
      * - MolmoAct
        - 八个 Task/Swap 单元
        - 1.5%
        - :ref:`表 3 <benchmark-source-p3>`
      * - NORA
        - 八个 Task/Swap 单元
        - 0.0%
        - :ref:`表 3 <benchmark-source-p3>`
      * - X-VLA
        - 八个 Task/Swap 单元
        - 3.8%
        - :ref:`表 3 <benchmark-source-p3>`
      * - AtomVLA
        - 八个 Task/Swap 单元
        - 6.3%
        - :ref:`表 3 <benchmark-source-p3>`
      * - π_RLinf
        - 八个 Task/Swap 单元
        - 50.0%
        - :ref:`表 3 <benchmark-source-p3>`
      * - Cap-X
        - 六个非 Long 单元
        - 18.2%
        - :ref:`表 3 <benchmark-source-p3>`
      * - RATS
        - 六个非 Long 单元
        - 43.8%
        - :ref:`表 3 <benchmark-source-p3>`

   .. benchmark-data-end: baseline-libero-pro

   Cap-X 和 RATS 仅涵盖 Spatial、Object、Goal 的 Task/Swap 六个单元，不与涵盖八个单元的 Overall 直接排名。

.. dropdown:: RoboCasa365 Target50

   .. benchmark-data-begin: baseline-robocasa

   .. list-table:: 参考方法
      :header-rows: 1
      :widths: 24 36 20 20

      * - 方法
        - 覆盖范围
        - 成功率
        - 来源
      * - RLDX-1
        - 全部三个划分，任务均权
        - 30.0%
        - :ref:`表 4 <benchmark-source-p4>`
      * - WorldDreamer
        - 全部三个划分，任务均权
        - 35.3%
        - :ref:`表 4 <benchmark-source-p4>`
      * - π0.5
        - 全部三个划分，任务均权
        - 16.9%
        - :ref:`表 4 <benchmark-source-p4>`
      * - π0
        - 全部三个划分，任务均权
        - 14.8%
        - :ref:`表 4 <benchmark-source-p4>`

   .. benchmark-data-end: baseline-robocasa

   RLDX-1 是冻结 VLA 的直接评测基线；其余方法为外部报告。Overall 保留来源报告的聚合口径。

.. dropdown:: RoboTwin C2R

   .. benchmark-data-begin: baseline-robotwin

   .. list-table:: 参考方法
      :header-rows: 1
      :widths: 24 36 20 20

      * - 方法
        - 覆盖范围
        - 成功率
        - 来源
      * - GR00T-N1.7
        - C2R
        - 20.7%
        - :ref:`表 6 <benchmark-source-p6>`
      * - π0.5
        - C2R
        - 47.9%
        - :ref:`表 6 <benchmark-source-p6>`
      * - StarVLA
        - C2R
        - 10.6%
        - :ref:`表 6 <benchmark-source-p6>`
      * - LingBot-VLA
        - C2R
        - 50.4%
        - :ref:`表 6 <benchmark-source-p6>`

   .. benchmark-data-end: baseline-robotwin

   LingBot-VLA 既是 RPent 的冻结接触策略后端，也是直接评测基线；不把 RPent 的 250 回合样本量套用到外部报告。

.. dropdown:: LIBERO-PRO Goal 零样本

   .. benchmark-data-begin: baseline-libero-pro-zero-shot

   .. list-table:: 参考方法
      :header-rows: 1
      :widths: 24 36 20 20

      * - 方法
        - 覆盖范围
        - 成功率
        - 来源
      * - Cap-X
        - Goal Task
        - 16.8%
        - :ref:`表 5 <benchmark-source-p5>`
      * - Cap-X
        - Goal Swap
        - 25.6%
        - :ref:`表 5 <benchmark-source-p5>`

   .. benchmark-data-end: baseline-libero-pro-zero-shot

   这些 Goal Task/Swap 数值对应不使用目标设置记忆的消融。

.. _id4:

指标、协议与来源
------------------------

结果核对日期为 2026-09-10。RPent 的 GPT-5.5 与 Opus-4.8 已报告成绩与
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
