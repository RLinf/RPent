:html_theme.sidebar_secondary.remove:

.. _benchmark-results:
.. _benchmark-leaderboard:
.. _leaderboard:

RPent Leaderboard
==========================================================================================

.. raw:: html

   <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/docs.css">
   <script defer src="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/table-sort.js"></script>
   <script defer src="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/leaderboard.js"></script>
   <script defer src="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/embed.js"></script>
   <div id="rpent-interactive-leaderboard" data-language="zh"
        data-results-url="https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/results.json">
     <div class="rpent-static-leaderboard">
       <p>完整结果表格见下文。</p>
     </div>
   </div>

**Codex / GPT-6 Astra / low / reasoning：Overall 92.63%（741/800）。**
:doc:`完整 80 任务与 800 个 seed 结果 <results/libero_pro_astra>`。

论文 v4、仓库复现与新增实验分别保留模型配置和来源，不视为仅替换模型的受控对比。缺失结果不等于零。各评测范围分别展示，表格按成功率降序排列。

``xhigh``、``max``、``low`` 是提供方的 effort 设置，不表示相等算力预算。全部 GPT-6 Astra 条目均使用 low 并开启 reasoning；GPT-5.6 无推理对照单独保留。

.. _libero-series:

LIBERO series
------------------------------------------------------------------------------------------

LIBERO · 标准操作评测
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

原始、无扰动 LIBERO。外部方法保留原论文的精度和评测范围。

.. csv-table:: 总体
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "AtomVLA", "97.0%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "96.0%", "384/400", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π_RLinf", "95.3%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π0", "94.2%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "NORA", "79.5%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "OpenVLA", "76.5%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "未报告", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-6 Astra / low / reasoning", "未报告", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"

.. csv-table:: Spatial
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "π_RLinf", "99.0%", "已评测：100", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "97.0%", "已评测：100", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π0", "96.8%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "AtomVLA", "96.4%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "NORA", "85.6%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "OpenVLA", "84.7%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "未报告", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-6 Astra / low / reasoning", "未报告", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"

.. csv-table:: Object
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Claude Code / Opus-4.8 / max / reasoning", "100.0%", "已评测：100", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "AtomVLA", "99.6%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π0", "98.8%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π_RLinf", "96.0%", "已评测：100", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "NORA", "89.4%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "OpenVLA", "88.4%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "未报告", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-6 Astra / low / reasoning", "未报告", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"

.. csv-table:: Goal
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "AtomVLA", "97.6%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π_RLinf", "97.0%", "已评测：100", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π0", "95.8%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "94.0%", "已评测：100", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "NORA", "80.0%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "OpenVLA", "79.2%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "未报告", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-6 Astra / low / reasoning", "未报告", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"

.. csv-table:: Long
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "AtomVLA", "94.4%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "93.0%", "已评测：100", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π_RLinf", "89.0%", "已评测：100", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π0", "85.2%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "NORA", "63.0%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "OpenVLA", "53.7%", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "未报告", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-6 Astra / low / reasoning", "未报告", "—", "`表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"

.. _libero-pro-long:
.. _libero-pro-across-task-families:

LIBERO-PRO · 指令与布局扰动
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

八套件 Overall 包含全部 Task/Swap 分项。新增模型分别探索与测试。Cap-X、RATS 保留已报告分项，不推算缺失的 Long 成绩或八套件 Overall。

.. csv-table:: 总体
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "92.63%", "741/800", "`GPT-6 Astra Long 评测 <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; `GPT-6 Astra 补充评测 <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; 用户确认的模型配置"
   "Claude Code / Opus 4.7", "82.50%", "已评测：800", "新增实验 · 2026-09-15"
   "Claude Code / Opus-4.8 / max / reasoning", "82.4%", "已评测：800", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.6 / xhigh / reasoning", "78.50%", "已评测：800", "新增实验 · 2026-09-15; 用户确认的模型配置"
   "Task card / Molmo", "72.63%", "已评测：800", "新增实验 · 2026-09-15"
   "Codex / GPT-5.5 / xhigh / reasoning", "72.1%", "已评测：800", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.6 / 无推理", "62.50%", "已评测：800", "新增实验 · 2026-09-15"
   "π_RLinf", "50.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "11.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "6.3%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "3.8%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "1.5%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.3%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Qwen", "未报告", "—", "新增实验 · 2026-09-15"

.. csv-table:: Spatial Task
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "100%", "100/100", "`GPT-6 Astra 补充评测 <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; 用户确认的模型配置"
   "Claude Code / Opus-4.8 / max / reasoning", "94.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "81.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "42.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "31.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "14.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "1.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "1.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Spatial Swap
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "98%", "98/100", "`GPT-6 Astra 补充评测 <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; 用户确认的模型配置"
   "Claude Code / Opus-4.8 / max / reasoning", "80.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "69.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "59.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "29.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "20.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "16.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "12.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Object Task
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "100%", "100/100", "`GPT-6 Astra 补充评测 <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; 用户确认的模型配置"
   "Codex / GPT-5.5 / xhigh / reasoning", "94.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "88.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "71.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "63.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "18.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "8.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "1.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Object Swap
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "99%", "99/100", "`GPT-6 Astra 补充评测 <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; 用户确认的模型配置"
   "Codex / GPT-5.5 / xhigh / reasoning", "91.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "90.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "78.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "61.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "22.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "17.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "10.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "6.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "2.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "2.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Goal Task
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "88%", "88/100", "`GPT-6 Astra 补充评测 <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; 用户确认的模型配置"
   "Claude Code / Opus-4.8 / max / reasoning", "87.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "75.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "45.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "36.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "17.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "11.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "9.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "2.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Goal Swap
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "99%", "99/100", "`GPT-6 Astra 补充评测 <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; 用户确认的模型配置"
   "Claude Code / Opus-4.8 / max / reasoning", "87.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "66.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "43.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "42.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "38.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "26.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "2.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "1.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Long Task
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "85%", "85/100", "`GPT-6 Astra Long 评测 <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; 用户确认的模型配置"
   "Claude Code / Opus-4.8 / max / reasoning", "71.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "52.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "49.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "10.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "9.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "6.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "1.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "未报告", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "未报告", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Long Swap
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "72%", "72/100", "`GPT-6 Astra Long 评测 <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; 用户确认的模型配置"
   "Claude Code / Opus-4.8 / max / reasoning", "62.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "49.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "14.0%", "已评测：100", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "8.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "1.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "0.0%", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "未报告", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "未报告", "—", "`表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

LIBERO-PRO · Goal · 零样本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

不使用目标设置的 Task Specific Memory 或 Global Memory。此消融与有记忆的八套件评测分开展示，下方包含表 5 的逐任务成绩。

.. csv-table:: Goal Task
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Claude Code / Opus-4.8 / max / reasoning", "79.0%", "已评测：100", "`表 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"
   "Cap-X", "16.8%", "—", "`表 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "未报告", "—", "`表 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"
   "Codex / GPT-6 Astra / low / reasoning", "未报告", "—", "`表 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"

.. csv-table:: Goal Swap
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Claude Code / Opus-4.8 / max / reasoning", "31.0%", "已评测：100", "`表 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"
   "Cap-X", "25.6%", "—", "`表 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "未报告", "—", "`表 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"
   "Codex / GPT-6 Astra / low / reasoning", "未报告", "—", "`表 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"

.. csv-table:: LIBERO-PRO Goal Task · 零样本，表 5
   :header: "方法", "任务 0", "任务 1", "任务 2", "任务 3", "任务 4", "任务 5", "任务 6", "任务 7", "任务 8", "任务 9", "均值"
   :class: table-sm

   "Opus-4.8", "10.0%", "100.0%", "90.0%", "100.0%", "20.0%", "80.0%", "90.0%", "100.0%", "100.0%", "100.0%", "79.0%"
   "Cap-X", "0.0%", "0.0%", "10.0%", "38.0%", "12.0%", "4.0%", "34.0%", "12.0%", "40.0%", "18.0%", "16.8%"

.. csv-table:: LIBERO-PRO Goal Swap · 零样本，表 5
   :header: "方法", "任务 0", "任务 1", "任务 2", "任务 3", "任务 4", "任务 5", "任务 6", "任务 7", "任务 8", "任务 9", "均值"
   :class: table-sm

   "Opus-4.8", "0.0%", "10.0%", "0.0%", "20.0%", "90.0%", "0.0%", "10.0%", "80.0%", "100.0%", "0.0%", "31.0%"
   "Cap-X", "0.0%", "4.0%", "0.0%", "36.0%", "22.0%", "60.0%", "4.0%", "2.0%", "62.0%", "66.0%", "25.6%"

RoboCasa365 · Target50
------------------------------------------------------------------------------------------

RPent Overall 对 50 个任务等权，不是按回合汇总。GPT-6 Astra 已完成 340 回合，仅提供 Overall。外部基线按表 4 引用，不声称采用相同 Target50 范围。仓库复现仍在下方独立保留。

.. csv-table:: 总体（原报告口径）
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "59.20%", "已评测：340", "新增实验 · 2026-09-15; 用户确认的模型配置"
   "Codex / GPT-5.5 / xhigh / reasoning", "57.1%", "已评测：340", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "48.6%", "已评测：340", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "WorldDreamer", "35.3%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "RLDX-1", "30.0%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0.5", "16.9%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0", "14.8%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Codex / GPT-5.6 / 无推理", "未报告", "—", "新增实验 · 2026-09-15"

.. csv-table:: Atomic-Seen
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-5.5 / xhigh / reasoning", "92.0%", "已评测：180", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "79.4%", "已评测：180", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "WorldDreamer", "66.3%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "RLDX-1", "60.0%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0.5", "39.6%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0", "34.6%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Codex / GPT-6 Astra / low / reasoning", "未报告", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"

.. csv-table:: Composite-Seen
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-5.5 / xhigh / reasoning", "61.0%", "已评测：80", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "47.5%", "已评测：80", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "WorldDreamer", "26.7%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "RLDX-1", "21.3%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0.5", "7.1%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0", "6.1%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Codex / GPT-6 Astra / low / reasoning", "未报告", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"

.. csv-table:: Composite-Unseen
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Claude Code / Opus-4.8 / max / reasoning", "15.0%", "已评测：80", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "13.8%", "已评测：80", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "WorldDreamer", "9.0%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "RLDX-1", "5.0%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0.5", "1.2%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0", "1.1%", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Codex / GPT-6 Astra / low / reasoning", "未报告", "—", "`表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"

RoboTwin · Clean → Randomized
------------------------------------------------------------------------------------------

RPent：50 个双臂任务，各 5 个随机 seeds。干净设置记忆直接迁移，不在随机设置中探索。外部方法保留其来源口径。

.. csv-table:: C2R
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "72.00%", "已评测：250", "贡献者 · RoboTwin C2R 成绩; 用户确认的模型配置"
   "Codex / GPT-5.5 / xhigh / reasoning · 仓库 · 2026-09-15", "62.4%", "156/250", "`RPent · RoboTwin 复现，2026-09-15 <https://github.com/RLinf/RPent/blob/849143bf740a562367345cec0de9ef4657dafd73/docs/source-en/rst_source/usage/robotwin.rst>`_"
   "Codex / GPT-5.6 / xhigh / reasoning", "61.20%", "已评测：250", "贡献者 · RoboTwin C2R 成绩; 用户确认的模型配置"
   "Claude Code / Opus-4.8 / max / reasoning", "58.4%", "已评测：250", "`表 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_"
   "Codex / GPT-5.5 / xhigh / reasoning · 论文 / 早期仓库", "58.0%", "145/250", "`表 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_; `RPent · RoboTwin 复现 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robotwin.rst>`_"
   "LingBot-VLA", "50.4%", "—", "`表 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_"
   "π0.5", "47.9%", "—", "`表 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_"
   "GR00T-N1.7", "20.7%", "—", "`表 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_"
   "StarVLA", "10.6%", "—", "`表 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_"

Task Card · Object · 仓库评测
------------------------------------------------------------------------------------------

仅 Object Task 与 Swap，此 200 回合报告不同于 Task card / Molmo 八套件 72.63%。

.. csv-table:: Object · Task + Swap
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / 无推理", "93.0%", "186/200", "`RPent · Task Card 评测 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/task_card.rst>`_"
   "Task card", "89.5%", "179/200", "`RPent · Task Card 评测 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/task_card.rst>`_"

RoboCasa365 · 仓库复现
------------------------------------------------------------------------------------------

GPT-5.6 xhigh 开启推理：Atomic 163/180、Seen 49/80、Unseen 12/80，任务等权 Overall 57.00%。早期 GPT-5.5 xhigh 开启推理为 55.40%，与论文 v4 的 57.1% 分开。GPT-6 Astra low 开启推理为 59.20%，与主榜为同一实验。下方 50 任务汇总仅属于 57.00% 这次实验。

.. csv-table:: 总体
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "59.20%", "已评测：340", "新增实验 · 2026-09-15; 用户确认的模型配置"
   "Codex / GPT-5.6 / xhigh / reasoning · 仓库复现", "57.00%", "已评测：340", "`RPent · Target50 复现 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/robots/robocasa/eval/target50_codex_results.md>`_; 用户确认的模型配置"
   "Codex / GPT-5.5 / xhigh / reasoning · 早期参考", "55.40%", "已评测：340", "`RPent · 早期参考列 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robocasa.rst>`_; 用户确认的模型配置"

.. csv-table:: Atomic
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-5.5 / xhigh / reasoning · 早期参考", "91.67%", "165/180", "`RPent · 早期参考列 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robocasa.rst>`_; 用户确认的模型配置"
   "Codex / GPT-5.6 / xhigh / reasoning · 仓库复现", "90.56%", "163/180", "`RPent · Target50 复现 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/robots/robocasa/eval/target50_codex_results.md>`_; 用户确认的模型配置"

.. csv-table:: Composite-Seen
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-5.6 / xhigh / reasoning · 仓库复现", "61.25%", "49/80", "`RPent · Target50 复现 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/robots/robocasa/eval/target50_codex_results.md>`_; 用户确认的模型配置"
   "Codex / GPT-5.5 / xhigh / reasoning · 早期参考", "56.25%", "45/80", "`RPent · 早期参考列 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robocasa.rst>`_; 用户确认的模型配置"

.. csv-table:: Composite-Unseen
   :header: "方法 / 模型 / 配置", "成功率", "成功 / 评测回合", "来源"
   :class: table-sm

   "Codex / GPT-5.6 / xhigh / reasoning · 仓库复现", "15.00%", "12/80", "`RPent · Target50 复现 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/robots/robocasa/eval/target50_codex_results.md>`_; 用户确认的模型配置"
   "Codex / GPT-5.5 / xhigh / reasoning · 早期参考", "13.75%", "11/80", "`RPent · 早期参考列 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robocasa.rst>`_; 用户确认的模型配置"

论文分析
------------------------------------------------------------------------------------------

调用次数、完成归因与 primitive 占比用于分析机制，不参与成功率排名。保留已发表原图，不估读曲线点值。Zhang 等，Harness VLA v4，CC BY 4.0。

.. container:: rpent-paper-figures

   .. figure:: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/invocations-libero.png
      :alt: LIBERO-PRO cumulative success by VLA invocation count
      :width: 520px
      :class: rpent-paper-figure
      :target: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/invocations-libero.png

      LIBERO-PRO

   .. figure:: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/invocations-robocasa.png
      :alt: RoboCasa365 cumulative success by VLA invocation count
      :width: 520px
      :class: rpent-paper-figure
      :target: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/invocations-robocasa.png

      RoboCasa365

   .. figure:: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/invocations-robotwin.png
      :alt: RoboTwin C2R cumulative success by VLA invocation count
      :width: 520px
      :class: rpent-paper-figure
      :target: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/invocations-robotwin.png

      RoboTwin C2R

.. image:: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/completion-attribution.png
   :alt: Harness VLA completion-attribution.png
   :width: 520px
   :align: center
   :class: rpent-paper-figure
   :target: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/completion-attribution.png

.. csv-table:: Primitive 使用统计 · 表 18
   :header: "Primitive", "LIBERO", "RoboTwin C2R", "RoboCasa365"
   :class: table-sm

   "MOVE_TO", "6263 (61.8%)", "685 (40.9%)", "3004 (38.7%)"
   "VLA_ACT", "1598 (15.8%)", "794 (47.4%)", "2746 (35.3%)"
   "SET_GRIPPER", "1137 (11.2%)", "71 (4.2%)", "371 (4.8%)"
   "RELEASE", "831 (8.2%)", "124 (7.4%)", "76 (1.0%)"
   "MOVE_POSE", "203 (2.0%)", "—", "—"
   "ROTATE_PITCH", "58 (0.6%)", "—", "66 (0.8%)"
   "ROTATE_WRIST", "44 (0.4%)", "1 (0.1%)", "—"
   "MOVE_BASE", "—", "—", "808 (10.4%)"
   "NAVIGATE_TO", "—", "—", "701 (9.0%)"
   "合计", "10134 (100.0%)", "1675 (100.0%)", "7772 (100.0%)"

.. csv-table:: Primitive 分类 · 表 19
   :header: "类别", "LIBERO", "RoboTwin C2R", "RoboCasa365"
   :class: table-sm

   "Analytic", "8536 (84.2%)", "881 (52.6%)", "5026 (64.7%)"
   "VLA", "1598 (15.8%)", "794 (47.4%)", "2746 (35.3%)"

协议与来源
------------------------------------------------------------------------------------------

任务成功以环境为准：LIBERO ``terminated``、RoboCasa ``state.success`` 或 RoboTwin ``TASK_ENV.eval_success``。Planner finish 不作为任务成功来源，探索回合不计入评测。冻结的 VLA 后端分别为 RLinf π0.5（LIBERO）、RLDX-1（RoboCasa）与后训练 LingBot-VLA（RoboTwin）。

.. _benchmark-source-p2:
.. _benchmark-source-r2:
.. _benchmark-source-p3:
.. _benchmark-source-r3:
.. _benchmark-source-p4:
.. _benchmark-source-p5:
.. _benchmark-source-r4:
.. _benchmark-source-p6:
.. _benchmark-source-r1:
.. _benchmark-protocol-r1:
.. _benchmark-source-astra-pro:

* `表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_
* `表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_
* `表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_
* `表 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_
* `表 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_
* `GPT-6 Astra Long 评测 <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_
* `GPT-6 Astra 补充评测 <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_
* 新增实验 · 2026-09-15 — 用户提供的结果图片，统计口径已明确确认：LIBERO-PRO 各模型完整八套件探索与测试；RoboCasa 全量 340 回合、50 任务等权 Overall。未提供逐回合结果及部分模型参数。
* `RPent · Task Card 评测 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/task_card.rst>`_ — Object Task 与 Swap，20 个任务各 10 seeds。原文提供汇总成功数，未指定本次评测的模型版本；与新增八套件 Molmo 结果分开。
* `RPent · RoboTwin 复现 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robotwin.rst>`_ — 仓库同样报告 Codex / GPT-5.5 xhigh 的 58.0%，并明确提供 145/250。该条成绩同时附上两个公开来源，不声称已核对逐回合一致性。
* `RPent · Target50 复现 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/robots/robocasa/eval/target50_codex_results.md>`_ — 已公开的 50 任务汇总记录，不是逐回合轨迹。与论文 v4 实验分别保留。
* `RPent · 早期参考列 <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robocasa.rst>`_ — 仓库文档保留的早期参考计数，与论文 v4 和仓库复现分别标注。
* `论文 · 图 4 <https://arxiv.org/html/2607.08448v4#S3.F4>`_
* `论文 · 图 6 <https://arxiv.org/html/2607.08448v4#S3.F6>`_
* `论文 · 表 18 <https://arxiv.org/html/2607.08448v4#A6.T18>`_
* `论文 · 表 19 <https://arxiv.org/html/2607.08448v4#A6.T19>`_
* 用户确认的模型配置 — 仅用于配置说明：GPT-5.6 推理使用 xhigh，GPT-6 Astra 使用 low 并开启 reasoning。62.50% 无推理对照不变，RoboCasa 复现实验模型标签由用户确认。成绩证据另行引用。
* 贡献者 · RoboTwin C2R 成绩 — RoboTwin C2R：GPT-5.6 xhigh 为 61.20%，GPT-6 Astra low 为 72.00%，均开启推理，各 250 回合。未提供精确成功数和逐任务结果。
* `RPent · RoboTwin 复现，2026-09-15 <https://github.com/RLinf/RPent/blob/849143bf740a562367345cec0de9ef4657dafd73/docs/source-en/rst_source/usage/robotwin.rst>`_ — Main #179 报告 GPT-5.5 xhigh：250 回合，156 成功、58 次任务失败、36 次超时，成功率 62.4%。表内 task/seed 绑定评测表的任务语言。与论文及早期 58.0% 分批保留。

* 原始 LIBERO；四套件，每套 100 回合；冻结 RLinf π0.5 策略。
* 使用目标设置记忆的 LIBERO-PRO；八个 Task/Swap 分项，每项 100 回合。
* Long 评测；seed-0 记忆冻结后测试 seed 1–10。运行版本 014a0fa，与 Spatial/Object/Goal 评测属于不同批次。
* Goal Task/Swap 不使用目标设置 Task Specific Memory 和 Global Memory；每项 100 回合。
* Target50：18 个原子任务各 10 seeds，16 个已见与 16 个未见复合任务各 5 seeds；总体为 50 个任务均权。
* 50 个任务各 5 个专家验证随机 seeds；记忆从 demo_clean 迁移至 demo_randomized。
* 四个标准套件；样本量沿用各外部报告。
* 八个 LIBERO-PRO Task/Swap 分项；样本量沿用各外部报告。
* 仅六个 Spatial/Object/Goal Task/Swap 分项，不含 Long。
* 无目标设置记忆的 Goal 消融；样本量沿用外部报告。
* 表 4 报告的基线数值。RLDX-1 为论文协议下直接评测的冻结策略，其他基线取自先前论文；未确认这些外部报告具有相同任务范围及聚合方式。
* 从干净设置到随机设置迁移；样本量沿用各外部报告。
* Spatial/Object/Goal Task 与 Swap；每项 10 个任务，各使用 seed 1–10。Seed 0 任务记忆经合并冻结后开始评测。
* 800 回合：Long 200 回合，Spatial/Object/Goal 600 回合。各批次使用对应的预先冻结记忆库；八个分项均为 100 回合。
* 完整 LIBERO-PRO 八套件，各模型独立完成探索与全量测试。不从 Overall 反推成功次数或子套件成绩。
* Target50 全量 340 回合，Overall 对 50 个任务等权。GPT-6 Astra 使用 low effort 并开启 reasoning；未提供 split 成绩及成功次数。
* Object Task + Object Swap；20 任务、200 回合。Task Card 179/200；Codex 无推理 186/200。不是八套件 Overall。
* Target50 仓库复现：18 个 Atomic 任务各 10 seeds，16 个 Seen 与 16 个 Unseen 任务各 5 seeds。Overall 按任务等权。
* 仓库中的早期参考列，不是 v4 论文实验。保留原有 split 计数和任务等权 Overall。
* RoboTwin C2R：50 任务各 5 seeds，每配置 250 回合。GPT-5.6 xhigh 开启推理，GPT-6 Astra low 开启推理，不从百分比反推成功次数。
* 50 任务，每任务 5 个 verified expert seeds，共 250 回合。场景 reset 后绑定评测表任务语言，只认最终 TASK_ENV.eval_success。Planner 时限 4800 秒，环境上限 10000 步。来源：RPent #179。

2026-09-15 核验的 Astra 快照包含完整八套件：741 成功、59 失败，共 800 回合。Long 与 Spatial/Object/Goal 分别使用各自冻结的 memory 批次，不声称共用一个快照。旧 600 回合快照留档，不作为第二次实验重复排名。

.. _benchmark-demo:

演示
------------------------------------------------------------------------------------------

**RPent 仿真演示：GPT-6 Astra low 开启推理，对比 GPT-5.6 xhigh，4 倍速。** 独立演示，不属于 GPT-5.5 评测。

.. raw:: html

   <video class="rpent-demo-video" controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/demo/demo-poster.jpg">
     <source src="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/demo/demo.mp4" type="video/mp4">
     <a href="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/demo/demo.mp4">下载演示视频</a>
   </video>

查看或下载 `完整 MP4（约 24 秒，5.4 MiB） <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/demo/demo.mp4>`_ 。

维护结果
------------------------------------------------------------------------------------------

展示代码、图片、数据和演示媒体位于 `RLinf/misc <https://github.com/RLinf/misc>`_，固定 commit ``c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1``。无 JavaScript 或远端资源时，原生表格仍可阅读。交互组件使用同一份 JSON 和隔离样式；RPent 文档构建不需要绘图代码。

更新时应使用核验后的数据同步中英文表格，保留模型与协议边界及缺失值，验证图表一致后再更新不可变资源版本。不从取整百分比推断未报告的成功次数。

.. toctree::
   :hidden:

   results/libero_pro_astra
