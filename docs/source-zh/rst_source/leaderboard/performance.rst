:html_theme.sidebar_secondary.remove:

评测成绩
========

.. rpent-leaderboard:: performance

.. _libero-pro-empty-memory-evaluation:

LIBERO-PRO 空记忆评测
---------------------

本实验于 2026 年 10 月评测 RPent 在冻结 Pi0.5 权重、使用空本地经验库时的任务完成能力。规划器为 Codex / ``gpt-6.1-sol`` / ``low``，使用常规感知与操作工具。分层清单共有 240 个任务与 seed 组合，226 项可评测，其中成功 76 次，成功率为 33.63%。排行榜以 ``RPent / GPT-6.1 Sol / no memory (stratified sample)`` 标识该配置，并保留各分项有效分母。

该结果衡量的是使用 Pi0.5、但不读取回合经验的智能体，仍包含规划器决策、感知和脚本操作工具，并非 Pi0.5 策略单独推理。动作模型为 ``RLinf-Pi05-LIBERO-130-fullshot-SFT``，评测时冻结权重，没有针对 LIBERO-PRO 额外微调。原始 Pi0.5 参考行与本实验的 SFT checkpoint 不应视作相同实验条件。

配置与评测范围
~~~~~~~~~~~~~~

实验配置为普通 evaluation，每个任务与 seed 组合运行一个 planner session，不启用探索模式。历史输出目录存在重复调用，其处理方式见“证据与限制”。

.. list-table:: 评测配置
   :header-rows: 1

   * - 设置
     - 记录的实验条件
   * - 环境
     - LIBERO-PRO；Spatial、Object、Goal、Long 各包含 Task/Swap 变体
   * - 规划器
     - Codex，``gpt-6.1-sol``，推理强度 ``low``
   * - 动作模型
     - 冻结的 ``RLinf-Pi05-LIBERO-130-fullshot-SFT``
   * - 感知与工具
     - SAM3，以及 RPent 常规 LIBERO 感知、Pi0.5 与脚本动作工具
   * - 记忆
     - ``--memory-profile local``；独立目录只包含空 ``MEMORY.md``，不读取 HF 或历史任务经验
   * - 静态指令
     - 仓库的感知、标定和工具说明仍然可用
   * - 预算
     - 最多 100 个 planner turn，planner timeout 为 1,200 秒，不加 ``--explore``
   * - 运行方式
     - A800 80 GB，4 路并行；各任务与 seed 组合使用独立输出目录
   * - 采样
     - 各套件变体包含 task 0–9；Spatial 使用 seed 0–5，其余使用 seed 0–1

清单包含 80 个不同的任务与变体配置、240 个名义回合，不是 240 种不同任务。Spatial 两类共 120 个名义回合，其余六类各 20 个。总体按有效回合做微平均，因此 Spatial 的权重高于八类等样本量评测。

八类任务结果
~~~~~~~~~~~~

成功依据记录的环境布尔字段 ``terminated`` / ``libero_terminated``。24 项明确的 planner timeout 计为失败；14 项记录为官方 PRO 初始状态缺失，单列为不可评测，不计入成功率分母。下表“失败”包含 timeout。

.. list-table:: 空记忆八类分项结果
   :header-rows: 1

   * - 套件变体
     - Seed
     - 名义回合
     - 有效回合
     - 成功
     - 失败
     - Timeout
     - 不可评测
     - 成功率
   * - Spatial Task
     - 0–5
     - 60
     - 48
     - 19
     - 29
     - 4
     - 12
     - 39.58%
   * - Spatial Swap
     - 0–5
     - 60
     - 60
     - 28
     - 32
     - 4
     - 0
     - 46.67%
   * - Object Task
     - 0–1
     - 20
     - 20
     - 7
     - 13
     - 2
     - 0
     - 35.00%
   * - Object Swap
     - 0–1
     - 20
     - 20
     - 6
     - 14
     - 2
     - 0
     - 30.00%
   * - Goal Task
     - 0–1
     - 20
     - 20
     - 7
     - 13
     - 2
     - 0
     - 35.00%
   * - Goal Swap
     - 0–1
     - 20
     - 20
     - 5
     - 15
     - 7
     - 0
     - 25.00%
   * - Long Task
     - 0–1
     - 20
     - 18
     - 2
     - 16
     - 1
     - 2
     - 11.11%
   * - Long Swap
     - 0–1
     - 20
     - 20
     - 2
     - 18
     - 2
     - 0
     - 10.00%
   * - 总体
     - 分层采样
     - 240
     - 226
     - 76
     - 150
     - 24
     - 14
     - 33.63%

Long Task/Swap 对应 ``libero_10_task`` / ``libero_10_swap``。不可评测项为 Spatial Task 的 task 3、7 各 seed 0–5，以及 Long Task 的 task 2、seed 0–1。它们保留在名义清单中，不作为模型失败。非 Spatial 批次为 29/118（24.58%），89 次失败包含 16 次 timeout。

Spatial 专项对照
~~~~~~~~~~~~~~~~

Spatial Task 和 Spatial Swap 覆盖最充分，每个任务使用 6 个 seed，其余六类使用 2 个 seed。两类合计为 47/108（43.52%），失败 61 次，其中 8 次 timeout；另有 12 项不可评测。这是本实验中采样最充分的空记忆能力记录。

.. list-table:: Spatial 空记忆结果与已发布的记忆辅助参考
   :header-rows: 1

   * - 配置
     - Spatial Task
     - Spatial Swap
     - 覆盖
   * - RPent / GPT-6.1 Sol / low / 空记忆
     - 19/48（39.58%）
     - 28/60（46.67%）
     - Seed 0–5；Spatial Task 有 12 项不可评测
   * - RPent / GPT-5.5 / xhigh / 已发布记忆
     - 81.0%
     - 69.0%
     - 已发布结果，各套件变体 100 回合
   * - RPent / GPT-6 Astra / low / 已发布记忆
     - 100/100（100%）
     - 98/100（98%）
     - 已发布结果，各套件变体 100 回合

以上是与本页已发布结果的描述性对照，规划器、记忆来源、采样、运行版本和推理条件存在差异。每任务使用 6 个 seed 并不等于严格配对的记忆消融；目前没有确认一组使用同一 GPT-6.1 Sol 配置、相同任务与 seed 清单的开启记忆评测，因此不能把差值全部归因于记忆。33.63% 的分层样本总体成功率也不能替代八类等样本量的 800 回合成绩。

证据与限制
~~~~~~~~~~

贡献者汇总记录位于 ``03-memory-tuned-evaluation/`` 下的 ``final_record_20261008/``，主要文档为 ``03-LIBERO_PRO_PI05_TEST_PROTOCOL.md``、``07-PROGRESS_LIBERO_PRO_NO_MEMORY_240.md`` 和 ``COMPARISON_LIBERO_PRO_NO_MEMORY.md``。排行榜的 :download:`results.json <../../../_static/leaderboard/results.json>` 中以 ``contributor-libero-pro-no-memory-20261008`` 记录八类计数、采样范围、timeout 口径和审计清单 SHA-256。原始日志、checkpoint 和生成的逐回合输出不放入代码仓库。

来源核对包含 201 份终态 JSON、24 项 timeout 日志失败，以及 1 项仅由 transcript 结算的失败。Long Swap task 8 / seed 0 没有 canonical task JSON，其串接 transcript 的最后一个可解析对象记录失败，并包含 ``terminated:false``，因此保留为失败，不从分母删除。14 个不可评测项中，8 项在核对的结果树里保留诊断日志；Spatial Task 7 的 seed 0–5 依据汇总的不可评测记录。

调度器曾因缺少 canonical JSON 而重跑已结束的任务，输出中存在重复调用及串接 transcript。统计采用归档终态记录；上述特殊回合采用最后一个可解析 transcript。若终态 JSON 已包含有效布尔结果，则优先于同一路径日志中的较早 timeout 字符串。这段选择历史意味着本归档不能解释为严格单次运行或控制变量的开启/关闭记忆实验。24/39 等历史探索结果单独保存，不合并进本次 240 行。

历史运行的确切源码、仿真器和资产版本尚未独立确认；空经验库也不代表模型未经过训练或没有静态指令。本结果不推导耗时、token 开销或记忆效果的统计结论。

运行空记忆条件
~~~~~~~~~~~~~~

先完成 :doc:`../simulators/libero` 和规划器配置。下面展示记录中的单个任务命令形式，并显式指定 planner 预算。覆盖本次实验需要遍历前述任务与 seed 清单、为各组合使用独立输出目录，并记录实际代码、资源和 checkpoint 版本；这不是对历史运行逐位复现的保证。

.. code-block:: bash

   NO_MEMORY_DIR="$(mktemp -d)"
   printf '# Empty memory\n' > "$NO_MEMORY_DIR/MEMORY.md"

   rpent --robot libero \
     --suite libero_spatial_task --task 0 --seed 0 \
     --libero-type pro --planner codex \
     --model gpt-6.1-sol --reasoning-effort low \
     --memory-profile local --memory-dir "$NO_MEMORY_DIR" \
     --max-turns 100 --planner-timeout-s 1200 \
     --cuda-device 0 --output-dir ./outputs/no-memory/spatial_task_t0_s0

评测全程保持经验库为空，不加 ``--explore``，也不替换为默认 HF 记忆。使用记录的冻结动作模型与感知模型权重。后续运行应固定任务与 seed 覆盖、规划器、动作模型、prompt、工具预算和失败结算方式；新运行单独记录，不静默替换历史结果。
