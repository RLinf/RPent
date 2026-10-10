RoboCasa 评测参考
======================

本页供核对评测配置、排查记忆文件和校验历史结果时查阅。运行步骤见 :ref:`reproduce-target50`。

.. _robocasa-task-list:

完整任务列表
----------------

Seen/Unseen 指任务是否出现在预训练数据中；target 厨房场景是独立的保留场景划分，详见 `RoboCasa 数据定义 <https://robocasa.ai/docs/build/html/datasets/datasets_overview.html>`_。50 个任务分三组：

- **Atomic (18)** —— 开合、搬运等单项操作任务： ``CloseBlenderLid``、 ``CloseFridge``、``CloseToasterOvenDoor``、``CoffeeSetupMug``、 ``NavigateKitchen``、``OpenCabinet``、``OpenDrawer``、 ``OpenStandMixerHead``、``PickPlaceCounterToCabinet``、 ``PickPlaceCounterToStove``、``PickPlaceDrawerToCounter``、 ``PickPlaceSinkToCounter``、``PickPlaceToasterToCounter``、 ``SlideDishwasherRack``、``TurnOffStove``、``TurnOnElectricKettle``、 ``TurnOnMicrowave``、``TurnOnSinkFaucet``。
- **Composite seen (16)** —— 预训练数据中出现过的组合任务： ``ScrubCuttingBoard``、``StackBowlsCabinet``、``WashLettuce``、 ``RinseSinkBasin``、``PreSoakPan``、``StirVegetables``、 ``LoadDishwasher``、``SteamInMicrowave``、``SetUpCuttingStation``、 ``GetToastedBread``、``DeliverStraw``、``KettleBoiling``、 ``PrepareCoffee``、``StoreLeftoversInBowl``、``SearingMeat``、 ``PackIdenticalLunches``。
- **Composite unseen (16)** —— 预训练数据中未出现过的组合任务： ``ArrangeBreadBasket``、``ArrangeTea``、 ``BreadSelection``、``CategorizeCondiments``、 ``CuttingToolSelection``、``GarnishPancake``、``GatherTableware``、 ``HeatKebabSandwich``、``MakeIceLemonade``、``PanTransfer``、 ``PortionHotDogs``、``RecycleBottlesByType``、 ``SeparateFreezerRack``、``WaffleReheat``、``WashFruitColander``、 ``WeighIngredients``。

任选一个传给 ``--task-name`` 即可。RoboCasa 完整目录更大，参见 `RoboCasa <https://robocasa.ai>`_ 上游。

动作参数
------------

Target50 的三个参数作用于每次 RLDX 工具调用：

.. list-table:: Target50 RLDX 参数
   :header-rows: 1
   :widths: 42 12 46

   * - 环境变量
     - 值
     - 含义
   * - ``RLDX_MAX_CHUNKS``
     - 40
     - 每次调用最多预测的动作块数；普通 RoboCasa 使用 70。
   * - ``RLDX_SETTLE_PATIENCE``
     - 999
     - 末端与夹爪连续多少个动作块几乎不动时，才按静止判定停止。该值超过 40 个动作块的上限。
   * - ``RLDX_ACTION_STEPS_PER_CHUNK``
     - 8
     - 每个预测动作块中执行的动作数。

记忆文件选择
----------------

HF 模式从数据集当前 main 读取 ``robocasa/``，为当前任务提供 ``<Task>_s0.json``、``<Task>_s0_recipe.jsonl``、可选的 ``<Task>.md`` 和 ``global/GLOBAL_MEMORY.md``。评测的 ``--seed`` 改变场景，参考记忆仍使用 ``_s0``。

本地评测使用 ``--memory-profile local --memory-dir <目录>``，支持两种任务文件命名：

- 发布语料：``task-specific/<Task>_s0.json`` 与 ``<Task>_s0_recipe.jsonl``。
- 探索产物：``task-specific/<Task>_<split>_s0.json`` 与 ``<Task>_<split>_s0_recipe.jsonl``。

JSON 与 recipe 必须成对存在，也可以同时缺失；同时缺失时仍可使用 global 与实时观测。若同一任务的两套文件同时存在，请用不同的 ``--memory-dir`` 分开。``<Task>.md`` 为可选文件，缺失时记录日志。

本地评测还开放 ``global/*.md``，以及 YAML frontmatter 同时匹配 ``suite: robocasa``、``regime: <split>``、``task_id: <Task>`` 的 ``task-family/*.md``。global 层至少需要一份可读文件。

提示词和 RPent 文件工具使用同一份可读文件列表。工具拒绝访问其他任务、其他 split、根索引 ``MEMORY.md`` 及 ``_internal/``；限制仅作用于 RPent 工具。CLI 在启动服务前检查记忆；Dashboard 在启动共享 VLA 前检查目录和 global，再于任务环境启动前检查任务文件。任务文件错误不会停掉已有的共享 VLA。

模型按需读取记忆。实时任务语言、RGB-D、任务进展和工具结果优先；有接触、持物或可见进展时继续调用 VLA，连续两次无接触且无进展后再定位并有限调整姿态。每次调用使用完整的实时任务语言；历史 ``vla_act`` 仅供理解策略，历史坐标不可回放。

每次运行单独记录文件选择、缺失层和实际读取，即使复用输出目录也会重新开始。零读取或部分读取都可产生有效环境结果；审计文件缺失或损坏单独报告。结果记录所用 profile 和任务族身份，供校验任务边界。

自定义记忆来源
~~~~~~~~~~~~~~~~~~

使用相同目录结构的其他 HF 数据集时，设置 ``RPENT_MEMORY_HF_REPO=<owner>/<dataset>`` 并使用 ``--memory-profile hf``。该变量接收仓库 ID。

自定义子目录或分支先下载到新目录，再使用 local 模式；选择分支时加上 ``--revision <branch>``：

.. code-block:: bash

   hf download <owner>/<dataset> --repo-type dataset \
      --include '<subpath>/**' --local-dir ./custom-memory

   # 在 RPent 运行命令中加上：
   # --memory-profile local --memory-dir ./custom-memory/<subpath>

所选目录应包含 ``task-specific/`` 与 ``global/``，并满足上面的文件选择规则。

协议与历史兼容性
--------------------

当前 ``target50.json`` 使用 ``robocasa-harness-vla-v2`` 协议和 ``1.1`` 结果格式，参考配置为 GPT-5.5。软件依赖由 ``pyproject.toml`` 管理。自定义任务、seed 或规划器配置时，将当前协议的清单传给校验器：

.. code-block:: bash

   python -m robots.robocasa.eval.validate_target50 /path/to/results \
      --manifest /path/to/manifest.json

比较实验时保留同一份记忆，并在本地记录 HF commit 或文件哈希，以及源码依赖 ``rpent`` 分支实际安装的提交。程序不锁定 memory 版本，校验器也不比较各次运行的记忆正文。

历史 v1 结果仅使用 task-specific memory，应在 `历史代码 <https://github.com/RLinf/RPent/tree/ec4e18fc2f6a73a00c6a5c035a8a3fdb17950b61>`_ 中使用对应的 `v1 清单 <https://github.com/RLinf/RPent/blob/ec4e18fc2f6a73a00c6a5c035a8a3fdb17950b61/robots/robocasa/eval/target50.json>`_ 校验：

.. code-block:: bash

   python -m robots.robocasa.eval.validate_target50 /path/to/historical-results \
      --manifest robots/robocasa/eval/target50.json

HF 的 `reproduce/memory 归档 <https://huggingface.co/datasets/RLinf/RPent-memory/tree/reproduce/memory>`_ 保留 ``d8c25a7f`` 的 GPT-5.5 Harness-VLA 资源，后续更新只进入 main。归档中的 RoboCasa 使用 ``task_only/``，与当前要求的 ``task-specific/`` 不兼容；其 README 描述历史用法。本指南尚未确立与该归档配套的 RoboCasa 代码/数据快照，当前运行命令使用 main 语料。

榜单成绩来自独立实验报告。采用当前协议并完成 340 次运行，并不能证明复现了某条榜单成绩；还需核对模型、记忆和代码配置。历史成绩保留原始来源。

Target50 报告成绩与历史结果
----------------------------

下表成功率以 :doc:`排行榜 <../leaderboard/performance>` 为准。RPent 的三个配置分别为 Codex / GPT-5.5 / xhigh / reasoning、Codex / GPT-6 Astra / low / reasoning，以及 Claude Code / Opus-4.7 / max.reasoning。Harness VLA 参考列采用 `论文表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_ 中的 GPT-5.5 结果。Overall 对 50 个任务等权计算，不是 340 回合中的成功回合占比。

.. list-table:: 已报告的 Target50 成功率
   :header-rows: 1
   :widths: 24 18 18 18 22

   * - Split
     - RPent / GPT-5.5
     - RPent / GPT-6 Astra
     - RPent / Opus-4.7
     - Harness VLA / GPT-5.5 参考值
   * - Atomic-Seen
     - 92.0%
     - 87.78%
     - 79.4%
     - 92.0%
   * - Composite-Seen
     - 61.0%
     - 43.75%
     - 47.5%
     - 61.0%
   * - Composite-Unseen
     - 13.8%
     - 42.50%
     - 15.0%
     - 13.8%
   * - 总体（任务加权）
     - 57.1%
     - 59.20%
     - 48.6%
     - 57.1%

Astra 的报告值为 **Overall 59.20%**，三个分项分别为 **87.78% / 43.75% / 42.50%**。回合数已根据 `实验贡献者确认的更正 <https://github.com/RLinf/RPent/pull/205#issuecomment-5749514622>`_ 同步为 **340（180/80/80）**；此前的 250 回合信息属于尚未同步的历史记录。此次更正保留已报告成功率，不由四舍五入后的比率推算成功次数，也不代表重新核验了全部 340 份原始结果。

历史 Codex 复现
~~~~~~~~~~~~~~~~

归档中的复现覆盖全部 340 个评测单元，按任务汇总的结果如下。这些历史数值不代表使用 v2 协议重新评测的结果：

.. list-table:: Codex Target50 复现结果
   :header-rows: 1
   :widths: 30 20 20 30

   * - 任务组
     - 成功次数 / 运行数
     - 成功率
     - Harness VLA 参考值
   * - Atomic
     - 163/180
     - 90.56%
     - 165/180 (91.67%)
   * - Composite-Seen
     - 49/80
     - 61.25%
     - 45/80 (56.25%)
   * - Composite-Unseen
     - 12/80
     - 15.00%
     - 11/80 (13.75%)
   * - 总体（任务加权）
     - 不适用
     - 57.00%
     - 55.40%

`历史逐任务结果表 <https://github.com/RLinf/RPent/blob/57088f6df30b227f2229ead985aa75403c0ce291/robots/robocasa/eval/target50_codex_results.md>`_ 给出每个任务的成功次数和成功率。这份历史记录仅提供任务级汇总数据，不包含各 seed 的执行记录、原始轨迹或失败分类，因此不能用于逐次复核运行过程。
