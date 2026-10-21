.. _release-v0-1:

RPent v0.1 版本发布 🎉
==================================

我们发布了 RPent 的首个公开版本 v0.1。

RPent（Recursive Physical Agent）是面向真实物理世界的开源具身智能体基础设施。它连接通用大模型、VLA 模型、记忆与机器人接口，通过任务规划、执行反馈和经验复用完成物理世界任务。

v0.1 支持使用 Codex、Claude Code 等 Agent Harness 进行任务规划，接入 LIBERO 系列、RoboCasa365、RoboTwin、RoboDojo 等仿真环境，以及单臂 Franka、双臂 Franka 和 YAM 真机，提供 Memory、Flash Mode、交互式 CLI 与 Web Dashboard。RPent 已在多种仿真环境和真机任务中完成验证，并在 LIBERO-PRO 等仿真基准上取得榜单所列方法中的领先任务成功率。

版本亮点
--------------------

真机演示
~~~~~~~~~~~~

- **真机任务演示**\ ：双臂 Franka 根据任务目标调整餐具放置位置，完成倾倒钢珠、持盘擦拭和收纳任务；YAM 演示按品牌标签分拣饮料、检查抓取状态，以及移开遮挡物寻找勺子等任务。 :doc:`双臂 Franka 演示 </rst_source/real_world_demos/franka>` ｜ :doc:`YAM 演示 </rst_source/real_world_demos/yam>`

仿真评测
~~~~~~~~~~~~

- **多仿真环境评测与公开榜单**\ ：使用 GPT-6 Astra 的 RPent 在 LIBERO-PRO（low reasoning）、RoboCasa365 Target50（high reasoning）上分别达到 92.63%、59.20% 成功率；使用 GPT-5.5 的 RPent 在 RoboTwin C2R 上达到 62.4%。公开榜单提供评测配置、成功率、耗时与 Token 开销数据。 :doc:`Leaderboard </rst_source/leaderboard/index>`

系统设计
~~~~~~~~~~~~

- **模块化**\ ：采用开放的模块化设计，将系统划分为用户层、智能层、接口层和环境层四大层级。

- **统一接口**\ ：将 VLA 模型与解析式原语封装为统一工具，通过 MCP 或原生工具调用接入规划器，通过 RPC 连接模型、机器人与仿真服务。 :doc:`核心接口 </rst_source/development/interfaces>`

Memory 设计
~~~~~~~~~~~~~~~~~~

- **积累与复用任务经验**\ ：探索模式积累经验，评测模式只读复用分层保存的通用经验、任务策略和执行记录。 :doc:`Memory </rst_source/development/memory>`

- **用任务卡加快执行**\ ：Flash Mode 将成功流程整理为任务卡，根据当前视觉定位更新动作目标，减少逐步规划开销。在 LIBERO-PRO 的 800 回合评测中，成功率为 **72.63%**\ ，平均耗时 **60.19 秒**\ ，较 Codex + GPT-6 Astra（low reasoning）实现约 **6.8 倍加速**\ 。该评测使用官方公开的 GPT-5.5 探索记忆，成绩采用 seed 0–9 中的最佳评测结果。 :doc:`Flash Mode </rst_source/guides/flash>` ｜ :doc:`评测数据 </rst_source/leaderboard/time-token-costs>`

论文
~~~~~~~~~~~~

- **Harness VLA**\ ：RPent 的首篇研究论文，介绍如何利用任务记忆组合 VLA 模型与解析式原语。无需更新 VLA 权重，即可应对任务变化、环境扰动和多步骤操作。 `论文 <https://arxiv.org/abs/2607.08448>`__ ｜ `项目主页 <https://harnessvla.github.io/>`__

功能特性
----------------

真机
~~~~~~~~~~~~

- **单臂与双臂 Franka**\ ：支持机械臂与夹爪控制、多相机观测和手眼标定。单臂可接入外部 VLA 抓取服务；双臂提供抓取、交接和放置技能，支持双节点部署与操作员参与的探索。两者均支持重放经人工审阅的 Flash 任务卡，并由操作员确认结果。 :doc:`单臂 </rst_source/real_world_robots/franka>` ｜ :doc:`双臂 </rst_source/real_world_robots/dual_franka>`

- **YAM**\ ：支持双臂末端运动、腕部旋转、夹爪控制和多相机观测，并接入 Pi0.5 动作模型。可通过终端或 Dashboard 开展探索、接收人工反馈并保存经验。 :doc:`文档 </rst_source/real_world_robots/yam>`

仿真
~~~~~~~~~~~~

- **LIBERO、LIBERO-PRO**\ ：覆盖 Spatial、Object、Goal 和 LIBERO-10 任务族，支持指令与场景布局扰动，以及探索、记忆复用评测和 Flash Mode。 :doc:`文档 </rst_source/simulators/libero>`

- **RoboCasa365**\ ：支持 PandaOmron 移动机械臂的导航、底盘移动与机械臂操作，提供 Target50 评测配置。 :doc:`文档 </rst_source/simulators/robocasa>`

- **RoboTwin**\ ：接入双臂仿真，支持 ``demo_clean``、``demo_randomized`` 场景、多相机观测和 50 个任务的评测配置。 :doc:`文档 </rst_source/simulators/robotwin>`

- **RoboDojo**\ ：仿真器基于 Isaac Sim / IsaacLab，采用 ARX-X5 双机械臂平台。当前支持安装。 :doc:`文档 </rst_source/simulators/robodojo>`

模型
~~~~~~~~~~~~

- **规划器与模型**\ ：支持 Codex、Claude Code 和 RPent 内置规划器，可接入 Claude、GPT、Qwen 等模型，并按规划器配置推理强度、规划轮数上限和超时时间。在 LIBERO、RoboCasa365 和 RoboTwin 中，规划器还可直接输出动作控制数值，由环境执行。 :doc:`配置指南 </rst_source/guides/configure_planner>`

- **VLA 模型**\ ：支持 Pi0.5（LIBERO、RoboDojo、双臂 Franka、YAM）、RLDX-1（RoboCasa365）和 LingBot-VLA（RoboTwin）。 :doc:`Pi0.5 </rst_source/simulators/libero>` ｜ :doc:`RLDX-1 </rst_source/simulators/robocasa>` ｜ :doc:`LingBot-VLA </rst_source/simulators/robotwin>`

- **感知模型**\ ：支持 SAM3 目标分割和 Molmo2-8B 视觉定位，结合深度与相机信息确定目标位置。 :doc:`视觉定位指南 </rst_source/guides/flash>`

系统
~~~~~~~~~~~~

- **扩展接口**\ ：支持接入新机器人、模型和工具，复用服务通信、任务执行与记忆管理组件。 :doc:`开发指南 </rst_source/development/add_robot>`

- **交互式 CLI 与 Dashboard**\ ：支持任务交互与运行记录保存；Dashboard 可展示规划过程、工具调用、相机画面和动作时间线，并提供中英文界面。 :doc:`CLI </rst_source/guides/cli>` ｜ :doc:`Dashboard </rst_source/guides/dashboard>`

- **服务部署**\ ：模型与环境服务可独立部署在本地或远程，连续任务可复用已启动的模型服务。 :doc:`部署指南 </rst_source/guides/advanced_deployment>`

- **Memory**\ ：提供公开的任务经验与任务卡数据集，支持本地记忆校验、合并和索引更新，方便不同智能体复用已有经验。 :doc:`使用指南 </rst_source/guides/memory>` ｜ `数据集 <https://huggingface.co/datasets/RLinf/RPent-memory>`__

Contributors
------------------------

感谢所有贡献者，排名不分先后。

`@qurakchin <https://github.com/qurakchin>`__ `@QuanluZhang <https://github.com/QuanluZhang>`__ `@zoeyuchao <https://github.com/zoeyuchao>`__ `@wilburx813 <https://github.com/wilburx813>`__ `@Elessar123 <https://github.com/Elessar123>`__ `@Sonorous281 <https://github.com/Sonorous281>`__ `@ye1457 <https://github.com/ye1457>`__ `@silence-breaker <https://github.com/silence-breaker>`__ `@huanming99 <https://github.com/huanming99>`__ `@jx-qiu <https://github.com/jx-qiu>`__ `@wkzCode <https://github.com/wkzCode>`__ `@6UOOON9 <https://github.com/6UOOON9>`__ `@zhexuanxu <https://github.com/zhexuanxu>`__ `@ucfyao <https://github.com/ucfyao>`__ `@catr1xLiu <https://github.com/catr1xLiu>`__ `@liushb9 <https://github.com/liushb9>`__ `@andylin-hao <https://github.com/andylin-hao>`__ `@chen00double00guang-cmyk <https://github.com/chen00double00guang-cmyk>`__ `@Code-Hit-ai <https://github.com/Code-Hit-ai>`__ `@Brunch-Life <https://github.com/Brunch-Life>`__ `@JamesHuang0319 <https://github.com/JamesHuang0319>`__ `@NewYork424 <https://github.com/NewYork424>`__ `@Travor278 <https://github.com/Travor278>`__ `@mintonight <https://github.com/mintonight>`__ `@Gaojr-2024 <https://github.com/Gaojr-2024>`__ `@Hiforever-h <https://github.com/Hiforever-h>`__ `@scilwb <https://github.com/scilwb>`__ `@ZhuWeiwen315 <https://github.com/ZhuWeiwen315>`__ `@skyswordx <https://github.com/skyswordx>`__ `@littleZ05 <https://github.com/littleZ05>`__
`@mvanhorn <https://github.com/mvanhorn>`__ `@akushonkamen <https://github.com/akushonkamen>`__ `@Yang-star-code <https://github.com/Yang-star-code>`__

RPent 评测结果
--------------------

公开基准成绩与评测配置见 :doc:`评测文档 </rst_source/leaderboard/performance>` ，本次发布测试结果见文末。

快速开始
------------

- **运行第一个任务**\ ：从 LIBERO-PRO 开始，完成环境安装、Pi0.5 与 SAM3 权重下载及规划器配置。 :doc:`安装与快速开始 </rst_source/get_started/quickstart>`

- **探索与经验复用**\ ：通过探索生成 Memory，或从成功轨迹生成可复用的 Flash 执行计划。 :doc:`LIBERO 探索 </rst_source/simulators/libero>` ｜ :doc:`Flash Mode </rst_source/guides/flash>`

- **真机部署**\ ：按平台指南安装机器人控制软件、配置相机、完成手眼标定，并获取和配置所需模型权重。 :doc:`Franka </rst_source/real_world_robots/franka>` ｜ :doc:`双臂 Franka </rst_source/real_world_robots/dual_franka>` ｜ :doc:`YAM </rst_source/real_world_robots/yam>`

RPent v0.1 Test Results
----------------------------------------------

RPent v0.1 在 LIBERO-PRO、RoboCasa365 和 RoboTwin 上完成 180 回合发布版本测试、180 回合论文配置抽样复现及 15 回合 Memory 复用测试，并在双臂 Franka 和 YAM 上完成三项真机任务验证。

每个任务在一个指定 seed 下运行一次，计为一个测试回合。仿真任务采用环境原生成功判定，失败及超时均计入测试回合数。成功率按成功回合数占比计算；发布版本测试与论文复现的耗时和 Token 用量为每回合平均值，汇总按回合数加权计算。本节为所选任务的发布验证结果，完整基准成绩见文末 Leaderboard。

1. 发布版本仿真测试
~~~~~~~~~~~~~~~~~~~~~~

使用 RPent ``release/v0.1`` 和 HF Memory ``release/v0.1``，评测期间只读 Memory。

1.1 LIBERO-PRO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 Codex + GPT-6 Astra（low）和 Pi0.5。覆盖 Spatial、Object、Goal 和 LIBERO-10 的 Task 与 Swap 共 8 个套件，每套件选择 2 个任务，每任务运行 seed 1 至 seed 5，共 80 回合。

.. list-table::
   :header-rows: 1
   :class: rpent-release-results


   * - 套件与任务
     - 成功回合
     - 测试回合
     - 成功率
     - 耗时（秒）
     - 输入 Token
     - 输出 Token
   * - libero_spatial_task · task 0
     - 5
     - 5
     - 100%
     - 288.54
     - 1,469,403.20
     - 4,355.00
   * - libero_spatial_task · task 4
     - 4
     - 5
     - 80%
     - 348.88
     - 1,587,509.80
     - 4,888.00
   * - libero_spatial_swap · task 0
     - 5
     - 5
     - 100%
     - 212.86
     - 741,139.60
     - 3,241.20
   * - libero_spatial_swap · task 4
     - 3
     - 5
     - 60%
     - 339.52
     - 1,370,942.40
     - 4,389.80
   * - libero_object_task · task 0
     - 5
     - 5
     - 100%
     - 180.36
     - 726,203.80
     - 2,863.00
   * - libero_object_task · task 4
     - 5
     - 5
     - 100%
     - 192.32
     - 929,688.80
     - 3,153.00
   * - libero_object_swap · task 0
     - 5
     - 5
     - 100%
     - 251.54
     - 1,311,042.20
     - 3,873.80
   * - libero_object_swap · task 4
     - 5
     - 5
     - 100%
     - 247.34
     - 1,057,305.80
     - 3,253.80
   * - libero_goal_task · task 0
     - 3
     - 5
     - 60%
     - 410.84
     - 1,657,833.40
     - 4,453.60
   * - libero_goal_task · task 7
     - 5
     - 5
     - 100%
     - 143.88
     - 434,249.00
     - 2,133.20
   * - libero_goal_swap · task 0
     - 5
     - 5
     - 100%
     - 523.70
     - 1,462,251.80
     - 3,985.20
   * - libero_goal_swap · task 7
     - 5
     - 5
     - 100%
     - 213.40
     - 316,889.40
     - 1,931.20
   * - libero_10_task · task 3
     - 5
     - 5
     - 100%
     - 507.38
     - 2,201,277.20
     - 5,133.20
   * - libero_10_task · task 9
     - 3
     - 5
     - 60%
     - 469.48
     - 2,895,074.00
     - 6,348.40
   * - libero_10_swap · task 3
     - 5
     - 5
     - 100%
     - 518.64
     - 3,341,025.40
     - 6,639.00
   * - libero_10_swap · task 9
     - 1
     - 5
     - 20%
     - 715.74
     - 5,376,288.20
     - 9,809.60
   * - **汇总**
     - **69**
     - **80**
     - **86.25%**
     - **347.78**
     - **1,679,882.75**
     - **4,403.19**

1.2 RoboCasa365
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 Codex + GPT-6 Astra（low）和 RLDX-1。覆盖 3 个 Atomic 任务，每个任务运行 seed 1 至 seed 10；4 个 Composite 任务，每任务运行 seed 1 至 seed 5，共 50 回合。

.. list-table::
   :header-rows: 1
   :class: rpent-release-results


   * - 套件与任务
     - 成功回合
     - 测试回合
     - 成功率
     - 耗时（秒）
     - 输入 Token
     - 输出 Token
   * - Atomic · OpenDrawer
     - 5
     - 10
     - 50%
     - 261.97
     - 504,529.00
     - 2,066.10
   * - Atomic · NavigateKitchen
     - 10
     - 10
     - 100%
     - 86.55
     - 154,368.10
     - 854.40
   * - Atomic · PickPlaceCounterToCabinet
     - 8
     - 10
     - 80%
     - 186.62
     - 318,589.00
     - 1,537.20
   * - Composite-Seen · PrepareCoffee
     - 0
     - 5
     - 0%
     - 697.10
     - 1,318,701.80
     - 4,115.00
   * - Composite-Seen · LoadDishwasher
     - 1
     - 5
     - 20%
     - 542.68
     - 898,868.40
     - 2,864.80
   * - Composite-Unseen · RecycleBottlesByType
     - 0
     - 5
     - 0%
     - 785.30
     - 1,926,543.40
     - 4,675.80
   * - Composite-Unseen · GatherTableware
     - 0
     - 5
     - 0%
     - 404.88
     - 695,211.80
     - 2,549.80
   * - **汇总**
     - **24**
     - **50**
     - **48.00%**
     - **350.02**
     - **679,429.76**
     - **2,312.08**

1.3 RoboTwin
^^^^^^^^^^^^^^^^^^^^^^^^

使用 Codex + GPT-5.5（xhigh）和 LingBot-VLA。在 ``demo_randomized`` 场景下测试 10 个任务，每个任务使用 5 个已验证的场景 seed，共 50 回合。

.. list-table::
   :header-rows: 1
   :class: rpent-release-results


   * - 套件与任务
     - 成功回合
     - 测试回合
     - 成功率
     - 耗时（秒）
     - 输入 Token
     - 输出 Token
   * - adjust_bottle
     - 4
     - 5
     - 80%
     - 624.46
     - 1,158,068.40
     - 6,144.20
   * - beat_block_hammer
     - 4
     - 5
     - 80%
     - 891.18
     - 3,216,841.00
     - 14,468.80
   * - handover_block
     - 2
     - 5
     - 40%
     - 2,369.52
     - 7,202,960.80
     - 29,643.80
   * - place_empty_cup
     - 5
     - 5
     - 100%
     - 431.60
     - 775,905.80
     - 4,554.60
   * - place_shoe
     - 5
     - 5
     - 100%
     - 672.50
     - 1,501,595.00
     - 6,732.80
   * - stack_bowls_three
     - 3
     - 5
     - 60%
     - 1,654.54
     - 3,416,140.80
     - 12,468.00
   * - blocks_ranking_rgb
     - 4
     - 5
     - 80%
     - 1,581.44
     - 4,454,111.00
     - 16,199.20
   * - lift_pot
     - 4
     - 5
     - 80%
     - 609.38
     - 1,751,153.40
     - 6,595.00
   * - dump_bin_bigbin
     - 3
     - 5
     - 60%
     - 1,161.22
     - 3,479,640.40
     - 13,112.80
   * - open_laptop
     - 4
     - 5
     - 80%
     - 1,693.66
     - 3,892,900.20
     - 15,953.20
   * - **汇总**
     - **38**
     - **50**
     - **76.00%**
     - **1,168.95**
     - **3,084,931.68**
     - **12,587.24**

2. 安装、探索与 Memory 复用验证
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 LIBERO-PRO、RoboCasa365 和 RoboTwin 上验证环境安装、探索生成记忆及重启后复用流程，每个环境使用 5 个 seed 测试，共 15 回合。成功率仅统计记忆复用阶段，探索回合单独记录。

LIBERO-PRO 与 RoboCasa365 使用 seed 0 探索、seed 1 至 seed 5 测试；RoboTwin 在 demo_clean 中探索，在 demo_randomized 的 seed 100000、100001、100003、100004、100005 上测试。表中耗时与 Token 用量对应探索阶段。

.. list-table::
   :header-rows: 1
   :class: rpent-release-results rpent-memory-results


   * - 环境与验证任务
     - 流程验证
     - 成功回合
     - 测试回合
     - 成功率
     - 耗时（秒）
     - 输入 Token
     - 输出 Token
   * - LIBERO-PRO · libero_object_swap · task 0
     - 通过
     - 5
     - 5
     - 100%
     - 194.16
     - 1,074,213.00
     - 3,224.60
   * - RoboCasa365 · OpenDrawer
     - 通过
     - 5
     - 5
     - 100%
     - 2,293.80
     - 1,812,551.00
     - 38,699.00
   * - RoboTwin · beat_block_hammer
     - 通过
     - 3
     - 5
     - 60%
     - 4,267.00
     - 5,088,881.00
     - 28,691.00

RoboDojo 完成环境安装与 ``put_bottles_into_dustbin`` 单次任务运行检查，验证范围为安装与运行流程，不包含任务成功率评测。

3. 论文配置抽样复现
~~~~~~~~~~~~~~~~~~~~~~

采用 Harness VLA 配套代码与 Memory，在 LIBERO-PRO、RoboCasa365 和 RoboTwin 上分别完成 80、50、50 回合测试，共 180 回合。各环境的任务及 seed 范围均与第 1 节一致，配置与结果如下。

3.1 LIBERO-PRO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

规划模型：Claude Opus 4.7（xhigh）；VLA：Pi0.5。共 16 个任务、80 回合。 :doc:`环境配置与运行指南 </rst_source/simulators/libero>`

.. list-table::
   :header-rows: 1
   :class: rpent-release-results


   * - 套件与任务
     - 成功回合
     - 测试回合
     - 成功率
     - 耗时（秒）
     - 输入 Token
     - 输出 Token
   * - libero_spatial_task · task 0
     - 4
     - 5
     - 80%
     - 804.10
     - 3,189,007.00
     - 31,817.00
   * - libero_spatial_task · task 4
     - 5
     - 5
     - 100%
     - 464.60
     - 1,707,359.00
     - 16,834.00
   * - libero_spatial_swap · task 0
     - 4
     - 5
     - 80%
     - 497.50
     - 1,877,124.00
     - 14,657.00
   * - libero_spatial_swap · task 4
     - 4
     - 5
     - 80%
     - 534.30
     - 1,629,280.00
     - 18,174.00
   * - libero_object_task · task 0
     - 3
     - 5
     - 60%
     - 721.10
     - 2,623,604.00
     - 18,532.00
   * - libero_object_task · task 4
     - 4
     - 5
     - 80%
     - 456.40
     - 1,706,284.00
     - 10,606.00
   * - libero_object_swap · task 0
     - 4
     - 5
     - 80%
     - 485.40
     - 1,565,337.00
     - 12,344.00
   * - libero_object_swap · task 4
     - 5
     - 5
     - 100%
     - 282.40
     - 913,677.00
     - 10,398.00
   * - libero_goal_task · task 0
     - 1
     - 5
     - 20%
     - 314.30
     - 730,283.00
     - 12,251.00
   * - libero_goal_task · task 7
     - 5
     - 5
     - 100%
     - 119.00
     - 362,758.00
     - 4,226.00
   * - libero_goal_swap · task 0
     - 0
     - 5
     - 0%
     - 1,066.90
     - 3,140,629.00
     - 44,427.00
   * - libero_goal_swap · task 7
     - 5
     - 5
     - 100%
     - 86.50
     - 318,485.00
     - 2,915.00
   * - libero_10_task · task 3
     - 1
     - 5
     - 20%
     - 1,207.30
     - 3,659,185.00
     - 31,154.00
   * - libero_10_task · task 9
     - 0
     - 5
     - 0%
     - 841.10
     - 1,797,568.00
     - 26,700.00
   * - libero_10_swap · task 3
     - 0
     - 5
     - 0%
     - 904.80
     - 2,799,185.00
     - 25,717.00
   * - libero_10_swap · task 9
     - 0
     - 5
     - 0%
     - 376.10
     - 896,539.00
     - 14,635.00
   * - **汇总**
     - **45**
     - **80**
     - **56.25%**
     - **572.60**
     - **1,807,269.00**
     - **18,462.00**

3.2 RoboCasa365
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

规划配置：Codex + GPT-5.5（xhigh）；VLA：RLDX-1。共 7 个任务、50 回合。 :ref:`环境配置与运行指南 <reproduce-target50>`

.. list-table::
   :header-rows: 1
   :class: rpent-release-results


   * - 套件与任务
     - 成功回合
     - 测试回合
     - 成功率
     - 耗时（秒）
     - 输入 Token
     - 输出 Token
   * - OpenDrawer
     - 8
     - 10
     - 80%
     - 339.40
     - 1,360,316.00
     - 10,583.00
   * - NavigateKitchen
     - 10
     - 10
     - 100%
     - 288.00
     - 1,286,723.00
     - 13,702.00
   * - PickPlaceCounterToCabinet
     - 8
     - 10
     - 80%
     - 317.20
     - 1,626,763.00
     - 10,042.00
   * - PrepareCoffee
     - 0
     - 5
     - 0%
     - 2,435.90
     - 9,453,302.00
     - 49,694.00
   * - LoadDishwasher
     - 2
     - 5
     - 40%
     - 1,744.40
     - 5,114,983.00
     - 32,871.00
   * - RecycleBottlesByType
     - 0
     - 5
     - 0%
     - 2,332.60
     - 8,435,321.00
     - 47,109.00
   * - GatherTableware
     - 0
     - 5
     - 0%
     - 1,760.80
     - 7,701,994.00
     - 51,364.00
   * - **汇总**
     - **28**
     - **50**
     - **56.00%**
     - **1,016.29**
     - **3,925,320.40**
     - **24,969.20**

3.3 RoboTwin
^^^^^^^^^^^^^^^^^^^^^^^^

规划配置：Codex + GPT-5.5（xhigh）；VLA：LingBot-VLA。共 10 个任务、50 回合，场景为 demo_randomized。 :doc:`环境配置与运行指南 </rst_source/simulators/robotwin>`

.. list-table::
   :header-rows: 1
   :class: rpent-release-results


   * - 套件与任务
     - 成功回合
     - 测试回合
     - 成功率
     - 耗时（秒）
     - 输入 Token
     - 输出 Token
   * - adjust_bottle
     - 5
     - 5
     - 100%
     - 272.40
     - 303,276.00
     - 1,861.00
   * - beat_block_hammer
     - 2
     - 5
     - 40%
     - 753.10
     - 4,075,393.00
     - 21,435.00
   * - handover_block
     - 0
     - 5
     - 0%
     - 2,317.00
     - 11,461,447.00
     - 45,969.00
   * - place_empty_cup
     - 5
     - 5
     - 100%
     - 343.30
     - 773,897.00
     - 5,453.00
   * - place_shoe
     - 4
     - 5
     - 80%
     - 434.90
     - 1,399,450.00
     - 8,048.00
   * - stack_bowls_three
     - 2
     - 5
     - 40%
     - 738.50
     - 2,084,738.00
     - 12,753.00
   * - blocks_ranking_rgb
     - 3
     - 5
     - 60%
     - 1,484.70
     - 8,932,639.00
     - 37,453.00
   * - lift_pot
     - 2
     - 5
     - 40%
     - 690.40
     - 3,016,443.00
     - 14,398.00
   * - dump_bin_bigbin
     - 3
     - 5
     - 60%
     - 651.90
     - 2,620,781.00
     - 11,887.00
   * - open_laptop
     - 5
     - 5
     - 100%
     - 349.70
     - 845,135.00
     - 5,620.00
   * - **汇总**
     - **31**
     - **50**
     - **62.00%**
     - **803.60**
     - **3,551,320.00**
     - **16,488.00**

代码使用 reproduce/libero、reproduce/robocasa 和 reproduce/robotwin 分支，Memory 使用 Hugging Face 数据集的 reproduce/memory 分支。

4. 真机演示验证
~~~~~~~~~~~~~~~~~~

在双臂 Franka 和 YAM 上完成餐具分类放置、倾倒及饮料标签识别与分拣三项任务。餐具分类任务包含一个碗和两个盘子；表中耗时与 Token 用量对应各任务的单次成功执行。

.. list-table::
   :header-rows: 1
   :class: rpent-release-results rpent-demo-results


   * - 平台与任务
     - 规划配置
     - 执行结果
     - 耗时（秒）
     - 输入 Token
     - 输出 Token
   * - 双臂 Franka · demo01：餐具分类放置
     - Codex + GPT-6 Astra (xhigh)
     - 已完成
     - 600.00
     - 2,996,944
     - 7,503
   * - 双臂 Franka · demo03：从放置到倾倒
     - GPT-6 Astra (low)
     - 已完成
     - 1,984.00
     - 14,163,904
     - 24,159
   * - YAM · demo02：饮料标签识别与分拣
     - GPT-6 Sol (medium)
     - 已完成
     - 1,886.85
     - 12,026,547
     - 38,909

完整基准成绩与评测配置见 :doc:`Leaderboard </rst_source/leaderboard/index>` 。
