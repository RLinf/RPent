RoboCasa365
============

.. figure:: https://raw.githubusercontent.com/robocasa/robocasa/main/docs/images/readme.webp
   :alt: RoboCasa365 环境概览
   :width: 90%
   :align: center

   RoboCasa365 的厨房场景、物体与任务。图片来源：`RoboCasa365 项目 <https://github.com/robocasa/robocasa>`_。

使用 RPent 在 RoboCasa365 中运行厨房操作任务，并复现 Target50 实验。当前集成使用 PandaOmron 移动机械臂和 RLDX-1 动作模型，CLI 名称为 ``robocasa``。

.. _robocasa-overview:

概览
------------

先确认所需模型、任务与运行环境，再按后续步骤安装并运行。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      RLDX-1

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``

   .. grid-item-card:: 任务

      Target50 厨房任务

   .. grid-item-card:: 硬件

      Linux、NVIDIA GPU；Python 3.10；CUDA 与 EGL。

任务
~~~~~~~~~~~~

Target50 包含以下任务类别，完整任务、seed 和运行限制见本页实验复现部分。

.. list-table::
   :header-rows: 1

   * - 类别
     - 任务数
     - 内容
   * - Atomic
     - 18
     - 单项厨房操作。
   * - Composite-Seen
     - 16
     - 已见类别的组合任务。
   * - Composite-Unseen
     - 16
     - 未见类别的组合任务。

.. _robocasa-observation-action:

观测与动作
~~~~~~~~~~~~

下表区分规划器使用的工具、模型输入及环境的成功判定。

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - 相机图像、深度／世界坐标及底盘、末端、夹爪状态。RLDX-1 接收三路相机的历史帧、状态和任务文字。
   * - 动作
     - 规划器调用 ``rldx_skill`` 和动作原语；RLDX-1 预测末端、夹爪、底盘运动和控制模式指令。
   * - 奖励与成功判定
     - 任务成功以环境 ``_check_success()`` 返回的结果为准，该结果通过 ``state.success`` 提供。
   * - 任务指令
     - 使用当前环境的完整 ``task_language``，历史记忆中的指令不能替代当前任务指令。

安装与资源准备
---------------------

需要 Linux、NVIDIA GPU、可用的 CUDA/EGL，以及 ``git`` 和 `uv <https://docs.astral.sh/uv/getting-started/installation/>`_。使用独立的 Python 3.10 环境；已有仓库请先进入该目录，再从第三行开始：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   uv venv --python 3.10 .venv-robocasa
   source .venv-robocasa/bin/activate

先按 `PyTorch 安装说明 <https://pytorch.org/get-started/locally/>`_ 安装与本机驱动兼容的 CUDA 版 Torch 和 torchvision，可将命令中的 ``pip`` 换为 ``uv pip``。RLDX 要求 Torch >= 2.7、torchvision >= 0.22，且两者版本匹配。再安装 RoboCasa：

.. list-table::
   :header-rows: 1

   * - 环境
     - 安装命令
     - 资源下载
   * - RoboCasa365
     - ``uv pip install -e ".[robocasa]"``
     - ``robocasa-download-assets`` （见下方命令）

依赖以 ``pyproject.toml`` 为准，包含 RoboCasa、RLDX 和 Robosuite 的 ``rpent`` 分支源码。不要另装提供同一导入包的 ``rlinf-robocasa365``。

检查依赖后，将约 10 GB 的厨房资源放到 Python 包目录之外，并在启动 RPent 的终端设置资源路径：

.. code-block:: bash

   uv pip check
   robocasa-download-assets --assets-path ~/.robocasa/assets --no-macros -y
   export ROBOCASA_ASSETS_PATH=~/.robocasa/assets

Target50 不需要数据集或遥操作配置，使用 ``--no-macros`` 即可。资源包含下载集合与随包场景文件；重复下载可加 ``--skip-existing`` 检查已有文件。模型文件按下一节准备。

.. dropdown:: 可选依赖与版本记录

   ``flash-attn`` 可选，未安装时使用 PyTorch SDPA。如需安装，参见 `FlashAttention 官方说明 <https://github.com/Dao-AILab/flash-attention#installation-and-features>`_。

   参考实验使用 Torch 2.7.0、torchvision 0.22.0、CUDA 12.6；实际安装应与本机兼容。源码分支可能更新，复现时保存依赖和代码版本：

   .. code-block:: bash

      uv pip freeze > installed-requirements.txt
      git rev-parse HEAD > rpent-revision.txt

.. _robocasa-vla-configuration:

VLA 配置
------------

RLDX-1 需要微调权重和基础模型的支持文件，两项都要下载。模型和资源各自遵循其许可证。

模型权重
~~~~~~~~~~~~

下载针对 RoboCasa365 微调的 ``RLDX-1-FT-RC365``，运行时将 ``--vla-model-path`` 指向该目录：

.. code-block:: bash

   hf download RLWRLD/RLDX-1-FT-RC365 \
      --revision 587e9ecdcc5e7184fcc17f58713908edff5af041 \
      --local-dir ./checkpoints/rldx-1-ft-rc365

基础模型支持文件
~~~~~~~~~~~~~~~~~~~~

微调权重还需要 ``RLWRLD/RLDX-1-VLM`` 的模型结构、图像处理和分词配置。以下命令只下载支持文件，无需基础模型权重：

.. code-block:: bash

   export HF_HOME="$PWD/.cache/huggingface"
   export HF_HUB_CACHE="$HF_HOME/hub"
   hf download RLWRLD/RLDX-1-VLM \
      --revision 4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7 \
      --include "*.json" "*.txt" "*.jinja" "*.md" "*.png" ".gitattributes" \
      --exclude "*.safetensors.index.json"

启动时保留上述缓存变量，避免 ``TRANSFORMERS_CACHE`` 指向其他空目录。普通运行、Target50 和单独启动的 RPent VLA 服务都会自动使用此支持文件版本；微调权重仍由 ``--vla-model-path`` 指定。支持文件共约 16.4 MB，包含 15 个配置、文档和图片文件。

运行一个任务
------------------

先按 :doc:`../guides/configure_planner` 配置模型服务，并用 ``rpent-check-llm`` 检查连接。以下命令运行种子为 1 的 ``OpenDrawer`` 任务：

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer \
         --split target \
         --seed 1 \
         --vla-model-path ./checkpoints/rldx-1-ft-rc365 \
         --planner claude_code \
         --model claude-opus-4-8

也可使用 ``--planner api`` 或 ``--planner codex``，配置方式见上述规划器指南。

查看结果
------------

任务是否成功由环境的 ``_check_success()`` 判定，其布尔结果记录在 ``state.success`` 中。规划器调用 ``finish`` 会结束对话，但其中声明的状态不作为评测结果。查看输出目录中的 ``result.json``、``transcript_*.json`` 和 ``run.log``；服务启动问题分别记录在 ``env_server.log``、``vla_server.log``。

可使用 ``--dashboard`` 观察相机和规划器输出，通用操作见 :doc:`../guides/dashboard`。

任务记忆
------------

默认从 `HF main <https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/robocasa>`_ 下载记忆，CLI 与 Dashboard 行为一致：向规划器提供当前任务记忆（task-specific）和通用记忆（global）。规划器按需读取，以实时观测和任务指令为准。

.. code-block:: text

   memory/robocasa/
   ├── task-specific/
   └── global/

global 文件必须存在。任务记录（JSON）与动作序列（recipe）必须成对存在，也可以同时缺失。任务 Markdown 为可选文件。RPent 文件工具只开放当前任务允许读取的记忆。

需要固定一份本地记忆完成评测时，使用 ``--memory-profile local --memory-dir <目录>``，下载与运行示例见 :ref:`reproduce-target50`。文件命名、本地探索产物和自定义来源见下方说明。

.. dropdown:: 记忆文件选择与自定义来源

   .. _robocasa-memory-selection:

   HF 模式从数据集当前 main 读取 ``robocasa/``，为当前任务提供 ``<Task>_s0.json``、``<Task>_s0_recipe.jsonl``、可选的 ``<Task>.md`` 和 ``global/GLOBAL_MEMORY.md``。评测的 ``--seed`` 改变场景，参考记忆仍使用 ``_s0``。

   本地评测使用 ``--memory-profile local --memory-dir <目录>``，支持两种任务文件命名：

   - 发布语料：``task-specific/<Task>_s0.json`` 与 ``<Task>_s0_recipe.jsonl``。
   - 探索产物：``task-specific/<Task>_<split>_s0.json`` 与 ``<Task>_<split>_s0_recipe.jsonl``。

   JSON 与 recipe 必须成对存在，也可以同时缺失；同时缺失时仍可使用 global 与实时观测。若同一任务的两套文件同时存在，请用不同的 ``--memory-dir`` 分开。``<Task>.md`` 为可选文件，缺失时记录日志。

   本地评测还开放 ``global/*.md``，以及 YAML frontmatter 同时匹配 ``suite: robocasa``、``regime: <split>``、``task_id: <Task>`` 的 ``task-family/*.md``。global 层至少需要一份可读文件。

   提示词和 RPent 文件工具使用同一份可读文件列表。工具拒绝访问其他任务、其他 split、根索引 ``MEMORY.md`` 及 ``_internal/``；限制仅作用于 RPent 工具。CLI 在启动服务前检查记忆；Dashboard 在启动共享 VLA 前检查目录和 global，再于任务环境启动前检查任务文件。任务文件错误不会停掉已有的共享 VLA。

   模型按需读取记忆。实时任务语言、RGB-D、任务进展和工具结果优先；有接触、持物或可见进展时继续调用 VLA，连续两次无接触且无进展后再定位并有限调整姿态。每次调用使用完整的实时任务语言；历史 ``vla_act`` 仅供理解策略，历史坐标不可回放。

   每次运行单独记录文件选择、缺失层和实际读取，即使复用输出目录也会重新开始。零读取或部分读取都可产生有效环境结果；审计文件缺失或损坏单独报告。结果记录所用 profile 和任务族身份，供校验任务边界。

   .. rubric:: 自定义记忆来源

   使用相同目录结构的其他 HF 数据集时，设置 ``RPENT_MEMORY_HF_REPO=<owner>/<dataset>`` 并使用 ``--memory-profile hf``。该变量接收仓库 ID。

   自定义子目录或分支先下载到新目录，再使用 local 模式；选择分支时加上 ``--revision <branch>``：

   .. code-block:: bash

      hf download <owner>/<dataset> --repo-type dataset \
         --include '<subpath>/**' --local-dir ./custom-memory

      # 在 RPent 运行命令中加上：
      # --memory-profile local --memory-dir ./custom-memory/<subpath>

   所选目录应包含 ``task-specific/`` 与 ``global/``，并满足上面的文件选择规则。

探索模式
------------

添加 ``--explore`` 后，规划器可复位并重试任务，将经验保存到本地。记忆目录可为空；默认最多 3 个规划会话，每个会话最多尝试 5 次：

.. code-block:: bash

   rpent --robot robocasa --task-name OpenDrawer --split target --seed 0 \
     --vla-model-path /path/to/rldx \
     --planner codex --reasoning-effort high --planner-timeout-s 7200 \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir /path/to/robocasa-memory

- **保存内容：** 探索笔记先写入当前任务的草稿目录（inbox）；探索时可读取根索引 ``MEMORY.md``。
- **合并结果：** 正常结束且无智能体执行错误时，自动将通过校验的经验写入 ``task-family/`` 和 ``global/``，成功尝试的记录与动作序列写入 ``task-specific/``，并更新索引。加 ``--no-auto-merge-memory`` 可关闭自动合并。
- **用于评测：** 使用 seed 为 0 的探索产物，通过 ``--memory-profile local`` 指向同一目录。先确认 global 已发布；评测仅开放当前任务和 split 的记忆。

.. _reproduce-target50:

实验复现（Target50）
----------------------

Target50 包含 50 个厨房任务，共运行 340 次。以下使用 Codex、GPT-5.5、xhigh，配合当前任务记忆与 global memory 完成评测。

1. 准备记忆和运行参数
~~~~~~~~~~~~~~~~~~~~~~~~

先完成上文的安装、资源下载和 Codex 登录。将记忆下载到新目录，整轮评测使用同一份内容：

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
      --include 'robocasa/**' --local-dir ./target50-memory

设置动作参数，并清除可能覆盖 ``--seed`` 的旧环境变量：

.. code-block:: bash

   export RLDX_MAX_CHUNKS=40
   export RLDX_SETTLE_PATIENCE=999
   export RLDX_ACTION_STEPS_PER_CHUNK=8
   unset RLDX_RESET_SEED

2. 按任务和 seed 运行
~~~~~~~~~~~~~~~~~~~~~~~~

先运行 ``OpenDrawer`` 的 seed 1：

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer --split target --seed 1 \
         --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 \
         --planner codex --model gpt-5.5 --reasoning-effort xhigh \
         --max-turns 100 --planner-timeout-s 1800 \
         --memory-profile local \
         --memory-dir ./target50-memory/robocasa \
         --output-dir ./runs/target50/atomic/OpenDrawer_s1

接着按下表，为每个任务与 seed 组合各运行一次。任务名见 :ref:`完整任务列表 <robocasa-task-list>`，清单位于 ``robots/robocasa/eval/target50.json``。

.. list-table:: RoboCasa Target50 矩阵
   :header-rows: 1
   :widths: 30 15 20 20 15

   * - 任务组
     - 任务数
     - 每个任务的 seed 范围
     - 单次运行时限
     - 运行数
   * - Atomic
     - 18
     - 1--10
     - 1800 秒
     - 180
   * - Composite-Seen
     - 16
     - 1--5
     - 3600 秒
     - 80
   * - Composite-Unseen
     - 16
     - 1--5
     - 3600 秒
     - 80
   * - **总计**
     - **50**
     -
     -
     - **340**

运行顺序为 Atomic、Composite-Seen、Composite-Unseen。两组组合任务将时限改为 ``--planner-timeout-s 3600``，输出目录分别改为 ``composite_seen/<Task>_s<seed>`` 和 ``composite_unseen/<Task>_s<seed>``，统一放在 ``./runs/target50/`` 下。

评测期间不复位环境。任务失败和规划器超时保留原结果；只有基础设施故障导致未产生有效环境结果时才重试。

3. 检查结果
~~~~~~~~~~~~~~~~

每次运行的结果保存在输出目录的 ``result.json``，成功与否以环境的 ``state.success`` 为准。运行以下命令汇总检查：

.. code-block:: bash

   python -m robots.robocasa.eval.validate_target50 ./runs/target50

完整评测应有 340 次运行通过检查：``valid_cells=340``、``expected_cells=340``，无校验错误，退出码为 0。只运行部分任务时，程序会列出缺失结果并返回非零退出码。总体成功率对 50 个任务等权计算。

.. _codex:

公开成绩见 :doc:`排行榜 <../leaderboard/performance>`。任务列表、评测参数和公开成绩可在下方展开查阅。

.. dropdown:: 完整任务列表

   .. _robocasa-task-list:

   Seen/Unseen 指任务是否出现在预训练数据中；target 厨房场景是独立的保留场景划分，详见 `RoboCasa 数据定义 <https://robocasa.ai/docs/build/html/datasets/datasets_overview.html>`_。50 个任务分三组：

   - **Atomic (18)** —— 开合、搬运等单项操作任务： ``CloseBlenderLid``、 ``CloseFridge``、``CloseToasterOvenDoor``、``CoffeeSetupMug``、 ``NavigateKitchen``、``OpenCabinet``、``OpenDrawer``、 ``OpenStandMixerHead``、``PickPlaceCounterToCabinet``、 ``PickPlaceCounterToStove``、``PickPlaceDrawerToCounter``、 ``PickPlaceSinkToCounter``、``PickPlaceToasterToCounter``、 ``SlideDishwasherRack``、``TurnOffStove``、``TurnOnElectricKettle``、 ``TurnOnMicrowave``、``TurnOnSinkFaucet``。
   - **Composite seen (16)** —— 预训练数据中出现过的组合任务： ``ScrubCuttingBoard``、``StackBowlsCabinet``、``WashLettuce``、 ``RinseSinkBasin``、``PreSoakPan``、``StirVegetables``、 ``LoadDishwasher``、``SteamInMicrowave``、``SetUpCuttingStation``、 ``GetToastedBread``、``DeliverStraw``、``KettleBoiling``、 ``PrepareCoffee``、``StoreLeftoversInBowl``、``SearingMeat``、 ``PackIdenticalLunches``。
   - **Composite unseen (16)** —— 预训练数据中未出现过的组合任务： ``ArrangeBreadBasket``、``ArrangeTea``、 ``BreadSelection``、``CategorizeCondiments``、 ``CuttingToolSelection``、``GarnishPancake``、``GatherTableware``、 ``HeatKebabSandwich``、``MakeIceLemonade``、``PanTransfer``、 ``PortionHotDogs``、``RecycleBottlesByType``、 ``SeparateFreezerRack``、``WaffleReheat``、``WashFruitColander``、 ``WeighIngredients``。

   任选一个传给 ``--task-name`` 即可。RoboCasa 完整目录更大，参见 `RoboCasa <https://robocasa.ai>`_ 上游。

.. dropdown:: 动作参数

   .. _robocasa-action-settings:

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

.. dropdown:: 评测协议

   .. _robocasa-protocol-history:
   .. _robocasa-evaluation-protocol:

   当前评测同时使用 task-specific 与 global memory。``target50.json`` 使用 ``robocasa-harness-vla-v2`` 协议和 ``1.1`` 结果格式，参考配置为 GPT-5.5。软件依赖由 ``pyproject.toml`` 管理。自定义任务、seed 或规划器配置时，将当前协议的清单传给校验器：

   .. code-block:: bash

      python -m robots.robocasa.eval.validate_target50 /path/to/results \
         --manifest /path/to/manifest.json

   比较实验时保留同一份记忆，并在本地记录 HF commit 或文件哈希，以及源码依赖 ``rpent`` 分支实际安装的提交。程序不锁定 memory 版本，校验器也不比较各次运行的记忆正文。

   榜单成绩来自独立实验报告。采用当前协议并完成 340 次运行，并不能证明复现了某条榜单成绩；还需核对模型、记忆和代码配置。

.. dropdown:: Target50 公开成绩

   .. _robocasa-reported-results:

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

.. _environment-smoke-tests:

环境自检
------------

先检查仿真器，再检查 VLA 推理。两项均需可用的 GPU/EGL；跳过的测试不算通过，自检也不产生榜单成绩。

仿真器检查
~~~~~~~~~~~~

安装依赖和厨房资源后运行，无需规划器授权或模型权重：

.. code-block:: bash

   uv pip install pytest pytest-timeout
   RPENT_RUN_ROBOCASA_INTEGRATION=1 \
      pytest tests/integration_tests/robots/robocasa/test_target50_runtime_smoke.py -v

预期四项通过：三个任务（``OpenDrawer``、``NavigateKitchen``、``PickPlaceCounterToCabinet``，seed 1）和移动相机检查。

.. dropdown:: 仿真器检查覆盖范围

   任务测试检查环境创建与复位、12D 动作、操作相机、导航 RGB-D/world map、成功判定和关闭流程。相机测试检查底盘执行八步后的相机位姿与画面变化。这些测试使用真实仿真器，与离线 CPU 单元测试分开执行。

VLA 推理检查
~~~~~~~~~~~~

完成 :ref:`VLA 配置 <robocasa-vla-configuration>` 后，选择空闲 GPU 和新的输出目录运行。需要测试依赖时安装 ``.[test]``：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 \
   RLDX_MODEL_PATH="$PWD/checkpoints/rldx-1-ft-rc365" \
   RPENT_E2E_OUTPUT_DIR="$PWD/e2e-robocasa" \
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
   MUJOCO_GL=egl python -m pytest -q \
      tests/e2e_tests/robocasa/test_components.py::test_rldx_component --timeout=300

此项检查 VLA 服务启动、RPC 通信和首次推理，不启动规划器。离线变量仅用于这条命令；正常运行时，HF 记忆同步仍需联网。

.. _robocasa-troubleshooting:

常见问题
------------

先查看输出目录中的日志：环境启动看 ``env_server.log``，VLA 启动看 ``vla_server.log``，任务执行看 ``run.log``。

下载与资源
~~~~~~~~~~~~

- **下载慢或中断：** 包下载可设置 ``UV_HTTP_TIMEOUT=600``；模型下载可重试或使用下方镜像。保持 TLS 校验开启，并核对下载完整性，不能只看文件大小。
- **磁盘空间不足：** 缓存和临时目录需同时容纳 ZIP 与解压文件；覆盖安装还要容纳原有资源。资源目录应支持硬链接。
- **资源缺失或冲突：** 先重跑资源下载命令，并加 ``--skip-existing`` 检查。确认要替换已有文件时，再使用下方 ``--overwrite`` 命令。

.. dropdown:: 资源替换与模型下载镜像

   资源目录需包含下载集合及随包的 scene、arena、fixture 文件，并保留署名文件。默认保留内容不同的已有文件；确认替换后运行：

   .. code-block:: bash

      robocasa-download-assets --assets-path ~/.robocasa/assets --no-macros --overwrite -y

   ``--overwrite`` 优先于 ``--skip-existing``，只替换安装范围内的文件。下载中断后可重跑。

   模型下载较慢时，可改用镜像，保持相同 revision：

   .. code-block:: bash

      HF_ENDPOINT=https://hf-mirror.com hf download RLWRLD/RLDX-1-FT-RC365 \
         --revision 587e9ecdcc5e7184fcc17f58713908edff5af041 \
         --local-dir ./checkpoints/rldx-1-ft-rc365

启动与推理
~~~~~~~~~~~~

- **旧环境缺少 RLDX 接口：** 重新安装 ``.[robocasa]``，并加 ``--reinstall-package rlinf-rldx``。旧 PyPI wheel 即使版本号相同，也可能缺少所需接口。
- **缺少导航相机：** 若报错包含 ``mobilebase0_navview``，重新安装 ``.[robocasa]`` 以更新 Robosuite，不要手改 XML。
- **目录权限错误：** 使用 ``.[robocasa]`` 指定的源码依赖，XML 转换应写入临时目录。将 ``NUMBA_CACHE_DIR`` 指向可写目录，无需修改包权限。
- **CUDA/EGL 不可用：** 按本机配置选择 Torch/torchvision，再运行 :ref:`环境自检 <environment-smoke-tests>` 验证 GPU 运算和渲染；仅查看驱动版本不足以确认可用。
- **离线找不到模型文件：** 核对 :ref:`支持文件和缓存路径 <robocasa-vla-configuration>`。``NO_ALBUMENTATIONS_UPDATE=1`` 只关闭更新检查，无需因此修改图像处理或 geometry fallback 参数。
- **RPC 连接受代理影响：** ``127.0.0.1`` 和 ``localhost`` 自动直连。其他服务若也需直连，将其准确主机名加入 ``NO_PROXY`` 和 ``no_proxy``，保留远程规划器的代理设置。

记忆读取
~~~~~~~~~~~~

``read_text_file`` 找不到当前任务记忆时，检查 ``memory/robocasa/task-specific/`` 或所选本地目录，以及任务名是否正确。详见 :ref:`记忆文件选择规则 <robocasa-memory-selection>`。Atomic 任务没有发布可选的 ``<Task>.md``；系统不会用其他任务的记忆替代。

实现说明
------------

开发接入方式见 :doc:`../development/add_robot`。RoboCasa 专用的观测格式和会话处理见下方。

.. dropdown:: 观测、会话与探索细节

   - **环境工具：** toolkit 提供动作、状态读取和 ``finish``。抓取检测与动作组装在环境服务中执行；动作组件通过环境客户端渲染和执行动作，通过模型客户端调用 RLDX-1。
   - **模型输入：** 三路相机的视频张量为 ``(1, T, H, W, 3)``，``T`` 是历史帧数；另有 ``state.*`` 和 ``annotation.*`` 字段。
   - **会话隔离：** ``RpcClient`` 在 ``wait_for_ready`` 时注册私有的 ``rpc_`` + UUID 十六进制 ID。服务器将其传给 ``predict`` / ``reset_session``，隔离各客户端的 RLDX 记忆和 RTC 状态。ID 不放入观测，``rldx_skill`` / ``vla_client`` 无需处理。
   - **会话清理：** 服务器定期清理空闲会话，默认超时 3600 秒；客户端退出时通过 ``atexit`` 发送 ``session.close``。
   - **探索提示词：** ``robots/robocasa/prompts/explore.py`` 规定移动底盘、``task_progress``、RLDX 连续执行及失败记录规则。``reset`` 使用环境原有复位流程，只导出最后一次复位后成功尝试的动作序列。
