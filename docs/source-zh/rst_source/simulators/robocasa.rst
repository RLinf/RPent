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

需要 Linux、NVIDIA GPU、可用的 CUDA/EGL 环境，以及 ``git`` 和 `uv <https://docs.astral.sh/uv/getting-started/installation/>`_。若已有 RPent 仓库，进入仓库后从创建虚拟环境开始。

RLDX-1 要求 Python ``3.10``。请创建独立环境，并通过 ``.[robocasa]`` 安装完整的 RoboCasa365 运行栈：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   uv venv --python 3.10 .venv-robocasa
   source .venv-robocasa/bin/activate

先使用 `PyTorch 官方安装选择器 <https://pytorch.org/get-started/locally/>`_ 根据本机 GPU、驱动和 Python 版本选择匹配的 CUDA 版 PyTorch 与 torchvision。在此环境执行所选命令，可将 ``pip`` 换为 ``uv pip``。RLDX 依赖要求 Torch >= 2.7、 torchvision >= 0.22；两者必须互相兼容，不能分别任意选版本。然后安装 RPent：

.. code-block:: bash

   uv pip install -e ".[robocasa]" \
      --constraint robots/robocasa/eval/target50-constraints.txt
   uv pip check

约束文件固定 Target50 所需的兼容性敏感依赖。``robocasa`` extra 从 RoboCasa、RLDX 和 Robosuite 的 ``rpent`` 分支安装；不要同时安装提供同一导入包的 ``rlinf-robocasa365``。

Torch、torchvision 与 CUDA 需要按本机配置选择。参考实验使用 Torch 2.7.0、torchvision 0.22.0 和 CUDA 12.6。每次复现都应保存实际依赖与 Git 版本，因为分支内容可能更新：

.. code-block:: bash

   uv pip freeze > installed-requirements.txt
   git rev-parse HEAD > rpent-revision.txt

``flash-attn`` 为可选依赖，未安装时 RLDX-1 使用 PyTorch SDPA。如需安装，请按 `FlashAttention 官方说明 <https://github.com/Dao-AILab/flash-attention#installation-and-features>`_ 选择兼容版本。

**安装后处理**

将厨房仿真资源（约 10 GB）下载到 ``site-packages`` 之外，重新安装 Python 包时即可保留这些资源。Target50 不使用 RoboCasa 数据集或遥操作配置；以下命令通过 ``--no-macros`` 跳过可选的本机宏配置：

.. code-block:: bash

   robocasa-download-assets --assets-path ~/.robocasa/assets --no-macros -y

命令结束时会打印需要导出的环境变量，把它加到启动 ``rpent`` 的 shell 里：

.. code-block:: bash

   export ROBOCASA_ASSETS_PATH=~/.robocasa/assets

安装器会补齐下载资源及随包的场景文件。重复运行时可加 ``--skip-existing`` 检查已有下载。资源冲突、磁盘空间和相机问题见本页的常见问题。

**RLDX-1 checkpoint**

下文的 ``--vla-model-path`` 需要指向本地 ``RLDX-1-FT-RC365`` 模型目录，即针对 RoboCasa365 微调的权重。先从 Hugging Face 下载：

.. code-block:: bash

   hf download RLWRLD/RLDX-1-FT-RC365 \
      --revision 587e9ecdcc5e7184fcc17f58713908edff5af041 \
      --local-dir ./checkpoints/rldx-1-ft-rc365

下载较慢时，可使用 Hugging Face 镜像：

.. code-block:: bash

   HF_ENDPOINT=https://hf-mirror.com hf download RLWRLD/RLDX-1-FT-RC365 \
      --revision 587e9ecdcc5e7184fcc17f58713908edff5af041 \
      --local-dir ./checkpoints/rldx-1-ft-rc365

**RLDX-1 backbone 支持文件**

FT checkpoint 虽包含权重，仍引用 ``RLWRLD/RLDX-1-VLM`` 的架构、processor 和 tokenizer。Target50 将这些支持文件固定到 ``4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7``，共 15 个不含权重的文件，约 16.4 MB（包括模型文档和图片）。下载到启动 RPent 时使用的同一缓存：

.. code-block:: bash

   export HF_HOME="$PWD/.cache/huggingface"
   export HF_HUB_CACHE="$HF_HOME/hub"
   hf download RLWRLD/RLDX-1-VLM \
      --revision 4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7 \
      --include "*.json" "*.txt" "*.jinja" "*.md" "*.png" ".gitattributes" \
      --exclude "*.safetensors.index.json"

无需额外下载基础模型权重。启动时保留上述缓存变量，不要通过 ``TRANSFORMERS_CACHE`` 指向空缓存。RoboCasa VLA worker 在普通运行和 Target50 中均自动使用上述支持文件 revision，单独启动的 RPent VLA 服务也相同；无需额外 revision 参数或手工修改缓存 ref。该固定值仅作用于 backbone 元数据，不改变 ``--vla-model-path`` 选择的微调权重。模型和 assets 的许可证独立于 RPent 代码许可证。

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

RoboCasa 不绑定具体 planner；可使用 ``api``、``claude_code`` 或 ``codex`` 规划器。配置方式参见 :doc:`../guides/configure_planner`。

查看结果
------------

任务是否成功由环境的 ``_check_success()`` 判定，其布尔结果记录在 ``state.success`` 中。规划器调用 ``finish`` 会结束对话，但其中声明的状态不作为评测结果。查看输出目录中的 ``result.json``、``transcript_*.json`` 和 ``run.log``；服务启动问题分别记录在 ``env_server.log``、``vla_server.log``。

可使用 ``--dashboard`` 观察相机和规划器输出，通用操作见 :doc:`../guides/dashboard`。

任务记忆
------------

使用 ``--memory-profile hf`` （默认值）时，CLI 与 Dashboard 从 `RLinf/RPent-memory 数据集 <https://huggingface.co/datasets/RLinf/RPent-memory/tree/release%2Fv0.1/robocasa>`_ 的 ``release/v0.1`` 分支按模型选择语料：

- ``gpt-5.5`` 对应 ``robocasa/GPT_5.5_xhigh/``。
- ``gpt-6-astra`` 对应 ``robocasa/GPT_6_astra_high/``。
- 其他模型使用 GPT-5.5 memory，并记录警告。

``--memory-version auto`` 根据实际解析出的 planner 模型选择语料；Codex 未显式指定模型时会使用 ``CODEX_MODEL``。支持 ``openai:gpt-6-astra`` 等 provider 前缀。可以通过 ``--memory-version GPT_5.5_xhigh`` 或 ``--memory-version GPT_6_astra_high`` 手动覆盖。选择语料不会改变评测模型或推理档位。

HF 分支由实现固定为 ``release/v0.1``。每次准备时在线解析该分支，仅下载所选子树，缓存按仓库和 commit 隔离。解析或下载失败会停止准备，不会改用另一份语料或 revision。离线运行应先下载，再使用 local profile。``--memory-version`` 仅用于 HF 评测；本地 memory 和探索仍通过 ``--memory-dir`` 指定目录。

所选语料根目录的结构为：

.. code-block:: text

   <selected-corpus-root>/
   ├── task-specific/
   │   ├── <Task>_s0.json
   │   ├── <Task>_s0_recipe.jsonl
   │   └── <Task>.md              # 可选
   └── global/
       └── GLOBAL_MEMORY.md

HF 评测时，RoboCasa 同时提供当前任务已有的 JSON、recipe、Markdown 和 ``global/GLOBAL_MEMORY.md``。规划器通过 ``read_text_file`` 按需查阅与当前任务相关的 task-specific 和 global 经验，自主决定读取时机和内容量；动作与任务结束不要求先读完全部文件。不提供关闭 global 层的选项。

提示词与文件工具使用相同的文件选择。RPent 文件工具拒绝读取其他任务的 memory；这是工具层限制，不是操作系统级隔离。JSON/JSONL 均缺失时，继续使用实时观测和 global；只有其中一个存在则报错。缺少可选 Markdown 会记录日志，global 文件必须存在。文件按任务名和目录直接发现，无需额外索引。CLI 在启动机器人服务前校验 memory。Dashboard 在确定任务模型后选择 HF memory，并在启动该任务的环境前完成校验；local memory 还会在共享 VLA 启动前检查 global 层。任务 memory 校验失败时，已有的共享 VLA 仍可供其他任务使用。

实时 ``task_language``、RGB-D、任务进展和工具返回优先于 memory。只有可见前提成立时才应用 global 策略。有接触、持有物体、fixture 进展或计数器上升时保持 VLA 连续调用；连续两次无接触且无可见进展后，重新定位并有限调整姿态。每次 VLA 调用都使用完整、逐字的实时任务语言。历史 ``vla_act`` 仅描述策略，执行使用当前工具，不回放历史坐标。评测时仍然不允许 reset。

使用本地 memory 时，下载到新目录后选择 local profile。使用新目录也可避免旧下载目录残留已从远端删除的文件：

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
      --revision release/v0.1 --include 'robocasa/GPT_5.5_xhigh/**' --local-dir ./target50-memory

   rpent --robot robocasa \
         --task-name OpenDrawer --split target --seed 1 \
         --vla-model-path /path/to/rldx \
         --planner codex --model gpt-5.5 --reasoning-effort xhigh \
         --memory-profile local --memory-dir ./target50-memory/robocasa/GPT_5.5_xhigh

按模型划分的语料发布到 HF ``release/v0.1``。 `reproduce/memory 归档 <https://huggingface.co/datasets/RLinf/RPent-memory/tree/reproduce/memory>`_ 保留 ``d8c25a7f`` 的 GPT-5.5 Harness-VLA 复现资源，正文和目录命名均保持原样。其中 RoboCasa 使用 ``task_only/``，当前加载器要求 ``task-specific/``，不会转换旧布局。因此，下载该归档后直接传给当前 ``--memory-profile local`` 加载器，不是受支持的复现命令。归档 README 描述的是历史行为，不是当前 CLI。本指南尚未确立与该归档配套的 RoboCasa 代码/数据快照；上面的命令选择 GPT-5.5 发布语料。

本地探索产物也可以直接用于评测，无需转换。对于 ``--task-name <Task> --split <split>``，local 评测从 ``task-specific/`` 中选择发布版 ``<Task>_s0.json`` / ``<Task>_s0_recipe.jsonl`` 文件对，或者原生 ``<Task>_<split>_s0.json`` / ``<Task>_<split>_s0_recipe.jsonl`` 文件对。任一候选只存在半对都会报错；两对同时存在时，必须用不同的 ``--memory-dir`` 目录分开，RPent 不会自动选择。两对都缺失时，可以仅使用 global 指导。

local 评测还提供 ``global/*.md``，以及 YAML frontmatter 中 ``suite: robocasa``、``regime: <split>``、``task_id: <Task>`` 均精确匹配的 ``task-family/*.md``。其他任务和 split 不会被提供；可选的 ``task-specific/<Task>.md`` 仍可读取。评测无需也不开放全语料的 ``MEMORY.md`` 索引和 ``_internal/``，而是直接列出已选择文件。包括 local profile 在内，缺少 global 都会阻止评测启动。提示词、文件权限和读取审计共用相同选择；结果记录实际 profile 和匹配的任务族身份，供校验器检查。

自定义记忆来源
~~~~~~~~~~~~~~

使用相同 ``robocasa/<memory-version>/`` 结构的其他 HF 数据集时，启动 RPent 前设置 ``RPENT_MEMORY_HF_REPO=<owner>/<dataset>``，并使用 ``--memory-profile hf``。这里接受数据集仓库 ID，不是浏览器页面 URL。

使用自定义子目录或维护分支时，下载对应子树到新目录后选择 local profile：

.. code-block:: bash

   hf download <owner>/<dataset> --repo-type dataset \
      --include '<subpath>/**' --local-dir ./custom-memory

   # 在 RPent 运行命令中加上：
   # --memory-profile local --memory-dir ./custom-memory/<subpath>

所选目录使用 ``task-specific/``，并且必须提供至少一个可读的 ``global/*.md`` 文件。选择分支时，在 HF 下载命令中加上 ``--revision <branch>``。

探索模式
------------

添加 ``--explore`` 后，规划器可以复位环境并重新尝试任务，将经验写入本地记忆。探索可从空目录开始，并可读取索引、写入当前任务的草稿目录。与 LIBERO 相同，每次运行默认最多包含 3 个规划会话，每个会话最多尝试 5 次：

.. code-block:: bash

   rpent --robot robocasa --task-name OpenDrawer --split target --seed 0 \
     --vla-model-path /path/to/rldx \
     --planner codex --reasoning-effort high --planner-timeout-s 7200 \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir /path/to/robocasa-memory

``reset`` 使用环境原有的复位流程。运行器只导出最后一次复位后成功尝试的动作序列。探索经验先写入当前任务的本地草稿目录（inbox）；运行正常结束、未发生智能体执行错误时，会自动合并草稿。传入 ``--no-auto-merge-memory`` 可关闭自动合并。探索提示词位于 ``robots/robocasa/prompts/explore.py``，规定了移动底盘的使用、``task_progress`` 检查、RLDX 连续执行以及失败尝试的记录方式。

共享记忆合并流程 将通过校验的提案发布到 ``task-family/`` 和 ``global/``，将成功的 audit/recipe 文件对复制到 ``task-specific/``，并刷新 ``MEMORY.md``。评测这些原生产物时，使用 seed-0 探索输出，以 ``--memory-profile local`` 指向同一目录。必须先有已发布的 global，评测才可启动；评测使用独立的任务/split 访问边界，探索则保留重试和 inbox 工作流程。

实验复现（Target50）
----------------------

当前 ``robots/robocasa/eval/target50_v2.json`` 协议（``robocasa-harness-vla-v2``）使用 task/global memory，不锁定数据版本。它保留 target 的 task/seed 矩阵、cell 时限、no-reset 规则、环境成功判据和 40/999/8 的 RLDX 参数。协议 ID 标识结果格式和评测规则，供校验器区分 v1 与 v2，不是 memory 数据版本选择参数。

结果记录固定的任务/global 文件选择、缺失文件和实际读取情况。校验器允许零读取和部分读取，仍检查任务访问边界和审计结构。审计文件缺失或损坏会单独报告；是否完整读取不决定环境结果的有效性或成功值。每次运行都会重新初始化读取审计，即使复用了输出目录也不继承旧记录。

校验器不比较不同运行之间的 memory 正文。HF ``release/v0.1`` 接收这些语料的更新， ``reproduce/memory`` 保持为不再变更的历史归档。使用当前布局进行可重复的对照实验时，只下载一次 memory，所有 cell 均用 ``--memory-profile local --memory-dir`` 指向同一份保持不变的目录。保留这些文件，并在本地实验记录中保存 HF commit 或哈希。需要固定 commit 时，使用 ``hf download --revision <commit>`` 下载后选择 local profile。结果元数据不包含数据版本标识。

清单定义评测矩阵和校验规则，:doc:`排行榜 <../leaderboard/performance>` 展示独立报告的成绩。340 个回合本身不能证明运行使用了哪份 memory、模型或代码配置。当前 v2 清单包含 GPT-5.5 参考配置，不能直接用于校验榜单上的所有模型。

- ``target50.json`` 保留历史 v1 task-specific 协议。校验相匹配的旧记录时传入 ``--manifest robots/robocasa/eval/target50.json``。
- ``target50_v2.json`` 描述当前 task-specific 与 global 一起使用的运行，是新结果 和校验器的默认清单。
- 榜单成绩保留各自报告的来源；没有匹配的运行证据时，不将其重新标为 v2 结果。

每个任务与 seed 的组合称为一个评测单元（cell）。源码依赖使用清单中指定的 ``rpent`` 分支；每次运行都需记录实际安装的提交版本。

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

Seen/Unseen 指任务是否出现在预训练数据中；target 厨房场景是独立的保留场景划分，详见 `RoboCasa 数据定义 <https://robocasa.ai/docs/build/html/datasets/datasets_overview.html>`_。50 个任务分三组：

- **Atomic (18)** —— 单步原语的开合与搬运任务: ``CloseBlenderLid``、 ``CloseFridge``、``CloseToasterOvenDoor``、``CoffeeSetupMug``、 ``NavigateKitchen``、``OpenCabinet``、``OpenDrawer``、 ``OpenStandMixerHead``、``PickPlaceCounterToCabinet``、 ``PickPlaceCounterToStove``、``PickPlaceDrawerToCounter``、 ``PickPlaceSinkToCounter``、``PickPlaceToasterToCounter``、 ``SlideDishwasherRack``、``TurnOffStove``、``TurnOnElectricKettle``、 ``TurnOnMicrowave``、``TurnOnSinkFaucet``。
- **Composite seen (16)** —— 预训练数据中出现过的组合任务： ``ScrubCuttingBoard``、``StackBowlsCabinet``、``WashLettuce``、 ``RinseSinkBasin``、``PreSoakPan``、``StirVegetables``、 ``LoadDishwasher``、``SteamInMicrowave``、``SetUpCuttingStation``、 ``GetToastedBread``、``DeliverStraw``、``KettleBoiling``、 ``PrepareCoffee``、``StoreLeftoversInBowl``、``SearingMeat``、 ``PackIdenticalLunches``。
- **Composite unseen (16)** —— 预训练数据中未出现过的组合任务： ``ArrangeBreadBasket``、``ArrangeTea``、 ``BreadSelection``、``CategorizeCondiments``、 ``CuttingToolSelection``、``GarnishPancake``、``GatherTableware``、 ``HeatKebabSandwich``、``MakeIceLemonade``、``PanTransfer``、 ``PortionHotDogs``、``RecycleBottlesByType``、 ``SeparateFreezerRack``、``WaffleReheat``、``WashFruitColander``、 ``WeighIngredients``。

任选一个传给 ``--task-name`` 即可。RoboCasa 完整目录更大，参见 `RoboCasa <https://robocasa.ai>`_ 上游。

进行可重复的 Target50 对照实验时，先按“任务记忆”中的下载命令准备当前语料。所有评测单元使用同一份保持不变的本地目录，并随结果保留来源 revision 或文件哈希。

运行 Target50 时，先准备上文的资源和本地记忆，再为清单中的每个任务与 seed 组合运行一次命令。参考实验使用 Codex，配置为 ``gpt-5.5``、``xhigh``、``max_turns=100``；RoboCasa 本身也支持其他规划器。场景由 ``--seed`` 指定，不要设置 ``RLDX_RESET_SEED``。普通 RoboCasa 使用 ``max_chunks=70``，Target50 将其设为 40。运行前设置 Target50 评测使用的 RLDX 参数：

.. code-block:: bash

   export RLDX_MAX_CHUNKS=40
   export RLDX_SETTLE_PATIENCE=999
   export RLDX_ACTION_STEPS_PER_CHUNK=8
   unset RLDX_RESET_SEED

以下运行 Atomic 组的 ``OpenDrawer`` 任务，使用第一个 seed：

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer --split target --seed 1 \
         --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 \
         --planner codex --model gpt-5.5 --reasoning-effort xhigh \
         --max-turns 100 --planner-timeout-s 1800 \
         --memory-profile local \
         --memory-dir ./target50-memory/robocasa/GPT_5.5_xhigh \
         --output-dir ./runs/target50/atomic/OpenDrawer_s1

Composite-Seen 与 Composite-Unseen 使用 ``--planner-timeout-s 3600``。按 Atomic、Composite-Seen、Composite-Unseen 的顺序执行。只有最终环境记录中的 ``state.success=true`` 才计为成功，规划器的 ``finish(status=...)`` 不作为评测结果。已产生有效结果的任务失败和规划器超时均不重试；仅在基础设施故障导致该评测单元未产生有效环境结果时，才允许重试。

每条完成的命令都会原子写入 ``<output-dir>/result.json``，成功标记取自最终环境状态的 ``state.success``。文件记录实际生效的评测参数，但不保存模型服务的错误原文或凭据。全部结果按 ``<results-root>/<manifest-split>/<Task>_s<seed>/result.json`` 保存后，运行以下命令检查评测数量是否完整，并计算按任务加权的成功率：

.. code-block:: bash

   python -m robots.robocasa.eval.validate_target50 ./runs/target50


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

.. _environment-smoke-tests:

环境自检
------------

安装 RoboCasa 及其 assets 后，可显式启用环境冒烟测试，检查仿真器安装与接口。测试不需要 planner 凭据或 VLA checkpoint：

.. code-block:: bash

   uv pip install pytest pytest-timeout
   RPENT_RUN_ROBOCASA_INTEGRATION=1 \
      pytest tests/integration_tests/robots/robocasa/test_target50_runtime_smoke.py -v

共四项测试：``OpenDrawer``、``NavigateKitchen``、``PickPlaceCounterToCabinet`` 各使用 seed 1，另加一项移动相机测试。任务测试检查构造/reset、12D action、操作相机、导航 RGB-D/world map、成功判定及关闭流程；相机测试检查底盘执行八步动作后的相机位姿与画面变化。这些真实仿真测试需要可用的 GPU/EGL 环境，与离线 CPU CI 分开运行；跳过不能计作通过。

常见问题
------------

资源目录必须包含下载集合和随包的 scene、arena、fixture 文件。请保留资源的署名文件。``--skip-existing`` 检查下载清单；若文件冲突，确认需要替换后再使用：

.. code-block:: bash

   robocasa-download-assets --assets-path ~/.robocasa/assets --no-macros --overwrite -y

``--overwrite`` 优先于 ``--skip-existing``，只替换安装范围内的资源文件。默认不覆盖内容不同的已有文件，并要求目标文件系统支持硬链接。磁盘应能容纳 ZIP 和解压数据；替换时还需保留原有数据所占空间。中断后可重新运行下载命令。

先执行 :ref:`环境冒烟测试 <environment-smoke-tests>`。四类资源下载完成后，使用现有 RoboCasa E2E 组件测试检查 VLA worker 启动、HTTP RPC 和首次推理。按需安装 ``.[test]``，选择一张可用 GPU，并使用新的输出目录：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 \
   RLDX_MODEL_PATH="$PWD/checkpoints/rldx-1-ft-rc365" \
   RPENT_E2E_OUTPUT_DIR="$PWD/e2e-robocasa" \
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
   MUJOCO_GL=egl python -m pytest -q \
      tests/e2e_tests/robocasa/test_components.py::test_rldx_component --timeout=300

这些检查不会启动 planner 或生成 benchmark 成绩，默认 skip 不能当作通过。离线变量仅作用于该自检命令；普通 HF memory 同步仍需要网络。保持远程 planner 的代理配置不变。轻量协议测试仍检查全部 50 个任务和固定的 340-cell 分母，全量评测单独执行。

- 下载慢时可使用 ``UV_HTTP_TIMEOUT=600``，并将缓存和临时目录放在空间足够的文件系统上。重新执行固定 revision 的 HF 下载命令；不能只看 shard 文件大小判断完整性，不要禁用 TLS 校验。
- 只读资源目录报错时，检查是否使用 ``.[robocasa]`` 指定的依赖，不要开放资源目录写权限；转换后的 XML 应写入临时目录。
- RLDX 离线缓存缺失时检查上述支持文件和缓存变量。 ``NO_ALBUMENTATIONS_UPDATE=1`` 只关闭导入时的更新检查，不改变图像处理；保持现有 image geometry fallback 参数。
- 通过 GPU 运算和 EGL render 验证所选 Torch/CUDA 构建，不能只看驱动显示版本；应使用与本机兼容的构建。
- 共享只读环境应将 ``NUMBA_CACHE_DIR`` 设置到当前用户可写目录，不要修改包的代码权限。

- 导航 RGB-D 或 world map 渲染报告缺少 ``mobilebase0_navview`` 时，应重新安装 ``.[robocasa]`` 以刷新 ``RLinf/robosuite`` 的 ``rpent`` 分支；不要手工修改已安装的 XML。
- ``read_text_file`` 报告缺少当前任务结果时，请检查 所选语料根目录下的 ``task-specific/`` 目录。RPent 不会读取其他任务的 memory 作为替代。 Markdown 为可选文件；Atomic 任务没有发布 ``<Task>.md``。
- 环境与 VLA 启动错误会分别记录在 ``<output_dir>/env_server.log`` 和 ``<output_dir>/vla_server.log``；运行级错误也可检查 ``<output_dir>/run.log``。
- 只有准确的 ``127.0.0.1`` 与 ``localhost`` 主机名会自动绕过 HTTP 代理。其他主机名与 IP 均遵循标准代理环境；只有该服务应当直连时，才需要把准确主机名加入 ``NO_PROXY`` 与 ``no_proxy`` 配置。

实现说明
------------

RoboCasa 通过 toolkit 提供动作、状态读取和 ``finish`` 工具。接入 RLDX-1 时需要注意两点：

- **环境服务中的辅助方法。** 抓取检测与动作组装需要访问运行中的仿真环境，因此由环境服务器通过 RPC 提供。动作组件通过环境客户端获取渲染结果、执行动作，并通过模型客户端调用 RLDX-1。接入方式见 :doc:`../development/add_robot`。
- **模型观测。** RLDX-1 接收三路相机的视频张量，形状为 ``(1, T, H, W, 3)``，其中 ``T`` 表示堆叠的历史帧；输入还包括 ``state.*`` 与 ``annotation.*`` 字段。

RPC 框架负责管理模型会话，观测中不包含会话 ID。``RpcClient`` 生成以 ``rpc_`` 开头、后接 UUID 十六进制字符串的私有 ID，并在 ``wait_for_ready`` 连接过程中向服务器注册。服务器将 ID 传给 ``predict`` 和 ``reset_session``，按客户端隔离 RLDX 的记忆与 RTC 策略状态；``rldx_skill`` 和 ``vla_client`` 不需要直接处理 ID。

服务器记录每个会话的空闲时间，并定期清理超时会话，默认超时为 3600 秒。客户端在进程退出时通过 ``atexit`` 发送 ``session.close``。
