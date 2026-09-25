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

先按 :doc:`configure_planner` 配置模型服务，并用 ``rpent-check-llm`` 检查连接。以下命令运行种子为 1 的 ``OpenDrawer`` 任务：

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer \
         --split target \
         --seed 1 \
         --vla-model-path ./checkpoints/rldx-1-ft-rc365 \
         --planner claude_code \
         --model claude-opus-4-8

RoboCasa 不绑定具体 planner；可使用 ``api``、``claude_code`` 或 ``codex`` 规划器。配置方式参见 :doc:`configure_planner`。

查看结果
------------

任务是否成功由环境的 ``_check_success()`` 判定，其布尔结果记录在 ``state.success`` 中。规划器调用 ``finish`` 会结束对话，但其中声明的状态不作为评测结果。查看输出目录中的 ``result.json``、``transcript_*.json`` 和 ``run.log``；服务启动问题分别记录在 ``env_server.log``、``vla_server.log``。

可使用 ``--dashboard`` 观察相机和规划器输出，通用操作见 :doc:`dashboard`。

任务记忆
------------

使用 ``--memory-profile hf`` （评测默认值）时，RPent 会在运行前通过记忆管理器，从 `RLinf/RPent-memory 数据集 <https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/robocasa/results>`_ 将 ``robocasa/**`` 同步到 ``memory/robocasa``。在此模式下，当前任务只能读取 ``results/`` 中与该任务对应的记忆文件：

.. code-block:: text

   memory/robocasa/results/<Task>_s0.json
   memory/robocasa/results/recipe_<Task>_s0.jsonl
   memory/robocasa/results/<Task>.md  # 可选

已发布的记忆数据包含 43 个任务运行记录（audit JSON）、43 个动作序列（recipe JSONL）和 25 个任务经验文件（Markdown），共 111 个文件，不包含通用记忆。每对 JSON/JSONL 文件保存经过审核的 seed-0 运行证据。可选的 Markdown 文件记录同一任务的探索经验，可能汇总多次尝试；全部 16 个 Composite-Seen 任务和 9 个 Composite-Unseen 任务都提供该文件。提示词要求规划器在执行动作前，通过 ``read_text_file`` 读取当前任务已有的全部记忆文件；RPent 不会把这些文件的内容自动写入提示词。

使用上述公开记忆时，规划器不读取通用记忆，也不使用其他任务的记忆作为替代。7 个 Composite-Unseen 任务没有任务记忆，但仍计入评测：``HeatKebabSandwich``、``PanTransfer``、``PortionHotDogs``、``SeparateFreezerRack``、``WaffleReheat``、``WashFruitColander`` 和 ``WeighIngredients``。这些任务依靠实时观测继续执行。记忆只提供执行方法的参考；历史坐标、位姿、像素和子任务提示词不能替代当前场景定位与本次任务的完整指令。

探索模式
------------

添加 ``--explore`` 后，规划器可以复位环境并重新尝试任务，将经验写入本地记忆。与 LIBERO 相同，每次运行默认最多包含 3 个规划会话，每个会话最多尝试 5 次：

.. code-block:: bash

   rpent --robot robocasa --task-name OpenDrawer --split target --seed 0 \
     --vla-model-path /path/to/rldx \
     --planner codex --reasoning-effort high --planner-timeout-s 7200 \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir /path/to/robocasa-memory

``reset`` 使用环境原有的复位流程。运行器只导出最后一次复位后成功尝试的动作序列。探索经验先写入当前任务的本地草稿目录（inbox）；运行正常结束、未发生智能体执行错误时，会自动合并草稿。传入 ``--no-auto-merge-memory`` 可关闭自动合并。探索提示词位于 ``robots/robocasa/prompts/explore.py``，规定了移动底盘的使用、``task_progress`` 检查、RLDX 连续执行以及失败尝试的记录方式。

实验复现（Target50）
----------------------

``robots/robocasa/eval/target50.json`` 定义 Harness VLA 在 RoboCasa Target50 上的复现要求，包括 ``target`` 环境划分、Hugging Face 资源版本、记忆使用范围、任务与 seed 的组合、运行时限、成功判定和重试规则。每个任务与 seed 的组合称为一个评测单元（cell），协议 ID 为 ``robocasa-harness-vla-v1``。源码依赖使用清单中指定的 ``rpent`` 分支；每次复现都需记录实际安装的提交版本：

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

50 个任务分三组：

- **Atomic (18)** —— 单步原语的开合与搬运任务: ``CloseBlenderLid``、 ``CloseFridge``、``CloseToasterOvenDoor``、``CoffeeSetupMug``、 ``NavigateKitchen``、``OpenCabinet``、``OpenDrawer``、 ``OpenStandMixerHead``、``PickPlaceCounterToCabinet``、 ``PickPlaceCounterToStove``、``PickPlaceDrawerToCounter``、 ``PickPlaceSinkToCounter``、``PickPlaceToasterToCounter``、 ``SlideDishwasherRack``、``TurnOffStove``、``TurnOnElectricKettle``、 ``TurnOnMicrowave``、``TurnOnSinkFaucet``。
- **Composite seen (16)** —— 训练时见过的厨房布局上的多步任务: ``ScrubCuttingBoard``、``StackBowlsCabinet``、``WashLettuce``、 ``RinseSinkBasin``、``PreSoakPan``、``StirVegetables``、 ``LoadDishwasher``、``SteamInMicrowave``、``SetUpCuttingStation``、 ``GetToastedBread``、``DeliverStraw``、``KettleBoiling``、 ``PrepareCoffee``、``StoreLeftoversInBowl``、``SearingMeat``、 ``PackIdenticalLunches``。
- **Composite unseen (16)** —— 训练时 **没** 见过的布局上的多步任务（泛化测试）： ``ArrangeBreadBasket``、``ArrangeTea``、 ``BreadSelection``、``CategorizeCondiments``、 ``CuttingToolSelection``、``GarnishPancake``、``GatherTableware``、 ``HeatKebabSandwich``、``MakeIceLemonade``、``PanTransfer``、 ``PortionHotDogs``、``RecycleBottlesByType``、 ``SeparateFreezerRack``、``WaffleReheat``、``WashFruitColander``、 ``WeighIngredients``。

任选一个传给 ``--task-name`` 即可。RoboCasa 完整目录更大，参见 `RoboCasa <https://robocasa.ai>`_ 上游。

HF 模式下的普通运行同步 Hugging Face ``main`` 分支；正式 Target50 评测使用固定的记忆快照 ``551fc3157b3e56b40a3d3a3b4c7ff81721ebe89b``：

.. code-block:: bash

   hf download RLinf/RPent-memory \
      --repo-type dataset \
      --revision 551fc3157b3e56b40a3d3a3b4c7ff81721ebe89b \
      --include "robocasa/**" \
      --local-dir ./target50-memory

复现 Target50 时，先下载上文指定版本的资源，再为清单中的每个任务与 seed 组合运行一次命令。参考实验使用 Codex，配置为 ``gpt-5.5``、``xhigh``、``max_turns=100``；RoboCasa 本身也支持其他规划器。场景由 ``--seed`` 指定，不要设置 ``RLDX_RESET_SEED``。普通 RoboCasa 使用 ``max_chunks=70``，Target50 将其设为 40。运行前设置 Target50 评测使用的 RLDX 参数：

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
         --memory-dir ./target50-memory/robocasa \
         --output-dir ./runs/target50/atomic/OpenDrawer_s1

Composite-Seen 与 Composite-Unseen 使用 ``--planner-timeout-s 3600``。按 Atomic、Composite-Seen、Composite-Unseen 的顺序执行。只有最终环境记录中的 ``state.success=true`` 才计为成功，规划器的 ``finish(status=...)`` 不作为评测结果。已产生有效结果的任务失败和规划器超时均不重试；仅在基础设施故障导致该评测单元未产生有效环境结果时，才允许重试。

每条完成的命令都会原子写入 ``<output-dir>/result.json``，成功标记取自最终环境状态的 ``state.success``。文件记录实际生效的评测参数，但不保存模型服务的错误原文或凭据。全部结果按 ``<results-root>/<manifest-split>/<Task>_s<seed>/result.json`` 保存后，运行以下命令检查评测数量是否完整，并计算按任务加权的成功率：

.. code-block:: bash

   python -m robots.robocasa.eval.validate_target50 ./runs/target50


已发布的 Target50 结果
----------------------------

已发布的 Codex 复现覆盖全部 340 个评测单元，按任务汇总的结果如下：

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

`完整逐任务结果表 <https://github.com/RLinf/RPent/blob/main/robots/robocasa/eval/target50_codex_results.md>`_ 给出每个任务的成功次数和成功率。当前发布的是任务级汇总数据，不包含各 seed 的执行记录、原始轨迹或失败分类，因此不能用于逐次复核运行过程。

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
- ``read_text_file`` 报告缺少当前任务结果时，请检查 ``memory/robocasa/results/`` 目录或所选本地目录。RPent 不会读取其他任务的 memory 作为替代。 Markdown 为可选文件；Atomic 任务没有发布 ``<Task>.md``。
- 环境与 VLA 启动错误会分别记录在 ``<output_dir>/env_server.log`` 和 ``<output_dir>/vla_server.log``；运行级错误也可检查 ``<output_dir>/run.log``。
- 只有准确的 ``127.0.0.1`` 与 ``localhost`` 主机名会自动绕过 HTTP 代理。其他主机名与 IP 均遵循标准代理环境；只有该服务应当直连时，才需要把准确主机名加入 ``NO_PROXY`` 与 ``no_proxy`` 配置。

实现说明
------------

RoboCasa 通过 toolkit 提供动作、状态读取和 ``finish`` 工具。接入 RLDX-1 时需要注意两点：

- **环境服务中的辅助方法。** 抓取检测与动作组装需要访问运行中的仿真环境，因此由环境服务器通过 RPC 提供。动作组件通过环境客户端获取渲染结果、执行动作，并通过模型客户端调用 RLDX-1。接入方式见 :doc:`../development/add_robot`。
- **模型观测。** RLDX-1 接收三路相机的视频张量，形状为 ``(1, T, H, W, 3)``，其中 ``T`` 表示堆叠的历史帧；输入还包括 ``state.*`` 与 ``annotation.*`` 字段。

RPC 框架负责管理模型会话，观测中不包含会话 ID。``RpcClient`` 生成以 ``rpc_`` 开头、后接 UUID 十六进制字符串的私有 ID，并在 ``wait_for_ready`` 连接过程中向服务器注册。服务器将 ID 传给 ``predict`` 和 ``reset_session``，按客户端隔离 RLDX 的记忆与 RTC 策略状态；``rldx_skill`` 和 ``vla_client`` 不需要直接处理 ID。

服务器记录每个会话的空闲时间，并定期清理超时会话，默认超时为 3600 秒。客户端在进程退出时通过 ``atexit`` 发送 ``session.close``。
