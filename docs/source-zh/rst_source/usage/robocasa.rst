RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ 是面向厨房场景的长时序操作仿真环境。
在 RPent 中由 **RLDX-1** VLA 策略驱动，默认通过 HTTP RPC 提供服务
（与 LIBERO 一致），也支持 pickle-framed socket 传输。详见
``robots/robocasa/vla_server.py`` 与 ``robots/robocasa/robot_spec.py``
中的传输选择逻辑。

.. note::

   当前 task/global 协议为 ``robots/robocasa/eval/target50_v2.json``，
   ``target50.json`` 保留旧 v1 协议。
   340 个 cell 均使用普通的单任务 ``rpent --robot robocasa`` 命令。

运行流程
--------

RoboCasa365 使用 PandaOmron 移动机械臂与冻结的 RLDX-1 策略。
接入层与 planner 无关，API planner、Claude Code 和 Codex 使用相同的 RoboCasa
工具接口。后端配置参见 :doc:`configure_planner`；凭据由用户在仓库外提供。

.. code-block:: text

   rpent CLI -> task-memory sync -> environment and VLA servers
             -> planner toolkit -> final environment state.success

未指定外部 endpoint 时，RPent 为每次运行启动环境服务器和 VLA 服务器。
Planner 根据实时任务语言和观测选择 primitive，RLDX-1 执行操作技能。
评测成功只取环境自身 ``_check_success()`` 暴露的 ``state.success``。
公开协议使用普通单-cell 命令，不包含批量启动器；Harness VLA 的总体说明参见
:doc:`../awesome_works/harnessvla`。

安装
----

RLDX-1 要求 Python ``3.10``。请创建独立环境，并通过 ``.[robocasa]``
安装完整的 RoboCasa365 运行栈：

.. code-block:: bash

   uv venv --python 3.10
   source .venv/bin/activate

先使用 `PyTorch 官方安装选择器 <https://pytorch.org/get-started/locally/>`_
根据本机 GPU、驱动和 Python 版本选择匹配的 CUDA 版 PyTorch 与 torchvision。
在此环境执行所选命令，可将 ``pip`` 换为 ``uv pip``。RLDX 依赖要求 Torch >= 2.7、
torchvision >= 0.22；两者必须互相兼容，不能分别任意选版本。然后安装 RPent：

.. code-block:: bash

   uv pip install -e ".[robocasa]" \
      --constraint robots/robocasa/eval/target50-constraints.txt
   uv pip check

RoboCasa 专用 constraints 文件固定经 Target50 复现验证的兼容性敏感包版本，
同时不会收窄 RPent 中 LIBERO 或 RoboTwin 的共享依赖。``robocasa`` extra 跟随
RoboCasa、RLDX 和 Robosuite 仓库维护中的 ``rpent`` 分支，普通运行与 Target50
使用同一安装方式。Manifest 记录分支，不冻结源码 commit。RoboCasa 的
``rpent`` 分支声明的发行包名为 ``rpent-robocasa365``；不要同时安装提供相同
import 包的 ``rlinf-robocasa365``。无需再次安装固定 SHA 的源码。
分支可能更新，因此每次评测都应记录实际解析的 Git commit 和安装版本：

.. code-block:: bash

   uv pip freeze > installed-requirements.txt

将此环境记录与实验产物一起保存；稍后再次安装同一分支并不保证源码相同。
下文的 checkpoint 和 backbone 支持文件仍使用固定 HF snapshot；
task/global memory 跟随所选分支。
constraints 不固定 Torch、torchvision 或 CUDA backend，安装时
保留已安装且兼容的版本组合；依赖冲突必须先解决再运行。Manifest 的
``reference_accelerator`` 仅记录此前使用的 Torch 2.7.0 / torchvision 0.22.0 /
CUDA 12.6，属于来源记录而非安装要求。请随结果记录实际版本，并执行下方组件
自检；不预先假定其他组合的数值结果完全相同。软件源镜像属于用户自行选择的
网络配置，不是评测协议的一部分。

.. note::

   flash-attn 是可选的，未安装时 RLDX-1 使用 PyTorch SDPA。若要安装，请按
   `FlashAttention 官方说明
   <https://github.com/Dao-AILab/flash-attention#installation-and-features>`_
   选择与本机 Python、Torch、CUDA 和 GPU 匹配的构建；本文不指定机器专用 wheel。

**安装后处理**

将厨房 assets（约 10 GB）下载到 ``site-packages`` 之外，重装不会丢。
Target50 不使用 RoboCasa dataset 或 teleop macros，因此跳过可选的 private
macros 配置：

.. code-block:: bash

   robocasa-download-assets --assets-path ~/.robocasa/assets --no-macros -y

命令结束时会打印需要导出的环境变量，把它加到启动 ``rpent`` 的 shell 里：

.. code-block:: bash

   export ROBOCASA_ASSETS_PATH=~/.robocasa/assets

外置根目录需要同时包含六类下载资源，以及发行包内的 scene、arena 和 fixture
静态文件。修正后的安装器会补齐静态文件，不覆盖内容不同的已有文件。
``--skip-existing`` 会检查成功下载的文件清单，不再把非空目录当作完整安装。
请保留官方 attribution 文件，并在实验开始前完成中断的下载。

资源以原子方式发布，中断复制不会留下半写入的正式文件。若旧安装器已经留下
内容冲突的资源，可在原命令中显式增加覆盖权限进行修复：

.. code-block:: bash

   robocasa-download-assets --assets-path ~/.robocasa/assets --no-macros --overwrite -y

``--overwrite`` 优先于 ``--skip-existing``，仅替换本次安装范围内的资源文件，
不删除无关文件或整个目录；未指定时，内容不同的已有文件仍受保护。
默认原子不覆盖发布要求目标文件系统支持硬链接。强制终止遗留的临时文件不会
阻止重试。

新资源集合主要需要 ZIP 与一份解压数据的空间，staging 发布时不再复制 payload；
替换已有安装时还需为旧资源占用预留空间。``--skip-existing`` 会跳过已完成集合的
下载和内容比较，随包静态文件单独检查。

**移动相机**

``robocasa`` extra 会安装 ``RLinf/robosuite`` 的 ``rpent`` 分支，该分支
包含 Omron 底盘固定的 ``navview`` 相机，其组合后的 MuJoCo 相机名为
``mobilebase0_navview``。导航 RGB-D 与 world map 渲染会在首次请求时验证该
相机，并在缺失时明确报错。无需手工修改
``site-packages`` 中的 XML。Target50 同样使用此维护分支；请按上文随环境
信息记录实际解析的 revision。

**RLDX-1 checkpoint**

下面运行命令的 ``--vla-model-path`` 期望一个本地 ``RLDX-1-FT-RC365``
checkpoint 路径（RoboCasa365 微调版）。从 HuggingFace 下载:

.. code-block:: bash

   hf download RLWRLD/RLDX-1-FT-RC365 \
      --revision 587e9ecdcc5e7184fcc17f58713908edff5af041 \
      --local-dir ./checkpoints/rldx-1-ft-rc365

下载慢的话用 HF 镜像:

.. code-block:: bash

   HF_ENDPOINT=https://hf-mirror.com hf download RLWRLD/RLDX-1-FT-RC365 \
      --revision 587e9ecdcc5e7184fcc17f58713908edff5af041 \
      --local-dir ./checkpoints/rldx-1-ft-rc365

**RLDX-1 backbone 支持文件**

FT checkpoint 虽包含权重，仍引用 ``RLWRLD/RLDX-1-VLM`` 的架构、processor
和 tokenizer。Target50 将此第四类资源固定到
``4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7``，共 15 个非权重文件，约
16.4 MB（包括模型文档和图片）。下载到启动 RPent 时使用的同一缓存：

.. code-block:: bash

   export HF_HOME="$PWD/.cache/huggingface"
   export HF_HUB_CACHE="$HF_HOME/hub"
   hf download RLWRLD/RLDX-1-VLM \
      --revision 4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7 \
      --include "*.json" "*.txt" "*.jinja" "*.md" "*.png" ".gitattributes" \
      --exclude "*.safetensors.index.json"

无需额外下载基础模型权重。启动时保留上述缓存变量，不要通过
``TRANSFORMERS_CACHE`` 指向空缓存。RoboCasa VLA worker 在普通运行和 Target50
中均自动使用上述支持文件 revision，单独启动的 RPent VLA 服务也相同；无需额外
revision 参数或手工修改缓存 ref。该固定值仅作用于 backbone 元数据，不改变
``--vla-model-path`` 选择的微调权重。
模型和 assets 的许可证独立于 RPent 代码许可证。

**任务与 Global Memory**

``--memory-profile hf``（默认值）下，CLI 与 Dashboard 均从
`RLinf/RPent-memory 数据集
<https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/robocasa>`_
当前的 ``main`` 分支同步 ``robocasa/**``，不锁定数据 commit。目录结构为：

.. code-block:: text

   memory/robocasa/
   ├── task-specific/
   │   ├── <Task>_s0.json
   │   ├── <Task>_s0_recipe.jsonl
   │   └── <Task>.md              # 可选
   └── global/
       └── GLOBAL_MEMORY.md

HF 评测时，RoboCasa 同时提供当前任务已有的 JSON、recipe、Markdown 和
``global/GLOBAL_MEMORY.md``。规划器通过 ``read_text_file`` 按需查阅与当前任务
相关的 task-specific 和 global 经验，自主决定读取时机和内容量；动作与任务结束
不要求先读完全部文件。不提供关闭 global 层的选项。

提示词与文件工具使用相同的文件选择。RPent 文件工具拒绝读取其他任务的 memory；
这是工具层限制，不是操作系统级隔离。JSON/JSONL 均缺失时，继续使用实时观测和
global；只有其中一个存在则报错。缺少可选 Markdown 会记录日志，global 文件
必须存在。文件按任务名和目录直接发现，无需额外索引。
CLI 在启动机器人服务前校验 memory。Dashboard 在启动共享 VLA 前检查 memory
目录和 global 层；选定任务后，先检查该任务的文件，再启动其环境。
任务 memory 校验失败时，已有的共享 VLA 仍可供其他任务使用。

实时 ``task_language``、RGB-D、任务进展和工具返回优先于 memory。
只有可见前提成立时才应用 global 策略。有接触、持有物体、fixture 进展或计数器
上升时保持 VLA 连续调用；连续两次无接触且无可见进展后，重新定位并有限调整姿态。
每次 VLA 调用都使用完整、逐字的实时任务语言。历史 ``vla_act`` 仅描述策略，
执行使用当前工具，不回放历史坐标。评测时仍然不允许 reset。

使用本地 memory 时，下载到新目录后选择 local profile。使用新目录也可避免旧
下载目录残留已从远端删除的文件：

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
      --include 'robocasa/**' --local-dir ./target50-memory

   rpent --robot robocasa \
         --task-name OpenDrawer --split target --seed 1 \
         --vla-model-path /path/to/rldx \
         --planner codex --model gpt-5.5 --reasoning-effort xhigh \
         --memory-profile local --memory-dir ./target50-memory/robocasa

使用维护中的复现 memory 分支时，在下载命令中加上
``--revision reproduce/memory``。它选择可更新的分支，不锁定数据版本。
切换分支请使用新目录。两个分支的 RoboCasa 均采用上述 task/global 结构；
后续更新对应的任务或 global 文件即可，无需修改代码。

本地探索产物也可以直接用于评测，无需转换。对于
``--task-name <Task> --split <split>``，local 评测从 ``task-specific/`` 中
选择发布版 ``<Task>_s0.json`` / ``<Task>_s0_recipe.jsonl`` 文件对，或者原生
``<Task>_<split>_s0.json`` / ``<Task>_<split>_s0_recipe.jsonl`` 文件对。
任一候选只存在半对都会报错；两对同时存在时，必须用不同的 ``--memory-dir``
目录分开，RPent 不会自动选择。两对都缺失时，可以仅使用 global 指导。

local 评测还提供 ``global/*.md``，以及 YAML frontmatter 中
``suite: robocasa``、``regime: <split>``、``task_id: <Task>`` 均精确匹配的
``task-family/*.md``。其他任务和 split 不会被提供；可选的
``task-specific/<Task>.md`` 仍可读取。评测无需也不开放全语料的 ``MEMORY.md``
索引和 ``_internal/``，而是直接列出已选择文件。包括 local profile 在内，缺少
global 都会阻止评测启动。提示词、文件权限和读取审计共用相同选择；结果记录
实际 profile 和匹配的任务族身份，供校验器检查。

自定义 memory 来源
~~~~~~~~~~~~~~~~~~

使用相同 ``robocasa/`` 结构的其他 HF 数据集时，启动 RPent 前设置
``RPENT_MEMORY_HF_REPO=<owner>/<dataset>``，并使用
``--memory-profile hf``。这里接受数据集仓库 ID，不是浏览器页面 URL。

使用自定义子目录或维护分支时，下载对应子树到新目录后选择 local profile：

.. code-block:: bash

   hf download <owner>/<dataset> --repo-type dataset \
      --include '<subpath>/**' --local-dir ./custom-memory

   # 在 RPent 运行命令中加上：
   # --memory-profile local --memory-dir ./custom-memory/<subpath>

所选目录使用 ``task-specific/``，并且必须提供至少一个可读的
``global/*.md`` 文件。选择分支时，在 HF 下载命令中加上
``--revision <branch>``。无需交付包或迁移脚本。

Harness VLA Target50 复现协议
-----------------------------

当前 ``robots/robocasa/eval/target50_v2.json`` 协议
（``robocasa-harness-vla-v2``）使用 task/global memory，不锁定数据版本。
它保留 target 的 task/seed 矩阵、cell 时限、no-reset 规则、环境成功判据和
40/999/8 的 RLDX 参数。协议 ID 标识结果格式和评测规则，供校验器区分 v1 与 v2，
不是 memory 数据版本选择参数。

结果记录固定的任务/global 文件选择、缺失文件和实际读取情况。校验器允许零读取
和部分读取，仍检查任务访问边界和审计结构。审计文件缺失或损坏会单独报告；
是否完整读取不决定环境结果的有效性或成功值。每次运行都会重新初始化读取审计，
即使复用了输出目录也不继承旧记录。

校验器不比较不同运行之间的 memory 正文。``main`` 和 ``reproduce/memory`` 都是
维护中的分支。需要可重复的对照实验时，只下载一次 memory，所有 cell 均用
``--memory-profile local --memory-dir`` 指向同一份保持不变的目录。保留这些文件，
并在本地实验记录中保存 HF commit 或哈希。RPent 不固定 memory，也不向结果元数据
添加数据版本标识。

清单定义评测矩阵和校验规则，:doc:`排行榜 <../leaderboard/performance>` 展示独立
报告的成绩。340 个回合本身不能证明运行使用了哪份 memory、模型或代码配置。
当前 v2 清单包含 GPT-5.5 参考配置，不能直接用于校验榜单上的所有模型。

- ``target50.json`` 保留历史 v1 task-specific 协议。校验相匹配的旧记录时传入
  ``--manifest robots/robocasa/eval/target50.json``。
- ``target50_v2.json`` 描述当前 task-specific 与 global 一起使用的运行，是新结果
  和校验器的默认清单。
- 榜单成绩保留各自报告的来源；没有匹配的运行证据时，不将其重新标为 v2 结果。

源码依赖跟随清单记录的 ``rpent`` 分支。

.. list-table:: RoboCasa Target50 矩阵
   :header-rows: 1
   :widths: 30 15 20 20 15

   * - Split
     - 任务数
     - 每任务 seed
     - Cell 时限
     - Cells
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

50 个任务分三组:

- **Atomic (18)** —— 单步原语的开合与搬运任务: ``CloseBlenderLid``、
  ``CloseFridge``、``CloseToasterOvenDoor``、``CoffeeSetupMug``、
  ``NavigateKitchen``、``OpenCabinet``、``OpenDrawer``、
  ``OpenStandMixerHead``、``PickPlaceCounterToCabinet``、
  ``PickPlaceCounterToStove``、``PickPlaceDrawerToCounter``、
  ``PickPlaceSinkToCounter``、``PickPlaceToasterToCounter``、
  ``SlideDishwasherRack``、``TurnOffStove``、``TurnOnElectricKettle``、
  ``TurnOnMicrowave``、``TurnOnSinkFaucet``。
- **Composite seen (16)** —— 训练时见过的厨房布局上的多步任务:
  ``ScrubCuttingBoard``、``StackBowlsCabinet``、``WashLettuce``、
  ``RinseSinkBasin``、``PreSoakPan``、``StirVegetables``、
  ``LoadDishwasher``、``SteamInMicrowave``、``SetUpCuttingStation``、
  ``GetToastedBread``、``DeliverStraw``、``KettleBoiling``、
  ``PrepareCoffee``、``StoreLeftoversInBowl``、``SearingMeat``、
  ``PackIdenticalLunches``。
- **Composite unseen (16)** —— 训练时 **没** 见过的布局上的多步任务
  （泛化测试）: ``ArrangeBreadBasket``、``ArrangeTea``、
  ``BreadSelection``、``CategorizeCondiments``、
  ``CuttingToolSelection``、``GarnishPancake``、``GatherTableware``、
  ``HeatKebabSandwich``、``MakeIceLemonade``、``PanTransfer``、
  ``PortionHotDogs``、``RecycleBottlesByType``、
  ``SeparateFreezerRack``、``WaffleReheat``、``WashFruitColander``、
  ``WeighIngredients``。

任选一个传给 ``--task-name`` 即可。RoboCasa 完整目录更大，参见
`RoboCasa <https://robocasa.ai>`_ 上游。

运行一个任务
------------

HTTP RPC endpoint 的主机名为 ``127.0.0.1`` 或 ``localhost`` 时一律直连，无论
worker 由 RPent 启动还是由用户指定。其他主机名与 IP 均遵循标准代理环境。Codex
只在其子进程环境中为本地 MCP 应用相同的两个主机名例外。如果 Hugging Face、
远程 planner 或其他远程服务需要 ``HTTP_PROXY`` 或 ``HTTPS_PROXY``，请保持原有
代理；默认运行不要求 shell 统一配置 ``NO_PROXY``。

如果用户指定的本地服务使用其他主机名或 IP 且应当直连，请将该准确值加入用户
已有的 ``NO_PROXY`` 与 ``no_proxy`` 配置。

RoboCasa 的 CLI 参数由 ``robots/robocasa/__init__`` 注册，可通过
``rpent --robot robocasa --help`` 查看:

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer \
         --split target \
         --seed 1 \
         --vla-model-path /path/to/rldx \
         --planner claude_code \
         --model claude-opus-4-8

RoboCasa 不绑定具体 planner；RPent 支持的任意 planner 都可用于该机器人。
配置方式参见 :doc:`configure_planner`。

正式 Target50 先按上文下载固定资源，再为 manifest 中每个 cell 调用一次普通
命令。Codex 参考 profile 为 ``gpt-5.5``、``xhigh``、``max_turns=100``；
RoboCasa 运行时本身仍与 planner 解耦。场景身份直接使用普通 ``--seed`` 参数，
不要设置 ``RLDX_RESET_SEED``。普通 RoboCasa 使用 ``max_chunks=70``，只有
Target50 将其覆盖为 40。运行前固定 Target50 的 RLDX 执行参数：

.. code-block:: bash

   export RLDX_MAX_CHUNKS=40
   export RLDX_SETTLE_PATIENCE=999
   export RLDX_ACTION_STEPS_PER_CHUNK=8
   unset RLDX_RESET_SEED

第一个 ``OpenDrawer`` Atomic cell 示例：

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer --split target --seed 1 \
         --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 \
         --planner codex --model gpt-5.5 --reasoning-effort xhigh \
         --max-turns 100 --planner-timeout-s 1800 \
         --memory-profile local \
         --memory-dir ./target50-memory/robocasa \
         --output-dir ./runs/target50/atomic/OpenDrawer_s1

Composite-Seen 与 Composite-Unseen 使用 ``--planner-timeout-s 3600``。执行顺序为
Atomic、Composite-Seen、Composite-Unseen。成功只认最终环境记录中的
``state.success=true``，planner 的 ``finish(status=...)`` 不是评测标签。有效任务
失败与 planner timeout 不重跑；只有没有产生有效环境结果的基础设施失败允许重跑。

每条完成的命令都会原子写入 ``<output-dir>/result.json``，成功值只来自最终环境
``state.success``。文件记录有效协议参数，但不保存 provider 错误原文或凭据。全部
cell 按 ``<results-root>/<manifest-split>/<Task>_s<seed>/result.json`` 落盘后，
用下面命令校验固定分母并输出任务加权指标：

.. code-block:: bash

   python -m robots.robocasa.eval.validate_target50 ./runs/target50

.. note::

   使用 ``--env-endpoint`` / ``--vla-endpoint`` 指向已运行的服务器
   (``[protocol://]host:port``)；不指定时，RPent 会就地启动 env 和 VLA
   子进程，日志分别写到 ``<output_dir>/env_server.log`` 和
   ``<output_dir>/vla_server.log``。

已报告成绩与历史记录
--------------------

当前 GPT-5.5 公开成绩以 :doc:`排行榜 <../leaderboard/performance>` 为准：
**Overall 57.1%**、**Atomic-Seen 92.0%**、**Composite-Seen 61.0%**、
**Composite-Unseen 13.8%**。Overall 对 50 个任务等权计算，不是 340 回合中的
成功回合占比。

Astra 的报告值为 **Overall 59.20%**，三个分项分别为
**87.78% / 43.75% / 42.50%**。回合数已根据
`实验贡献者确认的更正
<https://github.com/RLinf/RPent/pull/205#issuecomment-5749514622>`_
同步为 **340（180/80/80）**；此前的 250 回合信息属于尚未同步的历史记录。
此次更正保留已报告成功率，不由四舍五入后的比率推算成功次数，也不代表重新
核验了全部 340 份原始结果。

`归档的逐任务复现表格
<https://github.com/RLinf/RPent/blob/57088f6df30b227f2229ead985aa75403c0ce291/robots/robocasa/eval/target50_codex_results.md>`_
保留在原始 commit 中，属于独立的历史记录，不用于推导当前榜单数值或证明 v2 成绩。

.. _environment-smoke-tests:

环境冒烟测试
------------

安装 RoboCasa 及其 assets 后，可显式启用环境冒烟测试，检查仿真器安装与接口。
测试不需要 planner 凭据或 VLA checkpoint：

.. code-block:: bash

   uv pip install pytest pytest-timeout
   RPENT_RUN_ROBOCASA_INTEGRATION=1 \
      pytest tests/integration_tests/robots/robocasa/test_target50_runtime_smoke.py -v

共四项测试：``OpenDrawer``、``NavigateKitchen``、``PickPlaceCounterToCabinet``
各使用 seed 1，另加一项移动相机测试。任务测试检查构造/reset、12D action、
操作相机、导航 RGB-D/world map、成功判定及关闭流程；相机测试检查底盘执行
八步动作后的相机位姿与画面变化。这些真实仿真测试需要可用的 GPU/EGL 环境，
与离线 CPU CI 分开运行；跳过不能计作通过。

常见错误
--------

先执行 :ref:`环境冒烟测试 <environment-smoke-tests>`。四类资源下载完成后，
使用现有 RoboCasa E2E 组件测试检查 VLA worker 启动、HTTP RPC 和首次推理。
按需安装 ``.[test]``，选择一张可用 GPU，并使用新的输出目录：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 \
   RLDX_MODEL_PATH="$PWD/checkpoints/rldx-1-ft-rc365" \
   RPENT_E2E_OUTPUT_DIR="$PWD/e2e-robocasa" \
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
   MUJOCO_GL=egl python -m pytest -q \
      tests/e2e_tests/robocasa/test_components.py::test_rldx_component --timeout=300

这些检查不会启动 planner 或生成 benchmark 成绩，默认 skip 不能当作通过。
离线变量仅作用于该自检命令；普通 HF memory 同步仍需要网络。保持远程 planner
的代理配置不变。
轻量协议测试仍检查全部 50 个任务和固定的 340-cell 分母，全量评测单独执行。

- 下载慢时可使用 ``UV_HTTP_TIMEOUT=600``，并将缓存和临时目录放在空间足够的
  文件系统上。重新执行固定 revision 的 HF 下载命令；不能只看 shard 文件大小
  判断完整性，不要禁用 TLS 校验。
- 只读 assets 报错需要安装修正后的 RoboCasa 依赖，不要开放资源目录写权限；
  转换后的 XML 应写入临时目录。
- RLDX 离线缓存缺失时检查上述支持文件和缓存变量。
  ``NO_ALBUMENTATIONS_UPDATE=1`` 只关闭导入时的更新检查，不改变图像处理；
  保持现有 image geometry fallback 参数。
- 通过 GPU 运算和 EGL render 验证所选 Torch/CUDA 构建，不能只看驱动显示
  版本；应使用与本机兼容的构建。
- 共享只读环境应将 ``NUMBA_CACHE_DIR`` 设置到当前用户可写目录，不要修改包的
  代码权限。

- 导航 RGB-D 或 world map 渲染报告缺少 ``mobilebase0_navview`` 时，应重新
  安装 ``.[robocasa]`` 以刷新 ``RLinf/robosuite`` 的 ``rpent`` 分支；不要手工
  修改已安装的 XML。
- ``read_text_file`` 报告缺少当前任务结果时，请检查
  ``memory/robocasa/task-specific/`` 目录或所选本地目录。RPent 不会读取其他任务的
  memory 作为替代。
  Markdown 为可选文件；Atomic 任务没有发布 ``<Task>.md``。
- 环境与 VLA 启动错误会分别记录在 ``<output_dir>/env_server.log`` 和
  ``<output_dir>/vla_server.log``；运行级错误也可检查 ``<output_dir>/run.log``。
- 只有准确的 ``127.0.0.1`` 与 ``localhost`` 主机名会自动绕过 HTTP 代理。其他
  主机名与 IP 均遵循标准代理环境；只有该服务应当直连时，才需要把准确主机名
  加入 ``NO_PROXY`` 与 ``no_proxy`` 配置。

Toolkit 与 LIBERO 的差异
------------------------

RoboCasa toolkit 提供的工具 *形式* 与 LIBERO 相同（一次原语调用、
一次状态查看、一次 ``finish``），但有两处 RoboCasa 特有的差异:

- **Env 侧的辅助方法。** 抓取检测与动作组装需要运行中的仿真 env, 所以
  它们是 env_server 的 RPC。Agent 侧的 skill 因此同时持有 **两个**
  client: env client 做 render/step, model client 做 RLDX-1 推理。
  理由参见 :doc:`../development/add_robot`。
- **观测形状。** RLDX-1 看到的是 3 路相机的视频张量
  ``(1, T, H, W, 3)``, 按历史 ``T`` 堆叠, 加上 ``state.*`` 与
  ``annotation.*`` 字段。session id 不在观测里——它由 RPC 框架自动
  管理: ``RpcClient`` 生成 ``rpc_`` + uuid hex 的私有 session id,
  ``wait_for_ready`` 在连接时注册到服务端; 服务端跟踪每个 session
  的空闲时间, 后台 sweep 线程定期回收超时 (默认 3600 秒) 的 session,
  进程退出时客户端通过 atexit 发送 ``session.close``。业务代码
  (``rldx_skill`` / ``vla_client``) 从不直接看到 session id, 服务端
  把它注入到 ``predict`` / ``reset_session`` 中, 按客户端隔离
  RLDX memory/RTC 策略状态。

探索模式
--------

添加 ``--explore`` 后，规划器可以在新的 episode 中重新尝试任务，并写入本地
memory。与 LIBERO 相同，默认每次运行包含 3 个 planner session，每个 session
最多尝试 5 次：

.. code-block:: bash

   rpent --robot robocasa --task-name OpenDrawer --split target --seed 0 \
     --vla-model-path /path/to/rldx \
     --planner codex --reasoning-effort high --planner-timeout-s 7200 \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir /path/to/robocasa-memory

``reset`` 沿用环境原有的 episode reset。runner 只导出最后一次 reset 后的获胜
命令。探索产生的 memory 写入当前本地 inbox；除非传入
``--no-auto-merge-memory``，否则运行结束后会自动合并。探索 prompt 来自
``robots/robocasa/prompts/explore.py``，包含移动底盘、``task_progress``、
RLDX 连续性和失败 attempt 归档规则。

探索可以从空 memory 目录开始，并可读取索引、写入本轮 inbox。共享 memory merge
将通过校验的提案发布到 ``task-family/`` 和 ``global/``，将成功的 audit/recipe
文件对复制到 ``task-specific/``，并刷新 ``MEMORY.md``。评测这些原生产物时，
使用 seed-0 探索输出，以 ``--memory-profile local`` 指向同一目录。
必须先有已发布的 global，评测才可启动；评测使用独立的任务/split 访问边界，
探索则保留重试和 inbox 工作流程。
