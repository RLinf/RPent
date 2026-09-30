.. _memory-management:

使用记忆与探索模式
===========================

记忆用于复用任务经验；探索模式通过多次尝试生成本地经验。先完成 :doc:`../get_started/quickstart`，再按本页示例使用 LIBERO 记忆。其他环境的任务参数和成功判定见各自页面。

评测模式只读取记忆，不更新记忆；探索可以复位并尝试多次。正式复现应使用实验指定的记忆版本和评测模式。

使用公开记忆
------------------

评测默认使用 ``--memory-profile hf``，从公开数据集 `RLinf/RPent-memory <https://huggingface.co/datasets/RLinf/RPent-memory>`_ 下载记忆。LIBERO 默认使用 ``--memory-version auto`` 按模型选择：

.. list-table:: LIBERO memory 版本
   :header-rows: 1
   :widths: 25 35 40

   * - 当前运行模型
     - ``libero/`` 下的 memory 目录
     - 探索生成配置
   * - ``gpt-5.5``
     - ``GPT_5.5_xhigh``
     - Codex，Reasoning 开启，xhigh
   * - ``gpt-6-astra``
     - ``GPT_6_astra_low``
     - Codex，Reasoning 开启，low

支持 ``openai:`` 等提供方前缀。Codex 优先使用 ``--model``，其次使用 ``CODEX_MODEL``。其他模型、Claude 或无法确定的默认模型会回退到 ``GPT_5.5_xhigh``，并输出提示。Flash 重放默认选择 GPT-5.5。显式指定版本优先于自动选择，不改变当前模型或 reasoning effort；目录名中的 effort 只说明该 memory 的探索生成配置。

.. code-block:: bash

   # 自动选择 Astra memory。
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-6-astra --reasoning-effort low

   # 同一模型使用 GPT-5.5 memory。
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-6-astra --reasoning-effort low \
     --memory-version GPT_5.5_xhigh

CLI 和 Dashboard 都在每个任务开始前解析 memory 根目录。在 Dashboard 的 **下一任务的模型** 中修改模型后，下一任务会重新进行自动选择；正在运行的任务保留原模型和 memory。显式指定的 memory 版本不会随模型切换而改变。

仅下载所选版本。LIBERO 缓存位于 ``memory/libero/.versions/``，按仓库、提交和版本隔离，每次复用前校验文件集合完全一致及每份文件的哈希。额外文件会使缓存失效，固定 revision 时也不例外；联网同步会重建无效缓存。``HF_HUB_OFFLINE=1`` 要求所选版本及 revision 已有完整、未改动的缓存。下载失败不会改用另一模型的 memory；缓存缺失或不完整会明确报错。旧缓存记录未标明版本来源时，需要联网成功刷新一次；不复用旧的无版本缓存。其他机器人保持原有的 memory 同步行为。

独立下载与本地评测
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m robots.libero.memory sync --memory-version GPT_6_astra_low
   python -m robots.libero.memory sync --model gpt-6-astra \
     --revision <release-commit> --output-dir /path/to/new-astra-memory
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-6-astra --reasoning-effort low \
     --memory-profile local --memory-dir /path/to/new-astra-memory

``sync`` 输出实际 memory 根目录，``--output-dir`` 必须是尚不存在的目录。 ``--planner`` 默认是 ``api``，与 ``rpent`` 一致；希望在省略 ``--model`` 时读取 ``CODEX_MODEL``，需指定 ``--planner codex``。 ``--memory-profile local`` 不下载 memory；本地模式或 ``--explore`` 与显式远程 ``--memory-version`` 同时使用会报参数冲突。探索使用本地 memory，每次独立探索应指定单独的空目录。

分版本下载使用 LIBERO 专用命令；共享的 ``rpent-memory`` 继续提供 ``merge``、 ``validate`` 和 ``build-index`` 三个命令。

发布来源与兼容性
~~~~~~~~~~~~~~~~

数据集中的 ``libero/README.md`` 和 ``libero/manifest.json`` 记录版本、原始来源快照和发布文件。各版本的 ``files`` 保存相对于该版本根目录的路径与发布文件的 SHA-256，加载器据此校验实际内容。来源 revision 和来源哈希描述原始快照；目录、索引及正文引用调整后，发布哈希另行更新，不改写原始来源哈希。

两个版本根目录都使用 ``MEMORY.md``、``global/``、``task-family/`` 和 ``task-specific/``。GPT-5.5 还在 ``flash/`` 中包含 78 对 plan/anchor 文件，正文与已合入的 Flash 发布版本一致。Flash 从所选版本的这个目录读取计划。Astra 没有重放资产，显式选用它执行 Flash 时会报错。

Astra 发布版合并了 Long 与 Spatial/Object/Goal 两批探索 memory；三个重名但内容不同的 global 文件分别加来源后缀并保留两份。79 对任务 audit/recipe 保持原始内容，Long Swap task 6 没有专属经验，不补造。历史 **741/800** 成绩使用原先两份冻结快照按套件分别评测， **合并发布版尚未重新评测**。生成环境为运行提交 ``014a0fa``，属于场景 seed 修复前版本。原始来源 revision 分别为 ``cf5d14ce9b3ec6c72de5477ec0fea806a3884efa`` （Long）和 ``984c57f7c6caf48b572f5926851dd9134ae041d8`` （Spatial/Object/Goal）。

当前加载器要求 Hub 数据按模型分版本存放，不转换旧布局，也不回退到旧的无版本语料。代码与数据需要配套更新。当前目录名需搭配 `RPent #190 <https://github.com/RLinf/RPent/pull/190>`_ 的共享 memory 命名更新，以及 `对应数据更新 <https://huggingface.co/datasets/RLinf/RPent-memory/discussions/13>`_。数据集 ``main`` 持续更新，``reproduce/memory`` 保留历史内容和布局，供匹配的机器人历史复现分支使用。历史 GPT-5.5 复现使用匹配的历史客户端与原始数据 revision ``21a62795fe3b7e500c8381ac47938f6d713ebe18``：

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
     --revision 21a62795fe3b7e500c8381ac47938f6d713ebe18 \
     --include 'libero/*' --local-dir /path/to/legacy-download
   # 旧 RPent 客户端使用：
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-5.5 --memory-profile local \
     --memory-dir /path/to/legacy-download/libero

探索并生成本地记忆
---------------------------

以下命令使用独立的本地目录，最多运行 3 个规划会话，每个会话最多尝试 5 次。目录可以为空；``--explore`` 会启用本地模式。重新运行时，请使用新的输出目录：

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 0 \
     --planner claude_code --model claude-opus-4-8 \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir ./memory/libero-local \
     --output-dir logs/explore_10_task_t0_s0

每个规划会话创建独立的工具集（toolkit），会话内每次尝试前都会复位环境。状态和观测保存在 ``<output-dir>/sessions/session_NNN/``，经验草稿写入 ``<memory-dir>/_internal/inbox/<cell>/``。

正常结束时，运行器校验并合并草稿、更新索引。只有 LIBERO 判定成功后，才将成功任务的运行记录（audit）和动作序列加入可供后续任务读取的记忆。探索产生的数据保存在本地，不会自动上传。添加 ``--dashboard`` 可观看探索过程，操作方法见 :doc:`dashboard`。

使用本地记忆评测
------------------------

探索结束后，先检查本地目录和运行结果，再用该目录评测另一个场景：

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner claude_code --model claude-opus-4-8 \
     --memory-profile local --memory-dir ./memory/libero-local

``local`` 模式不下载记忆，也不开启探索。LIBERO 会检查目录中是否已有记忆；空目录会报错。``hf`` 模式不能与 ``--memory-dir`` 同用。

在对话记录中查看记忆读取工具的调用，确认它读取了当前任务允许使用的文件。历史坐标、像素和姿态只能作为参考，执行动作前仍需根据当前观测定位。

检查与人工合并
---------------------

检查记忆文件格式，并根据已有经验重建索引：

.. code-block:: bash

   rpent-memory --memory-dir ./memory/libero-local validate
   rpent-memory --memory-dir ./memory/libero-local build-index

如需先审核草稿，在探索命令中加入 ``--no-auto-merge-memory``。审核完成并核实 LIBERO 的成功结果后，可用以下命令合并对应任务；``--solved`` 是人工提供的成功标记，不能仅凭规划器的总结设置：

.. code-block:: bash

   rpent-memory --memory-dir ./memory/libero-local merge \
     --cell 10_task_t0_s0 --output-dir logs/explore_10_task_t0_s0 --solved

格式校验不证明任务真实成功。目录、权限和发布规则见 :doc:`../development/memory`。

其他环境
------------

- :doc:`../simulators/robocasa` 和 :doc:`../simulators/robotwin` 支持探索；任务参数、复位行为和记忆范围见各自页面。
- :doc:`../real_world_robots/dual_franka` 探索需要现场操作员复位场景并确认结果，默认不自动合并记忆。先完成部署与自检，再开始探索。
- :doc:`../real_world_robots/franka` 当前不支持探索模式。
