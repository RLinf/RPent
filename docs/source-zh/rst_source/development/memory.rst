Memory 管理
===========

.. _memory-management:

RPent 的 memory 按机器人维护，用于复用已验证的任务经验和操作策略，避免每次运行
都从头试错。

运行模式
--------

两种运行模式对 memory 的使用方式不同：

- **Evaluation** 读取已有 memory，但不会更新 memory。
- **Exploration** 用于生成和更新本地 memory，LIBERO、RoboCasa 和 RoboTwin 均支持；
  具体探索流程见对应机器人指南。

Exploration 和本地 memory Evaluation 的详细流程见
:ref:`LIBERO 探索文档 <libero-exploration>`。

目录结构
--------

发布到 Hugging Face 的 memory 与本地准备用于评测的 memory 使用相同的目录结构：

.. code-block:: text

   <memory-root>/
   |-- MEMORY.md
   |-- global/
   |-- task-family/
   `-- task-specific/
       |-- <cell>.json
       |-- <cell>_recipe.jsonl
       `-- <task_key>.md

默认本地目录为 ``memory/<robot>/``。Hugging Face 中 LIBERO 按模型版本分目录，
详见下文；其他机器人仍使用 ``<robot>/``。自定义 ``--memory-dir`` 可指向任意采用上述结构的目录。

各机器人按需提供实际使用的记忆层：

.. list-table:: 记忆层级
   :header-rows: 1
   :widths: 25 45 30

   * - 记忆层级
     - 保存内容
     - 复用范围
   * - 通用记忆（Global Memory，``global/``）
     - 跨任务通用规律与失败模式
     - 所有任务
   * - 任务族记忆（Task-family Memory，``task-family/``）
     - 特定任务族中验证过的策略与注意事项
     - 同类任务及其变体
   * - 单任务记忆（Task-specific Memory，``task-specific/``）
     - 单次任务的执行记录和操作流程
     - 仅供当前任务参考

使用记忆时，其适用前提和证据范围应与当前任务匹配。``MEMORY.md`` 是可选的
global 与 task-family 索引，不是额外的记忆层。RoboCasa 的 HF 发布语料使用
``global/GLOBAL_MEMORY.md``，本地语料则可提供多份 ``global/*.md`` 文件。

评测时，规划器只能读取当前机器人的 memory。各机器人的具体要求仍然适用：
RoboCasa 始终同时提供 task-specific 与 global memory，并要求 global 文件存在；
任务 JSON 和 recipe 必须同时存在或同时缺失。规划器按需查阅相关经验，机器人动作
不要求先完整读取全部文件。

更新旧语料
----------

当前目录名为 ``task-specific/`` 和 ``task-family/``。更新 RPent 时，请将配套的
新版 HF 语料下载到新目录。任务族文档采用 ``scope: task-family``，文件名形如
``task-family_libero10_task_t2.md``；其中表示 benchmark 身份的 ``suite`` 字段不变。
迁移自有文档时也需更新索引和链接。程序会对未迁移目录报错，避免误判为缺少任务记忆；
文件工具不会开放旧缓存路径。

RoboCasa 同时使用 task-specific 与 global memory，不提供记忆层选择参数。
历史结果应与生成结果时的代码及校验器一起保存。显式选择的历史 v1 清单继续保留，
本次迁移不改写历史结果。

使用 memory
-----------

RPent 从公开数据集 ``RLinf/RPent-memory`` 下载 memory。
LIBERO 默认使用 ``--memory-version auto`` 按模型选择：

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

支持 ``openai:`` 等提供方前缀。Codex 优先使用 ``--model``，其次使用 ``CODEX_MODEL``。
其他模型、Claude 或无法确定的默认模型会回退到 ``GPT_5.5_xhigh``，并输出提示。
Flash 重放默认选择 GPT-5.5。显式指定版本优先于自动选择，不改变当前模型或 reasoning
effort；目录名中的 effort 只说明该 memory 的探索生成配置。

.. code-block:: bash

   # 自动选择 Astra memory。
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-6-astra --reasoning-effort low

   # 同一模型使用 GPT-5.5 memory。
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-6-astra --reasoning-effort low \
     --memory-version GPT_5.5_xhigh

CLI 和 Dashboard 都在每个任务开始前解析 memory 根目录。在 Dashboard 的
**下一任务的模型** 中修改模型后，下一任务会重新进行自动选择；正在运行的任务保留原模型
和 memory。显式指定的 memory 版本不会随模型切换而改变。

仅下载所选版本。LIBERO 缓存位于 ``memory/libero/.versions/``，按仓库、提交和版本隔离，
每次复用前校验文件集合完全一致及每份文件的哈希。额外文件会使缓存失效，固定 revision
时也不例外；联网同步会重建无效缓存。``HF_HUB_OFFLINE=1`` 要求所选版本及 revision
已有完整、未改动的缓存。
下载失败不会改用另一模型的 memory；缓存缺失或不完整会明确报错。
旧缓存记录未标明版本来源时，需要联网成功刷新一次；不复用旧的无版本缓存。
其他机器人保持原有的 memory 同步行为。

独立下载与本地评测
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m robots.libero.memory sync --memory-version GPT_6_astra_low
   python -m robots.libero.memory sync --model gpt-6-astra \
     --revision <release-commit> --output-dir /path/to/new-astra-memory
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-6-astra --reasoning-effort low \
     --memory-profile local --memory-dir /path/to/new-astra-memory

``sync`` 输出实际 memory 根目录，``--output-dir`` 必须是尚不存在的目录。
``--planner`` 默认是 ``api``，与 ``rpent`` 一致；希望在省略 ``--model`` 时读取
``CODEX_MODEL``，需指定 ``--planner codex``。
``--memory-profile local`` 不下载 memory；本地模式或 ``--explore`` 与显式远程
``--memory-version`` 同时使用会报参数冲突。探索使用本地 memory，每次独立探索应指定单独的空目录。

分版本下载使用 LIBERO 专用命令；共享的 ``rpent-memory`` 继续提供 ``merge``、
``validate`` 和 ``build-index`` 三个命令。

发布来源与兼容性
~~~~~~~~~~~~~~~~

数据集中的 ``libero/README.md`` 和 ``libero/manifest.json`` 记录版本、原始来源快照和
发布文件。各版本的 ``files`` 保存相对于该版本根目录的路径与发布文件的 SHA-256，
加载器据此校验实际内容。来源 revision 和来源哈希描述原始快照；目录、索引及正文引用
调整后，发布哈希另行更新，不改写原始来源哈希。

两个版本根目录都使用 ``MEMORY.md``、``global/``、``task-family/`` 和 ``task-specific/``。
GPT-5.5 还在 ``flash/`` 中包含 78 对 plan/anchor 文件，正文与已合入的 Flash 发布版本
一致。Flash 从所选版本的这个目录读取计划。Astra 没有重放资产，显式选用它执行
Flash 时会报错。

Astra 发布版合并了 Long 与 Spatial/Object/Goal 两批探索 memory；三个重名但内容不同的
global 文件分别加来源后缀并保留两份。79 对任务 audit/recipe 保持原始内容，Long Swap task 6
没有专属经验，不补造。历史 **741/800** 成绩使用原先两份冻结快照按套件分别评测，
**合并发布版尚未重新评测**。生成环境为运行提交 ``014a0fa``，属于场景 seed 修复前版本。
原始快照保留在 Hub tag ``libero-astra-long-frozen-20260917`` 和
``libero-astra-spatial-object-goal-frozen-20260917``。

当前加载器要求 Hub 数据按模型分版本存放，不转换旧布局，也不回退到旧的无版本语料。
代码与数据需要配套更新。当前目录名需搭配
`RPent #190 <https://github.com/RLinf/RPent/pull/190>`_ 的共享 memory 命名更新，以及
`对应数据更新 <https://huggingface.co/datasets/RLinf/RPent-memory/discussions/13>`_。
数据集 ``main`` 持续更新，``reproduce/memory`` 保留历史内容和布局，供匹配的机器人历史
复现分支使用。历史复现使用匹配的历史客户端与数据 revision；迁移前数据
归档为 ``libero-gpt5.5-xhigh-before-versions-20260917``：

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
     --revision libero-gpt5.5-xhigh-before-versions-20260917 \
     --include 'libero/*' --local-dir /path/to/legacy-download
   # 旧 RPent 客户端使用：
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-5.5 --memory-profile local \
     --memory-dir /path/to/legacy-download/libero

也可以按相同的目录结构自行准备本地 memory，通过对应环境的 ``--memory-dir`` 选项或
本地 memory 配置使用。Hugging Face memory 和本地 memory 使用相同的目录规范，区别只
在于来源。

贡献 memory
-----------

Hugging Face 上的 memory 由 RPent 维护者审核和发布，仓库本身不提供自助上传入口。
如果希望新增或更新 memory，可以在 RPent 仓库提交 issue，附上对应的 memory 文件和
来源信息，由维护者审核后加入 ``RLinf/RPent-memory``。
