记忆机制
============

RPent 通过 ``MemoryManager`` 管理每个机器人的经验文件、读取权限和探索草稿。日常操作见 :doc:`../guides/memory`；本页说明目录和发布过程。

目录与访问范围
---------------------

发布到 Hugging Face 的 memory 与本地准备用于评测的 memory 使用相同的目录结构：

.. code-block:: text

   <memory-root>/
   ├── MEMORY.md
   ├── global/
   ├── task-family/
   ├── task-specific/
   │   ├── <cell>.json
   │   ├── <cell>_recipe.jsonl
   │   └── <task_key>.md
   └── _internal/inbox/<cell>/

默认本地目录为 ``memory/<robot>/``。Hugging Face 中 LIBERO 按模型版本分目录，详见 :doc:`../guides/memory`；其他机器人仍使用 ``<robot>/``。自定义 ``--memory-dir`` 可指向任意采用上述结构的目录。

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

使用记忆时，其适用前提和证据范围应与当前任务匹配。``MEMORY.md`` 是可选的 global 与 task-family 索引，不是额外的记忆层。RoboCasa 的 HF 发布语料使用 ``global/GLOBAL_MEMORY.md``，本地语料则可提供多份 ``global/*.md`` 文件。

评测时，规划器只能读取当前机器人的 memory。各机器人的具体要求仍然适用：RoboCasa 始终同时提供 task-specific 与 global memory，并要求 global 文件存在；任务 JSON 和 recipe 必须同时存在或同时缺失。规划器按需查阅相关经验，机器人动作不要求先完整读取全部文件。

每个工具集（toolkit）根据运行配置构造 ``MemoryManager``：评测时只读，探索时允许写入当前任务的草稿目录（inbox）。实际可读范围还受各环境的工具与任务规则约束。LIBERO 本地评测会检查记忆数据是否存在。

更新旧语料
~~~~~~~~~~

当前目录名为 ``task-specific/`` 和 ``task-family/``。更新 RPent 时，请将配套的新版 HF 语料下载到新目录。任务族文档采用 ``scope: task-family``，文件名形如 ``task-family_libero10_task_t2.md``；其中表示 benchmark 身份的 ``suite`` 字段不变。迁移自有文档时也需更新索引和链接。程序会对未迁移目录报错，避免误判为缺少任务记忆；文件工具不会开放旧缓存路径。

RoboCasa 同时使用 task-specific 与 global memory，不提供记忆层选择参数。历史结果应与生成结果时的代码及校验器一起保存。历史 v1 清单可用于校验与之匹配的结果。

探索与合并
---------------

探索会保留各次尝试的状态和工具调用记录，并生成注明来源的经验草稿。运行器根据环境结果决定任务是否成功，再调用 ``MemoryManager.merge_memory`` 合并草稿、更新索引；成功任务的运行记录（audit）和动作序列有单独的发布条件。

仿真环境默认自动合并，可用 ``--no-auto-merge-memory`` 保留草稿供人工检查。双臂 Franka 默认不自动合并，并通过人工评价确认成功。人工运行 ``rpent-memory merge`` 时，``--solved`` 是调用方提供的成功标记，只有核实环境或操作员结果后才能使用。

``rpent-memory validate`` 检查记忆文件结构，不能验证任务是否真实成功。工具权限和合并实现见 ``rpent/memory/manager.py``，运行模式与结果处理见各机器人的 ``robot_spec.py`` 和 ``rpent/cli/main.py``。

同步与贡献
---------------

HF 模式从公开数据集 ``RLinf/RPent-memory`` 同步记忆。LIBERO 会选择并校验对应模型的记忆版本；离线使用需要该版本及 revision 的完整缓存。下载、缓存要求及历史代码与数据的配套关系见 :doc:`../guides/memory`。

公开记忆由维护者审核发布。贡献经验时，在 RPent issue 中附上记忆文件、代码与模型版本、任务参数及成功证据；仓库没有自动上传记忆的入口。
