记忆机制
============

RPent 通过 ``MemoryManager`` 管理每个机器人的经验文件、读取权限和探索草稿。日常操作见 :doc:`../usage/memory`；本页说明目录和发布过程。

目录与访问范围
---------------------

本地记忆通常位于 ``memory/<robot>/``，也可通过 ``--memory-dir`` 指定。分层记忆使用以下结构；各层按需存在：

.. code-block:: text

   <memory-root>/
   ├── MEMORY.md
   ├── global/
   ├── suite/
   ├── task_only/
   └── _internal/inbox/<cell>/

``global/`` 保存通用经验，``suite/`` 保存按任务集组织的经验，``task_only/`` 保存同任务参考，``MEMORY.md`` 索引可检索的经验。探索草稿写入 ``_internal/inbox/<cell>/``。

公开记忆数据还可能采用环境专用结构。例如 RoboCasa Target50 使用 ``results/<Task>_s0.json``、``recipe_<Task>_s0.jsonl`` 和可选的任务 Markdown，详见 :doc:`../usage/robocasa`。不能把一种环境的目录假定为所有环境的格式。

每个工具集（toolkit）根据运行配置构造 ``MemoryManager``：评测时只读，探索时允许写入当前任务的草稿目录（inbox）。实际可读范围还受各环境的工具与任务规则约束。LIBERO 本地评测会检查记忆数据是否存在。

探索与合并
---------------

探索会保留各次尝试的状态和工具调用记录，并生成注明来源的经验草稿。运行器根据环境结果决定任务是否成功，再调用 ``MemoryManager.merge_memory`` 合并草稿、更新索引；成功任务的运行记录（audit）和动作序列有单独的发布条件。

仿真环境默认自动合并，可用 ``--no-auto-merge-memory`` 保留草稿供人工检查。双臂 Franka 默认不自动合并，并通过人工评价确认成功。人工运行 ``rpent-memory merge`` 时，``--solved`` 是调用方提供的成功标记，只有核实环境或操作员结果后才能使用。

``rpent-memory validate`` 检查记忆文件结构，不能验证任务是否真实成功。工具权限和合并实现见 ``rpent/memory/manager.py``，运行模式与结果处理见各机器人的 ``robot_spec.py`` 和 ``rpent/cli/main.py``。

同步与贡献
---------------

HF 模式从公开数据集 ``RLinf/RPent-memory`` 同步当前机器人目录；``HF_HUB_OFFLINE=1`` 跳过同步。正式复现应按环境页下载固定版本，并选择本地模式。

公开记忆由维护者审核发布。贡献经验时，在 RPent issue 中附上记忆文件、代码与模型版本、任务参数及成功证据；仓库没有自动上传记忆的入口。
