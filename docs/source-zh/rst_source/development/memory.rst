Memory 管理
===========

RPent 的 memory 按机器人维护，用于复用已验证的任务经验和操作策略，避免每次运行
都从头试错。

运行模式
--------

两种运行模式对 memory 的使用方式不同：

- **Evaluation** 读取已有 memory，但不会更新 memory。
- **Exploration** 用于生成和更新本地 memory，目前仅 LIBERO 支持。

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

默认本地目录为 ``memory/<robot>/``；Hugging Face 数据集中相同内容位于
``<robot>/`` 子目录下。自定义 ``--memory-dir`` 可指向任意采用上述结构的目录。

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
global 与 task-family 索引，不是额外的记忆层。RoboCasa 直接读取其单份 global 文件。

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

默认情况下，RPent 从 Hugging Face 数据集 ``RLinf/RPent-memory`` 把当前机器人的
memory 同步到 ``memory/<robot>/``。数据集是公开的，无需 token 即可下载。设
``HF_HUB_OFFLINE=1`` 可跳过同步，只用本地副本。memory 是可选的：如果某机器人在
数据集上没有 memory，或同步失败，运行也会用本地已有的内容继续。

也可以按相同的目录结构自行准备本地 memory，通过对应环境的 ``--memory-dir`` 选项或
本地 memory 配置使用。Hugging Face memory 和本地 memory 使用相同的目录规范，区别只
在于来源。

贡献 memory
-----------

Hugging Face 上的 memory 由 RPent 维护者审核和发布，仓库本身不提供自助上传入口。
如果希望新增或更新 memory，可以在 RPent 仓库提交 issue，附上对应的 memory 文件和
来源信息，由维护者审核后加入 ``RLinf/RPent-memory``。
