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
   |-- suite/
   `-- task_only/
       |-- <cell>.json
       |-- <cell>_recipe.jsonl
       `-- <task_key>.md

默认本地目录为 ``memory/<robot>/``；Hugging Face 数据集中相同内容位于
``<robot>/`` 子目录下。自定义 ``--memory-dir`` 可指向任意采用上述结构的目录。

各目录均按需存在，机器人只需提供实际使用的目录：

- ``global/`` 保存从成功经验中提炼的跨任务通用经验。
- ``suite/`` 保存探索过程中按 suite 组织的任务级经验，可汇总多次尝试，
  并在同一任务的不同 seed 间复用。
- ``task_only/`` 保存成功运行产生的 audit、recipe 等同一任务参考文件。
- ``MEMORY.md`` 用于索引 ``global/`` 和 ``suite/``。

评测时，规划器只能读取当前机器人的 memory；缺少某一类 memory 不会阻止任务运行。

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
