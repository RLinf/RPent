Memory 管理
===========

LIBERO agent 的全局 memory 位于 ``resources/libero/memory/``：一个
``MEMORY.md`` 索引，外加它索引的若干篇单条笔记。这是一份经过审阅的只读
知识库，在每次运行开始时载入。

托管方式
--------

``resources/`` 不随 git 仓库分发，而是托管在 HuggingFace 数据集
``RLinf/RPent-memory`` 上（按环境分层，例如 ``libero/memory/`` 与
``libero/results_*_pert/``）。``rpent.utils.resources.ensure_resources`` 会在
每次运行时从数据集增量同步该环境的子目录（只下载有变化的文件），使本地副本
保持最新。数据集是公开的，无需 token 即可下载；设 ``HF_HUB_OFFLINE=1`` 则
跳过同步、仅使用本地副本。memory 是可选的：若某环境在数据集上没有 memory，
或同步失败，运行也会用本地已有的内容继续。

更新 memory
-----------

发布 memory 是一项受控操作，由拥有 ``RLinf`` 组织写权限的维护者执行，仓库
本身不提供自助上传入口。如果你有效果更好的 memory，可以开一个 issue 附上
内容来贡献，由维护者审阅后发布。
