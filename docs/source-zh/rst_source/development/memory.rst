Memory 管理
===========

LIBERO agent 的全局 memory 位于 ``resources/libero/memory/``（一个
``MEMORY.md`` 索引加若干叶子笔记）。它是一份经过审阅的只读知识库，每次运行
开始时读取。

托管方式
--------

``resources/`` 不随 git 仓库分发，而是托管在 HuggingFace 数据集
``RLinf/RPent-memory``（按环境分层，例如 ``libero/memory/`` 与
``libero/results_*_pert/``）。``rpent.utils.resources.ensure_resources`` 会在
每次运行时从数据集增量同步该环境的子目录（只下载有变化的文件），使本地副本
保持最新。该数据集为公开，全新 clone 无需 token 即可下载。设
``HF_HUB_OFFLINE=1`` 可跳过同步、仅用本地副本。memory 是可选的：若某环境在
数据集上没有 memory，或同步失败，运行都会用本地已有内容继续。

更新 memory
-----------

发布 memory 是一项受控操作，由对 ``RLinf`` 组织拥有写权限的维护者执行；仓库
不提供自助上传入口。如果你有效果更好的 memory，可以提一个 issue 附上内容来
贡献，由维护者审阅后发布。

读取无需凭证；写入需要 ``RLinf`` 写权限。

