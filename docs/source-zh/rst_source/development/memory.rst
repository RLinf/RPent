Memory 管理
===========

RPent 的 memory 分两层，对应 ``resources/<robot>/`` 下两类只读参考语料。重点是记下什么时候、
在什么条件下调用 VLA 才靠谱，以及如何把已验证轨迹适配到新 seed 或扰动场景，免得每次都从头试错。

两层结构
--------

* **任务级参考。** 固定种子探索成功后，单次运行
  会写出 audit JSON（含 ``strategy_notes``、定性目标区域等）；``recipe_*.jsonl``
  则在运行结束时自动导出（仅含 ``move_to``、``pi0_pick`` 等 primitive 命令序列，
  不含读文件、感知工具调用）。经筛选后，这些文件进入 ``results_*_pert/`` 等
  参考目录，供同任务在其他 seed 上部署时阅读。planner 参考步骤顺序与策略，但须
  根据当前画面重新感知并计算坐标，不得照搬历史 xyz。
* **全局经验。** ``resources/<robot>/memory/`` 下的 Markdown 笔记
  （``MEMORY.md`` 索引及子笔记）记录跨任务的操作要点、参数范围与常见失败模式。
  planner 将它与任务级参考一并阅读，用于理解「为什么这样排步骤」以及失败后如何
  调整。

在 LIBERO 上，prompt 要求 planner 先扫 ``MEMORY.md`` 并读取相关笔记，再查看
``results_*_pert/`` 里同任务的 seed-0 参考（若有）。recipe 只提供命令顺序；真正适配新场景要靠
memory 笔记里的技巧、参数范围和失败模式。

托管方式
--------

``resources/`` 不随 git 仓库分发，而是托管在 Hugging Face 数据集 ``RLinf/RPent-memory``
上（按机器人分层，例如 ``libero/memory/`` 与 ``libero/results_*_pert/``）。``rpent.utils.resources.ensure_resources``
会在每次运行时从数据集增量同步该机器人的子目录（只下载有变化的文件），使本地副本保持最新。
数据集是公开的，无需 token 即可下载；设 ``HF_HUB_OFFLINE=1`` 则跳过同步、仅使用本地副本。
memory 是可选的：若某机器人在数据集上没有 memory，或同步失败，运行也会用本地已有的内容继续。

更新 memory
-----------

探索跑通后，audit 与 recipe 先落在当次 ``output_dir``；进入 ``results_*_pert/``
或 ``memory/`` 参考库须经过筛选。发布由维护者统一把关：只有拥有 ``RLinf`` 组织写权限的人能更新
Hugging Face 数据集；仓库本身不提供自助上传入口。如果你有效果更好的参考轨迹或
memory 笔记，可以开一个 issue 附上内容来贡献，由维护者审阅后发布。
