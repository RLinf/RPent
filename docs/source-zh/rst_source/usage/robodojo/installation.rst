RoboDojo 后端环境安装与复现
============================

本文档说明如何从零搭建复现本集成（``rpent --env robodojo``）所需的完整
环境。以下步骤在一台 Linux x86_64 工作站（NVIDIA RTX PRO 6000 Blackwell、
驱动 580.173.02、Ubuntu 24.04）上端到端验证通过。

运行时构成
----------

集成刻意保持三个隔离的运行时，外加大型资产与一个 planner 凭证；RPent
只负责编排，绝不混用：

.. list-table::
   :header-rows: 1

   * - 运行时
     - Python
     - 内容
     - 用途
   * - RPent venv
     - 3.11
     - rpent 本体 + SAM3
     - agent 循环 / 工具 / 记忆
   * - robodojo-sim（conda）
     - 3.11
     - Isaac Sim 5.1 / IsaacLab 0.54.3 / CuRobo
     - 仿真环境
   * - Pi_05 策略环境（uv）
     - 3.11
     - OpenPI（XPolicyLab 集成）
     - 策略服务

前置条件
--------

* Linux x86_64，NVIDIA GPU（Blackwell 需 cu128 torch），约 100–180 GB 磁盘；
* RoboDojo 官方 checkpoint（RoboDojo-sim-arx_x5-joint-0，约 44.7 GB）与
  资产（约 28.5 GB），来自 ModelScope；
* SAM3 checkpoint（约 3.45 GB）与 CLIP BPE 词表。

RPent + SAM3
------------

.. code-block:: bash

   uv pip install -e ".[rlinf,openpi,libero-pro,sam3]"   # 完整安装
   # 仅 RoboDojo：uv pip install -e ".[sam3]"
   # 固定 mcp>=1.23,<2；Blackwell 换 torch==2.7.1+cu128 / torchvision==0.22.1+cu128
   # SAM3 的 CLIP BPE 词表（bpe_simple_vocab_16e6.txt.gz）放在 sam3 包旁边，
   # 另下载 sam3.pt 指向 SAM3_CHECKPOINT_PATH

RoboDojo 源码
-------------

克隆 RoboDojo 官方仓库（含 XPolicyLab 子模块），并固定到与本地验证一致
的 pinned 提交（见集成日志与 ``config/runtime.env`` 接线）。

仿真环境（Isaac Sim / IsaacLab / CuRobo）
------------------------------------------

* ``isaacsim[all,extscache]==5.1.0``；
* 使用 pinned IsaacLab 0.54.3 fork，并把 ``isaacsim.asset.importer.urdf
  ==2.4.31`` 物化进 Kit 扩展缓存（Isaac Sim wheel 自带 2.4.30）；
* CuRobo（``cuda_core``）：固定 ``viser==0.1.34``、``tyro==0.9.0``、
  ``websockets==12.0``；Blackwell 设 ``TORCH_CUDA_ARCH_LIST=12.0``。

RoboDojo 资产
-------------

完整克隆 ModelScope 数据集仓库（资产与 checkpoint 共用），共 14,506 个
LFS 文件（声明约 41 GB）。其中 9,224 个 eval layout JSON 以 LFS 指针存储
但缺少 ``filter=lfs`` 属性，普通 ``git lfs checkout`` 拒绝物化；请使用
SHA-256 校验的 scoped materializer（集成日志 §11 附脚本）。

Pi_05 策略环境（OpenPI via XPolicyLab）
----------------------------------------

在 vendored OpenPI 中执行 ``uv sync --locked --no-dev --group lerobot``，
补最小 pinned extras 并 editable 安装 XPolicyLab；checkpoint 与
``dataset_stats.json`` 放入策略目录约定的 checkpoint 布局。

接线与冒烟
----------

``spec.py`` 读取 RoboDojo 工作区 ``config/runtime.env``
（``ROBODOJO_SIM_ENV``、``ROBODOJO_PI05_ENV``、``ROBODOJO_SOURCE_ROOT``
等）；``SAM3_CHECKPOINT_PATH`` 指向 sam3.pt。冒烟链路：

.. code-block:: text

   robodojo.sh doctor → 官方 Pi_05 debug gate → 裸 eval → rpent --env robodojo ...

单次运行约 45 GB 显存。完整命令序列、排障表与已知 pin 见集成日志与
``robots/robodojo/guides/interface.md``。
