BEHAVIOR
========

`BEHAVIOR-1K <https://behavior.stanford.edu/>`_ 基于 OmniGibson 提供长程家庭任务。
RPent 支持运行 ``turning_on_radio`` 和 ``picking_up_trash`` 两个 BEHAVIOR 任务。

概览
----

安装前先确认任务、模型和运行环境要求。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      Pi0.5 生成 32 步动作块。

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``。

   .. grid-item-card:: 任务

      ``turning_on_radio`` 和 ``picking_up_trash``。

   .. grid-item-card:: 硬件

      Linux、NVIDIA GPU；两个 Python 3.10 环境；CUDA 与 Isaac Sim。

任务
~~~~

使用 ``--task-name`` 和 ``--public-seed`` 选择任务。public seed 通过 ``robots/behavior/task_specs.py`` 固定映射到官方 activity instance。

.. list-table::
   :header-rows: 1
   :widths: 24 42 16 18

   * - 任务
     - 目标
     - Explore seeds
     - Eval seeds
   * - ``turning_on_radio``
     - 打开客厅桌上的收音机。
     - ``0``
     - ``1``-``9``
   * - ``picking_up_trash``
     - 把客厅的三个汽水罐放进厨房垃圾桶。
     - ``0``-``9``
     - ``10``-``19``

观测与动作
~~~~~~~~~~

规划器使用的观测与工具不同于 Pi0.5 的输入和动作。

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - 头部和左右腕部 RGB 图像、深度／世界坐标及机器人状态。Pi0.5 接收三路相机画面和 R1Pro 本体状态。
   * - 动作
     - 规划器调用 Pi0.5 或运动原语。Pi0.5 预测 32 步、每步 23 维的动作；任务成功或截断时，环境可能提前停止动作块。
   * - 奖励与成功判定
     - 只有当前 episode 的 ``info["done"]["success"] is True`` 才是官方任务成功；reward、终止和规划器的 ``finish`` 都不能替代。
   * - 任务指令
     - 固定任务指令来自 ``robots/behavior/task_specs.py``。

安装与资源准备
--------------

使用带 NVIDIA GPU 和兼容 CUDA 驱动的 Linux 主机，安装 ``git``、``bash``、``curl``、
``nvidia-smi`` 和 ``sha256sum``，并为模拟器、授权数据及模型权重留足磁盘空间。
RLinf 默认通过无需交互的 ``sudo`` 安装系统依赖；若主机已经安装这些依赖，
可设置 ``RLINF_NO_ROOT=1``，让安装器传入 RLinf 的 ``--no-root`` 选项。
上游下载缓慢时，可设置 ``RLINF_USE_MIRROR=1``，传入 RLinf 的 ``--use-mirror`` 选项。
源码 checkout 中的安装器会创建两个相互独立的 Python 3.10 环境：

- **RPent venv**：运行 CLI、planner、Dashboard 和 MemoryManager；
- **BEHAVIOR venv**：运行 RLinf、OmniGibson、Isaac Sim 和 Pi0.5。

非 editable 的 RPent wheel 包含 ``robots/behavior`` Python 包。``.[behavior]`` 只添加 RPent 侧依赖；两者都不包含 OmniGibson 运行环境、授权数据或 Pi0.5 checkpoint。双 venv 安装器本身需要从源码 checkout 执行。非 editable 运行时应将 ``RPENT_REPO_ROOT`` 设为同一提交的 checkout，并显式指定可写的 ``--memory-dir``；否则默认路径可能落在 site-packages 中。全新安装步骤：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   export RPENT_REPRO_ROOT="$PWD/../rpent-behavior-runtime"
   export UV_CACHE_DIR="$RPENT_REPRO_ROOT/uv-cache"
   bash robots/behavior/install_behavior_runtime.sh
   export RPENT_PYTHON="$RPENT_REPRO_ROOT/venvs/rpent/bin/python"
   export BEHAVIOR_PYTHON="$RPENT_REPRO_ROOT/venvs/behavior/bin/python"
   "$RPENT_PYTHON" -m robots.behavior.selfcheck

上述 runtime 目录位于源码树之外。安装器在规划器 venv 中以 editable 模式安装
RPent；ENV、VLA 和 DINO 在独立的模拟器 venv 中使用同一份源码。安装器调用固定
RLinf checkout 的 ``embodied --model openpi --env behavior`` 合装流程，不再覆盖其
Torch、OpenPI 或 LeRobot 版本；RPent 仅补装运动规划所需的 cuRobo，并将依赖 freeze
和源码版本写入 ``$RPENT_REPRO_ROOT/manifests``。全新安装请使用新的
``RPENT_REPRO_ROOT``；安装器不会覆盖 revision 不符或有未提交改动的 RLinf checkout。
selfcheck 通过只说明源码接口检查完成，不代表 GPU 服务或任务动作已经运行。

.. warning::

   RLinf 的合装流程最终为 OmniGibson 使用 Torch 2.5.1，但
   ``rlinf-openpi==0.1.1`` 的包元数据声明需要 ``torch==2.7.1``。
   安装器保留 ``uv pip check`` 的原始结果；非零结果不能称为依赖检查通过。
   除这项已知不一致外，出现其他冲突时安装器会停止。
   运行验收还必须通过 CUDA、真实 ENV reset、Pi0.5 推理和环境动作执行。
   仅创建 venv 或通过 selfcheck 不能证明这些结果。

运动规划在 BEHAVIOR venv 中使用 NVlabs/cuRobo v0.8.0，固定 commit
``4ea77366ca48ee453e7df139e39fa6532af49f3b``。应基于 RLinf 安装后的
runtime 安装 cuRobo，不改变其 Torch、NumPy 或 Isaac Sim 版本；无约束解析器
可能升级到 NumPy 2，破坏该仿真环境的兼容性。

仿真资产
~~~~~~~~

接受 BEHAVIOR/OmniGibson 许可后，选择独立数据根并使用标准资产命令。该命令会在
BEHAVIOR venv 中调用三个 OmniGibson 官方下载函数，不会把 OmniGibson import 到
RPent 环境：

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH="$RPENT_REPRO_ROOT/omnigibson-data"
   "$RPENT_PYTHON" -m robots.behavior.assets_cli --accept-license

不传 ``--accept-license`` 时，官方下载器会显示交互式许可确认。该参数代表明确的
非交互许可确认；仅在接受许可条款后使用。
三个官方下载步骤都完成后，下载器才将路径和文件大小清单写入 ``.rpent-behavior-download.json``。``--skip-existing`` 与 ``--verify`` 只接受与清单匹配的数据；没有记录的旧数据根应改用新目录重新下载并验证。该清单并非内容 SHA-256 校验。

最终数据根必须包含：

.. code-block:: text

   <OMNIGIBSON_DATA_PATH>/
     2025-challenge-task-instances/
     behavior-1k-assets/
       scenes/
     omnigibson-robot-assets/
     omnigibson.key

Pi0.5 checkpoint
~~~~~~~~~~~~~~~~

将已审查 checkpoint 下载到源码树之外：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH="$RPENT_REPRO_ROOT/downloads/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32"
   "$RPENT_REPRO_ROOT/venvs/rpent/bin/hf" download \
     RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 \
     --local-dir "$PI05_CHECKPOINT_PATH"

共享 Pi0.5 组件接收头部、左腕、右腕画面与原始 R1Pro 本体状态。原始 RPC 输出形状为 ``[1, 32, 23]``，客户端返回 ``[32, 23]``。启动前会检查 checkpoint 文件大小及固定的 SHA-256 身份。

DINOv2 派生缓存
~~~~~~~~~~~~~~~~

BEHAVIOR 保留经审查的 `DINOv2 <https://github.com/facebookresearch/dinov2>`_
ViT-S/14 部署，用于从官方 Memory 语料中生成整图 embedding。运行前提供 DINOv2
源码归档和 ``dinov2_vits14_pretrain.pth`` 权重：

.. code-block:: bash

   export DINOV2_SOURCE_ARCHIVE="$RPENT_REPRO_ROOT/downloads/dinov2-source.tar.gz"
   export DINOV2_WEIGHTS="$RPENT_REPRO_ROOT/downloads/dinov2_vits14_pretrain.pth"

   curl -fL \
     https://github.com/facebookresearch/dinov2/archive/7764ea0f912e53c92e82eb78a2a1631e92725fc8.tar.gz \
     -o "$DINOV2_SOURCE_ARCHIVE"
   curl -fL \
     https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
     -o "$DINOV2_WEIGHTS"

DINOv2 从官方 Memory 语料生成整图 embedding，不替代当前观测或 MemoryManager 的 Markdown/YAML 记录。源码 revision 和两个资产的 SHA-256 固定在 ``robots/behavior/dino_v2/encoder.py``。

运行任务前检查完整资产布局，以及 checkpoint 和 DINO 文件的固定身份：

.. code-block:: bash

   "$RPENT_PYTHON" -m robots.behavior.assets_cli --verify

DINO 缓存从已发布的 ``task-specific`` audit/recipe 对及其
``task-specific/artifacts/<cell>/`` 下经过校验的证据重建。仅 head RGB 帧参与
embedding，左右腕帧保留用于诊断。成功运行先在本地准备证据，再由 MemoryManager
在同一个 merge 锁内将证据和任务对一起发布。``dino_v2/`` 是可删除重建的派生数据，
不是另一套语料。使用 BEHAVIOR 解释器执行重建，不要手工修改缓存文件。

.. code-block:: bash

   "$BEHAVIOR_PYTHON" -m robots.behavior.build_memory_cli \
     --memory-dir "$RPENT_REPRO_ROOT/memory/behavior" \
     --source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --weights "$DINOV2_WEIGHTS" \
     --cuda-device 1

运行一个任务
------------

先按 :doc:`../guides/configure_planner` 配置规划器并检查连接。将 GPU 序号 ``0`` 和 ``1`` 换成空闲的物理 GPU；各组件只使用指定的设备。以下示例以空的本地记忆语料运行一次 Eval，因此不需下载 HF 记忆：

.. code-block:: bash

   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent-check-llm" \
     --planner codex --model gpt-5.5
   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent" --robot behavior \
     --task-name turning_on_radio --public-seed 1 \
     --planner codex --model gpt-5.5 \
     --memory-profile local \
     --memory-dir "$RPENT_REPRO_ROOT/memory/behavior" \
     --output-dir "$RPENT_REPRO_ROOT/runs/first-eval" \
     --behavior-repo "$RPENT_REPRO_ROOT/RLinf" \
     --behavior-python "$RPENT_REPRO_ROOT/venvs/behavior/bin/python" \
     --activity-instance-dir \
       "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --behavior-env-cuda-device 0 \
     --behavior-model-cuda-device 1 \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS"

首次加载环境可能需要数分钟。规划器检查通过后再启动任务；该检查只验证认证与连接，不验证模拟器。后续运行请使用新的输出目录。

查看结果
--------

上述 Eval 命令结束后，检查以下文件：

.. code-block:: text

   <output-dir>/run.log
   <output-dir>/transcript_*.json
   <output-dir>/states.json
   <output-dir>/episode.mp4
   <output-dir>/terminal_receipt.json       （调用 finish 后）
   <output-dir>/behavior_env_server.log
   <output-dir>/behavior_vla_server.log
   <output-dir>/behavior_dino_server.log

``states.json`` 索引每一步的观测和相机图像；transcript 记录规划器消息和 ``environment_success``。视频、进程正常退出、reward 或规划器 ``finish`` 都不是官方成功。只有当前 episode 的原始 ``info["done"]["success"] is True`` 才是成功判定。receipt 记录证据及其 SHA-256 摘要，不能制造成功。

Explore 的 ``run.log``、transcript 和服务日志留在运行根目录；``states.json``、``episode.mp4`` 与 ``terminal_receipt.json`` 写在 ``<output-dir>/sessions/session_001/`` 下，后续 session 依次递增。Dashboard 为每个 TaskRun 创建 ``<output-dir>/tasks/0001_<task-name>_s<seed>/``，状态和视频写在该目录；Explore TaskRun 还会增加 ``sessions/session_001/`` 子目录。浏览器查看方式见 :doc:`../guides/dashboard`。

任务记忆
--------

BEHAVIOR 使用和其他机器人相同的 Markdown/YAML ``MemoryManager`` 格式与公共 memory
工具。普通 Eval 使用共享默认值 ``--memory-profile hf``；Explore 使用本地 memory，
并且只通过公共 inbox 流程写入。DINO cache 从已审查的 Memory 语料派生；配置后，
它的 advisory 会附加到公开 tool receipt，并始终只作为历史建议。

- Eval 默认从 HF 语料构造 ``read_only`` MemoryManager；只有显式传入
  ``--memory-profile local`` 时才读取本地目录；
- Explore 只构造一个 ``inbox_write`` MemoryManager，写入范围限定为
  ``<memory-dir>/_internal/inbox/<recipe-tag>``；
- ``MEMORY.md``、``global/``、``task-family/`` 和 ``task-specific/``
  保存已发布语料，
  成功的 audit/recipe 对复制到 ``task-specific/``，recipe 文件使用共享
  MemoryManager 命名，例如 ``<tag>_recipe.jsonl``；
- merge 处理有效的根级草稿后，将该 cell 的 inbox 归档到
  ``_internal/merged/<recipe-tag>``。只有无效草稿的 inbox 保留原位；
  冲突文本归档到 ``_internal/conflicts/``。

缺失或空 corpus 是合法状态，但不会提供任何建议。只有需要共享本地已审查 memory
的运行才显式传入同一个 ``--memory-dir``。

探索模式
--------

使用标准 RPent Explore 入口运行一组有界 session：

.. code-block:: bash

   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent" --robot behavior \
     --explore \
     --explore-sessions 3 \
     --task-name picking_up_trash --public-seed 0 \
     --output-dir "$RPENT_REPRO_ROOT/runs/behavior-explore" \
     --memory-dir "$RPENT_REPRO_ROOT/memory/behavior" \
     --planner codex --model gpt-5.5 \
     --behavior-repo "$RPENT_REPRO_ROOT/RLinf" \
     --behavior-python "$RPENT_REPRO_ROOT/venvs/behavior/bin/python" \
     --activity-instance-dir \
       "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --behavior-env-cuda-device 0 \
     --behavior-model-cuda-device 1 \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS"

对 BEHAVIOR 而言，一个 session 就是一个 attempt。每个 session 都启动 fresh env
sidecar、new episode，并写入独立的 ``sessions/session_NNN`` 目录；VLA 与 DINO
sidecar 在各 session 之间共享。planner 无权在单次 invocation 内 reset，且
``--explore-attempts-per-session`` 大于零会被拒绝。

Runtime 与 Dashboard
--------------------

runtime 有四个 component role：

- ``env``：task-scoped 官方 BEHAVIOR/OmniGibson 环境；
- ``vla``：基于公共 Pi0.5 facade 的 ``robots/behavior/vla_server.py`` 适配服务，
  在 session 间共享；
- ``dino``：共享 ``robots/behavior/dino_v2/server.py`` episode-memory
  embedding 服务；
- ``memory``：task-scoped 官方 MemoryManager。

启动 Dashboard Session：

.. code-block:: bash

   export RPENT_BEHAVIOR_PYTHON="$RPENT_REPRO_ROOT/venvs/behavior/bin/python"
   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent" \
     --robot behavior --dashboard \
     --task-name turning_on_radio --public-seed 1 \
     --behavior-repo "$RPENT_REPRO_ROOT/RLinf" \
     --behavior-python "$RPENT_BEHAVIOR_PYTHON" \
     --activity-instance-dir \
       "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS" \
     --behavior-env-cuda-device 0 \
     --behavior-model-cuda-device 1 \
     --memory-profile local \
     --memory-dir "$RPENT_REPRO_ROOT/memory/behavior" \
     --output-dir "$RPENT_REPRO_ROOT/runs/behavior-dashboard"

Dashboard 使用公共 Start Session 流程与 head/left-wrist/right-wrist 相机视图。
BEHAVIOR 不增加 robot-local 手动按钮、手动控制 backend 或
``env.dashboard_*`` RPC。公开合同注册 9 个可执行 planner primitive：
``pi0_nav_pick``、``observe``、``pixel_to_world``、``navigate_to``、``move_to``、
``rotate_wrist``、``close``、``open`` 和 ``press``。
``move_to(hand=both)`` 通过 cuRobo 碰撞检查轨迹协调双臂，腕旋转复用同一规划器。
导航执行有界直线底盘运动或转向，路径被阻挡时拒绝执行。
RGB-D 投影使用当前物理相机帧；R1Pro 没有可动头部相机，因此拒绝非 center 的
``head_view`` 预设。``press`` 沿已对准的手部方向推进，最多 2 cm、10 秒，
遇外部接触或 episode 结束即停；接触不等于已验证按钮接触，视觉手部检查仍未验证。
规划、碰撞、跟踪和时长限制导致的失败均明确返回。
任务成功仅认原始 ``info["done"]["success"]``。运动原语返回最终观测供后续策略调用
和流式视频使用；VLA chunk 另外记录每个实际返回的环境帧。
