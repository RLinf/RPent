BEHAVIOR
========

`BEHAVIOR-1K <https://behavior.stanford.edu/>`_ 基于 OmniGibson 提供长程家庭任务。
RPent 当前在源码目录 ``robots/behavior`` 中提供 ``turning_on_radio`` 和
``picking_up_trash`` 两个 BEHAVIOR 任务。

和 LIBERO、RoboCasa、RoboTwin、Franka 一样，BEHAVIOR 通过
``get_robot_spec()`` 提供命令行、配置和运行时入口，通过 ``get_toolkit()``
提供公开工具。BEHAVIOR 专属的生命周期逻辑保留在 ``robots/behavior`` 内。
共享的 ``--explore`` 入口已支持 BEHAVIOR，并保持“每个 session 只运行一次
attempt”的环境生命周期。

安装状态
--------

BEHAVIOR 是源码目录中的机器人插件，并使用两个相互独立的 Python 3.10 环境：

- **RPent venv**：运行 CLI、planner、Dashboard 和 MemoryManager；
- **BEHAVIOR venv**：运行 RLinf、OmniGibson、Isaac Sim 和 Pi0.5。

普通 ``rpent`` 包只发布框架本体。``.[behavior]`` 只安装轻量的 RPent 侧辅助依赖，
不会把 ``robots/behavior`` 打进普通 wheel，也不包含完整模拟器、资产或
checkpoint。请在 RPent 源码目录中执行：

.. code-block:: bash

   python -m pip install -e ".[behavior]"
   export RPENT_REPRO_ROOT="$PWD/.behavior-runtime"
   export UV_CACHE_DIR="$RPENT_REPRO_ROOT/uv-cache"
   python -m robots.behavior.install_runtime

安装器会在两个 venv 中以源码方式安装 RPent，克隆已审查的 RLinf revision，
调用官方 RLinf BEHAVIOR 安装器，在 BEHAVIOR runtime venv 内应用已审查的
CUDA/OpenPI/LeRobot 兼容性 pin，验证关键 import 和 CUDA，并在
``$RPENT_REPRO_ROOT/manifests`` 写入 freeze 与源码身份。全新安装应使用新的
``RPENT_REPRO_ROOT``；脚本不会覆盖 revision 错误或 dirty 的 RLinf checkout。

运动规划在 BEHAVIOR venv 中使用 NVlabs/cuRobo v0.8.0，固定 commit
``4ea77366ca48ee453e7df139e39fa6532af49f3b``。安装器在最终 repin 前使用
constraints，保留 NumPy 1.26.4、Torch 2.5.1+cu124 和 Isaac Sim 4.5.0.0。
不要无约束安装 cuRobo：resolver 升级到 NumPy 2 会破坏此环境的兼容性。

仿真资产
--------

接受 BEHAVIOR/OmniGibson 许可后，选择独立数据根并使用标准资产命令。该命令会在
BEHAVIOR venv 中调用三个 OmniGibson 官方下载函数，不会把 OmniGibson import 到
RPent 环境：

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   export BEHAVIOR_PYTHON="$RPENT_REPRO_ROOT/venvs/behavior/bin/python"
   python -m robots.behavior.assets_cli --accept-license --skip-existing

不传 ``--accept-license`` 时，官方下载器会显示交互式许可确认。该参数代表明确的
非交互许可确认；仅在接受许可条款后使用。

最终数据根必须包含：

.. code-block:: text

   BEHAVIOR-1K-datasets/
     2025-challenge-task-instances/
     behavior-1k-assets/
       scenes/
     omnigibson-robot-assets/
     omnigibson.key

Pi0.5 checkpoint
----------------

将已审查 checkpoint 下载到源码树之外：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32
   "$RPENT_REPRO_ROOT/venvs/rpent/bin/hf" download \
     RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 \
     --local-dir "$PI05_CHECKPOINT_PATH"

``python -m robots.behavior.assets_cli --verify`` 会检查 OmniGibson 必需目录，以及
源码中固定的 checkpoint size/SHA binding。#136 的共享 Pi0.5 component 接收
head、left wrist、right wrist 和 raw R1Pro proprio；原始 RPC 输出为
``[1, 32, 23]``，公共 client 返回 ``[32, 23]``。

.. code-block:: bash

   python -m robots.behavior.assets_cli --verify

DINOv2 派生缓存
---------------

BEHAVIOR 保留经审查的 `DINOv2 <https://github.com/facebookresearch/dinov2>`_
ViT-S/14 部署，用于从官方 Memory 语料中生成整图 embedding。运行前提供 DINOv2
源码归档和 ``dinov2_vits14_pretrain.pth`` 权重：

.. code-block:: bash

   export DINOV2_SOURCE_ARCHIVE=/path/to/dinov2-source.tar.gz
   export DINOV2_WEIGHTS=/path/to/dinov2_vits14_pretrain.pth

   curl -L \
     https://github.com/facebookresearch/dinov2/archive/7764ea0f912e53c92e82eb78a2a1631e92725fc8.tar.gz \
     -o "$DINOV2_SOURCE_ARCHIVE"
   curl -L \
     https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
     -o "$DINOV2_WEIGHTS"

DINOv2 是共享视觉 memory component；它不是分割模型，不替代 SAM3 mask、当前
公开观察或 MemoryManager 的 Markdown/YAML 语料。接受的源码 revision 和两个资产
SHA-256 固定在 ``robots/behavior/dino_v2/encoder.py``；runtime 会拒绝不匹配的资产。

DINO 缓存从已发布的 ``task-specific`` audit/recipe 对及其
``task-specific/artifacts/<cell>/`` 下经过校验的证据重建。仅 head RGB 帧参与
embedding，左右腕帧保留用于诊断。成功运行先在本地准备证据，再由 MemoryManager
在同一个 merge 锁内将证据和任务对一起发布。``dino_v2/`` 是可删除重建的派生数据，
不是另一套语料。使用 BEHAVIOR 解释器执行重建，不要手工修改缓存文件。

.. code-block:: bash

   "$BEHAVIOR_PYTHON" -m robots.behavior.build_memory_cli \
     --memory-dir /path/to/memory/behavior \
     --source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --weights "$DINOV2_WEIGHTS" \
     --cuda-device 1

任务身份
--------

使用 ``--task-name`` 和 ``--public-seed``。public seed 通过
``robots/behavior/task_specs.py`` 固定映射到官方 activity instance。

.. list-table::
   :header-rows: 1
   :widths: 24 42 16 18

   * - 任务
     - 指令
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

运行一次 Eval
-------------

每个 CUDA 子进程必须显式绑定一个物理 GPU：

.. code-block:: bash

   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent" --robot behavior \
     --task-name turning_on_radio --public-seed 1 \
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

首次加载环境通常需要数分钟。env、VLA 和 DINO 是不同进程，每个进程只接收自己
显式选择的 GPU。

官方 MemoryManager
------------------

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

使用标准 RPent Explore 入口运行一组有界 session：

.. code-block:: bash

   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent" --robot behavior \
     --explore \
     --explore-sessions 3 \
     --task-name picking_up_trash --public-seed 0 \
     --output-dir /path/to/behavior-explore \
     --memory-dir /path/to/behavior-memory \
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
- ``vla``：共享 ``rpent/robots/components/pi05_vla_server.py`` 服务；
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
     --output-dir /path/to/behavior-dashboard-run

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

主要日志：

.. code-block:: text

   <output-dir>/run.log
   <output-dir>/behavior_vla_server.log
   <output-dir>/behavior_dino_server.log
   <output-dir>/tasks/<task-run>/behavior_env_server.log
   <output-dir>/tasks/<task-run>/episode.mp4
   <output-dir>/tasks/<task-run>/terminal_receipt.json
