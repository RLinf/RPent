BEHAVIOR
========

`BEHAVIOR-1K <https://behavior.stanford.edu/>`_ 基于 OmniGibson 提供长程家庭任务。
RPent 当前把 ``turning_on_radio`` 和 ``picking_up_trash`` 作为标准 sibling
robot plugin 接入，源码位于 ``robots/behavior``。

接入合同与 LIBERO、RoboCasa、RoboTwin 相同：``get_robot_spec()`` 提供
CLI/config/runtime hooks，``get_toolkit()`` 提供公开工具。BEHAVIOR 生命周期逻辑
留在 ``robots/behavior`` 内。公共 ``--explore`` loop 仍只属于 LIBERO；
BEHAVIOR 使用 ``--behavior-mode explore`` 和自己的外层 harness。

安装状态
--------

BEHAVIOR 以源码 editable 方式运行，并使用两个相互独立的 Python 3.10 环境：

- **RPent venv**：运行 CLI、planner、Dashboard 和 MemoryManager；
- **BEHAVIOR venv**：运行 RLinf、OmniGibson、Isaac Sim 和 Pi0.5。

``.[behavior]`` 只安装 RPent 侧依赖，不包含完整模拟器、资产或 checkpoint；
普通 wheel 不承诺可直接运行 BEHAVIOR。请在源码 checkout 中执行：

.. code-block:: bash

   python -m pip install -e ".[behavior]"
   export RPENT_REPRO_ROOT="$PWD/.behavior-runtime"
   export UV_CACHE_DIR="$RPENT_REPRO_ROOT/uv-cache"
   behavior-install-runtime

安装器会在两个 venv 中保持 RPent editable，克隆已审查的 RLinf revision，调用
官方 RLinf BEHAVIOR 安装器，应用已审查的 CUDA/OpenPI 兼容性 pin，验证关键 import
和 CUDA，并在 ``$RPENT_REPRO_ROOT/manifests`` 写入 freeze 与源码身份。fresh install
应使用新的 ``RPENT_REPRO_ROOT``；脚本不会覆盖 revision 错误或 dirty 的 RLinf
checkout。

仿真资产
--------

接受 BEHAVIOR/OmniGibson 许可后，选择独立数据根并使用标准资产命令。该命令会在
BEHAVIOR venv 中调用三个 OmniGibson 官方下载函数，不会把 OmniGibson import 到
RPent 环境：

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   export BEHAVIOR_PYTHON="$RPENT_REPRO_ROOT/venvs/behavior/bin/python"
   behavior-download-assets --accept-license --skip-existing

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

``behavior-download-assets --verify`` 会检查 OmniGibson 必需目录，以及源码中固定的
checkpoint size/SHA binding。#136 的共享 Pi0.5 component 接收 head、left wrist、
right wrist 和 raw R1Pro proprio；原始 RPC 输出为 ``[1, 32, 23]``，公共 client
返回 ``[32, 23]``。

.. code-block:: bash

   behavior-download-assets --verify

DINOv2 配置
-------------

BEHAVIOR 保留经审查的 `DINOv2 <https://github.com/facebookresearch/dinov2>`_
ViT-S/14 部署，用于整图 embedding 与 episode memory 检索。运行前提供 DINOv2
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
     --behavior-mode eval \
     --planner codex --model gpt-5.5 \
     --behavior-repo "$RPENT_REPRO_ROOT/RLinf" \
     --behavior-python "$RPENT_REPRO_ROOT/venvs/behavior/bin/python" \
     --activity-instance-dir \
       "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --behavior-env-cuda-device 0 \
     --behavior-model-cuda-device 1 \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS" \
     --memory-profile local \
     --memory-dir /path/to/behavior-memory \
     --behavior-memory-dir /path/to/reviewed-behavior-episode-memory

首次加载环境通常需要数分钟。env、VLA 和 DINO 是不同进程，每个进程只接收自己
显式选择的 GPU。

官方 MemoryManager
------------------

BEHAVIOR 使用和其他机器人相同的 Markdown/YAML ``MemoryManager`` 格式与公共 memory
工具。DINO episode-memory catalog 是独立的视觉经验检索源；配置后，它的 advisory
会附加到公开 tool receipt，并始终只作为历史建议。

- Eval 只构造一个 ``read_only`` MemoryManager；
- Explore 只构造一个 ``inbox_write`` MemoryManager，写入范围限定为
  ``<memory-dir>/_inbox/<recipe-tag>``；
- ``MEMORY.md``、``global/``、``suite/``、``task/``、``_inbox/`` 和
  ``_merged/`` 保持 RPent 标准语义。

缺失或空 corpus 是合法状态，但不会提供任何建议。需要共享已审查 memory 的运行应
显式传入同一个 ``--memory-dir``。

``--behavior-memory-dir`` 只用于经审查的 DINO episode-memory catalog。省略该参数
会选择合法的空 episode catalog，不会下载或静默替换为特定任务 memory。

多次 Explore attempt 必须通过 BEHAVIOR 外层 harness 执行。它为每次 attempt 启动
fresh RPent 进程和 episode，让全部 attempt 指向同一个官方 corpus，并在结束后调用
现有 ``MemoryManager.merge_memory()``。planner 不能在单次 invocation 内 reset。

.. code-block:: bash

   python -m robots.behavior.harness explore \
     --attempts 3 \
     --output-dir /path/to/behavior-explore \
     --memory-dir /path/to/behavior-memory \
     -- \
     --task-name picking_up_trash --public-seed 0 \
     --planner codex --model gpt-5.5

如果 review 合同要求先保留 inbox、不立即发布，可传
``--no-auto-merge-memory``。只有 terminal receipt 携带官方成功时，task audit/recipe
pair 才会晋升。

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
     --behavior-mode eval \
     --behavior-repo "$RPENT_REPRO_ROOT/RLinf" \
     --behavior-python "$RPENT_BEHAVIOR_PYTHON" \
     --activity-instance-dir \
       "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS" \
     --memory-profile local --memory-dir /path/to/behavior-memory \
     --output-dir /path/to/behavior-dashboard-run

Dashboard 使用公共 Start Session 流程与 head/left-wrist/right-wrist 相机视图。
BEHAVIOR 不增加 robot-local 手动按钮、手动控制 backend 或
``env.dashboard_*`` RPC。``pi0_nav_pick``、``observe``、``navigate_to``、
``move_to``、``press``、``open``、``close`` 等 planner primitive 是否可用，以
active tool schema 和 backend capability 为准。

主要日志：

.. code-block:: text

   <output-dir>/run.log
   <output-dir>/behavior_vla_server.log
   <output-dir>/behavior_dino_server.log
   <output-dir>/tasks/<task-run>/behavior_env_server.log
   <output-dir>/tasks/<task-run>/episode.mp4
   <output-dir>/tasks/<task-run>/terminal_receipt.json

结果复现
--------

以下记录是一次有界的运行链路验收，不是长程 benchmark 结果。复现时应使用新的
``RPENT_REPRO_ROOT``，执行前文安装与下载命令并验证 checkpoint 和资产，然后启动
已知的 ``turning_on_radio`` development instance：activity definition ``0``、
activity instance ``242``、public seed ``0``。reset 后执行一次 ``env.step`` 和一次
由 Pi0.5 动作驱动的 ``env.chunk_step``，再检查 component metadata、观察和动作 shape、
Dashboard snapshot 与生成的 ``episode.mp4``。这次有界运行没有使用 held-out 布局；
视频只保留为运行 artifact，不写入仓库。

**运行链路复现。**

- fresh 独立 runtime root 生成了相互分离的 RPent 与 BEHAVIOR venv，使用
  ``uv 0.12.7`` CLI 和 Python ``3.10.12``。最终一次重试只复用该新 runtime root
  内的 venv，完成兼容性 repin：``torch 2.5.1+cu124``、
  ``torchaudio 2.5.1+cu124``、``torchcodec 0.2.0+cu124``、
  ``torchvision 0.20.1+cu124``、``transformers 4.53.2``；CUDA smoke 与关键
  import 检查通过，安装器退出码为 ``0``。report-only 依赖 metadata 检查仍记录了
  ``15`` 个上游 pin 不兼容项。
- checkpoint 校验记录的 ``model.safetensors`` 大小精确为
  ``7,233,650,408`` bytes，SHA-256 为
  ``7e257666d835f6af701de493676a6c86a0421b2efc737a0f911d782b7a09f635``。
  三个 OmniGibson 源归档合计精确为 ``31,887,356,541`` bytes；解压后，三个已验证
  资产目录共含 ``118,491`` 个文件、精确为 ``37,532,605,007`` bytes。
- 真实 GPU observation 包含 head RGB ``[720, 720, 3] uint8``、按 left/right
  排列的 wrist RGB ``[2, 480, 480, 3] uint8``，以及有限值 proprio
  ``[256] float32``。Pi0.5 返回有限值 raw action ``[1, 32, 23] float32``，client
  返回 ``[32, 23] float32``。
- 环境真实执行了一次单步和一个完整的 32-step chunk，共精确执行 ``33`` 个 env
  steps；输出视频精确包含 ``34`` 帧。ENV、VLA、MemoryManager 与 Dashboard 均报告
  ready。
- 后续 DINOv2 GPU smoke 使用已恢复的 ``behavior_dino`` RPC service，并指定 CUDA
  device ``2``。``healthz`` 返回 ``status=ok``，``dino.get_meta`` 报告
  ``facebookresearch/dinov2_vits14`` 在 revision
  ``facebookresearch/dinov2@7764ea0f912e53c92e82eb78a2a1631e92725fc8`` 上的
  dimension 为 ``384``。source archive 精确为 ``2,869,642`` bytes，SHA-256 为
  ``c27dcdaf50e9fb5bbdf2bb529da357716372e19c6afab17d5350f3f0094aed4b``；weights
  文件精确为 ``88,283,115`` bytes，SHA-256 为
  ``b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9``。对一张
  RGB ``[224, 224, 3] uint8`` 测试图执行 encode 后，raw RPC 与
  ``BehaviorDinoClient`` 均返回有限值 ``[384] float32`` 向量；client 归一化后的
  L2 norm 为 ``0.9999999997354404``。连续两次 encode 的最大绝对差为 ``0.0``。
  owner shutdown 返回 ``ok=true``，server 退出码为 ``0``，端口已释放。

**探索阶段边界。**

长程任务的 benchmark recipe 仍在探索中，现阶段不提供汇总任务完成指标。官方成功
证据只接受 ``terminal_receipt.json`` 中 ``task_success=true``，且其内嵌 receipt 的
``source`` 为 ``info["done"]["success"]``。这次运行记录为
``task_success=false``（探针字段名为 ``official_task_success``），因此不声称任务完成。
primitive 结果不能替代官方成功，运行链路可执行也不代表已具备 benchmark readiness。

成功与诊断
----------

官方 task success 只等于当前 episode 的
``info["done"]["success"] is True``。reward、``terminated``、``truncated``、
primitive success、截图、视频和进程退出都不能替代它。receipt 只能记录 raw evidence，
不能制造成功。

启动仿真前可运行轻量源码检查：

.. code-block:: bash

   python -m robots.behavior.selfcheck

self-check 验证 plugin discovery、CLI/config、任务映射、memory profile 和公开工具数量；
它不会加载资产、启动 GPU 服务、执行动作或证明 task success。
