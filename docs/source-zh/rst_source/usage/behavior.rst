BEHAVIOR
========

`BEHAVIOR-1K <https://behavior.stanford.edu/>`_ 是面向长程家庭活动的仿真基准，
提供照片级、可交互的家庭环境。RPent 当前通过 source-editable
``robots/behavior`` 接入两个已审查任务族：``turning_on_radio`` 和
``picking_up_trash``。默认 VLA 为 **Pi0.5**，由
``robots/behavior/vla_server.py`` 提供服务。

安装
----

建议使用 Ubuntu 22.04、Isaac Sim 4.5 支持的 NVIDIA RTX GPU 和较新的 NVIDIA
驱动。源码安装脚本会检查宿主机命令，并创建两个彼此独立的 Python 3.10 环境：

- **RPent 环境**运行 CLI、planner 与 Dashboard；
- **BEHAVIOR 环境**运行 RLinf、OmniGibson、Isaac Sim、Pi0.5、DINOv2 以及
  所有 BEHAVIOR sidecar 进程。

不要合并这两个环境。Isaac Sim 和 OpenPI 的兼容版本与通用 RPent 依赖不同。

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent

   export RPENT_REPRO_ROOT="$PWD/.behavior-runtime"
   bash scripts/install_behavior_runtime.sh

安装脚本会固定 ``uv`` 与已审查的 RLinf revision，调用官方 RLinf BEHAVIOR
installer，并在所有安装动作结束后执行一次最终兼容性回钉。FastAPI/Pydantic 会在
Isaac 安装后恢复到已验证版本；OpenPI transformer replacement 只会在最终
Transformers 版本安装完成后复制，之后不再执行任何 package install。

完整 package freeze、源码 revision、安装日志和 ``uv pip check`` 报告均写入
``$RPENT_REPRO_ROOT``。BEHAVIOR 环境会 editable 安装 RPent，确保直接启动的
sidecar script 能导入其源码，因此 ``pip check`` 也会看到只应在独立 RPent 环境
运行的 planner package metadata，并报告它们要求更高版本的 Pydantic/Starlette。
报告还可能保留已审查的 upstream 冲突，包括 ``rlinf-openpi``、``lerobot`` 的
torch/torchvision/torchcodec、``tensorflow-addons`` 的 typeguard 和
``tensorflow-metadata`` 的 protobuf 约束。最终回钉后不要为消除这些报告再次升级
package；安装脚本会单独强制验证精确运行版本、关键 import、CUDA tensor smoke 和
BEHAVIOR self-check。

BEHAVIOR 资产
-------------

策略 checkpoint 不包含 OmniGibson 完整数据。请在 BEHAVIOR 环境中准备已接受许可
的 BEHAVIOR-1K 数据：

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   export BEHAVIOR_PYTHON="$RPENT_REPRO_ROOT/venvs/behavior/bin/python"
   mkdir -p "$OMNIGIBSON_DATA_PATH"

   "$BEHAVIOR_PYTHON" -c \
     "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
   "$BEHAVIOR_PYTHON" -c \
     "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
   "$BEHAVIOR_PYTHON" -c \
     "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"

BEHAVIOR task assets 超过 30 GB。解压后的数据根目录必须同时包含以下四项；缺少
``scenes`` 会导致 env 启动失败，缺少 ``omnigibson.key`` 则无法加载加密 USD
资产。

.. code-block:: text

   BEHAVIOR-1K-datasets/
     2025-challenge-task-instances/
     behavior-1k-assets/
       scenes/
     omnigibson-robot-assets/
     omnigibson.key

VLA 配置
--------

下载 BEHAVIOR Pi0.5 checkpoint
`RLinf-Pi05-BEHAVIOR-1K-PT50-CS32
<https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_，再将
``PI05_CHECKPOINT_PATH`` 指向下载目录：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/your/pi05-behavior-model
   hf download RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 \
     --local-dir "$PI05_CHECKPOINT_PATH"

策略读取三路 RGB 图像和紧凑的 R1Pro 状态。其 ``predict`` RPC 返回
``[1, T, 23]`` batch tensor，executor 再逐个消费 ``[T, 23]`` action chunk。
checkpoint 应保存在 Python package 之外，并通过 ``PI05_CHECKPOINT_PATH`` 或
``--policy-checkpoint`` 显式绑定到每次运行。

DINOv2 配置
------------

BEHAVIOR 使用经过审查的 `DINOv2
<https://github.com/facebookresearch/dinov2>`_ ViT-S/14 部署生成整图
embedding，并检索 episode memory。运行时需要 DINOv2 source archive 和
``dinov2_vits14_pretrain.pth`` 权重：

.. code-block:: bash

   export DINOV2_SOURCE_ARCHIVE=/path/to/dinov2-source.tar.gz
   export DINOV2_WEIGHTS=/path/to/dinov2_vits14_pretrain.pth

   curl -L \
     https://github.com/facebookresearch/dinov2/archive/7764ea0f912e53c92e82eb78a2a1631e92725fc8.tar.gz \
     -o "$DINOV2_SOURCE_ARCHIVE"
   curl -L \
     https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
     -o "$DINOV2_WEIGHTS"

DINOv2 在 BEHAVIOR runtime 中承担共享视觉 memory component 的角色，但它不是
分割模型，也不会生成 SAM3 mask；当前目标定位依赖 fresh observation 和公开几何
工具。允许使用的 DINOv2 source revision 及两份资产的 SHA-256 identity 均固定在
``robots/behavior/memory_embeddings_dinov2.py``；runtime 会拒绝不匹配该公开
contract 的资产。

任务选择
--------

运行 BEHAVIOR 任务时，可通过以下参数选择任务：

- ``--task-name`` —— 选择 ``turning_on_radio`` 或
  ``picking_up_trash``。
- ``--public-seed`` —— 选择稳定的公开 seed；每个 seed 映射到一个官方
  BEHAVIOR activity instance。
- ``--behavior-mode`` —— 选择 ``eval`` 或 ``explore``，默认为 Eval。
  Explore attempt 由下文的外层 harness 启动。
- ``--max-episode-steps`` —— 设置 episode step budget。

``--task`` 和 ``--seed`` 是 ``--task-name`` 与 ``--public-seed`` 的兼容别名；
新命令应优先使用 BEHAVIOR 的显式参数名。

.. _behavior-core-tasks:

BEHAVIOR 核心任务一览
~~~~~~~~~~~~~~~~~~~~~

公开 seed 划分属于 source-controlled task spec。Explore 与 Eval 使用互不重叠的
官方 activity instance。

.. list-table::
   :header-rows: 1
   :widths: 24 38 18 20

   * - 任务
     - 指令
     - Explore seeds
     - Eval seeds
   * - ``turning_on_radio``
     - 打开客厅桌上的 radio receiver。
     - ``0``
     - ``1``-``9``
   * - ``picking_up_trash``
     - 将客厅的三个 soda can 放入厨房 trash can。
     - ``0``-``9``
     - ``10``-``19``

完整的 public-seed-to-instance 映射定义在
``robots/behavior/task_specs.py``。原生 activity instance ID 属于部署细节，不应
代替 CLI 中的 public seed。

验证资产并运行
------------

启动 Isaac Sim 前，先验证完整数据树、policy checkpoint contract 和两份固定
DINOv2 资产的 SHA-256：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/your/pi05-behavior-model
   export DINOV2_SOURCE_ARCHIVE=/path/to/dinov2-source.tar.gz
   export DINOV2_WEIGHTS=/path/to/dinov2_vits14_pretrain.pth

   scripts/verify_behavior_assets.sh

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

首次 env 加载通常需要 1.5 至 5 分钟。xFormers unavailable、Isaac
deprecation、audio 与 headless GLFW warning 可能不是致命错误；应结合 Dashboard
章节列出的 component log 判断真实启动失败。切换 planner 的方法见
:doc:`configure_planner`。

探索模式与本地 Memory 评测
--------------------------

RPent 支持两种 BEHAVIOR 运行模式：

- **Exploration** 是 memory 生成流程。外层 harness 可以执行多次 attempt，但
  每次 attempt 都拥有新的 RPent process、planner invocation、env server 和
  episode。BEHAVIOR 不会在同一个 planner invocation 内 reset episode。
- **Evaluation** 是默认的单次运行路径。提供 ``--behavior-memory-dir`` 时，它会
  读取经过审查的 episode-memory catalog，且不会重试 episode。

使用 Eval seed 运行本地 memory 评测：

.. code-block:: bash

   rpent --robot behavior \
     --task-name turning_on_radio --public-seed 1 \
     --behavior-mode eval \
     --planner codex --model gpt-5.5 \
     --behavior-memory-dir /path/to/reviewed-behavior-memory \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS"

省略 ``--behavior-memory-dir`` 时会使用合法的空 episode catalog，不会下载或
静默替换任务专用 memory。

重复 Explore attempt 必须通过 BEHAVIOR 自己的外层 harness 启动：

.. code-block:: bash

   python -m robots.behavior.harness explore \
     --attempts 3 \
     --output-dir /path/to/behavior-explore \
     -- \
     --task-name picking_up_trash --public-seed 0 \
     --planner codex --model gpt-5.5 \
     --dino-source-archive "$DINOV2_SOURCE_ARCHIVE" \
     --dino-weights "$DINOV2_WEIGHTS"

Explore artifact 可经人工审查后晋升为 recipe、task memory 或 DINO 索引的
episode memory；candidate Explore 证据必须与 held-out Eval artifact 分开。
成功只认 terminal receipt 中记录的官方原始
``info["done"]["success"]``，planner 或 primitive 完成不能替代该信号。

进程分工
--------

- **env_server** （``robots/behavior/env_server.py``）持有官方
  BEHAVIOR/OmniGibson 环境，并通过 RPent RPC 暴露 reset、observation、action、
  Dashboard control 和官方成功 receipt。
- **vla_server** （``robots/behavior/vla_server.py``）持有 Pi0.5 BEHAVIOR
  checkpoint，并通过 RPent RPC 暴露 ``predict``。
- **dino_server** （``robots/behavior/dino_server.py``）持有 DINOv2-S/14
  encoder，为 episode-memory retrieval 提供 embedding。
- **toolkit** （``robots/behavior/toolkit.py``）定义 planner 可调用的公开工具，并
  记录 observation、action trace 和 terminal receipt。

env process 使用独立 GPU；VLA 与 DINOv2 默认共享 model GPU。每个本地 CUDA
child 只接收一个显式物理 ``CUDA_VISIBLE_DEVICES`` 值。

Planner 能调用的工具
--------------------

BEHAVIOR 工具分为三组；每次运行以 active toolkit schema 为准。

**VLA 动作工具：**

- ``pi0_nav_pick(instruction, chunks)`` —— 使用 Pi0.5 完成导航与抓取。

**观测与解析动作工具：**

- ``observe(...)`` —— 读取 fresh head 或 wrist-camera observation。
- ``pixel_to_world(...)`` —— 将 fresh image pixel 反投影到场景中。
- ``navigate_to(...)`` —— 规划移动底盘轨迹。
- ``move_to(...)``、``move_both_to(...)`` —— 规划单臂或双臂运动。
- ``rotate_wrist(...)`` —— 调整腕部姿态。
- ``close(...)``、``open(...)`` —— 控制夹爪。
- ``press(...)`` —— 执行带保护的接触动作。

公开 schema 描述的是已审查的 planner 路线，但若当前 official RLinf backend 没有
reviewed manual-motion adapter，部署会返回 ``manual_motion_unavailable``。此时不能
把这些 manual motion tool 当作可执行 fallback；该部署已验证的运动入口仍是
``pi0_nav_pick``。

**安全、状态与终止工具：**

- ``get_prepared_motion_status(...)`` —— 读取 prepared motion 的执行状态。
- ``save_robot_state_checkpoint(...)`` —— 记录 planner 可见的状态标记。
- ``finish(status, summary)`` —— 结束 planner run 并写入 receipt。

物理动作工具会推进环境；observation、status 和 state checkpoint 本身不能证明
任务成功。

Dashboard
---------

加上 ``--dashboard`` 可启动长生命周期的本地 Dashboard Session。VLA 与
DINOv2 服务会在 TaskRun 之间共享，每个 TaskRun 则使用 fresh environment：

.. code-block:: bash

   TASK_NAME=turning_on_radio PUBLIC_SEED=1 \
     scripts/run_behavior_dashboard.sh

打开终端输出的 URL，确认 Session 配置，然后在页面中启动 TaskRun：

.. code-block:: text

   /rpent-task turning_on_radio 1

Dashboard 会显示 planner reasoning、head/wrist-camera frame 和 action timeline。
新的 ``/rpent-task`` 会启动 fresh environment。Dashboard 不会改变官方成功
定义。添加 ``--dashboard-language zh-cn`` 可切换中文界面。

launcher 会打印本次 output directory。component 启动问题应从以下日志定位：

.. code-block:: text

   <output-dir>/run.log
   <output-dir>/behavior_vla_server.log
   <output-dir>/behavior_dino_server.log
   <output-dir>/tasks/<task-run>/behavior_env_server.log

接入自定义 VLA
----------------

如果已有非 Pi0.5 的 BEHAVIOR-compatible VLA，可在不修改环境的情况下替换 model
client：

1. 暴露相同的 ``predict`` RPC contract，并按 BEHAVIOR policy layout 返回有限值
   ``[1, T, 23]`` action tensor。
2. 使用 ``--vla-endpoint [protocol://]host:port`` 指向该服务。
3. 只有 public tool surface 需要改变时，才修改
   ``robots/behavior/toolkit.py``。

工具扩展流程见 :doc:`../development/add_primitive`。

结果复现
--------

BEHAVIOR workflow 和 benchmark recipe 仍在探索中；RPent 现阶段暂不声称
BEHAVIOR benchmark success rate。

为了让运行可复现，应记录 RPent commit、pinned RLinf/OmniGibson/Isaac
environment、policy checkpoint digest、DINOv2 source 与 weight digest、
task/public-seed mapping version、planner 和 model、GPU binding，以及完整
output directory。正式运行前可先验证 RPent 侧轻量 contract：

.. code-block:: bash

   python -m robots.behavior.selfcheck

self-check 会验证 plugin import、RobotSpec/CLI parsing、task/seed mapping 和
public tool count；它不会渲染 prompt、启动 simulator，也不能证明任务成功。只有
``terminal_receipt.json`` 中存在 ``official_success_receipt``，且其 ``source``
为 ``info["done"]["success"]``、``raw_done.success`` 为 ``true``，运行结果才
能报告为成功。
