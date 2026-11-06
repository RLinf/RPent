.. _libero-data-flywheel:

采集轨迹与数据飞轮
========================

使用 ``api`` 或 ``codex`` 规划器时，可通过 ``--record-trajectory`` 保存任务、规划器和工具调用记录。已支持动作采集的环境还会记录实际执行的动作及观测，供后续导出为 LeRobot 训练数据。

安装
----

以下示例使用 LIBERO-PRO。环境准备和仿真资源下载见 :doc:`../get_started/quickstart`，API 认证配置见 :doc:`configure_planner`。在 RPent 仓库根目录安装 LIBERO-PRO 及可选的导出依赖：

.. code-block:: bash

   uv pip install -e ".[libero-pro,flywheel]"

采集轨迹
--------

采集默认关闭，支持 ``--planner api`` 和 ``--planner codex``。LIBERO、RoboCasa365、RoboTwin 和 RoboDojo 仅在评测模式下采集实际执行的动作及观测。探索模式仍会保存任务、规划器和工具调用记录，但不包含可导出的 VLA 动作数据。

RoboDojo 的原生动作工具记录的是脚本动作。当前规划器工具集中没有 VLA 工具，因此这类运行不会产生可供 ``rpent-flywheel`` 导出的 VLA 训练片段。

开启采集并指定运行目录：

.. code-block:: bash

   rpent --robot libero \
     --suite libero_goal --task 0 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 \
     --record-trajectory \
     --output-dir /path/to/run

- ``--record-trajectory``：开启上述轨迹采集。
- ``--output-dir /path/to/run``：指定本次运行的输出根目录，请将 ``/path/to/run`` 替换为实际保存路径。

上述评测示例的记录保存在 ``/path/to/run/trajectories/<task_id>/`` 下，包括 ``task.json``、``planners.jsonl``、``actions.jsonl``、``predictions.jsonl`` 和存放附件的 ``artifacts/`` 目录。

导出训练数据
------------

将同一机器人的轨迹导出到新目录：

.. code-block:: bash

   rpent-flywheel \
     --vla \
     --input-root /path/to/run \
     --output-dir /path/to/lerobot/goal-task-00

``--vla`` 表示仅导出 VLA 训练数据；不加该参数时也默认导出 VLA 数据。

``--input-root`` 可以指向包含 ``task.json`` 的单条轨迹目录，也可以指向包含多次运行的上级目录。不同机器人应分别导出；``--output-dir`` 必须尚不存在。

RoboTwin 支持 LingBot 使用的 ``robotwin.action.ee`` 和 Fast-WAM 使用的 ``robotwin.action.qpos`` 格式。导出时会根据记录中的动作格式，自动选择配套的状态和动作布局。EE 数据的状态和动作均为 16 维；qpos 数据以 ``robot_state.qpos_target14`` 中的 14 维关节目标值作为状态，并保留实际执行的 14 维动作。

如果多次运行分别保存在下面的目录中，可以一起导出：

.. code-block:: text

   /path/to/
   ├── run_001/trajectories/<task_id_1>/
   └── run_002/trajectories/<task_id_2>/

将 ``--input-root`` 设为共同的上级目录，用 ``--pattern`` 选择以 ``run_`` 开头的运行目录：

.. code-block:: bash

   rpent-flywheel \
     --vla \
     --input-root /path/to \
     --pattern 'run_*' \
     --output-dir /path/to/lerobot/combined

程序会查找这些运行目录及其子目录中的轨迹，按下文规则筛选后，统一导出到 ``/path/to/lerobot/combined/train/``。请保留命令中 ``run_*`` 两侧的引号。

添加 ``--image-writer-threads 4``，即可使用 4 个线程同时压缩和保存图片。默认值为 ``0``，即逐张压缩和保存。

只有环境确认成功的任务才参与导出，规划器结束不代表任务成功。导出程序按以下规则筛选和组织训练数据：

- 成功判定：任务结果中的 ``environment_success`` 必须明确为 ``true``；失败、结果缺失或无法确认成功的任务均不导出。
- 动作来源：只保留实际执行、且能唯一关联到有效 VLA 预测记录的动作。预测记录必须包含非空指令，并与动作属于同一次工具调用。脚本动作、其他非 VLA 动作，以及未执行的预测动作均不参与导出。
- 数据组织：每个连续片段保存为独立的 LeRobot episode，每个实际执行的动作对应一帧，包含执行前观测、动作和对应的 VLA 指令。

数据集保存在 ``--output-dir`` 指定目录下的 ``train/`` 子目录中，不会自动划分验证集。输出目录中的 ``report.json`` 记录导出数量和排除原因；没有符合条件的动作时，只生成报告。原始记录不会被修改。

使用 RLinf 训练
---------------

对于上述 LIBERO 示例，可使用导出的 LeRobot 数据集，在独立的 RLinf 环境中进行 Pi0.5 监督微调。RPent 负责采集和导出；模型训练、日志和权重保存由 RLinf 管理。

环境安装和训练配置请参考 `RLinf OpenPI 监督微调指南 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/sft_openpi.html>`_。训练数据路径应指向导出的 LeRobot 数据集目录，即包含 ``meta/info.json`` 的目录，而不是原始轨迹目录。

先准备支持 LIBERO LeRobot 数据加载的 RLinf 环境，并将 Pi0.5 LIBERO SFT 配置保存为 ``/path/to/sft-config/libero_pi05_sft.yaml``，再运行：

.. code-block:: bash

   cd /path/to/RLinf
   unset PYTHONPATH
   export PYTHONPATH="$PWD"
   export EMBODIED_PATH="$PWD/examples/sft"

   .venv/bin/python examples/sft/train_vla_sft.py \
     --config-path /path/to/sft-config \
     --config-name libero_pi05_sft \
     data.train_data_paths=/path/to/lerobot/goal-task-00/train \
     actor.model.model_path=/path/to/pi05-checkpoint \
     runner.logger.log_path=/path/to/new-training-output

根据可用 GPU 设置配置中的资源分配和 batch size，并使用与所选模型和数据集相匹配的 checkpoint 及归一化设置。

SFT 实验结果
------------

LIBERO-PRO VLA 微调结果
~~~~~~~~~~~~~~~~~~~~~~~

使用 RPent 采集的成功轨迹，对 ``RLinf-Pi05-LIBERO-130-fullshot-SFT`` 进行微调，并在 RPent 中替换 VLA 模型进行对比评测。

评测使用 ``libero_goal_swap`` 的 task 0–9、seed 25–34，每组共 100 场。两组使用相同的 DeepSeek-V4.1-Flash-1017 规划器（reasoning effort 为 ``none``）和官方 memory，仅替换 VLA 模型。

完整 RPent 系统的任务成功率如下：

.. list-table::
   :header-rows: 1

   * - 模型
     - 成功数 / 场次
     - 成功率
   * - 原模型
     - 70/100
     - 70%
   * - SFT 模型（step 300）
     - 73/100
     - 73%

本次评测中，SFT 后的模型多成功 3 场，成功率提高 3 个百分点。

RoboTwin VLA 微调结果
~~~~~~~~~~~~~~~~~~~~~

使用 RPent 采集的成功轨迹，对 ``LingBot-VLA-RoboTwin-EEF-ckpt1500`` 进行微调，并在 RPent 中替换 VLA 模型进行对比评测。

评测覆盖 RoboTwin ``demo_randomized`` 中与训练相同的 10 个任务，每任务 5 个固定测试场景，每组共 50 场。完整 RPent 系统的任务成功率如下：

.. list-table::
   :header-rows: 1

   * - 模型
     - 成功数 / 场次
     - 成功率
   * - 原模型
     - 24/50
     - 48%
   * - SFT 模型（step 500）
     - 36/50
     - 72%

本次评测中，SFT 后的模型多成功 12 场，任务成功率从 48% 提升至 72%，提高 24 个百分点。
