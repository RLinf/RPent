.. _libero-data-flywheel:

采集轨迹与数据飞轮
========================

使用 ``api``、``claude_code`` 或 ``codex`` 规划器时，可通过 ``--record-trajectory`` 保存任务、规划器和工具调用记录。已支持动作采集的环境还会记录实际执行的动作及观测。
``rpent-flywheel`` 支持导出 LeRobot 格式的 VLA 训练数据，暂不支持导出 VLM 训练数据。

安装
----

以下示例使用 LIBERO-PRO。环境准备和仿真资源下载见 :doc:`../get_started/quickstart`，API 认证配置见 :doc:`configure_planner`。在 RPent 仓库根目录安装 LIBERO-PRO 及可选的导出依赖：

.. code-block:: bash

   uv pip install -e ".[libero-pro,flywheel]"

采集轨迹
--------

采集默认关闭。用 ``--planner claude_code`` 采集轨迹时，需要 ``claude-agent-sdk 0.2.154+`` 和 ``Claude Code 2.1.274+``。LIBERO、RoboCasa365、RoboTwin 和 RoboDojo 仅在评测模式下采集实际执行的动作及观测。

开启采集并指定运行目录：

.. code-block:: bash

   rpent --robot libero \
     --suite libero_goal --task 0 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 \
     --record-trajectory \
     --output-dir /path/to/run

- ``--record-trajectory``：开启上述轨迹采集。
- ``--output-dir /path/to/run``：指定本次运行的输出根目录，请将 ``/path/to/run`` 替换为实际保存路径。

上述评测示例的轨迹保存在 ``/path/to/run/trajectories/<task_id>/`` 下，附件存放在轨迹目录中的 ``artifacts/`` 子目录。

轨迹数据格式
------------

``task.json`` 保存任务指令、运行配置、起止时间和任务结果。以下三个 JSONL 文件每行保存一个 JSON 对象。

.. list-table:: ``planners.jsonl`` 每行记录一次模型调用
   :header-rows: 1

   * - 字段
     - 含义
   * - ``planner_id``
     - 本次模型调用在轨迹内的唯一标识。
   * - ``backend``
     - 发起调用的规划器后端，取值为 ``api``、``claude_code`` 或 ``codex``。
   * - ``status``
     - 本次模型调用的状态，取值为 ``open``、``completed``、``failed`` 或 ``cancelled``。
   * - ``request``
     - 发给模型的请求，包含对话上下文和可用工具，具体结构取决于后端。
   * - ``response``
     - 本次调用的模型输出，可包含文本回复或工具调用请求。

.. list-table:: ``predictions.jsonl`` 每行记录一次 VLA 预测
   :header-rows: 1

   * - 字段
     - 含义
   * - ``prediction_id``
     - 本次预测在轨迹内的唯一标识。
   * - ``instruction``
     - VLA 在本次预测中使用的指令。
   * - ``action_ids``
     - 本次预测中实际执行的动作 ID，按执行顺序排列，对应 ``actions.jsonl`` 中的 ``action_id``。

.. list-table:: ``actions.jsonl`` 每行记录一步实际执行的环境动作
   :header-rows: 1

   * - 字段
     - 含义
   * - ``action_id``
     - 这一步动作在轨迹内的唯一标识。
   * - ``source``
     - 策略预测的动作记为 ``vla``，工具自行计算并执行的动作记为 ``tool_computed``，通过直接动作工具提交的控制量记为 ``direct``。
   * - ``action.format``
     - 机器人使用的动作格式。
   * - ``action.data``
     - 环境在这一步执行的控制指令。
   * - ``before_observation``
     - 执行动作前的环境观测。

导出训练数据
------------

将同一机器人的轨迹导出到新目录：

.. code-block:: bash

   rpent-flywheel \
     --vla \
     --input-root /path/to/run \
     --output-dir /path/to/lerobot/goal-task-00

``--vla`` 表示仅导出 VLA 训练数据；不加该参数时也默认导出 VLA 数据。

导出程序默认逐张压缩并保存图像（``--image-writer-threads 0``）。要并行写入图像，可在上面的命令中指定线程数，例如 ``--image-writer-threads 4``。

``--input-root`` 可以指向包含 ``task.json`` 的单条轨迹目录，也可以指向包含多次运行的上级目录。不同机器人应分别导出；``--output-dir`` 必须不存在。

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

导出程序按以下规则筛选和组织训练数据：

- 成功判定：任务结果中的 ``environment_success`` 必须明确为 ``true``；失败、结果缺失或无法确认成功的任务均不导出。
- 动作来源：只保留实际执行、且能唯一关联到有效 VLA 预测记录的动作。预测记录必须包含非空指令，并与动作属于同一次工具调用。非 VLA 动作（如工具自行计算并执行的动作）和未执行的预测动作均不参与导出。
- 数据组织：每个连续片段保存为独立的 LeRobot episode，每个实际执行的动作对应一帧，包含执行前观测、动作和对应的 VLA 指令。

数据集保存在 ``--output-dir`` 指定目录下的 ``train/`` 子目录中，不会自动划分验证集。输出目录中的 ``report.json`` 记录导出数量和排除原因；没有符合条件的动作时，只生成报告。原始记录不会被修改。

使用 RLinf 训练
---------------

对于上述 LIBERO 示例，可使用导出的 LeRobot 数据集，在独立的 RLinf 环境中进行 Pi0.5 监督微调。RPent 负责采集和导出；模型训练、日志和权重保存由 RLinf 管理。

环境安装和训练配置请参考 `RLinf OpenPI 监督微调指南 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/sft_openpi.html>`_。将 ``data.train_data_paths`` 设为导出目录下的 ``train/`` 子目录，其中包含 ``meta/info.json``。

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

评测使用 ``libero_goal_swap`` 的 task 0–9、seed 25–34，每组共 100 场。两组使用相同的 DeepSeek-V4.1-Flash-1017 规划器（reasoning effort 为 ``none``）并加载 Hugging Face 数据集 ``RLinf/RPent-memory`` 中的 ``libero/GPT_5.5_xhigh`` 任务记忆。

成功率如下：

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

评测覆盖 RoboTwin ``demo_randomized`` 中与训练相同的 10 个任务，每任务 5 个固定测试场景，每组共 50 场。两组使用相同的 DeepSeek-V4.1-Flash-1017 规划器（reasoning effort 为 ``none``）并加载 Hugging Face 数据集 ``RLinf/RPent-memory`` 中的 ``robotwin/`` 任务记忆（GPT-5.5 生成，推理强度为 ``xhigh``）。

成功率如下：

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
