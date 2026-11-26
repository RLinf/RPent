.. _libero-data-flywheel:

采集轨迹与数据飞轮
========================

RPent 数据飞轮用于记录从任务规划到动作执行的轨迹，并从成功轨迹中生成训练数据。

使用 ``api``、``claude_code`` 或 ``codex`` 规划器时，开启 ``--record-trajectory`` 可记录视觉语言模型（VLM）的请求、回复、工具调用及其结果。在支持动作采集的环境中，还会记录视觉语言动作模型（VLA）的子任务指令、实际执行的动作及对应的观测。

``rpent-flywheel`` 支持导出 VLA 或 VLM 训练数据：

- ``--vla``\ （默认）：将确认成功轨迹中的 VLA 动作片段导出为 LeRobot 数据集，供 RLinf 监督微调。在本次对照评测中，微调后 RPent 在 LIBERO-PRO 上的任务成功率从 70% 升至 73%，在 RoboTwin 上从 48% 升至 72%。
- ``--vlm``：将 ``api`` 规划器记录的对话轮次导出为 ``samples.jsonl`` 与去重后的 ``images/`` 目录，用于 VLM 训练。默认仅筛选环境确认成功的轨迹，也可通过 ``--include-unsuccessful`` 包含失败或结果未知的轨迹。在下方 RoboTwin 对照评测中，用微调后的 ``Qwen3.8-27B`` 替换原规划器，RPent 的任务成功率从 16% 升至 30%。

安装
----

以下示例使用 LIBERO-PRO。环境准备和仿真资源下载见 :doc:`../get_started/quickstart`，API 认证配置见 :doc:`configure_planner`。在 RPent 仓库根目录安装 LIBERO-PRO 及可选的导出依赖：

.. code-block:: bash

   uv pip install -e ".[libero-pro,flywheel]"

可选依赖 ``flywheel`` 会安装 LeRobot，仅在导出 VLA 数据（``--vla``）时需要；使用 ``--vlm`` 导出 VLM 样本只需 RPent 基础安装即可。

采集轨迹
--------

采集默认关闭。用 ``--planner claude_code`` 采集轨迹时，需要 ``claude-agent-sdk 0.2.154+`` 和 ``Claude Code 2.1.274+``。LIBERO、RoboCasa365、RoboTwin 和 RoboDojo 仅在评测模式下采集实际执行的动作及观测。如果后续要导出 VLM 训练样本，请使用 ``--planner api`` 采集轨迹；``rpent-flywheel --vlm`` 仅支持导出该后端记录的轮次。

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

以下是轨迹的目录结构及相关文件的数据格式

.. code-block:: text

   /path/to/run/trajectories/<task_id>/
   ├── task.json
   ├── planners.jsonl
   ├── predictions.jsonl
   ├── actions.jsonl
   └── artifacts/

.. list-table:: ``task.json`` 记录一次任务运行的指令、配置和结果
   :header-rows: 1

   * - 字段
     - 含义
   * - ``task.task_id``
     - 本次运行的唯一编号，对应轨迹目录名 ``<task_id>``。
   * - ``task.instruction``
     - 本次运行的任务指令。
   * - ``task.config``
     - 本次运行的配置，保存在 ``task.config.data`` 中，包含运行参数、``system_prompt`` 及已注册的 ``tools`` 列表。
   * - ``task.result``
     - 任务运行的结果。 ``task.result.data.environment_success`` 表示任务是否成功。

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

``rpent-flywheel`` 提供 ``--vla``\ （默认）和 ``--vlm`` 两种互斥模式，分别用于导出 LeRobot 格式的 VLA 数据集和 API 规划器的 VLM 训练样本。

在两种模式下，``--input-root`` 既可以指向包含 ``task.json`` 的单条轨迹目录，也可以指向包含多次运行的上级目录；``--output-dir`` 必须不存在，原始记录不会被修改。如果多次运行分别保存在多个目录中，可将共同的上级目录设为 ``--input-root``，并用 ``--pattern``\ （通配符需加引号）筛选匹配的运行目录：

.. code-block:: text

   /path/to/
   ├── run_001/trajectories/<task_id_1>/
   └── run_002/trajectories/<task_id_2>/

导出 VLA 训练数据
~~~~~~~~~~~~~~~~~

将同一机器人的轨迹导出到新目录：

.. code-block:: bash

   rpent-flywheel \
     --vla \
     --input-root /path/to/run \
     --output-dir /path/to/lerobot/goal-task-00

``--vla`` 表示仅导出 VLA 训练数据，不加该参数时也默认导出 VLA 数据。不同机器人应分别导出。导出程序默认逐张压缩并保存图像（``--image-writer-threads 0``）；要并行写入图像，可在上面的命令中指定线程数，例如 ``--image-writer-threads 4``。

要将名称以 ``run_`` 开头的多个运行目录一起导出：

.. code-block:: bash

   rpent-flywheel \
     --vla \
     --input-root /path/to \
     --pattern 'run_*' \
     --output-dir /path/to/lerobot/combined

程序会查找这些运行目录及其子目录中的轨迹，按下文规则筛选后，统一导出到 ``/path/to/lerobot/combined/train/``。请保留命令中 ``run_*`` 两侧的引号。

VLA 导出程序按以下规则筛选和组织训练数据：

- 成功判定：任务结果中的 ``environment_success`` 必须明确为 ``true``；失败、结果缺失或无法确认成功的任务均不导出。
- 动作来源：只保留实际执行、且能唯一关联到有效 VLA 预测记录的动作。预测记录必须包含非空指令，并与动作属于同一次工具调用。非 VLA 动作（如工具自行计算并执行的动作）和未执行的预测动作均不参与导出。
- 数据组织：每个连续片段保存为独立的 LeRobot episode，每个实际执行的动作对应一帧，包含执行前观测、动作和对应的 VLA 指令。

数据集保存在 ``--output-dir`` 指定目录下的 ``train/`` 子目录中，不会自动划分验证集。输出目录中的 ``report.json`` 记录导出数量和排除原因；没有符合条件的动作时，只生成报告。

导出 VLM 训练数据
~~~~~~~~~~~~~~~~~

将 ``api`` 规划器记录的对话轮次导出为 ``samples.jsonl`` 与去重图像：

.. code-block:: bash

   rpent-flywheel \
     --vlm \
     --input-root /path/to/run \
     --output-dir /path/to/vlm/goal-task-00

VLM 导出程序按以下规则筛选和整理训练样本：

- 轨迹筛选：默认要求任务结果中的 ``environment_success`` 明确为 ``true``；传入 ``--include-unsuccessful``\ （须与 ``--vlm`` 一起使用）后，失败或缺少环境成功判定的轨迹也会纳入导出。
- 后端与轮次有效性：仅导出 ``api`` 规划器已完成的调用，要求请求和回复内容完整、模型正常结束（排除因长度截断、内容过滤或报错中止的回复）、工具调用参数为合法 JSON 对象，且至少包含可见文本或工具调用。``claude_code`` 和 ``codex`` 的轮次会被跳过。
- 样本组织与思考内容过滤：每条样本对应一轮模型调用，输入为该轮发给模型的对话历史和可用工具定义，监督目标为该轮的单条助手回复。如果模型开启了深度思考（reasoning），导出时会去掉模型内部的思考过程，对话历史和监督目标中都只保留最终可见文本与工具调用。
- 图像去重：所有引用到的不同图像按内容哈希各复制一份到 ``images/`` 目录，并将 ``samples.jsonl`` 中的图像路径改写为相对路径，使导出目录可独立迁移。引用的图像文件缺失时，该轮次会被排除。

输出目录结构如下，不会自动划分验证集：

.. code-block:: text

   /path/to/vlm/goal-task-00/
   ├── samples.jsonl
   ├── images/
   └── report.json

输出目录中的 ``report.json`` 记录导出数量和排除信息。``samples.jsonl`` 每行对应一轮待监督的规划器调用，主要字段如下：

.. list-table:: ``samples.jsonl`` 每行记录一轮待监督的规划器调用
   :header-rows: 1

   * - 字段
     - 含义
   * - ``episode_id``
     - 该轮所属轨迹的任务编号（``task.task_id``）。
   * - ``turn_id``
     - 本次规划器调用在轨迹内的唯一标识（``planner_id``）。
   * - ``messages``
     - 发给模型的对话历史，按顺序包含 ``system``、``user``、``assistant`` 和 ``tool`` 消息，包含文本、``images/`` 下的图像引用以及历史工具调用。
   * - ``tools``
     - 该轮提供给模型的工具定义列表。
   * - ``target``
     - 本轮用于监督训练的单条 ``assistant`` 回复，包含可见文本或工具调用。

使用 RLinf 训练
---------------

对于上述 LIBERO 示例，可使用 ``--vla`` 导出的 LeRobot 数据集，在独立的 RLinf 环境中监督微调 Pi0.5。RPent 负责采集和导出；模型训练、日志和权重保存由 RLinf 管理。

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

RoboTwin VLM 微调结果
~~~~~~~~~~~~~~~~~~~~~

使用 RPent 采集的成功轨迹，经 ``rpent-flywheel --vlm`` 导出后微调 ``Qwen3.8-27B``，并在 RPent 中替换 VLM 规划器与原模型对比评测。

评测覆盖 RoboTwin ``demo_randomized`` 中与训练相同的 10 个任务，每任务 5 个 seed，每组共 50 场。两组均使用 ``LingBot-VLA-RoboTwin-EEF-ckpt1500`` 作为 VLA 模型，规划器 reasoning effort 设为 ``medium``，并加载 Hugging Face 数据集 ``RLinf/RPent-memory`` 中的 ``robotwin/`` 任务记忆。

成功率如下：

.. list-table::
   :header-rows: 1

   * - 模型
     - 成功数 / 场次
     - 成功率
   * - 原模型
     - 8/50
     - 16%
   * - SFT 模型（step 48）
     - 15/50
     - 30%

本次评测中，SFT 后的模型多成功 7 场，任务成功率从 16% 提升至 30%，提高 14 个百分点。
