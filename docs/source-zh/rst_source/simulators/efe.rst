Embodied First Exam
===================

通过 `Harbor <https://github.com/laude-institute/harbor>`_ 使用 RPent 运行 \
`Embodied First Exam（EFE） <https://www.embodied-first-exam.ai/>`_ 机器人评测任务。\
本页以仅需 CPU 的 AutoBio 实验室任务为例，说明在配有 Docker 与模型 API 的 Linux \
宿主机上的完整运行与评分流程；从 `运行一个任务`_ 开始即可跑通一次评测。

概览
----

EFE 提供带固定场景的机器人任务、HTTP 仿真服务（``rcb_client``）与独立评分程序\
（Verifier）。每次任务运行对应一个 Harbor **Trial**：由 Harbor 启动仿真容器与 \
Agent 容器、运行 RPent、记录 Verifier 评分并清理容器。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      无独立动作模型（解析 IK ``Skills`` 或原生 ``Raw`` 动作）

   .. grid-item-card:: 规划器

      Codex

   .. grid-item-card:: 任务

      36 个基准中的 1,066 个 ``standard`` 模式任务（如 AutoBio、RoboARC）

   .. grid-item-card:: 硬件

      Linux、Docker；CPU 或 NVIDIA GPU（视任务而定）

任务
~~~~

可在 `EFE 任务浏览页 <https://www.embodied-first-exam.ai/tasks/>`_ 查找任务，\
任务定义位于 `EFE 仓库 <https://github.com/mll-lab-nu/embodied-first-exam>`_ 的 \
``tasks/<benchmark>/<task>`` 目录。

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - 模式
     - 任务数
     - 支持情况与行为
   * - ``standard``
     - 1066
     - 支持 ``eai-standard/2.2`` 与 ``2.4`` 协议。仿真器在收到 Agent 动作后推进步进。
   * - ``standard-realtime``
     - 84
     - 暂不支持（使用 ``eai-standard/2.3`` 或 ``eai-standard-realtime/1.0`` 协议）。
   * - ``privileged``
     - 959
     - 暂不支持（要求提交完整轨迹或策略文件，而非逐步交互控制）。

.. note::

   运行前请检查任务 ``task.toml`` 中的 ``metadata.protocol``。非 \
   ``eai-standard/2.2`` 或 ``2.4`` 协议的任务会在连接阶段直接报错退出。

观测与动作
~~~~~~~~~~

RPent 通过 ``skills`` 参数提供两种控制模式（详见 `对照 Skills 与 Raw`_）：

- **Skills**\ （``skills=true``，默认）：提供 ``move_to_pose`` 与 ``set_gripper``。\
  若任务的动作布局或运动学不支持 IK 技能，工具集会自动回退为 **Raw**。
- **Raw**\ （``skills=false``）：提供 ``efe_submit``，直接按任务原生动作格式提交数值。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 项目
     - 说明
   * - 观测
     - 相机 RGB 图像、可选深度图、``camera_calibration``、``frame_extrinsics_cv``\ （逐帧世界到相机变换）、关节位置、工具末端位姿、夹爪开度及动作规格说明。
   * - 动作
     - **Skills**：目标末端位置、可选 ``wxyz`` 四元数朝向与夹爪开度。**Raw**：符合 ``efe_spec`` 顺序的原生动作向量。
   * - 奖励与成功判定
     - 由 EFE 评分程序独立判定，结果写入 ``verifier/reward.json`` 与 ``result.json``\ （见 `查看结果`_）。
   * - 任务指令
     - Harbor 传入的任务原始 ``instruction.md``。

安装与资源准备
--------------

Linux 宿主机前置要求：

- ``git``、`uv <https://docs.astral.sh/uv/getting-started/installation/>`_ 以及带 Compose v2 和 BuildKit 的 Docker Engine（参见 `EFE 宿主机要求 <https://github.com/mll-lab-nu/embodied-first-exam#quick-start>`_）。
- 运行 GPU 任务需安装 NVIDIA 驱动与 NVIDIA Container Toolkit（下文 AutoBio 示例仅用 CPU，无需 GPU）。
- Python 3.12 与 Harbor 0.23.0（在步骤 2 中安装）。

1. 克隆源码仓库（已有本地仓库可跳过）：

   .. code-block:: bash

      git clone https://github.com/RLinf/RPent.git
      git clone --recurse-submodules https://github.com/mll-lab-nu/embodied-first-exam.git

2. 设置仓库路径并安装 EFE 与 Harbor（``0.23.0``）：

   .. code-block:: bash

      export RPENT_ROOT=/absolute/path/to/RPent
      export EFE_ROOT=/absolute/path/to/embodied-first-exam
      uv sync --project "$EFE_ROOT" --python 3.12
      uv pip install --python "$EFE_ROOT/.venv/bin/python" -e "$RPENT_ROOT"
      docker info
      uv run --project "$EFE_ROOT" harbor --version
      test -f "$RPENT_ROOT/robots/efe/harbor_agent.py"

   - ``uv pip install ... -e "$RPENT_ROOT"`` 仅供宿主机手动调试（``rpent --robot efe``）使用；Harbor Trial 会在任务容器内自动打包并安装 RPent。
   - 确认 ``docker info`` 正常输出、``harbor --version`` 显示 ``0.23.0``，且 ``test -f`` 返回 ``0``。

3. 选择 AutoBio 离心机示例任务并配置模型 API：

   .. code-block:: bash

      export EFE_TASK="$EFE_ROOT/tasks/autobio/autobio-insert-centrifuge-5430-i00-standard"
      export EFE_MODEL=openai/your-model
      export OPENAI_API_KEY=your-key
      export OPENAI_BASE_URL=https://api.openai.com/v1
      export MODEL_HOST=api.openai.com
      test -f "$EFE_TASK/task.toml"

   - ``MODEL_HOST``：传给 ``--allow-agent-host`` 的模型 API 纯主机名（不带 ``https://`` 或路径），用于放行容器网络访问。使用自定义端点时需同步修改 ``OPENAI_BASE_URL`` 与 ``MODEL_HOST``。
   - 可选连通性自检：在宿主机执行 ``rpent-check-llm --planner codex --model <模型> --json``\ （宿主机需设置 ``CODEX_API_KEY``；容器内若未设置 ``CODEX_*``，适配器会自动从 ``OPENAI_*`` 映射）。

运行一个任务
------------

使用默认的 Skills 工具运行单次 Harbor Trial：

.. code-block:: bash

   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" \
     --n-concurrent 1 \
     --allow-agent-host "$MODEL_HOST" \
     --yes \
     --jobs-dir "$RPENT_ROOT/logs/efe-rpent"

- ``--agent``：通过 ``PYTHONPATH`` 加载 ``RPentEfeAgent``。
- ``--n-concurrent 1``：单任务串行运行；``--yes`` 跳过确认提示。
- 首次运行会自动拉取或构建任务镜像并在容器内安装依赖，结果写入 ``$RPENT_ROOT/logs/efe-rpent``。

查看结果
--------

每次 ``harbor run`` 会在 ``--jobs-dir`` 下生成带时间戳的 Job 目录：

.. code-block:: text

   <jobs-dir>/<job_name>/
   ├── result.json              # Job 汇总
   └── <trial_name>/
       ├── result.json          # 评分结果、Token 用量、总耗时
       ├── agent/
       │   ├── setup/           # 容器内 Agent 安装日志
       │   └── rpent/           # 对应容器内 /logs/agent/rpent/
       │       ├── transcript_*.json # CLI 对话与规划器统计
       │       ├── efe-result.json   # EFE 指标与收尾结果
       │       ├── codex_*.txt       # 规划器对话记录
       │       ├── codex-home/       # 本次 Trial 独立的 CODEX_HOME
       │       └── memory/           # 本次 Trial 的记忆目录
       ├── verifier/            # 评分程序日志与 reward.json
       ├── artifacts/           # 采集的产物
       ├── config.json
       ├── lock.json
       └── trial.log

若 Trial 中途失败导致缺少 ``result.json`` 或 ``transcript_*.json``，请先排查 \
``trial.log`` 与 ``agent/rpent/run.log``。

成功判定
~~~~~~~~

- **任务成功指标**：以 ``<trial_name>/result.json`` 中的 ``verifier_result.rewards.success`` 为准（源自 ``verifier/reward.json``）。支持重试的任务还会记录首次尝试的 ``first_success``，最终得分取最后一次尝试。
- 规划器调用 ``finish(status="completed")`` 或 CLI 退出码为 ``0`` 仅代表进程正常结束，不代表任务成功。
- 规划器用完时间预算时，Trial 仍会评分：适配器把预算耗尽当作正常结束，评分程序照常检查最后一次尝试的最终状态。

批量运行 benchmark
------------------

将 ``--path`` 指向仅包含受支持 ``standard`` 模式任务的 benchmark 目录（例如 \
``$EFE_ROOT/tasks/autobio``）：

.. code-block:: bash

   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_ROOT/tasks/<benchmark>" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" \
     --n-concurrent 8 \
     --allow-agent-host "$MODEL_HOST" \
     --yes \
     --jobs-dir "$RPENT_ROOT/logs/efe-benchmark"

- 批量运行前请确认目标目录下均为 ``eai-standard/2.2`` 或 ``2.4`` 的 ``standard`` 任务，不含 ``realtime`` 或 ``privileged`` 任务。
- 对比不同配置时，请保持模型、API 端点、任务版本与超时设置一致。

常用参数
--------

通过 ``--agent-kwarg key=value`` 向适配器传参：

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - 参数
     - 默认值
     - 说明
   * - ``skills=false``
     - ``true``
     - 关闭 IK 技能，仅注册原生动作工具（``efe_submit``），见 `对照 Skills 与 Raw`_。
   * - ``reasoning_effort=low``
     - ``none``
     - Codex 推理强度：``none``、``low``、``medium``、``high`` 或 ``xhigh``。
   * - ``max_turns=N``
     - ``200``
     - 规划器的模型回复轮数上限（非 Shell 命令次数）。
   * - ``memory_source=/path``
     - 未设置
     - 上传已有 RPent 记忆目录供本次 Trial 使用（未设置时从空 ``memory/`` 开始）。
   * - ``timeout_s=N``
     - 自动读取
     - RPent Trial 时间上限（自动从 Harbor 配置或本地 ``task.toml`` 读取，并预留规划器关闭与收尾时间）。
   * - ``connect_timeout_s=N``
     - ``60``
     - 等待 EFE 仿真服务就绪的最长秒数（从 Trial 总预算中扣除）。
   * - ``agent_timeout_s=N``
     - 自动读取
     - 任务非本地目录时手动指定 Harbor 超时秒数，需与 ``--agent-timeout-sec`` 保持一致。
   * - ``sandbox=full-access``
     - ``read-only``
     - Shell 沙箱模式。保持默认 ``read-only`` 以确保仅通过 RPent 工具控制机器人，见 `运行隔离与测量方式`_。
   * - ``workdir=/path``
     - ``/app``
     - 任务容器内的工作目录。
   * - ``source_root=/path``
     - 自动探测
     - 用于构建 Wheel 的 RPent 源码根目录（从已安装包导入 ``RPentEfeAgent`` 时需显式指定）。

任务记忆
--------

- 传入 ``--agent-kwarg memory_source=/path`` 可上传已有 RPent 记忆目录；未指定时 \
  Trial 从空的 ``memory/`` 开始。
- 每次 Trial 使用独立的 ``codex-home/``，模型由 ``--model`` 指定，密钥由 Harbor \
  从环境变量转发。

对照 Skills 与 Raw
------------------

使用 ``--agent-kwarg skills=false`` 可在同一任务上对比原生动作（**Raw**）与默认 \
IK 技能（**Skills**）：

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - 类别
     - 默认（``skills=true``）
     - Raw（``skills=false``）
   * - 读取与控制
     - ``efe_spec``、``efe_observe``、``efe_status``、``efe_recover``、``view_env_state``
     - 同左
   * - 动作
     - ``move_to_pose``、``set_gripper``
     - ``efe_submit``
   * - 条件工具
     - ``efe_reset``\ （在允许重试的任务中复位重来）、``efe_end_episode``\ （在不会自动结束的任务中主动结束并提交评分）
     - 同左

- **Skills**\ （``move_to_pose(x, y, z, orientation_wxyz=None)``、``set_gripper(opening)``）：通过逆运动学（IK）计算关节目标并闭环发送短动作块，直至位置误差 ``<= 1 cm`` 且朝向误差 ``<= 0.05 rad``\ （默认最多迭代 20 次，上限 50 次）。要求动作列顺序为 ``[*spec.kinematics.joint_names, gripper]`` 且 ``tcp_frames`` 中的工具坐标系包含在 ``ik_targets`` 内；不满足时自动回退保留 ``efe_submit``，并在 ``efe-result.json`` 的 ``skills_error`` 中记录原因。
- **Raw**\ （``efe_submit``）：按 ``efe_spec`` 返回的列顺序直接提交关节目标动作块。

.. code-block:: bash

   # Raw 模式（仅注册 efe_submit 等原始工具）
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 \
     --allow-agent-host "$MODEL_HOST" \
     --agent-kwarg skills=false \
     --yes \
     --jobs-dir "$RPENT_ROOT/logs/efe-raw"

   # Skills 模式（默认：move_to_pose 与 set_gripper）
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 \
     --allow-agent-host "$MODEL_HOST" \
     --yes \
     --jobs-dir "$RPENT_ROOT/logs/efe-skills"

手动运行（调试）
----------------

不经过 Harbor 时，可直接用 ``rpent`` 连接已启动的 EFE 任务服务以调试 Prompt 或\
工具调用（该命令不会启动仿真器，也不运行评分程序）：

.. code-block:: bash

   rpent --robot efe \
     --instruction-file "$EFE_TASK/instruction.md" \
     --planner codex --model <模型> --reasoning-effort low \
     --codex-sandbox read-only \
     --memory-profile local --memory-dir ./memory/efe \
     --output-dir logs/efe-manual

- ``--instruction-file``：任务 ``instruction.md`` 路径（省略 ``--task-name`` 时默认取其父目录名）。
- ``--model``：不带 ``openai/`` 前缀的模型名称。
- ``--efe-base-url``：仿真服务地址（省略时依次探测 ``sim:8000`` 与 ``localhost:8000``）。
- ``--efe-connect-timeout-s`` / ``--efe-trial-timeout-s``：连接等待上限（默认 ``3600`` 秒）与 Trial 总截止时间。
- ``--efe-no-skills``：仅注册 Raw 工具。

运行隔离与测量方式
------------------

- **Harbor 默认沙箱（``sandbox=read-only``）**：禁止 Shell 写文件或访问网络，确保所有机器人动作均通过 RPent 工具执行并计入 ``efe-result.json`` 的 ``metrics``。
- **完全放开 Shell（``sandbox=full-access``）**：允许 Shell 脚本直接调用 ``rcb_client`` 绕过 RPent 工具与统计，结果不可与 ``read-only`` 运行直接对比。
- 使用通用 CLI 手动复现 Harbor 默认配置时，需显式传入 ``--planner codex --codex-sandbox read-only``。

复现一次评测
------------

复现对比实验时，请固定 RPent 与 EFE 仓库的 commit，保持任务目录、模型标识、\
``reasoning_effort``、时间上限与尝试次数完全一致，并归档保存 Harbor Job 目录下的 \
``result.json`` 与 ``config.json``。

下一步
------

- 选择并配置模型与规划器：:doc:`../guides/configure_planner`
- 通用 CLI 的完整参数：:doc:`../guides/cli`
- 任务记忆与探索模式：:doc:`../guides/memory`
- 接入新的机器人：:doc:`../development/add_robot`
