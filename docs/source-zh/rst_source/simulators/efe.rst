EFE 基准评测
============

使用 RPent 在 EFE（Embodied First Exam）中运行机器人任务。EFE 通过 Harbor 启动 Agent 容器与 Simulator 容器，RPent 作为 Installed Agent 接入，通过任务自带的 ``rcb_client`` 操作机器人。该接入为实验性，状态是 EFE-compatible / non-official。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Agent

      RPent ``CodexPlanner``，只通过 RPent 工具操作 Episode。

   .. grid-item-card:: 工具层

      默认 ``move_to_xyz``、``set_gripper``；Raw 组使用 ``efe_submit``。

   .. grid-item-card:: 任务

      EFE ``standard`` 与 ``standard-realtime``，共 1150 个。

   .. grid-item-card:: 运行环境

      Docker 宿主；Harbor 0.23.0；Python 3.12。

Harbor 把一次任务运行称为 Trial：为一个任务启动一对容器、运行一次 Agent，并产生一份结果。

.. code-block:: text

   Harbor Trial
   ├── Agent 容器（main）
   │   └── RPent / CodexPlanner ── rcb_client ──▶ sim:8000
   └── Simulator 容器（sim）
       └── EFE 任务服务

1. Harbor 创建容器，并在 Simulator 容器中启动任务。
2. Adapter 打包当前 checkout 的 ``rpent/``、``robots/``、安装元数据和 ``pyproject.toml`` 声明的 package data，上传到 Agent 容器的 ``/opt/rpent-efe/source``，再用任务镜像自带的 Python 创建 venv 完成安装。容器不需要预装 ``uv``。
3. Runner 在 Agent 容器内启动现有 ``CodexPlanner``，作为该 Trial 唯一的 Agent 进程。``robots/efe`` 使用任务镜像提供的 ``rcb_client.Robot`` 连接 Simulator 服务，并把它注册为 RPent 的机器人。
4. Planner 只通过 RPent 工具操作 Episode。Harbor 管理容器、隔离和执行预算，EFE Verifier 独立评分。

RPent 不启动 Simulator，也不读取特权状态或修改任务、评分。``finish(status="completed")`` 只表示 Agent 完成了自己的工作，不表示任务成功；``solved()`` 只采信服务自己的结束原因，Verifier 仍然评分。CLI 退出码只报告 Runner 是否出错。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 模式
     - 任务数
     - 说明
   * - ``standard``
     - 1066
     - ``eai-standard/2.1``、``2.2``、``2.4``；仿真按控制步骤推进。
   * - ``standard-realtime``
     - 84
     - ``eai-standard/2.3``；Planner 思考时仿真时间继续推进。
   * - ``privileged``
     - 959
     - 暂不支持；交付整段控制，而不是逐步动作。

``standard-realtime`` 中，Planner 思考时 Simulator 的时间继续推进；服务在等待新动作期间保持上一次的目标，因此同一套工具仍然适用。

EFE 的其余 959 个任务属于 ``privileged`` 模式，我们暂时不支持：该模式让 Agent 与 Simulator 同处一个容器，可直接读取仿真状态，交付物是整段控制——可重放的轨迹文件 ``trajectory.npz``，或训练出的策略程序/模型。Verifier 在 Simulator 中重放轨迹或运行策略并评分，Agent 没有逐步观察和动作的机会。RPent 通过客户端逐步观察和动作，与该模式不匹配，因此暂未接入。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - 相机 RGB 图像（请求时含深度）、机器人本体感受，以及动作布局、关节名、相机和运动学。Agent 只看到客户端返回的内容。
   * - 动作
     - 默认组调用 ``move_to_xyz`` 移动工具坐标系、``set_gripper`` 设置夹爪开口；Raw 组用 ``efe_submit`` 提交关节目标动作块。两者都通过 ``rcb_client`` 与 Simulator 服务通信。
   * - 奖励与成功判定
     - EFE 的 Verifier 独立评分，写出 ``verifier/reward.json``；Harbor 记录为 ``result.json`` 的 ``verifier_result.rewards``，其中 ``success`` 是任务成功标记，其余字段是诊断信息。
   * - 任务指令
     - Harbor 从 EFE 任务读取指令并传给 Agent，RPent 不修改任务文本。

任务声明 Reset 预算时注册 ``efe_reset``：它放弃当前 Episode，从任务冻结的初始状态重新开始，并消耗一次预算。只有最后一个 Episode 参与评分；RPent 不会自行 Reset。任务隐藏成功信号时另注册 ``efe_end_episode``。

安装与资源准备
--------------

宿主需要 EFE 官方要求的 Docker 环境、云端模型凭据和 EFE checkout。按 EFE 的说明准备宿主和任务；Simulator 的资源要求由任务决定。首次运行仍需下载容器和任务资源。保留 EFE 原始任务、Simulator、Verifier 和资源限制。宿主的 ``PYTHONPATH`` 必须能导入本仓库中的 Adapter。

.. code-block:: bash

   export RPENT_ROOT=/absolute/path/to/RPent
   export EFE_ROOT=/absolute/path/to/embodied-first-exam
   export EFE_TASK="$EFE_ROOT/tasks/<family>/<task>"
   export EFE_MODEL=openai/your-model
   export OPENAI_API_KEY=your-key

不要将凭据写入仓库。Adapter 只转发模型相关环境变量和显式的 Harbor ``--agent-env`` 设置。它把 ``OPENAI_API_KEY``、``OPENAI_BASE_URL`` 映射到 RPent 的 ``CODEX_API_KEY``、``CODEX_BASE_URL``；已有 ``CODEX_*`` 设置优先。Adapter 不转发宿主的 ``CODEX_HOME``，Runner 会在 Agent 容器内为每个 Trial 新建该目录。

运行一个任务
------------

模型由 Harbor 的 ``--model`` 指定，凭据从当前终端的环境变量转发。以下命令运行一个 Trial：

.. code-block:: bash

   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
     --agent-kwarg source_root="$RPENT_ROOT" \
     --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
     --agent-kwarg reasoning_effort=none \
     --jobs-dir "$RPENT_ROOT/logs/efe-rpent"

``RPentEfeAgent`` 从当前 checkout 安装 RPent。``source_root`` 默认指向 Adapter 所在 checkout 的仓库根目录；若 Adapter 从已安装的 Wheel 导入，必须显式传入。源码包不包含宿主 ``.git``、测试、凭据文件和运行日志。``robots/`` 是现有 RPent distribution 所需的包，不会启动任何 Robot Runtime。

该自定义 Agent 不声明模型 Endpoint，Harbor 无法推断推理主机，需传入 ``--allow-agent-host <model-host>``。显式名单会替换默认放行，因此还要放行安装所需的包主机：``--allow-agent-host pypi.org --allow-agent-host files.pythonhosted.org``。首次容器安装运行 ``pip``，Harbor 默认 Setup 超时为 360 秒；冷安装较慢时加上 ``--agent-setup-timeout-multiplier 4``。

查看结果
--------

``--jobs-dir`` 是 Harbor 的 Job 根目录；默认每次 ``harbor run`` 在下面新建一个以时间命名的 Job 目录。

.. code-block:: text

   <jobs-dir>/<job_name>/
   ├── result.json              # Job 汇总
   └── <trial_name>/
       ├── result.json          # Verifier 结果、Token、耗时
       ├── agent/
       │   ├── setup/           # Agent 安装日志
       │   └── rpent/
       │       ├── session.json # Agent 执行指标、错误、各 Session 统计
       │       ├── codex*.txt   # Planner 对话
       │       ├── codex-home/  # 本次 Trial 的 CODEX_HOME
       │       └── memory/      # 本次 Trial 的 Memory 根目录
       ├── verifier/            # Verifier 日志和 reward.json
       ├── artifacts/           # 收集的产物
       ├── config.json
       ├── lock.json
       └── trial.log

容器内的 ``/logs/agent/rpent/`` 对应宿主上的 ``<trial_name>/agent/rpent/``。EFE 的 Verifier 写出 ``verifier/reward.json``，Harbor 将它记录为 ``result.json`` 的 ``verifier_result.rewards``；其中 ``success`` 是任务成功标记，其余字段是诊断信息。Agent 执行指标在 ``agent/rpent/session.json`` 的 ``stats`` 和各 Session 的 ``metrics``。Trial 未完成时可能没有 ``result.json`` 或 ``session.json``，先看 ``trial.log`` 和 ``agent/rpent/run.log``。

常用参数
--------

- ``--agent-kwarg source_root=/path``：指定要安装的 RPent checkout；默认取 Adapter 所在仓库。
- ``--agent-kwarg workdir=/path``：任务的工作目录；默认 ``/app``。
- ``--agent-kwarg timeout_s=N``：RPent 的内层时间上限，必须小于任务的 Agent 超时。两者相等时 Harbor 先超时，``session.json`` 会丢失。
- ``--agent-kwarg max_turns=N``：Planner 的模型响应次数上限，不等于 Shell 命令数。
- ``--agent-kwarg sandbox=full-access``：允许 Planner 用 Shell 直接编写脚本操作 Simulator。默认 ``read-only`` 让 Shell 无法访问 Simulator，只有 RPent 工具会作用于 Episode。

Harbor 的任务预算始终是外层硬限制；RPent 在 Trial 结束时写出 ``session.json``，所以要留出余量。

任务记忆与探索模式
------------------

- ``--agent-kwarg memory_source=/path``：上传已有 RPent 记忆语料，替换本次 Trial 的空记忆目录。
- ``--agent-kwarg explore_sessions=N``：在同一 Episode 上运行多个独立 Session。
- 每个 Trial 从空的 ``memory/`` 和独立的 ``codex-home/`` 开始；模型与凭据只来自显式环境变量。

对照 Skills 与 Raw
------------------

两组使用同一个 Runner、模型、预算和任务，只改变动作工具。

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - 类别
     - 默认（``skills=true``）
     - Raw（``skills=false``）
   * - 读取与控制
     - ``efe_spec``、``efe_observe``、``efe_status``、``efe_recover``
     - 同左
   * - 动作
     - ``move_to_xyz``、``set_gripper``
     - ``efe_submit``
   * - 条件工具
     - 任务有 Reset 预算时注册 ``efe_reset``；隐藏成功信号时注册 ``efe_end_episode``
     - 同左

默认组用两个高层技能代替直接提交关节目标：

- ``move_to_xyz(x, y, z)`` 把世界坐标目标解成 IK，提交短动作块，并反复读取本体感受，直到工具坐标系进入 1 cm 容差。它保持当前末端姿态，Planner 因此可以在笛卡尔空间推理。
- ``set_gripper(opening)`` 在保持当前臂位姿的同时设置夹爪开口。

Raw 组直接使用服务接口：``efe_spec`` 返回每个动作向量的列含义和顺序（即动作布局），Planner 再按该布局用 ``efe_submit`` 发送关节目标动作块。

动作布局定义一行动作里每个数的含义。技能只处理一种布局：前几列是各关节的目标角度，顺序与 ``spec.kinematics.joint_names`` 一致；最后一列是夹爪开度。例如：

.. code-block:: text

   joint_names = ["joint1", "joint2", "joint3"]
   layout      = ["q_joint1", "q_joint2", "q_joint3", "gripper"]

前 3 列去掉 ``q_`` 前缀后与 ``joint_names`` 同序，最后一列是 ``gripper``，技能可以直接把 IK 结果写入这些列。布局换序、缺列或多列时，技能无法确定每个关节对应哪一列，因此不注册：工具集保留 ``efe_submit``，把原因写入 ``session.json`` 的 ``skills_error``，Prompt 只列出当前 Session 实际注册的工具。

.. code-block:: bash

   # Raw：只注册 efe_submit 等原始工具
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
     --agent-kwarg source_root="$RPENT_ROOT" \
     --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
     --agent-kwarg skills=false \
     --jobs-dir "$RPENT_ROOT/logs/efe-raw"

   # Skills：默认组，注册 move_to_xyz 和 set_gripper
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
     --agent-kwarg source_root="$RPENT_ROOT" \
     --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
     --jobs-dir "$RPENT_ROOT/logs/efe-skills"

两组必须固定模型、端点、推理强度、任务版本、时间上限和尝试次数。单次 Trial 不构成对照。

运行隔离与测量模式
------------------

- Runner 默认使用 Codex 的 ``read-only`` Sandbox，只有 RPent 工具会作用于 Episode。``full-access`` 测量 Agent 的自由行为，Shell 动作不计入 Agent 执行指标。
- 通用 ``rpent --robot efe`` 不传 Sandbox 设置，始终以 ``full-access`` 运行；正式评测和对照使用 Harbor Adapter。
