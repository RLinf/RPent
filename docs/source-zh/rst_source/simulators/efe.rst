Embodied First Exam
===================

使用 RPent 运行一个仿真机器人任务，再检查机器人是否完成了目标。本指南以不需要 GPU 的 AutoBio 实验室任务为例。你需要一台可运行 Docker 的 Linux 主机，以及模型 API 的访问权限。

概览
----

EFE（Embodied First Exam）是一组机器人任务，提供预先确定的场景和评分规则。RPent 根据相机图像和机器人观测，让语言模型选择下一步动作。Harbor 负责启动任务、运行 RPent、保存结果，并在结束后清理容器。Harbor 把一次任务运行称为 **Trial**。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      使用 RPent 动作工具或任务原生控制指令。

   .. grid-item-card:: 规划器

      CodexPlanner 连接模型 API，选择下一次工具调用。

   .. grid-item-card:: 任务

      EFE 的 standard 和 standard-realtime 任务。

   .. grid-item-card:: 硬件

      Linux 与 Docker。GPU 要求取决于任务；AutoBio 使用 CPU。

Harbor 把 RPent 和仿真器放在不同容器中运行。RPent 通过 EFE 提供的客户端库 ``rcb_client`` 向仿真器发送动作。EFE 的评分程序（Verifier）独立检查任务结果，不以模型的最终回答作为评分依据。

任务
~~~~

在 `EFE 任务浏览 <https://www.embodied-first-exam.ai/tasks/>`_ 中选择任务。任务文件位于 `EFE 仓库 <https://github.com/mll-lab-nu/embodied-first-exam>`_ 的 ``tasks/<benchmark>/<task>`` 中，benchmark 表示一组相关任务。下表保留本集成文档中的任务清单数量，不代表 RPent 已评测的任务数量；上游还可能新增任务。

.. list-table::
   :header-rows: 1

   * - 模式
     - 任务数
     - 行为
   * - ``standard``
     - 1066
     - 支持。仿真器收到动作后推进仿真。
   * - ``standard-realtime``
     - 84
     - 支持。模型思考期间，仿真器仍持续运行。
   * - ``privileged``
     - 959
     - 暂不支持。Agent 提交完整轨迹或策略，由评分程序评测。

在 ``standard-realtime`` 中，等待新动作时，机器人继续执行最近一次指令指定的关节位置或其他控制目标。场景在等待期间仍会变化，因此模型响应时间会影响任务结果。

观测与动作
~~~~~~~~~~

RPent 提供两种控制方式。默认使用 **Skills**：模型通过 ``move_to_xyz`` 指定工具末端要到达的位置，通过 ``set_gripper`` 指定夹爪开度，RPent 再把请求转换为关节控制指令。使用 **Raw** 时，模型通过 ``efe_submit`` 直接提交任务原生的动作数值。首次运行保留默认配置即可；适用范围与对照命令见 `对照 Skills 与 Raw`_。如果机器人的控制方式超出 Skills 的支持范围，RPent 会自动改用 Raw。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 项目
     - 说明
   * - 观测
     - 相机 RGB 图像、可选深度图、关节位置、工具末端位姿、夹爪开度，以及可用控制指令的说明。
   * - 动作
     - Skills 指定末端移动和夹爪开度；Raw 按任务规定的顺序提交原生动作数值。
   * - 奖励与成功判定
     - EFE 的评分程序检查机器人的实际结果。读取 ``verifier/reward.json`` 或 Trial 的 ``result.json``，具体见 `查看结果`_。
   * - 任务指令
     - Harbor 将任务原始指令传给 RPent。

部分任务允许重新尝试。这类任务提供 ``efe_reset`` 工具：放弃当前 Episode（一次尝试），回到相同的初始场景，并消耗一次任务允许的重置次数。只有最后一个 Episode 参与评分，RPent 不会自动重置。

另一些任务不向 Agent 返回是否成功的信号。对这类任务，RPent 提供 ``efe_end_episode`` 工具，让模型主动结束动作执行，再由评分程序检查结束时的状态。调用这个工具不代表成功；目标尚未完成就结束，可能被判定为失败。

安装与资源准备
--------------

以下命令均在 Linux 宿主机终端执行。需要 git、`uv <https://docs.astral.sh/uv/getting-started/installation/>`_ 和带有 Compose v2、BuildKit 的 Docker Engine。先按 EFE 的 `宿主要求与安装指南 <https://github.com/mll-lab-nu/embodied-first-exam#quick-start>`_ 准备环境。GPU 任务还需要 NVIDIA 驱动和 NVIDIA Container Toolkit。本集成面向 Harbor 0.23.0 和 Python 3.12。

1. 获取两个仓库的源码。已有源码时无需重新克隆，在下一步使用它们的绝对路径。

   .. code-block:: bash

      git clone https://github.com/RLinf/RPent.git
      git clone --recurse-submodules https://github.com/mll-lab-nu/embodied-first-exam.git

2. 设置两个仓库的源码路径，并安装 EFE 项目依赖（其中包含 Harbor）。将前两个路径替换为实际路径。``RPENT_ROOT`` 指 RPent 仓库根目录，``EFE_ROOT`` 指 Embodied First Exam 仓库根目录。

   .. code-block:: bash

      export RPENT_ROOT=/absolute/path/to/RPent
      export EFE_ROOT=/absolute/path/to/embodied-first-exam
      uv sync --project "$EFE_ROOT" --python 3.12
      docker info
      uv run --project "$EFE_ROOT" harbor --version
      test -f "$RPENT_ROOT/robots/efe/harbor_agent.py"

   Docker 应显示服务正在运行，Harbor 应显示版本 0.23.0，文件检查应正常退出。首次 Trial 会构建或拉取任务镜像并下载资源。

3. 选择示例任务并配置模型。这个任务要求机器人把试管放入离心机转子。将 ``your-model`` 替换为 API 提供方和 CodexPlanner 支持的模型标识。

   .. code-block:: bash

      export EFE_TASK="$EFE_ROOT/tasks/autobio/autobio-insert-centrifuge-5430-i00-standard"
      export EFE_MODEL=openai/your-model
      export OPENAI_API_KEY=your-key
      export OPENAI_BASE_URL=https://api.openai.com/v1
      export MODEL_HOST=api.openai.com
      test -f "$EFE_TASK/task.toml"

   ``MODEL_HOST`` 指定允许任务容器访问的模型 API 主机，供运行命令中的 ``--allow-agent-host`` 使用。使用兼容的自定义 API 时，同时替换 ``OPENAI_BASE_URL`` 和 ``MODEL_HOST``。``MODEL_HOST`` 只填写主机名，不含 ``https://`` 或路径。例如，API 地址为 ``https://api.example.com/v1`` 时，填写 ``api.example.com``。不要把 API key 写入版本控制。已有的 ``CODEX_API_KEY``、``CODEX_BASE_URL`` 会覆盖对应的 ``OPENAI_*`` 变量；若要使用上述配置，先取消旧的 ``CODEX_*`` 设置。

.. _trial:

运行一个任务
------------

继续使用同一个终端，以保留上面设置的变量。以下命令用默认 Skills 工具运行一个 Trial。Harbor 会自动在任务容器中安装 RPent，无需手动进入容器安装。

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

``--agent`` 指定 Harbor 加载 RPent 的入口，保留示例值即可。``PYTHONPATH`` 让 Harbor 能从 RPent 源码目录找到它。``--n-concurrent 1`` 表示同时只运行一个任务，``--yes`` 跳过 Harbor 的确认提示。首次运行需要安装依赖，准备阶段可能较长。执行和评分结束后，到 ``logs/efe-rpent`` 查看结果；失败时先检查 ``trial.log`` 和 ``agent/rpent/run.log``。

查看结果
--------

在宿主机上查看 ``$RPENT_ROOT/logs/efe-rpent``。
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

容器内的 ``/logs/agent/rpent/`` 对应宿主机上的 ``<trial_name>/agent/rpent/``。EFE 的 Verifier 写出 ``verifier/reward.json``，Harbor 将它记录为 ``result.json`` 的 ``verifier_result.rewards``。Agent 执行指标在 ``agent/rpent/session.json`` 的 ``stats`` 和各 Session 的 ``metrics``。Trial 未完成时可能没有 ``result.json`` 或 ``session.json``，先看 ``trial.log`` 和 ``agent/rpent/run.log``。

判断任务是否成功，读取 Trial 的 ``verifier_result.rewards.success``。任务允许重试时，``first_success`` 表示首次尝试的结果。模型返回 ``finish(status="completed")`` 只表示 Agent 停止工作；CLI 正常退出只表示执行代码没有报告错误，两者都不代表评分成功。

批量运行 benchmark
------------------

要批量运行 benchmark，请将下方 ``<benchmark>`` 替换为所选 benchmark 的名称。先确认目录中只有受支持的 standard 或 standard-realtime 任务，不要混入 privileged 任务。

``--path`` 指向 benchmark 目录，其中每个子目录是一项任务，合起来就是要运行的任务集。``-d embodied-first-exam@all`` 表示 EFE 注册的全部任务。例如，EFE 文档中的 ``harbor run -d embodied-first-exam@all -p tasks/roboarc`` 只运行 RoboARC 任务集。

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

每个任务对应一个 Trial 和一份 ``result.json``；``--n-concurrent`` 默认 4。先跑通一个 Trial，再批量运行，并固定模型、端点、任务版本和超时后再比较结果。

常用参数
--------

- ``--agent-kwarg source_root=/path``：指定 RPent 源码目录；默认使用 RPentEfeAgent 所在的仓库。从已安装的 Wheel 导入时，需要显式指定该路径。
- ``--agent-kwarg workdir=/path``：任务的工作目录；默认 ``/app``。
- ``--agent-kwarg timeout_s=N``：RPent 的内层时间上限，必须小于任务的 Agent 超时。两者相等时 Harbor 先超时，``session.json`` 会丢失。
- ``--agent-kwarg max_turns=N``：Planner 的模型响应次数上限，不等于 Shell 命令数。
- ``--agent-kwarg sandbox=full-access``：允许 Planner 用 Shell 直接编写脚本操作 Simulator。默认 ``read-only`` 让 Shell 无法访问 Simulator，只有 RPent 工具会作用于 Episode。

任务记忆与探索模式
------------------

- ``--agent-kwarg memory_source=/path``：上传已有 RPent 记忆目录供本次 Trial 使用；未指定时，Trial 从空的 ``memory/`` 开始。
- ``--agent-kwarg explore_sessions=N``：在同一次任务尝试中最多运行 ``N`` 个规划会话；任务结束或时间耗尽时提前停止。后续会话可读取前序会话的工作笔记。
- 每个 Trial 使用独立的 ``codex-home/``；模型由 ``--model`` 指定，认证信息来自转发的环境变量。

对照 Skills 与 Raw
------------------

对同一个任务分别使用 Skills 与 Raw，比较两种工具配置的效果。模型和执行预算保持一致，只改变动作工具。

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

- ``move_to_xyz(x, y, z)`` 将末端移动到世界坐标系中的目标位置：用逆运动学（IK）计算关节位置、提交短动作块，再读回末端位置进行修正。位置误差不超过 1 cm 时停止；也可能因迭代上限、求解失败或任务结束而提前停止。它保持当前姿态，Planner 因此可以在笛卡尔空间推理。
- ``set_gripper(opening)`` 设置夹爪开合。

Raw 组直接使用服务接口：``efe_spec`` 返回每个动作向量的列含义和顺序（即动作布局），Planner 再按该布局用 ``efe_submit`` 发送关节目标动作块。

动作布局定义一行动作里每个数的含义。技能只支持一种布局：前几列是各关节的目标角度，顺序与 ``spec.kinematics.joint_names`` 一致；最后一列是夹爪开度。布局换序、缺列或多列时技能不注册：工具集保留 ``efe_submit``，把原因写入 ``session.json`` 的 ``skills_error``，Prompt 只列出当前 Session 实际注册的工具。

.. code-block:: bash

   # Raw：只注册 efe_submit 等原始工具
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 \
     --allow-agent-host "$MODEL_HOST" \
     --agent-kwarg skills=false \
     --yes \
     --jobs-dir "$RPENT_ROOT/logs/efe-raw"

   # Skills：默认组，注册 move_to_xyz 和 set_gripper
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 \
     --allow-agent-host "$MODEL_HOST" \
     --yes \
     --jobs-dir "$RPENT_ROOT/logs/efe-skills"

两组必须固定模型、端点、推理强度、任务版本、时间上限和尝试次数。单次 Trial 不构成对照。

运行隔离与测量模式
------------------

- Harbor Runner 默认使用 Codex 的 ``read-only`` 沙箱。它限制 Shell，不限制 RPent 工具：Shell 不能写文件或访问网络；机器人动作通过 RPent 工具执行。
- 使用 ``full-access`` 时，Agent 可以用 Shell 写脚本直接调用 ``rcb_client``，绕过 RPent 工具。这些调用不经过 RPent 的客户端，``session.json`` 的 ``metrics`` 不会统计，因此这种运行不能与走工具的运行对比。
- 通用 ``rpent --robot efe`` 没有 Sandbox 选项，固定以 ``full-access`` 运行，用于手动和交互调试。正式评测与对照请通过 Harbor 运行。

开发者集成说明
--------------

Harbor 通过 ``--agent`` 加载 RPent 的 ``RPentEfeAgent``。该入口会打包 RPent 源码、安装元数据和 ``pyproject.toml`` 声明的包数据，再用任务镜像自带的 Python 安装到 ``/opt/rpent-efe/venv``；容器不需要 ``uv``。打包内容不含 Git 数据、测试、API key 文件和运行日志。``robots/`` 是 RPent 安装包的一部分，不会因此启动其他机器人服务。

Runner 指 RPent 在容器内执行任务的代码，负责启动 CodexPlanner、通过 ``rcb_client.Robot`` 与仿真器交互，并保存 ``session.json``。Harbor 管理容器隔离、执行预算和清理。``solved()`` 只采信服务返回的结束原因。RPent 不启动仿真器、不读取特权状态，也不修改 EFE 的评分规则。

RPentEfeAgent 会转发模型环境变量和显式的 Harbor ``--agent-env`` 设置。若尚未设置 ``CODEX_API_KEY``、``CODEX_BASE_URL``，则分别从 ``OPENAI_API_KEY``、``OPENAI_BASE_URL`` 映射。宿主机的 ``CODEX_HOME`` 不会转发，每个 Trial 都会新建独立目录。

.. code-block:: text

   Harbor Trial
   ├── Agent 容器（main）
   │   └── RPent / CodexPlanner ── rcb_client ──▶ sim:8000
   └── 仿真器容器（sim）
       └── EFE 任务服务

支持的服务版本为 standard 模式的 ``eai-standard/2.1``、``2.2``、``2.4``，以及 standard-realtime 模式的 ``2.3``。privileged 任务则要求提交可重放的 ``trajectory.npz``，或策略程序、模型。评分程序通过重放轨迹或运行策略来评测，与本集成逐步观测和执行动作的方式不匹配。
