Embodied First Exam
===================

本页用 AutoBio 的一个实验室任务，演示如何通过 Harbor 用 RPent 运行 EFE\
（Embodied First Exam）任务，以及如何读取评分结果。示例任务只用 CPU；\
宿主机需要能运行 Docker，并配置好模型 API。

「运行一个任务」给出单次评测的完整流程；之后的小节依次说明结果文件、\
批量运行、工具配置、手动调试和常见故障。

概览
----

EFE 是一组带固定场景和评分规则的机器人任务。RPent 根据相机图像和机器人\
观测，让语言模型逐步选择动作。Harbor 负责启动任务容器、运行 RPent、\
保存结果，并在结束后清理容器。Harbor 把一次任务运行称为 **Trial**，\
本页沿用这一说法。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      使用 RPent 的运动工具（Skills），或任务原生的动作数值（Raw）。

   .. grid-item-card:: 规划器

      Harbor 路径固定使用 Codex 规划器。它连接模型 API，并选择下一步工具调用。

   .. grid-item-card:: 任务

      EFE 的 standard 模式任务。

   .. grid-item-card:: 硬件

      Linux 与 Docker。AutoBio 使用 CPU；其他任务的 GPU 要求见 EFE 文档。

RPent 和仿真器运行在两个容器里。RPent 通过 EFE 提供的客户端库 \
``rcb_client`` 向仿真器发送动作；EFE 的评分程序（Verifier）独立检查任务\
结果，不以模型的最终回答作为评分依据。

任务
~~~~

在 `EFE 任务浏览 <https://www.embodied-first-exam.ai/tasks/>`_ 中选择任务。\
任务文件位于 `EFE 仓库 <https://github.com/mll-lab-nu/embodied-first-exam>`_ \
的 ``tasks/<benchmark>/<task>`` 中，其中 benchmark 是一组相关任务。表中的\
数量是集成时上游给出的清单，不代表 RPent 评测过的任务数；上游可能继续新增。

.. list-table::
   :header-rows: 1

   * - 模式
     - 任务数
     - 行为
   * - ``standard``
     - 1066
     - 服务协议为 ``eai-standard/2.2`` 或 ``2.4`` 时可用。仿真器收到动作后推进仿真。
   * - ``standard-realtime``
     - 84
     - 暂不支持。这类任务使用 ``eai-standard/2.3`` 或 ``eai-standard-realtime/1.0``，本集成只处理 ``2.2`` 和 ``2.4``。
   * - ``privileged``
     - 959
     - 暂不支持。Agent 提交完整轨迹或策略，由评分程序评测。

本集成只处理服务协议为 ``eai-standard/2.2`` 或 ``2.4`` 的 standard 模式任务。\
选择 benchmark 前，先查看任务 ``task.toml`` 中的 ``metadata.protocol``；\
使用其他协议的任务会在连接阶段直接报错，不会拖到运行中途才失败。

观测与动作
~~~~~~~~~~

RPent 有两种控制方式，由 ``skills`` 参数切换。默认使用 **Skills**：模型调用 \
``move_to_pose`` 指定工具末端位姿，调用 ``set_gripper`` 指定夹爪开度。\
设为 **Raw** 后，模型通过 ``efe_submit`` 直接提交任务原生的动作数值；\
不受 Skills 支持的动作布局会保留 Raw 工具。两种方式的工具清单和对照命令见 \
`对照 Skills 与 Raw`_。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 项目
     - 说明
   * - 观测
     - 相机 RGB 图像、可选深度图、相机标定信息、关节位置、工具末端位姿、夹爪开度，以及可用控制指令的说明。
   * - 动作
     - Skills 指定末端位置、可选朝向和夹爪开度；Raw 按任务规定的顺序提交原生动作数值。
   * - 奖励与成功判定
     - EFE 的评分程序按任务规则检查机器人的实际结果。读取 ``verifier/reward.json`` 或 Trial 的 ``result.json``，见 `查看结果`_。
   * - 任务指令
     - Harbor 把任务原始指令传给 RPent。

服务提供标定信息时，``efe_observe`` 在 ``camera_calibration`` 中返回标定\
数据，在 ``frame_extrinsics_cv`` 中返回逐帧的世界到相机变换。需要把像素\
位置与深度换算到世界坐标时，使用这两项数据。

带重置预算的任务允许最多 N 次尝试。这类任务会注册 ``efe_reset``：模型\
调用后放弃当前这次尝试，回到冻结的初始状态，并消耗一次重置；只有最后一次\
尝试参与评分。规则见 EFE `standard 模式协议 <https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/standard_mode.md>`_。

另一类任务不向 Agent 返回成功信号。RPent 为此注册 ``efe_end_episode``，\
让模型主动结束动作执行，再由评分程序检查结束时的状态。调用它只表示动作\
结束，不代表成功；目标尚未完成就结束，很可能被判为失败。

安装与资源准备
--------------

以下命令在 Linux 宿主机终端执行。需要 git、`uv <https://docs.astral.sh/uv/getting-started/installation/>`_，以及带 Compose \
v2 和 BuildKit 的 Docker Engine。先按 EFE 的 `宿主机要求与安装指南 <https://github.com/mll-lab-nu/embodied-first-exam#quick-start>`_ 准备环境。\
GPU 任务还需要 NVIDIA 驱动和 NVIDIA Container Toolkit。本集成适配 Harbor \
0.23.0 与 Python 3.12。

1. 克隆两个仓库。已有源码时跳过这一步，下一步直接填绝对路径。

   .. code-block:: bash

      git clone https://github.com/RLinf/RPent.git
      git clone --recurse-submodules https://github.com/mll-lab-nu/embodied-first-exam.git

2. 设置两个仓库的路径，并安装 EFE 的依赖（其中包含 Harbor）。把前两行换成\
   实际路径：``RPENT_ROOT`` 指向 RPent 仓库根目录，``EFE_ROOT`` 指向 EFE \
   仓库根目录。

   .. code-block:: bash

      export RPENT_ROOT=/absolute/path/to/RPent
      export EFE_ROOT=/absolute/path/to/embodied-first-exam
      uv sync --project "$EFE_ROOT" --python 3.12
      docker info
      uv run --project "$EFE_ROOT" harbor --version
      test -f "$RPENT_ROOT/robots/efe/harbor_agent.py"

   ``docker info`` 应显示 Docker 服务正常，``harbor --version`` 应输出 \
   0.23.0，``test -f`` 应无报错退出。首次 Trial 还要构建或拉取任务镜像并\
   下载资源，准备阶段会更长。

3. 选择示例任务并配置模型。该任务要求机器人把试管放入离心机转子。把 \
   ``your-model`` 换成 API 提供方和 Codex 规划器都支持的模型标识。

   .. code-block:: bash

      export EFE_TASK="$EFE_ROOT/tasks/autobio/autobio-insert-centrifuge-5430-i00-standard"
      export EFE_MODEL=openai/your-model
      export OPENAI_API_KEY=your-key
      export OPENAI_BASE_URL=https://api.openai.com/v1
      export MODEL_HOST=api.openai.com
      test -f "$EFE_TASK/task.toml"

   ``MODEL_HOST`` 是任务容器允许访问的模型 API 主机，运行命令会把它传给 \
   ``--allow-agent-host``。使用兼容的自定义 API 时，``OPENAI_BASE_URL`` 和 \
   ``MODEL_HOST`` 要一起改。``MODEL_HOST`` 只写主机名，不带 ``https://`` \
   和路径：API 地址为 ``https://api.example.com/v1`` 时，填 \
   ``api.example.com``。

   API Key 不要写入版本控制。已设置的 ``CODEX_API_KEY``、``CODEX_BASE_URL`` \
   会覆盖对应的 ``OPENAI_*`` 变量；要使用上面的配置，先取消这些 ``CODEX_*`` \
   设置。

   如果想先确认密钥和端点可用，可以在宿主机运行 ``rpent-check-llm --planner codex --model <模型> --json``。\
   宿主机上的 Codex 规划器读取 ``CODEX_API_KEY``，不会自动使用 ``OPENAI_API_KEY``；\
   容器内则由适配器完成映射。

运行一个任务
------------

继续在同一个终端里执行，上面设置的变量才有效。下面的命令用默认的 Skills \
配置运行一次 Trial。RPent 由 Harbor 自动安装到任务容器，不需要手动进入\
容器操作。

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

``--agent`` 指定 Harbor 加载的 RPent 入口，保持原样即可；``PYTHONPATH`` \
让 Harbor 从 RPent 源码目录找到它。``--n-concurrent 1`` 表示同时只运行\
一个任务，``--yes`` 跳过 Harbor 的确认提示。首次运行需要安装依赖，准备\
阶段可能较长。运行和评分结束后，到 ``logs/efe-rpent`` 下查看结果；失败时\
先看 ``trial.log`` 和 ``agent/rpent/run.log``。

查看结果
--------

结果写在宿主机的 ``$RPENT_ROOT/logs/efe-rpent`` 下。``--jobs-dir`` 是 \
Harbor 的 Job 根目录，每次 ``harbor run`` 默认在下面新建一个按时间命名的 \
Job 目录。

.. code-block:: text

   <jobs-dir>/<job_name>/
   ├── result.json              # Job 汇总
   └── <trial_name>/
       ├── result.json          # 评分程序结果、Token、耗时
       ├── agent/
       │   ├── setup/           # Agent 安装日志
       │   └── rpent/
       │       ├── transcript_*.json # 通用 CLI 对话与规划器统计
       │       ├── efe-result.json   # EFE 指标与收尾结果
       │       ├── codex_*.txt       # 规划器对话
       │       ├── codex-home/  # 本次 Trial 的 CODEX_HOME
       │       └── memory/      # 本次 Trial 的记忆根目录
       ├── verifier/            # 评分程序日志和 reward.json
       ├── artifacts/           # 收集的产物
       ├── config.json
       ├── lock.json
       └── trial.log

容器内的 ``/logs/agent/rpent/`` 对应宿主机的 \
``<trial_name>/agent/rpent/``。EFE 的评分程序把分数写入 \
``verifier/reward.json``，Harbor 再把它记到 ``result.json`` 的 \
``verifier_result.rewards``。规划器统计在 ``agent/rpent/transcript_*.json``，\
EFE 指标在 ``agent/rpent/efe-result.json``。Trial 没有跑完时，这些文件可能\
缺失，先查 ``trial.log`` 和 ``agent/rpent/run.log``。

成功判定
~~~~~~~~

任务是否成功，只看 Trial 的 ``result.json`` 中的 \
``verifier_result.rewards.success``。支持重试的任务还会记录 \
``first_success``，表示首次尝试的结果；最终得分以最后一个尝试为准。

模型调用 ``finish(status="completed")`` 只表示它停止工作，CLI 退出码为 0 \
只表示执行代码没有报错，二者都不是评分结果。

批量运行 benchmark
------------------

把下面的 ``<benchmark>`` 换成要运行的 benchmark 名称。``--path`` 指向 \
benchmark 目录，其中每个子目录是一项任务，合起来就是要运行的任务集；\
运行前先确认目录里只有受支持的 standard 模式任务，不要混入 realtime 或 \
privileged 任务。``-d embodied-first-exam@all`` 表示 EFE 注册的全部任务，\
例如 EFE 文档中的 ``harbor run -d embodied-first-exam@all -p tasks/roboarc`` \
只运行 RoboARC 任务集。

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

每个任务对应一次 Trial 和一份 ``result.json``。``--n-concurrent`` 默认 4。\
先跑通单个 Trial，再批量运行；比较结果前，固定模型、端点、任务版本和\
时间上限。

常用参数
--------

以下参数都通过 ``--agent-kwarg key=value`` 传给适配器。

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - 参数
     - 默认值
     - 说明
   * - ``source_root=/path``
     - RPentEfeAgent 所在仓库
     - 构建 Wheel 用的 RPent 源码目录。从已安装的 Wheel 导入 Agent 时必须显式指定。
   * - ``workdir=/path``
     - ``/app``
     - 任务容器内的工作目录。
   * - ``timeout_s=N``
     - 自动读取
     - RPent Trial 的时间上限。默认从 Harbor Trial 配置读取，本地任务还会读 ``task.toml``；无论哪种来源，都会给规划器关闭和 CLI 收尾留出时间。
   * - ``connect_timeout_s=N``
     - 60
     - 等待 EFE 服务的最长秒数。这段时间从 RPent Trial 预算里扣除，剩余部分才分配给规划器。
   * - ``agent_timeout_s=N``
     - 自动读取
     - 手动指定 Harbor 实际使用的超时，用于任务配置无法作为本地目录读取的情况，取值要与 Harbor 一致。Harbor 0.23 在内部解析这个超时，但不传给 Installed Agent，所以适配层自己读 Trial 配置和 ``task.toml``。
   * - ``max_turns=N``
     - 200
     - 规划器的模型回复次数上限，不是 Shell 命令数。
   * - ``sandbox=full-access``
     - ``read-only``
     - 设为 ``full-access`` 后，规划器可以用 Shell 脚本直接操作仿真器；默认的 ``read-only`` 让 Shell 无法访问服务，只有 RPent 工具作用于当前这次尝试。
   * - ``skills=false``
     - ``true``
     - 只注册 ``efe_submit`` 等原始工具，见 `对照 Skills 与 Raw`_。
   * - ``memory_source=/path``
     - 未设置
     - 上传已有记忆目录供本次 Trial 使用；不设置时，Trial 从空的 ``memory/`` 开始。
   * - ``reasoning_effort=low``
     - ``none``
     - 传给 Codex 规划器的推理强度，可选 ``none``、``low``、``medium``、``high``、``xhigh``。

任务记忆
--------

- ``--agent-kwarg memory_source=/path``：上传已有记忆目录供本次 Trial 使用；\
  不指定时，Trial 从空的 ``memory/`` 开始。
- 每次 Trial 使用独立的 ``codex-home/``；模型由 ``--model`` 指定，认证信息\
  来自转发的环境变量。

对照 Skills 与 Raw
------------------

对同一个任务分别使用 Skills 与 Raw，比较两种工具配置的效果。模型和执行\
预算保持一致，只改变动作工具。

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
     - 任务有重置预算时注册 ``efe_reset``；隐藏成功信号时注册 ``efe_end_episode``
     - 同左

默认组用两个高层技能代替直接提交关节目标：

- ``move_to_pose(x, y, z, orientation_wxyz=None)`` 把工具末端移动到世界\
  坐标系中的目标位置，并可选指定朝向。四元数顺序为 ``w, x, y, z``；\
  省略朝向时保持当前朝向。技能用逆运动学（IK）计算关节位置，提交短动作\
  块，再读回位姿修正。位置误差不超过 1 cm、朝向误差不超过 0.05 rad \
  时停止；也可能因迭代次数用尽（默认 20 次，上限 50 次）、求解失败或任务\
  结束而提前停止。
- ``set_gripper(opening)`` 设置夹爪开度。

Raw 组直接调用服务接口。``efe_spec`` 返回动作向量每一列的含义和顺序，\
也就是动作布局；规划器据此用 ``efe_submit`` 发送关节目标动作块。

技能只支持一种布局：前面各列是关节目标，顺序与 \
``spec.kinematics.joint_names`` 一致，最后一列是夹爪开度。任务的 \
``tcp_frames`` 中，还要有一个坐标系出现在 ``ik_targets`` 里。本体感知\
必须包含该坐标系的位姿列和所有关节列。布局换序、缺列、多列，或缺少可用\
的工具坐标系时，Skills 不会注册。工具集保留 ``efe_submit``，原因写入 \
``efe-result.json`` 的 ``skills_error``；Prompt 也只列出当前 Session \
实际注册的工具。

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

   # Skills：默认组，注册 move_to_pose 和 set_gripper
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 \
     --allow-agent-host "$MODEL_HOST" \
     --yes \
     --jobs-dir "$RPENT_ROOT/logs/efe-skills"

两组必须固定模型、端点、推理强度、任务版本、时间上限和尝试次数。单次 \
Trial 不构成对照。

手动运行（调试）
----------------

不经过 Harbor 时，可以直接调用通用 CLI 调试工具调用或 Prompt。这个入口\
只连接已经运行的任务服务，不启动仿真器，也不负责评分。

.. code-block:: bash

   rpent --robot efe \
     --instruction-file "$EFE_TASK/instruction.md" \
     --planner codex --model <模型> --reasoning-effort low \
     --codex-sandbox read-only \
     --memory-profile local --memory-dir ./memory/efe \
     --output-dir logs/efe-manual

``--instruction-file`` 必填，指向任务的说明文件。``--task-name`` 不填时，\
任务名取该文件所在目录的名字；Harbor 路径会把说明文件放到别的目录，因此\
显式传入任务目录名。``--efe-base-url`` 指定服务地址，不填时先探测 \
``sim:8000``，再探测 ``localhost:8000``。``--efe-connect-timeout-s`` 默认 \
3600 秒；``--efe-trial-timeout-s`` 给整个 Trial 设截止时间，包含最初的\
连接等待。``--efe-no-skills`` 只注册原始工具。``--memory-repo`` 不受支持，\
需要使用记忆时改用 ``--memory-profile local`` 配合 ``--memory-dir``。

Harbor 路径传给 ``rpent`` 的 ``--model`` 不带 ``openai/`` 前缀，适配器会\
先去掉它；手动运行时同样只写模型名。

运行隔离与测量方式
------------------

- Harbor 默认给 Codex 使用 ``read-only`` 沙箱。它限制的是 Shell，不是 \
  RPent 工具：Shell 不能写文件或访问网络，机器人动作只能通过 RPent 工具\
  执行。
- 设为 ``full-access`` 后，Agent 可以用 Shell 写脚本直接调用 \
  ``rcb_client``，绕过 RPent 工具。这些调用不经过 RPent 的客户端，\
  ``efe-result.json`` 的 ``metrics`` 统计不到，因此不能与走工具的运行对比。
- 通用 CLI 的 ``--codex-sandbox`` 默认 ``full-access``；手动复现 Harbor 的\
  默认行为时，加上 ``--codex-sandbox read-only``。
- ``rpent`` 默认使用 ``--planner api``，没有 Codex 的 Shell 沙箱。手动运行 \
  ``rpent --robot efe`` 时，要同时加上 ``--planner codex`` 才与 Harbor \
  一致。

复现一次评测
------------

复现时固定两侧的代码版本：RPent 的 commit，以及提供任务定义与镜像的 EFE \
仓库 commit。同时记录任务目录名、模型标识、推理强度、时间上限和尝试次数，\
并保留 Harbor Job 目录中的 ``result.json`` 与 ``config.json``。

比较不同配置时，这些设置必须完全一致，并且要说明结果的汇总方式。单次 \
Trial 的差异不足以区分配置效果。

常见问题
--------

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - 现象
     - 原因
     - 处理
   * - 连接阶段报协议不支持
     - 任务使用 ``2.1``、``2.3`` 或 realtime 协议
     - 换用 ``2.2`` 或 ``2.4`` 的 standard 任务；其他版本需要单独的适配层。
   * - Trial 启动后很快失败，``run.log`` 报连接超时
     - 任务服务没起来，或端口不对
     - 查看 ``trial.log`` 和仿真器容器日志；必要时调大 ``connect_timeout_s``。
   * - 模型请求返回 401 或无法连接
     - 密钥、端点或主机白名单不对
     - 检查 ``OPENAI_API_KEY``、``OPENAI_BASE_URL``、``MODEL_HOST``，并确认没有残留的 ``CODEX_*`` 覆盖。
   * - 只有 ``efe_submit``，没有 Skills
     - 动作布局或运动学不满足要求
     - 查看 ``efe-result.json`` 的 ``skills_error``，按 `对照 Skills 与 Raw`_ 核对布局和工具坐标系。
   * - Harbor 在预算用尽时终止 Trial
     - 时间分配不够
     - 调大 ``--agent-timeout-sec``，并用 ``agent_timeout_s`` 把同一数值传给适配器。
   * - ``transcript_*.json`` 或 ``result.json`` 缺失
     - Trial 没有跑完
     - 先看 ``trial.log`` 和 ``agent/rpent/run.log``；``finish`` 和退出码都不是评分结果。

开发者集成说明
--------------

Harbor 通过 ``--agent`` 加载 ``RPentEfeAgent``。这个入口从 RPent 源码构建 \
Wheel 并上传；Wheel 按 ``pyproject.toml`` 的包发现和包数据规则生成，不\
包含宿主机的 Git 数据、测试、缓存、API Key 文件和运行日志。Harbor 再用\
任务镜像自带的 Python 把 Wheel 安装到 ``/opt/rpent-efe/venv``，容器里不\
需要 ``uv``。``robots/`` 是 RPent 安装包的一部分，安装它不会启动其他\
机器人服务。

适配器在容器内运行已安装的 ``rpent``，参数固定包含 ``--robot efe`` 和 \
``--planner codex``，并显式传入 ``--memory-profile local``、\
``--memory-dir``、``--task-name`` 等。EFE 的 ``RobotSpec`` 把通用 CLI 接到 \
``EfeEnvClient`` 和 ``EfeToolkit``：CLI 生成 ``transcript_*.json``，\
``finalize_run`` 钩子写出 ``efe-result.json``。适配器把 Trial 截止时间传给 \
``EfeEnvClient``，并为规划器关闭和 CLI 收尾留出时间；transcript 记录规划器\
预算耗尽时，按 Agent 正常退出处理，交给 Harbor 评分。``solved()`` 只采信\
服务返回的结束原因。RPent 不启动仿真器、不读取特权状态，也不修改 EFE 的\
评分规则。

RPentEfeAgent 会转发模型环境变量，以及 Harbor ``--agent-env`` 显式设置的变量。\
未设置 ``CODEX_API_KEY``、``CODEX_BASE_URL`` 时，分别从 \
``OPENAI_API_KEY``、``OPENAI_BASE_URL`` 映射。宿主机的 ``CODEX_HOME`` 不\
转发，每次 Trial 新建独立目录。

.. code-block:: text

   Harbor Trial
   ├── Agent 容器（main）
   │   └── RPent / Codex 规划器 ── rcb_client ──▶ sim:8000
   └── 仿真器容器（sim）
       └── EFE 任务服务

支持的服务版本是 standard 模式的 ``eai-standard/2.2`` 和 ``2.4``。``2.1`` \
和 ``2.3`` 在 ``/spec`` 中发布 ``control_dt`` 而不是 ``control_hz``，\
``rcb_client`` 也不同（``get_obs(include_depth=...)``、``Result.depth``、\
没有 ``wait_ready``）。``EfeEnvClient`` 遇到这些版本会直接报错，而不是等到\
缺少 spec 字段才失败；接入它们需要单独的适配层，``2.3`` 及其 \
``standard-realtime`` 任务同样如此。

``privileged`` 任务要求提交可重放的 ``trajectory.npz``，或策略程序、模型；\
评分程序通过重放轨迹或运行策略来评测。这种评测方式与 RPent 逐步观测、\
逐步执行动作的流程不匹配。

下一步
------

- 选择并配置模型与规划器：:doc:`../guides/configure_planner`
- 通用 CLI 的完整参数：:doc:`../guides/cli`
- 任务记忆与探索模式：:doc:`../guides/memory`
- 接入新的机器人：:doc:`../development/add_robot`
