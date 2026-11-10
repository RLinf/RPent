BEHAVIOR
========

用 RPent 的规划器在官方 BEHAVIOR-1K 仿真器中运行一条 R1Pro 轨迹。下面以用微波炉做爆米花为例。规划器选择工具，官方评测器（evaluator）通过 WebSocket 接收动作、推进仿真，并负责回合重置、终止和评分。

.. _behavior-overview:

概览
----

每个 Env 桥接服务对应一个 R1Pro 环境和一个官方评测会话。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      RLinf Pi0.5 BEHAVIOR；可选夹爪、手臂和底盘动作工具。

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``

   .. grid-item-card:: 任务

      100 个挑战赛任务，ID 范围为 ``0..99``。

   .. grid-item-card:: 硬件

      Linux、NVIDIA GPU，以及兼容的 OmniGibson 图形依赖和模型 CUDA 环境。

任务
~~~~

任务 ID、名称和完整任务指令由 ``robots/behavior/tasks.py`` 提供。切换任务时，这三项须指向同一任务。

.. list-table::
   :header-rows: 1

   * - 范围
     - 任务选择
     - 运行条件
   * - 100 个挑战赛任务
     - 选择 ID（``0..99``）、官方名称和完整任务指令。
     - 每个 Env 连接一个 R1Pro 环境、运行一条轨迹。任务成功率需要另行测评。
   * - 公开测试示例
     - 任务 ID 为 ``40``，官方名称为 ``make_microwave_popcorn``，``public_test`` 下标 ``0`` 对应场景实例 ``301``。
     - 使用 ``--num-envs 1`` 并保留官方步数预算。

.. _behavior-observation-action:

观测与动作
~~~~~~~~~~

规划器通过工具获取观测或执行动作，任务分数由官方评测器给出。

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - 头部、左腕、右腕三路 RGB 图像。Pi0.5 适配器将公开的 61 维机器人状态映射到模型布局，也接受已经编码的 256 维状态。输入其他长度会报错。
   * - 动作
     - 环境接收工具生成的 23 通道原生动作。BEHAVIOR 的 Pi0.5 预设使用 BF16 和 5 步去噪，每次预测 32 步。
   * - 奖励与成功判定
     - 读取官方评测器 JSON 中的 success 和 QScore。规划器的 ``finish`` 单独记录其判断。
   * - 任务指令
     - 主任务模式每次推理都使用配置的完整指令。

安装与资源准备
--------------

按 `BEHAVIOR-1K 官方说明 <https://github.com/StanfordVL/BEHAVIOR-1K>`_ 安装仿真器和评测器，准备挑战赛资源及所选任务的公开测试实例。RPent 支持 Python 3.10–3.12。为仿真器、Pi0.5 模型和规划器分别准备满足各自 Python、CUDA 和图形依赖要求的环境。

模型环境需要支持 ``pi05_behavior`` 的 RLinf/OpenPI、与训练相匹配的归一化统计及 tokenizer。示例可使用 `公开的 PT50-CS32 权重 <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_；按所用 RLinf 版本的要求用 ``OPENPI_DATA_HOME`` 指定本地 tokenizer 资源。按 :doc:`../guides/configure_planner` 配置可接收图像的规划器模型；其服务地址和凭据放在规划器配置中。

将下面的占位路径替换为本机实际路径。``RPENT_RUN`` 指向源码目录外的一个新目录。每个终端都需要加载同一组变量，并在源码目录里运行命令。

.. code-block:: bash

   export RPENT_REPO=/path/to/RPent
   export RPENT_RUN=/path/to/behavior-runs/run-name
   export VLA_PYTHON=/path/to/vla-env/bin/python
   export ENV_PYTHON=/path/to/env-service-env/bin/python
   export SIM_PYTHON=/path/to/simulator-env/bin/python
   export PLANNER_PYTHON=/path/to/planner-env/bin/python
   export VLA_CHECKPOINT=/path/to/checkpoint
   export PLANNER_MODEL=provider:vision-model-name
   export TASK_ID=40
   export TASK_NAME=make_microwave_popcorn
   export TASK_LANGUAGE='In the kitchen, take the popcorn bag from the countertop, put it into the microwave, and heat it until the popcorn is cooked so the cooked popcorn ends up inside the bag.'
   cd "$RPENT_REPO"

分别在规划器、Env 桥接服务和 Pi0.5 模型的环境中安装 RPent 依赖。``.[behavior-vla]`` 还需要上述模型运行环境，仿真器继续使用自己的解释器。

.. code-block:: bash

   "$PLANNER_PYTHON" -m pip install -e '.[behavior]'
   "$ENV_PYTHON" -m pip install -e '.[behavior]'
   "$VLA_PYTHON" -m pip install -e '.[behavior-vla]'

运行一个任务
------------

首次运行可选 ``vla_only.yaml``。此配置开放观测、VLA 动作和结束三个工具，由视觉语言规划器决定何时调用；它不依赖 SAM3 或 CuRobo。长期运行的进程各占一个终端。

**1. 启动共享 Pi0.5 服务。** 该服务加载 ``--model-path`` 指定的权重，等待动作预测请求。公开 PT50-CS32 的 ``comet`` 状态布局将两个夹爪排在双臂之后。如果权重训练时把夹爪紧跟在各自手臂之后，请使用含 ``openpi_pytorch`` 的 RLinf 环境，并在同一命令后追加 ``--behavior-state-layout abs_joint``。两种布局都需使用配套的归一化统计。要复现首次预测，请用相同的 ``--seed`` 重新启动服务。

.. code-block:: bash

   "$VLA_PYTHON" -m rpent.robots.components.pi05_vla_server \
     --embodiment behavior --model-path "$VLA_CHECKPOINT" \
     --host 127.0.0.1 --port 28011 --transport socket

**2. 模型就绪后，启动 Env 桥接服务。** 桥接服务在 ``28010`` 提供 RPent 内部 RPC，在 ``28012`` 监听评测器的策略 WebSocket，回合重置时通知同一个 Pi0.5 服务。

.. code-block:: bash

   "$ENV_PYTHON" -m robots.behavior.env_server \
     --task-id "$TASK_ID" --task-language "$TASK_LANGUAGE" \
     --am-endpoint socket://127.0.0.1:28011 \
     --host 127.0.0.1 --port 28010 --transport socket \
     --policy-host 127.0.0.1 --policy-port 28012

**3. 启动官方评测器。** 评测器连接端口 ``28012``，发送首帧后等待原生动作。用 ``--num-envs 1`` 跑一次轨迹，保留任务的官方步数预算。桥接协议每步交换一次观测和动作，评测器客户端不应重放缓存的动作块。

.. code-block:: bash

   "$SIM_PYTHON" -m omnigibson.eval.eval \
     --task-name "$TASK_NAME" --mode public_test --instance-indices 0 \
     --num-rollouts 1 --num-envs 1 --host 127.0.0.1 --port 28012 \
     --write-video --video-fps 120 --output-dir "$RPENT_RUN/official"

默认的 RGB 观测可用于 VLA 推理和二维分割。若要调用 ``pixel_to_robot_base`` 等三维工具，请在上述评测器命令中追加 ``--env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper``。坐标投影需要与所选 RGB 帧对应的深度图和相机标定。

**4. 收到首帧后，启动规划器。** ``--model`` 选择视觉语言规划器。它与 Env 服务共享任务 ID、完整指令和 Pi0.5 端点。运行配置和模型权重分别通过 ``--behavior-config`` 和模型服务的 ``--model-path`` 选择。

.. code-block:: bash

   "$PLANNER_PYTHON" -m rpent.cli.main --robot behavior --planner api \
     --model "$PLANNER_MODEL" \
     --behavior-config robots/behavior/configs/vla_only.yaml \
     --task-id "$TASK_ID" --task-language "$TASK_LANGUAGE" \
     --env-endpoint socket://127.0.0.1:28010 \
     --am-backend pi05 --am-endpoint socket://127.0.0.1:28011 \
     --output-dir "$RPENT_RUN/agent"

如需使用分割和手臂运动工具，请换用 ``robots/behavior/configs/agent.yaml``。先在独立环境准备 SAM3 权重并启动共享分割服务：

.. code-block:: bash

   export SAM3_PYTHON=/path/to/sam3-env/bin/python
   export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
   "$SAM3_PYTHON" -m pip install -e '.[sam3]'
   "$SAM3_PYTHON" -m rpent.robots.components.sam3_server \
     --host 127.0.0.1 --port 28013 --transport socket

在 Env 环境安装 CuRobo v0.7.8 和与之兼容的 PyTorch、Warp、CUDA，向第 2 步追加 ``--motion-backend curobo``；第 4 步改用 ``agent.yaml`` 并追加 ``--sam3-endpoint socket://127.0.0.1:28013``，任务信息和动作模型端点沿用同一组设置。

查看结果
--------

``$RPENT_RUN/agent/`` 保存规划器对话、观测和工具结果；``$RPENT_RUN/official/`` 保存官方 JSON 和视频。任务成功与 QScore 以评测器 JSON 为准。示例中视频以 120 FPS 编码，该值只决定视频播放速度。模型服务日志记录所选本体配置和权重路径。

基线由团队自训练的 VLA 直接执行任务，RPent 使用同一 VLA 作为动作模型。每个任务评测 10 个实例，表中每项得分为 BEHAVIOR 官方评测器对这 10 个实例给出的 QScore 的平均值。

.. list-table::
   :header-rows: 1

   * - 任务
     - 纯 VLA（基线）
     - RPent + VLA
   * - 打开收音机
     - 0.20
     - 0.80
   * - 制作微波爆米花
     - 0.70
     - 0.90
   * - 捡垃圾
     - 0.33
     - 0.73

常用参数
--------

规划器的命令行参数优先于 ``--behavior-config`` 指定的 YAML。以下模板值适用于随包提供的 ``agent.yaml`` 和 ``vla_only.yaml``。

.. list-table:: 规划器
   :header-rows: 1

   * - 参数
     - 用途与约束
   * - ``--behavior-config``、``--task-id``、``--task-language``
     - 选择工具配置及任务 ID。规划器可按任务 ID 读取官方指令；Env 命令要求显式提供完整指令。
   * - ``--control-mode``、``--instruction-mode``
     - 两份模板均使用 ``agent`` 控制，每次 VLA 调用后交回规划器。在 ``main`` 模式下，规划器连接 Env 桥接服务时核对完整任务指令。``override`` 模式允许工具调用单独指定指令。控制模式还可选择 ``scripted``。
   * - ``--execution-horizon``、``--max-chunks-per-call``
     - 两份模板每次使用 32 步、一次工具调用最多预测 1 个动作块。减少执行步数可只提交预测的前几步。
   * - ``--tool-allowlist``
     - 选择开放的机器人工具，必须包含 ``view_env_state`` 和 ``finish``。RPent 共享的文件与记忆工具继续可用。
   * - ``--env-endpoint``、``--am-backend``、``--am-endpoint``、``--sam3-endpoint``
     - 使用内部 RPC 地址。动作后端为 ``pi05``，原 ``--vla-endpoint`` 仍可使用。只有启用 ``segment`` 时需要 SAM3 端点。
   * - ``--env-rpc-timeout-s``
     - 模板设为 1230 秒，给 Env 的整个动作块超时和 RPC 开销留出时间。
   * - ``--model``、``--memory-dir``、``--output-dir``
     - 选择规划器模型、本地记忆目录和本次运行的输出目录。输出目录必填。

.. list-table:: 模型服务与 Env 桥接服务
   :header-rows: 1

   * - 所属进程与参数
     - 用途与取值
   * - Pi0.5：``--embodiment``、``--model-path``
     - 选择 ``behavior`` 和本地权重目录。未给出路径时，服务读取 ``PI05_CHECKPOINT_PATH``。
   * - Pi0.5：``--norm-stats-path``、``--repo-id``
     - ``--norm-stats-path`` 可指向 ``norm_stats.json`` 或其所在目录，优先于 ``--repo-id``。仅指定数据集 ID 时，服务会在权重根目录及其 ``assets/`` 子目录下查找该 ID 对应的统计文件。公开 PT50-CS32 权重默认使用 ``assets/behavior-1k/2025-challenge-demos/norm_stats.json``。找不到所需统计时，模型服务会在启动时报错。
   * - Pi0.5：``--behavior-state-layout``、``--seed``、``--cuda-device``
     - 默认状态布局为 ``comet``，也可指定 ``abs_joint``。BEHAVIOR 的模型随机种子默认是 ``0``；可指定该进程可用的 GPU。
   * - Env：``--am-endpoint``、``--task-id``、``--task-language``
     - 指向同一 Pi0.5 服务，并与评测器的任务身份一致。
   * - ``--host``、``--port``、``--transport``；Env 的 ``--policy-host``、``--policy-port``
     - 前三项配置内部 RPC，示例显式选用 ``socket``；后两项配置供官方评测器连接的策略 WebSocket。
   * - Env：``--simulation-timeout-s``、``--action-timeout-s``、``--chunk-timeout-s``
     - 等待仿真观测、等待下一条动作以及整个动作块的超时，默认分别为 600、180 和 1200 秒。
   * - Env：``--motion-backend``、``--motion-device``
     - 后端默认 ``none``；启用 ``curobo`` 后可指定设备，默认 ``cuda:0``。

任务记忆
--------

规划器通过共享的 ``list_dir`` 和 ``read_text_file`` 读取本地经验库。默认使用 ``memory/behavior/``，可由 ``--memory-dir`` 指定其他目录。先查看初始观测，再读 ``MEMORY.md`` 和最多三篇相关条目；物体位置与抓取结果仍需用当前观测确认。评测期间经验库只读，不从远程同步。索引、``global/`` 和 ``task-family/`` 的组织方式见 :doc:`../development/memory`。

工具与运行约定
--------------

``agent.yaml`` 开放下列 12 个机器人工具。读取保存帧和查询坐标均不刷新相机，需要新观测时请查看动作执行后保存的帧。

.. list-table::
   :header-rows: 1

   * - 工具
     - 用途
     - 推进仿真
   * - ``view_env_state``
     - 查看保存的三视角、实测状态和执行反馈。
     - 否
   * - ``read_robot_state``
     - 读取所选记录的关节实测值和速度。
     - 否
   * - ``segment``
     - 用物体文字描述或原图中的 ``[行, 列]`` 点请求 SAM3 分割。
     - 否
   * - ``pixel_to_robot_base``
     - 利用同帧深度和标定，把像素投影到该帧的机器人基座坐标系。
     - 否
   * - ``vla_act``
     - 用任务指令请求动作并执行指定前缀，每个动作块后更新观测。
     - 是
   * - ``set_gripper``
     - 控制一个夹爪，``val=0`` 闭合，``val=1`` 打开。
     - 是
   * - ``move_to``
     - 经 CuRobo 逆运动学把单臂移到附近目标位姿。
     - 是
   * - ``move_eef_delta``
     - 沿 ``move_to`` 的路径按 xyz 位移小幅移动单臂。
     - 是
   * - ``rotate_wrist``
     - 用 ``move_to`` 路径让末端执行器绕机器人基座 Z 轴旋转。
     - 是
   * - ``rotate_wrist_joint``
     - 不经逆运动学，按相对角度转动选定的腕部关节。
     - 是
   * - ``move_base``
     - 执行有限步数的局部底盘速度命令，最后发送零速度。
     - 是
   * - ``finish``
     - 结束规划器会话并保存其判断。
     - 否

可在 ``tool_allowlist`` 中额外启用 ``release`` 来打开选定夹爪。工具返回 ``ToolResult(data, images)``。动作后保存的结果带有三视角、``state`` 和 ``log.result``；``step`` 是保存记录的编号。已下发、已确认和未确认的动作分别统计。不确定是否执行的动作保留为未知，超时后不会自动重放动作块。

几何坐标只适用于所查询的保存帧，移动后请重新定位。``move_to``、``move_eef_delta`` 和 ``rotate_wrist`` 需要 Env 端的 CuRobo。它们从稳定姿态开始规划，保持躯干、另一臂和双夹爪命令，底盘 0–2 通道置零，仅调整左臂 7–13 或右臂 15–21 通道。工具按动作确认和实测位姿误差判断完成，无需末端速度降为零。求解和插值不检查整条路径中的场景碰撞；保持目标朝向失败时，会只约束位置重试并报告 ``orientation_relaxed`` 和实测朝向误差。IK 失败时不会下发规划的手臂轨迹，结果可给出最近可达位置。

``rotate_wrist`` 接受 −180° 至 180° 的相对偏航角，保持末端目标位置；中途手指仍会扫过周围空间。该工具适用于空手调整朝向。``rotate_wrist_joint`` 以线性关节轨迹转动单个 roll、pitch 或 yaw 关节，接受 −90° 至 90°，保持其余关节和双夹爪命令。手的位置会绕腕关节划过一小段弧线，结果报告实测转角和关节限位截断情况。``move_base`` 提供局部速度命令，没有可靠里程计或到达保证。夹爪命令完成也需结合观测判断是否抓住物体。

若无观测，先检查 Env、VLA 和评测器日志，确认官方 ``ws://`` 地址连接策略端口，``--env-endpoint`` 指向内部 RPC 端口。规划器退出时只关闭客户端连接。请显式停止独立启动的服务，并为下一条轨迹启动新的 Env 服务、选择新的输出目录。此接入每次只处理一个评测器会话，服务须从命令行启动；Dashboard 暂不支持启动此环境。

开发与测试
----------

从规划器到仿真的主要调用关系如下。表中机器人文件均相对 ``robots/behavior/``。

.. list-table::
   :header-rows: 1

   * - 入口
     - 职责
   * - ``tasks.py``、``robot_spec.py``、``configs/``
     - 官方任务 ID、名称和指令；通过机器人包入口 ``__init__.py`` 注册，配置并连接外部服务；``vla_only.yaml`` 与 ``agent.yaml`` 工具模板。
   * - ``prompts/evaluate.py``、``prompt_bundle.py``
     - 通用决策和工具选择规则；按已启用的工具组装提示词。任务目标在运行时传入。
   * - ``toolkit.py``、``tools.py``
     - 工具注册与资源绑定；观测解码（``snapshot_observation``、``extract_public_proprio``）、SAM3 掩码与叠加图、坐标查询和 ``finish``。
   * - ``primitives.py``、``motion.py``、``assets/r1pro/``
     - VLA、手臂、夹爪和底盘工具生成 23 通道动作，按带类型标注的 ``@tool`` 定义约束参数。Env 中的 ``R1ProMotionModel`` 读取随包 URDF 和 CuRobo 配置，``R1ProMotionPlanner`` 执行 IK 并在速度上限内插值七个手臂关节，返回路点供工具执行。
   * - ``pi05.py``、仓库根目录下的 ``rpent/robots/components/{pi05_vla_client,pi05_vla_server,am_runtime}.py``
     - R1Pro 三路图像、公开状态和动作的转换；共享 Pi0.5 客户端、模型预设、服务和标准动作模型选择。
   * - ``env_client.py``、``env_server.py``、``eval/evaluator.py``
     - 工具使用内部 RPC 访问 Env；``WebsocketPolicyServer`` 处理官方 WebSocket 消息，``OfficialEvaluatorPolicy`` 实现 ``reset()`` 和 ``act()``，``EvaluatorStepBridge`` 用下一帧观测确认上一动作的执行。评测器负责推进仿真。

在 Linux CPU 环境中，将 ``PLANNER_PYTHON`` 指向该环境的解释器。在源码目录安装测试依赖和 CPU 版 PyTorch 后，运行以下离线测试，检查机器人工具、服务接口和共享 Pi0.5：

.. code-block:: bash

   "$PLANNER_PYTHON" -m pip install -e '.[test,behavior]'
   "$PLANNER_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
   "$PLANNER_PYTHON" -m pytest tests/unit_tests/robots/behavior \
     tests/unit_tests/rpent/robots/components/test_pi05_vla_client.py \
     tests/unit_tests/rpent/robots/components/test_pi05_vla_server_contracts.py -q

要检查真实服务，请为一次专用轨迹依次完成启动流程的第 1–3 步。收到首帧后，用下列测试代替第 4 步的规划器。它们借用已运行的 Env 和 VLA，无需规划器模型或凭据；请串行运行，避免同时启动其他动作客户端或使用 ``pytest-xdist``。

.. code-block:: bash

   export RPENT_RUN_BEHAVIOR_INTEGRATION=1
   export RPENT_BEHAVIOR_TASK_ID="$TASK_ID"
   export RPENT_BEHAVIOR_TASK_LANGUAGE="$TASK_LANGUAGE"
   export RPENT_BEHAVIOR_ENV_ENDPOINT=socket://127.0.0.1:28010
   export RPENT_BEHAVIOR_VLA_ENDPOINT=socket://127.0.0.1:28011
   export RPENT_E2E_OUTPUT_DIR="$RPENT_RUN/integration"
   "$PLANNER_PYTHON" -m pytest tests/e2e_tests/behavior -v -s

未设置 ``RPENT_RUN_BEHAVIOR_INTEGRATION=1`` 时，三项测试均跳过，不读取端点或建立连接。仅运行 ``tests/e2e_tests/behavior/test_components.py`` 可检查三路 RGB、61 维公开状态和形状为 ``[T, 23]`` 的模型输出；它增加一次推理请求，不下发动作。``test_policy_chain.py`` 通过 ``BehaviorToolkit`` 直接调用 VLA，把预测前缀交给 Env，最多执行并确认 32 步，然后检查三路新观测并调用 ``finish(blocked)``。只有每个下发的动作都由后续观测确认，动作链检查才会通过。

在 ``RPENT_E2E_OUTPUT_DIR`` 查看模型原始预测、实际提交的动作、确认计数和 Toolkit 保存的状态与图像。``blocked`` 表示本次有限步数的检查到此结束；CLI、自主视觉语言决策和整项任务成功率需另行验证。测试不负责启动、重置或停止服务，也不由 ``run_gpu_suite.sh`` 启动。测试结束后，评测器可能仍在等待动作，按上文说明清理服务。
