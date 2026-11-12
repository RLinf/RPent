BEHAVIOR
========

使用 RPent 的视觉语言规划器，在 BEHAVIOR-1K 仿真环境中完成家庭操作任务。规划器根据相机观测调用共享 Pi0.5 和机器人动作工具，动作由仿真中的 R1Pro 双臂移动机器人执行。官方评测器负责推进仿真、结束回合和评分。

本文从 BEHAVIOR-1K 挑战赛的 100 个任务中选取“制作微波爆米花”（任务 ID：40）作为运行示例，介绍环境准备、任务启动，以及视频和评分结果的查看方式。

.. _behavior-overview:

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      RPent 共享 Pi0.5 的 BEHAVIOR 配置；可选夹爪、手臂和底盘工具。

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``

   .. grid-item-card:: 任务

      100 个挑战赛任务 ID；已公布 3 个任务的对比结果。

   .. grid-item-card:: 硬件

      Linux、NVIDIA GPU，以及兼容的 OmniGibson 图形依赖和模型 CUDA 环境。

每个 Env 桥接服务连接一个评测器、运行一个回合。

任务
~~~~

任务 ID、官方名称和完整指令见 ``robots/behavior/tasks.py``。选择任务时，这三项须对应同一任务；识别全部任务 ID 不代表已验证全部任务的效果。

.. list-table::
   :header-rows: 1

   * - 范围
     - 选择方式
   * - 100 个挑战赛任务
     - ID 为 ``0..99``。规划器只指定 ``--task-id`` 时使用官方任务指令；Env 启动命令仍需显式提供同一指令。
   * - 本文的公开测试示例
     - ID ``40``，名称 ``make_microwave_popcorn``；``public_test`` 下标 ``0`` 对应实例 ``301``。使用单环境并保留官方步数预算。

.. _behavior-observation-action:

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - 头部、左腕和右腕三路相机图像，以及机器人关节、夹爪等实测状态，供规划器和动作模型判断当前情况。
   * - 动作
     - 规划器调用 VLA 或机器人控制工具执行动作，再根据新的观测决定下一步。两份模板每次 VLA 调用最多执行 32 步，可通过参数缩短。
   * - 奖励与成功判定
     - 官方评测器根据仿真中的任务状态给出成功标志和 QScore。评分结果的文件位置见“查看结果”。
   * - 任务指令
     - 默认每次调用 VLA 都使用所选任务的完整指令。规划器可以选择不同工具，但不把临时生成的子任务指令替换为 VLA 的输入指令。

安装与资源准备
--------------

按 `BEHAVIOR-1K 官方说明 <https://github.com/StanfordVL/BEHAVIOR-1K>`_ 安装仿真器和评测器，准备挑战赛资源及所选任务的公开测试实例。RPent 支持 Python 3.10–3.12；仿真器、Pi0.5 和规划器分别使用满足各自 Python、CUDA 和图形依赖要求的环境。未启用 CuRobo 时，Env 桥接服务可使用规划器环境。

模型环境需要支持 ``pi05_behavior`` 的 RLinf/OpenPI，以及权重配套的归一化统计和 tokenizer。下面使用 `公开的 PT50-CS32 权重 <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_；按所用 RLinf 版本的要求，通过 ``OPENPI_DATA_HOME`` 指定本地 tokenizer 资源。视觉语言规划器的模型、服务地址和凭据按 :doc:`../guides/configure_planner` 配置，与 VLA 权重分别选择。

将占位路径替换为本机路径，``RPENT_RUN`` 使用源码目录外的新目录。在每个终端加载以下变量并进入源码目录，也可将它们保存到本地 shell 文件后分别加载。

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

在规划器、Env 和模型环境中安装 RPent。仿真器继续使用自己的解释器；``.[behavior-vla]`` 不能替代上述 RLinf/OpenPI 环境。

.. code-block:: bash

   "$PLANNER_PYTHON" -m pip install -e '.[behavior]'
   "$ENV_PYTHON" -m pip install -e '.[behavior]'
   "$VLA_PYTHON" -m pip install -e '.[behavior-vla]'

运行一个任务
------------

先选择规划器可以调用的工具。两份模板都由视觉语言规划器选择下一步动作，并都保留 VLA；区别在于是否提供额外的感知和本体控制工具。``vla_only.yaml`` 因此不是下文实验中的纯 VLA 基线。

.. list-table::
   :header-rows: 1

   * - 配置
     - 规划器可用的机器人工具
     - 额外依赖
   * - ``vla_only.yaml``
     - 查看观测、调用 VLA、结束会话。
     - 无 SAM3 或 CuRobo 依赖，适合首次运行。
   * - ``agent.yaml``
     - 保留观测、VLA 和结束工具，增加详细机器人状态读取、分割、三维定位，以及夹爪、手臂和底盘控制；共 12 个机器人工具。
     - SAM3、CuRobo；三维定位需要 RGB-D 观测。

下面使用 ``vla_only.yaml``，四个进程各占一个终端。

**1. 启动共享 Pi0.5 服务。** 加载权重和配套归一化统计，等待服务在 ``28011`` 端口就绪。以下统计文件路径属于公开 PT50-CS32 权重；更换权重时需同时选择其配套统计。

.. code-block:: bash

   "$VLA_PYTHON" -m rpent.robots.components.pi05_vla_server \
     --embodiment behavior --model-path "$VLA_CHECKPOINT" \
     --norm-stats-path "$VLA_CHECKPOINT/assets/behavior-1k/2025-challenge-demos/norm_stats.json" \
     --host 127.0.0.1 --port 28011 --transport socket

**2. 启动 Env 桥接服务。** ``28010`` 接收 RPent 的内部 RPC 请求，``28012`` 等待官方评测器的 WebSocket 连接。

.. code-block:: bash

   "$ENV_PYTHON" -m robots.behavior.env_server \
     --task-id "$TASK_ID" --task-language "$TASK_LANGUAGE" \
     --host 127.0.0.1 --port 28010 --transport socket \
     --policy-host 127.0.0.1 --policy-port 28012

**3. 启动官方评测器。** 评测器加载场景，连接 ``28012``，发送首帧后等待动作。保持 ``--num-envs 1``，每个仿真步交换一次观测和动作，不让客户端重放缓存的动作块。

.. code-block:: bash

   "$SIM_PYTHON" -m omnigibson.eval.eval \
     --task-name "$TASK_NAME" --mode public_test --instance-indices 0 \
     --num-rollouts 1 --num-envs 1 --host 127.0.0.1 --port 28012 \
     --write-video --video-fps 120 --output-dir "$RPENT_RUN/official"

**4. 收到首帧后启动规划器。** 任务 ID 和指令须与 Env 一致，动作模型地址指向第 1 步的服务。

.. code-block:: bash

   "$PLANNER_PYTHON" -m rpent.cli.main --robot behavior --planner api \
     --model "$PLANNER_MODEL" \
     --behavior-config robots/behavior/configs/vla_only.yaml \
     --task-id "$TASK_ID" --task-language "$TASK_LANGUAGE" \
     --env-endpoint socket://127.0.0.1:28010 \
     --am-backend pi05 --am-endpoint socket://127.0.0.1:28011 \
     --output-dir "$RPENT_RUN/agent"

规划器退出只关闭客户端连接。运行结束后，手动停止独立启动的服务；下一回合使用新的 Env 服务和输出目录。

查看结果
--------

一次运行会生成两类结果：RPent 保存工具调用与观测，官方评测器保存评分与视频。以下列出关键文件；额外深度文件仅在观测提供深度时生成。

.. code-block:: text

   $RPENT_RUN/
   ├── agent/
   │   ├── states.json
   │   ├── head_rgb.png/00.png
   │   ├── left_wrist_rgb.png/00.png
   │   ├── right_wrist_rgb.png/00.png
   │   ├── proprio.npy/00.npy
   │   └── head_depth.npy/00.npy
   └── official/
       ├── json/<task_name>_<instance_id>_<rollout_id>.json
       └── videos/<task_name>_<instance_id>_<rollout_id>.mp4

``00`` 为初始观测，随后为 ``01``、``02``……；路径中的 ``head_rgb.png`` 等名称是目录。左、右腕深度按相同方式保存。该编号是保存记录的编号，一次工具调用可执行多个仿真动作。

.. list-table:: 查看哪些字段
   :header-rows: 1

   * - 文件与字段
     - 含义
   * - 官方 JSON：``success``、``q_score.final``
     - 本次任务是否完成，以及最终 QScore。评测器在一次运行结束时写出；中断时不保证已有完整评分。
   * - 官方 JSON：``instance_id``、``rollout_id``、``steps``
     - 场景实例、重复运行编号和实际评测步数。
   * - ``states.json``：``steps[i].command``、``steps[i].result``
     - 工具调用及执行反馈，包括请求、下发、确认和未确认的动作数；未知值保留为 ``null``。
   * - ``states.json``：``steps[i].state.robot_state``、``steps[i].state.artifacts.views``
     - 该记录的实测机器人状态与对应图像路径，可用于检查动作前后的变化。

视频由 ``--write-video`` 开启。示例的 ``--video-fps 120`` 控制编码播放速度，不改变仿真控制频率。查看任务效果时，以官方评分和视频为依据；保存了工具调用记录不等于任务成功。

启用更多工具
------------

运行 ``agent.yaml`` 前，在 Env 环境准备 CuRobo v0.7.8 及兼容的 PyTorch、Warp、CUDA，并在独立环境准备 SAM3 权重。按前述顺序启动新回合时，做三处调整：

1. Env 命令增加 ``--motion-backend curobo``。
2. 官方评测器命令增加 ``--env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper``，提供三维定位所需的深度和标定。
3. 先启动下面的 SAM3 服务，再将规划器配置改成 ``robots/behavior/configs/agent.yaml``，并增加 ``--sam3-endpoint socket://127.0.0.1:28013``。

.. code-block:: bash

   export SAM3_PYTHON=/path/to/sam3-env/bin/python
   export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
   "$SAM3_PYTHON" -m pip install -e '.[sam3]'
   "$SAM3_PYTHON" -m rpent.robots.components.sam3_server \
     --host 127.0.0.1 --port 28013 --transport socket

默认 RGB 观测足以用于 VLA 和二维分割。使用三维坐标时，RGB、深度和相机标定必须来自同一帧。

``agent.yaml`` 默认开放以下 12 个机器人工具，此外还可使用 RPent 通用的文件与记忆工具。

.. list-table:: 工具选择
   :header-rows: 1

   * - 工具
     - 用途与主要限制
   * - ``view_env_state``、``read_robot_state``
     - 查看已保存的图像和机器人状态，不刷新相机、不推进仿真。
   * - ``segment``、``pixel_to_robot_base``
     - 分割目标、查询像素对应的机器人基座坐标。像素按原图 ``[行, 列]`` 给出；机器人或目标移动后需重新定位。
   * - ``vla_act``
     - 预测并执行动作前缀，每个动作块后获取新观测。
   * - ``set_gripper``
     - ``val=0`` 闭合、``val=1`` 打开。插值与最终保持动作合计最多 5000 步。可额外在 allowlist 中启用 ``release`` 作为打开夹爪的简写。
   * - ``move_to``、``move_eef_delta``
     - 在当前机器人基座坐标系中移动单臂，位置单位为米。前者指定目标，后者指定小幅位移；均需要 CuRobo。
   * - ``rotate_wrist``
     - 经 CuRobo 调整末端绕基座 Z 轴的朝向，接受 −180° 至 180°，保持目标位置；运动中手指仍会扫过周围空间。
   * - ``rotate_wrist_joint``
     - 不经逆运动学，转动选定的 roll、pitch 或 yaw 关节，接受 −90° 至 90°；查看实测转角和限位截断提示。
   * - ``move_base``
     - 执行局部底盘速度命令，最后发送零速度；不保证到达指定位置。
   * - ``finish``
     - 停止规划器的决策循环并记录原因；不结束官方评测回合。

官方评测器根据环境返回的终止或截断信号结束回合并保存评分。``finish`` 只停止 RPent 规划器，不通知官方评测器结束回合，也不关闭仿真服务。提前调用它时，评测器可能仍在等待动作，不能将这种停止视为正常完成评测；需要手动清理仍在运行的服务。

手臂规划从稳定姿态开始，保持躯干、另一臂和夹爪命令。它只求解局部目标并插值关节轨迹，不检查整条路径的场景碰撞；请根据图像判断空间是否足够。工具完成需结合动作确认与实测位姿误差判断，夹爪闭合也不证明已抓住物体。

未显式指定 ``quat`` 的 ``move_to`` 和 ``move_eef_delta``，在无法保持当前朝向时允许只约束位置重试，并报告 ``orientation_relaxed``。显式指定 ``quat`` 的 ``move_to`` 和 ``rotate_wrist`` 保留朝向要求，不进行这种放松。IK 失败时不下发规划的手臂轨迹，可返回最近可达位置供调整目标。

常用参数
--------

命令行参数覆盖 ``--behavior-config`` 中的对应设置。通用规划器参数见 :doc:`../guides/cli`；以下是运行 BEHAVIOR 时常用的设置。

.. list-table::
   :header-rows: 1

   * - 设置
     - 用途与默认值
   * - ``--behavior-config``、``--tool-allowlist``
     - 选择模板或覆盖机器人工具列表；列表须保留 ``view_env_state`` 和 ``finish``。共享文件与记忆工具继续可用。
   * - ``--control-mode``、``--instruction-mode``
     - 本文使用 ``agent`` 控制，由规划器选择工具。``--instruction-mode main`` 表示每次 VLA 调用均使用完整任务指令，连接 Env 时会核对该指令。``override`` 允许为单次调用提供不同指令，需所选模型支持这种用法。
   * - ``--execution-horizon``、``--max-chunks-per-call``
     - 两份模板每块最多执行 32 步、每次工具调用最多预测 1 块；减少 horizon 可缩短执行前缀。
   * - ``--env-endpoint``、``--am-endpoint``、``--sam3-endpoint``
     - 接收 ``socket://`` 或 ``http://`` 内部服务地址。动作后端选择 ``--am-backend pi05``；``--vla-endpoint`` 仍作为动作模型地址的别名。
   * - Pi0.5 的 ``--model-path``、``--norm-stats-path``、``--repo-id``
     - 指定权重、配套归一化统计和 RLinf 数据配置的数据集标识。未传权重路径时读取 ``PI05_CHECKPOINT_PATH``；``--cuda-device`` 选择模型 GPU。
   * - Env 的 ``--simulation-timeout-s``、``--action-timeout-s``、``--chunk-timeout-s``
     - 等待观测、等待动作、整个动作块的超时分别默认 600、180、1200 秒。两份规划器模板的 ``--env-rpc-timeout-s`` 为 1230 秒。
   * - Env 的 ``--motion-backend``、``--motion-device``
     - 后端默认 ``none``；启用 ``curobo`` 时默认使用 ``cuda:0``。

任务记忆
--------

``--memory-dir`` 指定本地经验库，默认使用仓库下的 ``memory/behavior/``。规划器先查看初始观测，再读 ``MEMORY.md`` 和最多三篇相关条目；历史物体位置和抓取结果需由当前观测确认。评测期间经验库只读，不从远程同步。索引及 ``global/``、``task-family/`` 的组织方式见 :doc:`../development/memory`。

实验结果
--------

以下对比使用团队自训练的同一个 VLA：基线由 VLA 直接执行任务，另一组由 RPent 规划器调用该 VLA 和工具。每个任务、每种方法评测 10 个实例，报告官方评测器给出的平均 QScore。

.. list-table::
   :header-rows: 1

   * - 任务
     - 纯 VLA 基线
     - RPent + 同一 VLA
   * - 打开收音机
     - 0.20
     - 0.80
   * - 制作微波爆米花
     - 0.70
     - 0.90
   * - 捡垃圾
     - 0.33
     - 0.73

这些数值是平均任务得分，不是成功率，也不是前述公开 PT50-CS32 权重的测评结果。复现本表需要相同的自训练权重、实例选择、规划器与记忆配置及动作预算；公开权重示例用于运行接入流程。

.. _runtime-contracts-and-troubleshooting:

常见问题
--------

.. list-table::
   :header-rows: 1

   * - 现象
     - 检查方式
   * - 是否需要 Dashboard
     - 本文通过命令行完成启动，无需 Dashboard；当前 BEHAVIOR 没有图形化启动入口。
   * - 规划器拿不到首帧
     - 检查评测器场景加载和连接日志。官方 WebSocket 连接 ``28012``，RPent 的 ``--env-endpoint`` 使用内部 RPC 地址 ``28010``。
   * - 模型加载或输入转换失败
     - 核对 RLinf/OpenPI 是否支持 ``pi05_behavior``、权重与统计文件是否配套，以及 tokenizer 路径。模型启动日志会记录本体配置和权重路径。
   * - 动作超时或执行计数矛盾
     - 查看记录中的已下发、已确认和未知计数。不能确认执行结果时保持未知，不重放动作块。后续 VLA 块失败时会保留此前已知执行记录；不可信的保持命令不能用于继续运动。
   * - 无法查询三维坐标或手臂规划被拒绝
     - 三维查询检查同帧 RGB-D 与标定；手臂运动检查 CuRobo、稳定起始状态、目标范围和朝向要求。根据反馈调整请求，避免重复提交相同失败目标。

开发与测试
----------

配置与注册入口为 ``robots/behavior/robot_spec.py``；工具实现位于同目录的 ``primitives.py`` 和 ``tools.py``。动作模型复用 ``rpent/robots/components/pi05_vla_client.py`` 和 ``pi05_vla_server.py``。

在 Linux CPU 环境中安装测试依赖，运行 BEHAVIOR 和共享 Pi0.5 离线测试：

.. code-block:: bash

   "$PLANNER_PYTHON" -m pip install -e '.[test,behavior]'
   "$PLANNER_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
   "$PLANNER_PYTHON" -m pytest tests/unit_tests/robots/behavior \
     tests/unit_tests/rpent/robots/components/test_pi05_vla*.py -q

真实服务检查借用已启动的服务。为专用回合完成启动步骤 1–3、收到首帧后，用下列命令代替步骤 4；无需规划器模型或凭据。串行执行测试，不同时运行其他动作客户端或 ``pytest-xdist``。

.. code-block:: bash

   export RPENT_RUN_BEHAVIOR_INTEGRATION=1
   export RPENT_BEHAVIOR_TASK_ID="$TASK_ID"
   export RPENT_BEHAVIOR_TASK_LANGUAGE="$TASK_LANGUAGE"
   export RPENT_BEHAVIOR_ENV_ENDPOINT=socket://127.0.0.1:28010
   export RPENT_BEHAVIOR_VLA_ENDPOINT=socket://127.0.0.1:28011
   export RPENT_E2E_OUTPUT_DIR="$RPENT_RUN/integration"
   "$PLANNER_PYTHON" -m pytest tests/e2e_tests/behavior -v -s

未设置开关时，三项测试跳过且不连接服务。``test_components.py`` 检查观测与推理，不下发动作；``test_policy_chain.py`` 通过 Toolkit 最多执行并确认 32 步，保存新观测并调用 ``finish(blocked)``。``blocked`` 表示有限步数检查结束，不表示任务失败。

在 ``RPENT_E2E_OUTPUT_DIR`` 查看预测、提交动作、确认计数和保存图像。这些测试验证组件和动作链，不代表自主 Planner 决策或整项任务效果；它们不启动、重置、停止服务，也不通过 ``run_gpu_suite.sh`` 启动。检查结束后手动清理仍在等待动作的服务。
