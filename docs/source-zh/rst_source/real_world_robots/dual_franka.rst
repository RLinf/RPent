双臂 Franka
=============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka-deploy.jpg
   :alt: 安装在同一工作台旁的两台 Franka 机械臂与相机
   :figclass: rpent-robot-figure
   :align: center

   用于真机实验的双臂 Franka 系统。

RPent 可以通过 RLinf ``RealWorldEnv`` worker 控制双节点双臂 Franka 系统。 任务视频见 :doc:`双臂 Franka 演示 <../real_world_demos/franka>`。

安装
----

请先按照 `libfranka 官方快速安装指南 <https://docs.ros.org/en/humble/p/libfranka/__README.html#quick-install>`_ 安装 libfranka 0.19.0。

克隆 RPent 并安装 Python 依赖。若已有仓库，进入仓库后执行 ``uv sync``：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   uv sync --extra franka --extra sam3

该命令将 RLinf ``release/v0.4``、``rpent-openpi``、SAM3 以及 Franka 的相机、夹爪和遥操作依赖安装到 ``.venv``。其中的 ``franky-control`` wheel 已包含 libfranka 0.19.0。

标定（Calibration）
----------------------------------------

``base_camera`` 和 ``d455_camera`` 都需要标定到 **右臂基座**。可使用
``calibration_tools/``，也可继续使用 ROS
`easy_handeye <https://github.com/IFL-CAMP/easy_handeye>`_，最终加载格式均为 YAML。
腕部相机在默认配置中仅用于观测；若使用腕部 RGBD 相机进行投影，需要按下文
配置手眼标定和投影视图。

本工具支持 D435，已在 D435 配置下验证；其他型号和流配置尚未验证，不能保证
仅修改参数即可使用。零畸变系数可直接使用；非零系数仅支持
``distortion.brown_conrady``。不支持的非零畸变模型会拒绝采样，并显示模型和
系数，不会自动转换或忽略畸变。

**1. 准备环境**

在相机节点激活已有的 RLinf Franka 运行环境，然后进入 RPent 仓库根目录，
检查所需接口：

.. code-block:: bash

   source /absolute/path/to/franka-env/bin/activate
   cd /absolute/path/to/RPent
   PYTHONPATH=calibration_tools python -c "import numpy, scipy, yaml, pyrealsense2; from common import check_opencv; check_opencv()"

检查包含 ChArUco、PnP、``calibrateHandEye`` 及所需方法常量。
若现有环境不满足依赖，可另建标定环境，避免改动策略运行环境：

.. code-block:: bash

   python3 -m venv .venv-calibration
   source .venv-calibration/bin/activate
   pip install -r calibration_tools/requirements.txt pyrealsense2

在右臂控制节点使用匹配机器人固件的 libfranka 开发库编译状态读取程序：

.. code-block:: bash

   cmake -S calibration_tools -B calibration_tools/build \
     -DCMAKE_PREFIX_PATH=/absolute/path/to/libfranka/install
   cmake --build calibration_tools/build --parallel

产物为 ``calibration_tools/build/read_franka_state``，通过 ``readOnce()`` 读取末端
位姿、关节速度和机器人状态并输出 JSON，不发送运动指令。Python 控制包的安装
不保证包含此编译步骤所需的头文件和 CMake 配置。
下文替换相机序列号、SSH 别名、读取程序绝对路径和机器人 IP；读取程序在本机时
省略 ``--ssh-host``，需要指定共享库目录时追加 ``--library-dir /path/to/lib``。

**2. 启动相机并采集**

退出占用设备的程序，启动相机服务：

.. code-block:: bash

   python calibration_tools/raw_camera_service.py \
     --base-serial BASE_SERIAL --d455-serial D455_SERIAL

将 ChArUco 板固定在右臂末端，相机保持不动。默认板为 6×8 格、格边长 25 mm、
标记边长 18 mm、DICT_4X4_100、非 legacy 布局。其他板可通过
``--squares-x``、``--squares-y``、``--square-m``、``--marker-m`` 和
``--dictionary`` 配置，长度单位为米。

在同一相机节点另开终端，激活上一步选用的环境后运行：

.. code-block:: bash

   python calibration_tools/base_handeye_collect.py \
     --arm right --camera-serial BASE_SERIAL \
     --camera-url http://127.0.0.1:8765/raw/base \
     --ssh-host robot-right \
     --reader /absolute/path/to/read_franka_state --robot-ip ROBOT_IP \
     --output calibration_tools/sessions/base-to-right

打开 ``http://127.0.0.1:8767``，人工调整右臂，释放引导按钮并停稳后点击
``Capture pose``。建议采集 20–30 个不同姿态，覆盖多个旋转轴；求解至少需要十组。
工具只读取状态，不会移动机械臂。不传 ``--output`` 时，默认写入脚本所在目录下的
``calibration_tools/sessions/``，与启动时的工作目录无关。

在已保存样本表格中查看角点数量和重投影 RMS（像素）。点击 ``Preview`` 打开带标注的样本图像，再通过 ``Exclude`` 将不合适的样本排除出下一次求解；文件仍保留在会话的 ``excluded/`` 目录中。``Active samples`` 只统计参与求解的有效样本；样本编号保持不变，出现空缺是正常的。增加或排除样本不会自动重算已有标定结果，应重新运行求解器后再导出。

按 ``Ctrl+C`` 停止采集器。中断后重新开始或整批重采时，重新运行上面的命令，并为 ``--output`` 指定新会话目录。旧会话文件仍保留在磁盘上，但不会加载到新的采集会话中。

标定 D455 时重复上述命令，将序列号换成 ``D455_SERIAL``、URL 换成 ``/raw/d455``、输出目录换成 ``calibration_tools/sessions/d455-to-right``。D455 流配置需先确认符合上面的畸变模型支持范围。

**3. 求解并导出**

.. code-block:: bash

   python calibration_tools/solve_base_handeye.py \
     calibration_tools/sessions/base-to-right --arm right
   python calibration_tools/export_dual_franka.py \
     calibration_tools/sessions/base-to-right/base_camera_extrinsic_candidate.json \
     --output calibration_tools/exports/base_to_right.yaml

D455 使用同样步骤，将会话目录和输出文件改为 ``d455-to-right`` 和
``d455_to_right.yaml``。检查会话中的 ``quality_report.json``，并通过独立实物
测量验证候选外参；求解成功或导出 YAML 不代表精度已验证。导出不会覆盖已有文件。
重新求解后，导出时使用新文件名，例如 ``calibration_tools/exports/base_to_right_v2.yaml``。
验证新结果后，再更新 ``perception.calibration`` 中的文件路径。

**4. 配置 RPent**

在自己的 robot config 中填写经过验证的 YAML 路径：

.. code-block:: yaml

   perception:
     calibration:
       base_camera: /absolute/path/to/base_to_right.yaml
       d455_camera: /absolute/path/to/d455_to_right.yaml

通过下文的 ``--robot-config`` 加载配置。启动 RPent 前停止标定相机服务。
保留并核对 ``perception.base_frames`` 和定位边界；本工具不会自动标定左右基座关系。

腕部投影为可选功能。在示例配置中取消 ``perception.calibration`` 和
``perception.projection_views`` 下腕部条目的注释，并配置具有对齐深度和 RGB
内参的 RGBD 相机。每份腕部标定 YAML 顶层需包含 ``arm: left`` 或 ``arm: right``，
设置 ``parameters.eye_on_hand: true``，并将
``parameters.robot_effector_frame`` 设置为 ``left_ee_O_T_EE`` 或
``right_ee_O_T_EE``。``transformation`` 将相机坐标转换到对应末端坐标系。
RPent 将其与所选快照中的末端位姿组合；若位姿使用 ``left_base``，还会应用
配置的基座变换，最终输出 ``right_base`` 坐标。快照 TCP 位姿必须对应 O_T_EE，
并声明基座坐标系。默认的 Lumos 观测相机不会自动启用 RGBD 投影；选择腕部锚点前
必须完成这些配置。

开发配置
--------

启用机械臂运动前，请检查并修改仓库中的开发默认值：

* ``robots/dual_franka/config/example.yaml`` 包含机器人身份（两台机器人 IP、相机序列号/类型、夹爪连接）、工作空间几何（目标位姿、安全边界）、easy_handeye YAML 映射（见上方标定说明）和感知定位边界 + base-frame 变换。示例设置 ``realtime_config: ignore``；在 PREEMPT_RT 内核上可改为 ``enforce``。

RPent 会将该机器人配置转换成内部双节点 RLinf cluster 和环境对象。如需使用其他文件，请传入 ``--robot-config /path/to/robot_config.yaml``。

``rotate_delta`` 旋转时，位置目标保持在本次旋转开始时的 TCP 位置。真实硬件采用柔顺/阻抗控制，实际位姿可能因负载、接触或有限的跟踪精度而偏离目标。

``rotation_hold.max_drift_m`` 默认为 ``0.10`` m（10 cm），限制 TCP 相对本次旋转起始位置的欧氏距离；它与位置到位容差 ``move_tolerance_m`` （0.006 m）不同。漂移小于或等于上限时，旋转结束后仅在位置误差超过到位容差时复用 ``move_delta`` 回位。只有补偿成功（如需补偿），且最终位置和角度均达到各自容差，才返回 ``ok=True`` ，否则返回 ``False`` 。漂移超过上限时，旋转循环中或结束时的检查会抛出 RPC 错误，停止本次调用，不再继续平移补偿。

机器人自身的保护仍可能独立触发。

启动双节点 Ray 集群
--------------------

两个节点的角色不同（定义在 ``robots/dual_franka/runtime_config.py`` 中）：

* 节点 ``0`` 是 Ray head 节点：运行双臂 Franka 环境 worker （全部相机、感知以及双臂和夹爪状态）和左臂的实时控制器。VLA 任务的本地 VLA 服务也运行在该节点上。
* 节点 ``1`` 是 Ray worker 节点：只运行右臂的实时控制器，不接相机，也不运行 RPent 进程。

.. warning::

   每个控制节点都必须在启动 Ray 前设置 ``RLINF_NODE_RANK``。

节点 ``0``：

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray stop --force
   ray start --head --port=6379 --node-ip-address=HEAD_IP

节点 ``1``：

.. code-block:: bash

   export RLINF_NODE_RANK=1
   ray stop --force
   ray start --address=HEAD_IP:6379 --node-ip-address=WORKER_IP

运行冒烟测试
------------

先按 :doc:`../guides/configure_planner` 配置规划器与模型服务。

任务 ``0`` 用于逐臂测试小幅平移、旋转和夹爪动作：

.. code-block:: bash

   uv run --extra franka rpent --robot dual_franka --task-id 0 \
     --planner claude_code --model claude-opus-4-8 \
     --robot-config robots/dual_franka/config/example.yaml

RPent 使用当前解释器启动 ``robots/dual_franka/env_server.py``，加载 RPent robot config 并生成内部 RLinf adapter config，然后连接 Ray，等待 ``healthz``，并将初始状态记录为 step ``0``。任务 ``0`` 不会加载 VLA。

VLA 抓取演示
-------------

RPent 提供了一个使用 VLA 抓取物品的演示。任务 ``1`` 提供 ``vla_right_grasp`` / ``vla_handoff`` / ``vla_left_place`` 工具，并可在本地启动双臂 Franka VLA 服务。``PI05_CHECKPOINT_PATH`` 指向训练好的 Pi-05 checkpoint，``DUAL_FRANKA_REPO_ID`` 是用于查找对应归一化统计的数据集 ID：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/checkpoints/global_step_N
   export DUAL_FRANKA_REPO_ID=org/dual-franka-tcp-rot6d

   uv run --extra franka rpent --robot dual_franka --task-id 1 \
     --cuda-device 0 \
     --planner claude_code --model claude-opus-4-8 \
     --robot-config robots/dual_franka/config/example.yaml

checkpoint 必须包含：

.. code-block:: text

   actor/model_state_dict/full_weights.pt
   <DUAL_FRANKA_REPO_ID>/norm_stats.json

**预训练 checkpoint**

ModelScope 上发布了一个可直接使用的 task ``1`` checkpoint： `Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000 <https://modelscope.cn/models/Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000>`_。下载后将 ``PI05_CHECKPOINT_PATH`` 指向下载目录，并将 ``DUAL_FRANKA_REPO_ID`` 设置为包含 ``norm_stats.json`` 的子目录：

.. code-block:: bash

   modelscope download \
     --model Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000 \
     --local_dir /path/to/pi05-dualfranka-clean-desk

   export PI05_CHECKPOINT_PATH=/path/to/pi05-dualfranka-clean-desk

.. warning::

	该 checkpoint 仅在我们的内部测试环境（机器人位姿、相机、工作空间布局和物体）上训练，切换到不同的环境时预计表现会较差。若要部署到你自己的机器上，请使用 RLinf 采集示教数据并微调你自己的 checkpoint，参见 `RLinf 双臂 Franka 指南 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/dual_franka.html>`_ （采集 GELLO 示教数据、转换为 tcp_rot6d、运行 SFT，然后部署）。

未设置 ``--vla-endpoint`` 时，RPent 会启动 ``rpent/robots/components/pi05_vla_server.py``，并只加载一次 ``pi05_dualfranka_tcp_rot6d``。

也可以单独启动 VLA 服务：

.. code-block:: bash

   uv run --extra franka python -m rpent.robots.components.pi05_vla_server \
     --embodiment dual_franka \
     --model-path /path/to/checkpoints/global_step_N \
     --repo-id org/dual-franka-tcp-rot6d \
     --cuda-device 0 --transport http --host 0.0.0.0 --port 6000

然后向 ``rpent`` 传入 ``--vla-endpoint http://VLA_HOST:6000``。外部 endpoint 始终优先于本地自动启动。

连接外部环境服务
----------------

连接已经运行的双臂 Franka 环境服务：

.. code-block:: bash

   uv run --extra franka rpent --robot dual_franka --task-id 0 \
     --env-endpoint http://ROBOT_HOST:PORT \
     --planner claude_code --model claude-opus-4-8 \
     --robot-config robots/dual_franka/config/example.yaml

工具与状态产物
--------------

双臂 Franka 扩展提供 ``view_env_state``、``view_camera_meta``、 ``move_delta``、``rotate_delta``、``open_gripper``、``close_gripper`` 和 ``vla_right_grasp`` / ``vla_handoff`` / ``vla_left_place``。每次解析式运动只会作用于一条臂（``left`` 或 ``right``）。所有会改变环境状态的工具都会在 RPent 统一的 ``EnvState`` 中保存每条臂的状态以及同步的 left-wrist、base 和 right-wrist 图像。

安全要求
--------

两条机械臂的急停按钮旁都必须有操作员值守。先使用极小幅度的单臂动作验证任务 ``0``，再尝试抓取。当相机与状态结果不一致、目标运动没有到位，或任何标定存在疑问时，应立即停止。

手动技能测试
------------

普通终端评测不加 ``--interactive`` 和 ``--dashboard``，在 TTY 中按提示完成人工确认。``--explore`` 不加 ``--interactive`` 也能按提示答复。需要通过终端对话指导探索时，添加 ``--interactive`` 或 ``-i``；Codex/Claude 的运行中输入和 API 的回复后追加输入方式见 :doc:`../guides/dashboard`。

使用 ``--dashboard`` 时，在 Dashboard 消息输入框中提交任务、消息和确认回复，无需 TTY，也不加 ``--interactive``；两个参数不能同时使用。Dashboard 支持 ``api``、``claude_code`` 和 ``codex`` （当前真机实验仅验证 Codex）。普通任务中，``/success``、``/failure`` 和 ``/continue`` 用于回答待处理的 ``request_operator_verdict``，随后智能体继续执行并调用 ``finish``。普通任务连接环境时复位，探索则等待场景确认后再复位。Dashboard 操作见 :doc:`../guides/dashboard`。

部署脚本位于 ``robots/dual_franka/``。在仓库根目录运行：

.. code-block:: bash

   robots/dual_franka/run_manual_skill.sh --list-primitives
   robots/dual_franka/run_manual_skill.sh --schema vla_right_grasp

通过 ``--primitive NAME --params JSON`` 调用工具。``--task-id`` 为命名 VLA 技能选择任务配置中的 ``vla_instruction``，规划器的阶段 prompt 单独记入日志。当前 clean-desk 任务继续使用 checkpoint 原来的训练指令。 ``--robot-config`` 选择机器参数，其中的 ``perception.calibration`` 映射指向 easy_handeye 手眼标定 YAML。本地 SAM3 需要安装 ``sam3`` extra；远端服务可通过 ``--sam3-endpoint`` 接入。

机器人 Codex 运行配置隔离
-------------------------

上述启动脚本使用独立的 ``RPENT_CODEX_HOME`` （默认仓库内 ``.codex-rpent-live``），不继承编程终端的 ``CODEX_HOME``。本地记忆默认放在该目录的 ``memory`` 中，Codex 状态数据库也使用独立目录。按需在独立目录创建私人 ``config.toml``，不要覆盖已有私人配置。

API 部署需显式设置 ``RPENT_CODEX_API_KEY`` 和可选的 ``RPENT_CODEX_BASE_URL``。脚本会清除继承的 ``CODEX_API_KEY``、``CODEX_BASE_URL``、``OPENAI_API_KEY`` 和 ``OPENAI_BASE_URL``。不使用 API key 时，应在独立目录下单独登录；可将凭据配置为保存在文件中。不要提交真实配置、凭据或日志。已有私人配置若使用系统钥匙串，须另行检查账号共享。

模型、推理强度、服务档位分别通过 ``RPENT_CODEX_MODEL``、 ``RPENT_REASONING_EFFORT``、``RPENT_CODEX_SERVICE_TIER`` 设置，默认保持 ``gpt-5.5``、``medium``、``fast``。这些规则只适用于上述部署脚本，不改变直接调用 RPent CLI 的通用环境变量接口。建议通过启动脚本运行；手动 source 会修改当前终端的环境变量。

目录隔离不是权限沙盒，也不隔离共享工作区文件。当前 planner 显式使用不请求交互审批、完整文件访问的运行参数。仅修改私人配置不能覆盖 planner 显式传入的权限；连接检测仍使用只读沙盒。

探索模式（人工复位与判定）
--------------------------

``dual_franka --explore`` 等待人工确认场景后复位机器人，并记录人工判定、相机及机器人状态。按前文配置规划器和 VLA 参数后，例如使用任务 0：

.. code-block:: bash

   rpent --robot dual_franka --task-id 0 --explore \
     --robot-config /path/to/robot.yaml \
     --memory-dir /path/to/memory/dual_franka \
     --explore-attempts-per-session 3 --explore-sessions 2 \
     --output-dir /path/to/new-run

每个规划器会话先通过 ``request_scene_reset`` 请求场景确认。处理夹持物体、恢复场景后回复 ``done``；机器人复位成功并取得相机和状态观测后开始一次尝试，复位失败时不允许运动。``--explore-attempts-per-session`` 限制每个会话的尝试次数。通过 ``--env-endpoint`` 接入的外部服务必须声明 ``explicit_reset_only=True``，才能支持先确认、后复位。

``request_operator_verdict`` 请求 ``success``、``failure``、``continue`` 或 ``abort``，可附备注。其中 success/failure 答复判定当前尝试，``continue`` 继续尝试，``abort`` 停止探索。后续运动会清除先前判定；关闭终端输入也会停止探索。

使用 Codex/Claude 并加 ``--interactive`` 时，确认回复使用显示的请求 ID，例如 ``/operator <request-id> done`` 或 ``/operator <request-id> success 物体已提起``；普通消息发送给智能体。不加 ``--interactive`` 时直接按提示答复；使用 ``--dashboard`` 时在 Dashboard 消息输入框中输入下述命令。

日志与记忆
~~~~~~~~~~

``sessions/session_<NNN>/`` 保存相机和机器人状态，``exploration.json`` 记录尝试边界，``operator_events.json`` 记录人工反馈。失败尝试的记录会保留，工作笔记位于 ``<memory-dir>/_internal/inbox/<recipe_tag>/wip/``。

默认启用 memory 合并。正常结束后合并合法的 ``global`` / ``task-family`` 草稿，包括失败尝试的经验；成功时还将完整的 audit/recipe 文件对保存到 ``task-specific``。使用 ``--no-auto-merge-memory`` 可保留记录而不合并。运行错误、中止、切换任务或关闭 Dashboard 会话时不合并。记忆目录说明见 :doc:`../guides/memory`。

探索命令
~~~~~~~~

在 Codex/Claude 终端交互（``--explore --interactive``）或 Dashboard 探索中，可使用：

* ``/success`` 或 ``/failure``：结束整个任务，而不只是判定当前尝试。成功要求场景已确认且复位成功；终端中可在首次复位前用 ``/failure`` 结束任务，Dashboard 中则需先用 ``/done`` 或 ``/abort`` 回答待处理的复位请求。
* ``/done``：确认待处理的场景复位请求。
* ``/continue``：回答待处理的判定请求并继续尝试。
* ``/abort``：停止任务，相机不可用时也可使用。

每条命令单独输入。不带斜杠的 ``success``、``failure`` 等词语是发给智能体的普通消息。已发送的机器人动作或 RPC 可能需返回后才能结束任务；这些命令不是急停。

独立 VLA 诊断控制台
~~~~~~~~~~~~~~~~~~~

原语测试复用 ``robots/dual_franka/run_manual_skill.sh``，VLA 诊断使用独立入口。不注册 task103/104，也不经过共享 runner 分发。``--task-id`` 选择已有 VLA 任务配置（默认 1），指令来自其 ``vla_instruction``，可用 ``--instruction`` 覆盖。请在源码仓库中运行以下命令。诊断只初始化环境和 VLA 组件，即使配置了 SAM3，也不会启动或连接其服务。

.. code-block:: bash

   python -m tests.e2e_tests.dual_franka.dual_franka_vla --task-id 1 \
     --robot-config /path/to/robot.yaml \
     --vla-model-path /path/to/checkpoint --vla-repo-id org/dataset

支持 ``prompt <指令>``、``infer`` （不执行）、``step`` （重新推理并执行）、 ``run N`` （1–20 块）、``reset``、``quit``。初始化可能复位。执行前将输入与预测保存为 JSON/NPZ，拒绝无效观测或动作；执行结果不确定时禁止继续运动，需重启会话。 RPC 成功不代表任务成功。动作校验默认要求每块预测包含 20 步；如果 checkpoint 使用不同块长度，请通过 ``--expected-action-steps`` 显式指定匹配的值。外部模型服务使用标准 VLA 推理和健康检查 RPC。

Flash 回放
----------

双臂使用共享的 :ref:`Franka Flash 流程 <franka-flash>`。生成任务卡时使用
``--robot dual_franka --task dual_franka_t0``，回放时使用
``--robot dual_franka --planner flash --task-id 0``，并提供双臂任务卡及配置。
平移标注使用具有深度和有效标定的 ``base``、``d455``、``left_wrist`` 或
``right_wrist`` 相机。移动原语通过 ``arm: left`` 或 ``arm: right`` 指定机械臂；
左臂工作空间检查会把共享的右基座目标坐标转换到左基座坐标系。

停止运行
------------

在终端按 Ctrl+C 请求结束 RPent。已经发送的机器人动作或正在执行的 RPC 需要返回后才能完成清理；紧急情况下应使用硬件急停按钮。停止后检查两条机械臂与夹爪状态，再按控制系统的停机流程关闭服务。
