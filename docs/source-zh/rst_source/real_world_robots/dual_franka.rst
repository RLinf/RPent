双臂 Franka
=============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka-deploy.jpg
   :alt: 安装在同一工作台旁的两台 Franka 机械臂与相机
   :figclass: rpent-robot-figure
   :align: center

   用于真机实验的双臂 Franka 系统。

RPent 通过 RLinf ``RealWorldEnv`` worker 控制双节点双臂 Franka 系统。任务视频见 :doc:`双臂 Franka 演示 <../real_world_demos/franka>`。

安装
----

在两台 Linux 控制节点分别安装 RPent，保持代码、Python、Ray 和 RLinf 版本一致。本地运行 VLA 或 SAM3 的节点需要 NVIDIA GPU。以下沿用 :doc:`../get_started/quickstart` 的安装方式：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[franka,sam3]"

下载 Python 依赖较慢时，可使用 `清华 PyPI 镜像 <https://mirrors.tuna.tsinghua.edu.cn/help/pypi/>`_ 替换上面的安装命令：

.. code-block:: bash

   uv pip install -e ".[franka,sam3]" \
     --index-url https://pypi.tuna.tsinghua.edu.cn/simple

该选项只替换 Python 包索引，不代理 Git 仓库、直接下载链接或模型权重。

连接机械臂前，请核对 libfranka 与机器人固件的兼容性。标定状态读取程序还需要 libfranka 开发库，见下文的标定步骤。

后续命令在仓库根目录和该环境中执行。新开终端时，重新激活 ``.venv`` 并设置所需的模型环境变量。

需要本地分割时，在环境节点下载 SAM3 权重：

.. code-block:: bash

   uv pip install modelscope
   modelscope download --model facebook/sam3 sam3.pt \
     --local_dir ./checkpoints/sam3
   export SAM3_CHECKPOINT_PATH="$PWD/checkpoints/sam3/sam3.pt"

未设置该变量或 ``--sam3-endpoint`` 时，分割功能不可用；仅安装 extra 不会下载权重。VLA checkpoint 配置见 :ref:`dual-franka-vla`。

标定
----

``base_camera`` 和 ``d455_camera`` 都需要标定到 **右臂基座**。可使用 ``calibration_tools/`` 或 ROS `easy_handeye <https://github.com/IFL-CAMP/easy_handeye>`_，最终加载格式均为 YAML。腕部相机默认仅用于观测；若使用腕部 RGBD 相机进行投影，需要配置手眼标定和投影视图。

采集器使用 RealSense RGB 图像，支持零畸变系数或非零的 ``distortion.brown_conrady`` 系数；其他非零畸变模型会拒绝采样。

**1. 准备环境**

在相机节点的 RPent 仓库根目录使用已安装的环境。若需独立安装标定依赖，可创建单独环境：

.. code-block:: bash

   uv venv .venv-calibration
   source .venv-calibration/bin/activate
   uv pip install -r calibration_tools/requirements.txt pyrealsense2

在右臂控制节点，先按 `libfranka 官方安装指南 <https://frankarobotics.github.io/docs/doc/libfranka/README.html#quick-install>`_ 安装与机器人固件兼容的开发库（含头文件和 CMake 配置），再编译状态读取程序：

.. code-block:: bash

   cmake -S calibration_tools -B calibration_tools/build \
     -DCMAKE_PREFIX_PATH=/absolute/path/to/libfranka/install
   cmake --build calibration_tools/build --parallel

编译后得到 ``calibration_tools/build/read_franka_state``。标定时用它读取机械臂状态，不会移动机械臂。

* 本机读取：读取程序与采集器在同一节点，不传 ``--ssh-host``。
* 远程读取：用 ``--ssh-host`` 指定读取程序所在节点，并提前配置 SSH 免密登录。

下文命令中的相机序列号、机器人 IP 和程序路径均需替换为实际值。

**2. 启动相机并采集**

退出占用设备的程序，启动相机服务：

.. code-block:: bash

   python calibration_tools/raw_camera_service.py \
     --base-serial BASE_SERIAL --d455-serial D455_SERIAL

将 ChArUco 板固定在右臂末端，相机保持不动。默认板为 6×8 格、格边长 25 mm、
标记边长 18 mm、DICT_4X4_100、非 legacy 布局。其他板可通过
``--squares-x``、``--squares-y``、``--square-m``、``--marker-m`` 和
``--dictionary`` 配置，长度单位为米。
请实测打印后的格边长，确保与配置一致。

在同一相机节点另开终端，激活上一步选用的环境后运行：

.. code-block:: bash

   python calibration_tools/base_handeye_collect.py \
     --arm right --camera-serial BASE_SERIAL \
     --camera-url http://127.0.0.1:8765/raw/base \
     --ssh-host robot-right \
     --reader /absolute/path/to/read_franka_state --robot-ip ROBOT_IP \
     --output calibration_tools/sessions/base-to-right

打开 ``http://127.0.0.1:8767``，按以下步骤采样：

1. 确认机械臂处于 Idle 且无报错。人工调整右臂后，释放引导按钮，停稳后点击 ``Capture pose``。
2. 采集 20–30 个不同位置和角度，覆盖多个旋转轴；求解至少需要 10 组。
3. 用 ``Preview`` 查看样本，``Exclude`` 排除不合适的样本。样本调整后需重新求解。
4. 采完按 ``Ctrl+C`` 停止采集器。重新采集时，为 ``--output`` 指定新目录。

若从自己的电脑访问网页，先转发端口，再打开上面的地址：

.. code-block:: bash

   ssh -L 8767:127.0.0.1:8767 CAMERA_HOST

标定 D455 时复用采集命令，将序列号改为 ``D455_SERIAL``、URL 改为 ``http://127.0.0.1:8765/raw/d455``、输出目录改为 ``calibration_tools/sessions/d455-to-right``。

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

通过下文的 ``--robot-config`` 加载配置。
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

标定结束后，停止本次标定启动的全部相机服务、采集器和机器人状态读取程序，再启动 RPent。

机器人配置
----------

若标定使用了独立 Python 环境，执行后续命令前请重新激活 RPent 的 ``.venv``。

启用机械臂运动前，请检查并修改仓库中的开发默认值：

* ``robots/dual_franka/config/example.yaml`` 包含机器人身份（两台机器人 IP、相机序列号/类型、夹爪连接）、工作空间几何（目标位姿、安全边界）、easy_handeye YAML 映射（见上方标定说明）和感知定位边界 + base-frame 变换。示例设置 ``realtime_config: ignore``；在 PREEMPT_RT 内核上可改为 ``enforce``。

RPent 会将该机器人配置转换成内部双节点 RLinf cluster 和环境对象。如需使用其他文件，请传入 ``--robot-config /path/to/robot_config.yaml``。

示例配置中，各相机的用途如下：

* ``base``、``left_wrist``、``right_wrist``：提供 RGB 图像，用于观察场景。
* ``d455``：提供 RGB 和对齐深度图像，可将图像中的选点转换为三维坐标。反投影需要该相机的有效标定。

``perception.projection_views`` 用于选择反投影视角，不控制相机是否采集深度；示例配置中应使用 ``d455``。

``rotate_delta`` 调整末端朝向，并尽量保持旋转开始时的位置。``rotation_hold.max_drift_m`` 设置旋转时允许的位置偏移上限，默认 ``0.10`` m（10 cm）；超出上限时中止本次调用并报错。

任务选择
--------

主要示例使用任务 ``2``。首次连接时可先用任务 ``0`` 检查小幅平移、旋转和夹爪动作；该任务由规划器驱动，不是固定自动测试序列，也不需要 VLA。

.. list-table::
   :header-rows: 1

   * - ID
     - 内容
     - 是否需要 VLA
   * - 0
     - 规划器驱动的单技能检查
     - 否
   * - 1
     - 碗盘脏污分类，包含杯子和筷勺
     - 是
   * - 2
     - 碗盘及两个杯子脏污分类，无筷勺
     - 是

.. note::

   任务编号用于启动命令，分类任务使用任务名称生成 recipe 和记忆标识。修改任务时，请保持编号、名称与任务内容的对应关系，不要随意重排编号或将已有名称用于不同任务。

任务 ``1`` 中，碗盘内的蛋挞或蛋挞锡盘标记脏污：脏碗盘放入金属筐，干净碗盘和杯子放入纸盒；筷勺插入杯中。

任务 ``2`` 中，碗内的鸡块、盘子或杯子内的蛋挞锡盘标记脏污：脏物放入金属筐，干净物放入纸盒，不含筷勺。

两个分类任务均提供 ``vla_right_grasp`` / ``vla_handoff`` / ``vla_left_place`` 技能。

.. _dual-franka-vla:

VLA 配置
--------

``PI05_CHECKPOINT_PATH`` 指向训练好的 Pi0.5 checkpoint，``DUAL_FRANKA_REPO_ID`` 是用于查找对应归一化统计的数据集 ID。

上述环境变量对应 ``--vla-model-path`` 和 ``--vla-repo-id``。repo ID 是 checkpoint 内对应归一化统计的相对目录，不是模型下载仓库名。checkpoint 必须包含：

.. code-block:: text

   actor/model_state_dict/full_weights.pt
   <DUAL_FRANKA_REPO_ID>/norm_stats.json

**预训练 checkpoint**

从 Hugging Face 下载 `双臂 Franka clean-desk checkpoint <https://huggingface.co/RLinf/RLinf-Pi05-RealWorld-DualFranka-Clean_Desk>`_。它用于执行 VLA 技能，物体的脏污分类由 RPent 规划器判断。

.. code-block:: bash

   uv pip install huggingface_hub
   hf download RLinf/RLinf-Pi05-RealWorld-DualFranka-Clean_Desk \
     --revision 4f43adc723dfe98fae806a131ad61999e1f95e4f \
     --local-dir ./models/pi05-dualfranka-clean-desk

   export PI05_CHECKPOINT_PATH="$PWD/models/pi05-dualfranka-clean-desk"
   export DUAL_FRANKA_REPO_ID=id_0

该 checkpoint 已包含 ``id_0/norm_stats.json``，请与权重一起保留，并按上例将 repo ID 设置为 ``id_0``。

.. warning::

	该 checkpoint 仅在我们的内部测试环境（机器人位姿、相机、工作空间布局和物体）上训练，切换到不同的环境时预计表现会较差。若要部署到你自己的机器上，请使用 RLinf 采集示教数据并微调你自己的 checkpoint，参见 `RLinf 双臂 Franka 指南 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/dual_franka.html>`_ （采集 GELLO 示教数据、转换为 tcp_rot6d、运行 SFT，然后部署）。

.. _dual-franka-codex-isolation:

Codex 配置
----------------

使用 Codex 时，先按\ :ref:`planner-codex-isolation`\ 配置独立的运行目录并登录。

启动双节点 Ray 集群
--------------------

启动 Ray 前，在两个节点分别激活安装好的 ``.venv``。``robots/dual_franka/runtime_config.py`` 的默认部署为：

* 节点 ``0`` 是 Ray head：运行 RPent、环境 worker、左臂及夹爪控制器、base 相机和左腕相机。额外的 D455 由 RPent 在此本地打开；按需启动的本地 VLA、SAM3 服务也运行在这里。
* 节点 ``1`` 是 Ray worker：运行右臂及夹爪控制器和右腕相机，需要安装环境并启动 Ray，但不需要另起 RPent runner。

RLinf 按配置的节点创建相机句柄，不会跨节点搜索被移动的 USB 相机。腕部相机应接在对应机械臂的控制节点，并按实际设备填写型号和序列号。

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

手动技能测试
------------

可先列出可调用的技能，再查看某个技能接受的参数及其类型。在仓库根目录运行：

.. code-block:: bash

   python robots/dual_franka/dual_franka_manual_call.py --list-primitives
   python robots/dual_franka/dual_franka_manual_call.py --schema vla_right_grasp

未指定服务地址时，脚本自动启动并连接所需服务，无需手动填写端口；退出时只关闭本次启动的服务。传入 ``--env-endpoint``、``--vla-endpoint`` 或 ``--sam3-endpoint`` 可复用已有服务。连接已有 Env 不会复位。若新启动 Env，手动动作前先请求场景确认并执行一次复位；显式调用手动 ``reset`` 也会等待确认。列出技能和查看参数不启动服务。

例如，先看 schema，再执行左臂小幅平移：

.. code-block:: bash

   python robots/dual_franka/dual_franka_manual_call.py --schema move_delta
   python robots/dual_franka/dual_franka_manual_call.py \
     --primitive move_delta --params '{"arm":"left","delta_xyz":[0.01,0,0]}' \
     --robot-config robots/dual_franka/config/example.yaml

通过 ``--primitive NAME --params JSON`` 调用工具，不选择任务编号。示例配置的 ``vla.instruction`` 已填写 clean-desk 策略指令；更换 checkpoint 时可修改该值，或用 ``--instruction`` 覆盖。该配置只用于手动调用，任务 runner 仍使用已注册任务的指令。``--params`` 中的 ``prompt`` 记录技能意图，不替代策略指令。

设置好上文的 VLA checkpoint 和归一化统计变量后，可以执行一次抓取技能：

.. code-block:: bash

   python robots/dual_franka/dual_franka_manual_call.py \
     --primitive vla_right_grasp \
     --params '{"prompt":"grasp the bowl","max_chunks":1}' \
     --robot-config robots/dual_franka/config/example.yaml

独立 VLA 诊断控制台
~~~~~~~~~~~~~~~~~~~

该控制台独立运行 VLA，不经过智能体或注册任务。策略指令来自机器人配置中的 ``vla.instruction``，可用 ``--instruction`` 覆盖。请在源码仓库中运行；控制台按需启动环境和 VLA 服务，不启动 SAM3。

.. code-block:: bash

   python -m tests.e2e_tests.dual_franka.dual_franka_vla \
     --robot-config robots/dual_franka/config/example.yaml

启动后，在控制台输入：

* ``run N``：预测并执行 N 个动作块（每块是模型一次输出的一段连续动作）；不带 N 时持续运行，不在技能边界处停止。
* ``reset``：等待操作员回复 ``done`` 后，在同一环境中复位双臂；回复 ``abort`` 则不执行复位。
* ``quit``：退出控制台。

启动时不会自动回位。运行中按 ``Ctrl+C`` 或创建日志目录下的 ``STOP`` 文件可停止；正在执行的 RPC 仍可能完成，这不是硬件急停。
输入和预测保存为 JSON/NPZ。如果提示执行状态不确定，请检查现场后重启会话。

每块动作步数自动从 VLA 服务读取，也可用 ``--expected-action-steps`` 显式指定。

运行一个任务
------------

完成相机标定、机器人配置和 Ray 集群启动，并按 :doc:`../guides/configure_planner` 配置规划器。使用 Codex 前，先完成 :ref:`planner-codex-isolation`。

设置好上面的模型变量后，以下以任务 ``2`` 为例运行；可通过 ``--task-id`` 选择其他已注册任务。RPent 自动启动所需的环境和模型服务，无需另开终端手动启动 VLA：

.. code-block:: bash

   rpent --robot dual_franka --task-id 2 \
     --planner codex --model gpt-6-astra --reasoning-effort xhigh \
     --robot-config robots/dual_franka/config/example.yaml \
     --memory-profile local --memory-dir ./memory/dual_franka

连接环境会初始化控制器和夹爪，但不复位机器人姿态。智能体先调用 ``request_scene_reset``：处理夹持物、恢复桌面并确认双臂回位路径安全后回复 ``done``，工具才执行复位并采集新观测。未确认或复位失败时禁止运动，每次新运行或会话都需要重新确认。

示例使用 Codex、``gpt-6-astra`` 和 ``xhigh`` 推理强度。也可将规划器和模型参数换为 ``--planner claude_code --model claude-opus-4-8``。模型与推理强度的配置见 :doc:`../guides/configure_planner`。

在同一条命令中按需追加参数：

* ``--explore``：探索并积累任务记忆，见下文。
* ``--explore --interactive``：探索时在终端输入反馈；普通任务不支持单独使用 ``--interactive``。
* ``--dashboard``：通过网页输入框提交任务、消息和确认；探索时同时加 ``--explore``。不能与 ``--interactive`` 同用。界面操作见 :doc:`../guides/dashboard`。

查看结果
--------

每次运行使用新的 ``--output-dir``。重点查看：

* ``transcript_*.json``：规划器消息、工具调用与结束结果。
* ``run.log`` 及各组件 ``*_server.log``：运行或启动错误。
* ``states.json`` 及其引用的图像、深度和机器人状态文件：每步执行后的观测证据。
* ``operator_events.json``：复位确认与人工判定。

探索模式的逐会话状态和人工记录位于 ``sessions/session_<NNN>/``。真机任务成功以操作员判定为准，RPC 成功或规划器自己的 ``finish`` 文本不能替代该判定。通用目录说明见 :ref:`run-output-files`。

工具与状态产物
--------------

双臂 Franka 扩展提供 ``view_env_state``、``view_camera_meta``、 ``move_delta``、``rotate_delta``、``open_gripper``、``close_gripper`` 和 ``vla_right_grasp`` / ``vla_handoff`` / ``vla_left_place``。每次解析式运动只会作用于一条臂（``left`` 或 ``right``）。所有会改变环境状态的工具都会在 RPent 统一的 ``EnvState`` 中保存每条臂的状态以及同步的 left-wrist、base 和 right-wrist 图像。

探索模式（人工复位与判定）
--------------------------

在任务运行命令后追加 ``--explore``，即可启用探索模式。``--explore-attempts-per-session`` 设置每个规划器会话的尝试次数，``--explore-sessions`` 设置会话数；终端交互再加 ``--interactive``，网页交互改用 ``--dashboard``。

每个规划器会话先通过 ``request_scene_reset`` 请求场景确认。处理夹持物体、恢复场景后回复 ``done``；机器人复位成功并取得相机和状态观测后开始一次尝试，复位失败时不允许运动。``--explore-attempts-per-session`` 限制每个会话的尝试次数。普通任务和探索模式通过 ``--env-endpoint`` 接入的外部服务都必须声明 ``explicit_reset_only=True``。

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

Flash 回放
----------

双臂使用共享的 :ref:`Franka Flash 流程 <franka-flash>`。以下以任务 ``0`` 为例；其他任务应同时替换生成和回放参数中的任务编号。生成任务卡时使用
``--robot dual_franka --task dual_franka_t0``，回放时使用
``--robot dual_franka --planner flash --task-id 0``，并提供双臂任务卡及配置。
平移标注使用具有深度和有效标定的 ``base``、``d455``、``left_wrist`` 或
``right_wrist`` 相机。移动原语通过 ``arm: left`` 或 ``arm: right`` 指定机械臂；
左臂工作空间检查会把共享的右基座目标坐标转换到左基座坐标系。
