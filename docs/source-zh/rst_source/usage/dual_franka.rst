Dual Franka
===========

RPent 可以通过 RLinf ``RealWorldEnv`` worker 控制双节点双臂 Franka 系统。

安装
----

.. note::

	以下的步骤只会安装 Python 侧依赖（自定义的 RLinf Franka 分支和
	``rlinf-openpi``），并 **不会** 构建双臂真正需要的机器人节点控制栈。在安装
	RPent 之前，请先按照 RLinf 双臂 Franka 指南配置两个机器人节点：选择兼容的
	``LIBFRANKA_VERSION``，构建 ``franka-franky`` （franky/libfranka）控制栈，配置
	PREEMPT_RT 实时内核与相关权限，并安装 GELLO 遥操作与夹爪依赖。参见
	`RLinf 双臂 Franka 指南
	<https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/dual_franka.html>`_。

在 RPent 仓库根目录运行：

.. code-block:: bash

	uv sync --extra franka --extra sam3

该命令将自定义 RLinf Franka 分支和 ``rlinf-openpi`` 安装到 ``.venv``。

标定（Calibration）
----------------------

手眼标定使用 ROS `easy_handeye
<https://github.com/IFL-CAMP/easy_handeye>`_ 完成。它为每台投影相机（
``base_camera`` 和 ``d455_camera``）生成一个 YAML，默认保存在
``~/.ros/easy_handeye/`` 下。

RPent 会读取一个 JSON 文件（``hand_eye_calibration.json``），其中包含每台相机的
``source_name``、``parameters`` 和 ``transformation``。生成方式是从每个
``easy_handeye`` YAML 中复制这些字段。

该文件的位置可通过 ``--calibration-path`` 配置（默认
``~/.ros/easy_handeye/hand_eye_calibration.json``）。

开发配置
--------

启用机械臂运动前，请检查并修改仓库中的开发默认值：

* ``robots/dual_franka/config/example.yaml`` 包含机器人身份（两台机器人 IP、相机
  序列号/类型、夹爪连接）、工作空间几何（目标位姿、安全边界）和感知定位
  边界 + base-frame 变换。

RPent 会将该机器人配置转换成内部双节点 RLinf cluster 和环境对象。如需使用
其他文件，请传入 ``--robot-config /path/to/robot_config.yaml``。

启动双节点 Ray 集群
--------------------

两个节点的角色不同（定义在 ``robots/dual_franka/runtime_config.py``
中）：

* 节点 ``0`` 是 Ray head 节点：运行双臂 Franka 环境 worker
  （全部相机、感知以及双臂和夹爪状态）和**左臂**的实时控制器。VLA 任务的
  本地 VLA 服务也运行在该节点上。
* 节点 ``1`` 是 Ray worker 节点：只运行**右臂**的实时控制器，不接相机，
  也不运行 RPent 进程。

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

任务 ``0`` 用于测试稳妥的单臂解析式运动和夹爪 primitives：

.. code-block:: bash

	uv run --extra franka rpent --robot dual_franka --task-id 0 \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

RPent 使用当前解释器启动 ``robots/dual_franka/env_server.py``，加载 RPent
robot config 并生成内部 RLinf adapter config，然后连接 Ray，等待 ``healthz``，
并将初始状态记录为 step ``0``。任务 ``0`` 不会加载 VLA。

VLA 抓取 DEMO
-------------

RPent 提供了一个使用 VLA 抓取物品的 DEMO。task-id ``1`` 会暴露 ``vla_right_grasp`` / ``vla_handoff`` / ``vla_left_place``，
并可在本地启动双臂 Franka VLA 服务。``PI05_CHECKPOINT_PATH`` 指向 
训练好的 Pi-05 checkpoint，``DUAL_FRANKA_REPO_ID`` 是用于查找对应归一化统计的数据集 ID：

.. code-block:: bash

	export PI05_CHECKPOINT_PATH=/path/to/checkpoints/global_step_N
	export DUAL_FRANKA_REPO_ID=org/dual-franka-tcp-rot6d

	uv run --extra franka rpent --robot dual_franka --task-id 1 \
	  --cuda-device 0 \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

checkpoint 必须包含：

.. code-block:: text

	actor/model_state_dict/full_weights.pt
	<DUAL_FRANKA_REPO_ID>/norm_stats.json

**预训练 checkpoint**

ModelScope 上发布了一个可直接使用的 task ``1`` checkpoint：
`Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000
<https://modelscope.cn/models/Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000>`_。
下载后将 ``PI05_CHECKPOINT_PATH`` 指向下载目录，并将 ``DUAL_FRANKA_REPO_ID``
设置为包含 ``norm_stats.json`` 的子目录：

.. code-block:: bash

	modelscope download \
	  --model Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000 \
	  --local_dir /path/to/pi05-dualfranka-clean-desk

	export PI05_CHECKPOINT_PATH=/path/to/pi05-dualfranka-clean-desk

.. warning::

	该 checkpoint 仅在我们的内部测试环境（机器人位姿、相机、工作空间布局和物体）
	上训练，切换到不同的环境时预计表现会较差。若要部署到你自己的机器上，请使用
	RLinf 采集示教数据并微调你自己的 checkpoint，参见
	`RLinf 双臂 Franka 指南
	<https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/dual_franka.html>`_
	（采集 GELLO 示教数据、转换为 tcp_rot6d、运行 SFT，然后部署）。

未设置 ``--vla-endpoint`` 时，RPent 会启动
``robots/dual_franka/vla_server.py``，并只加载一次
``pi05_dualfranka_tcp_rot6d``。

也可以单独启动 VLA 服务：

.. code-block:: bash

	uv run --extra franka python -m robots.dual_franka.vla_server \
	  --model-path /path/to/checkpoints/global_step_N \
	  --repo-id org/dual-franka-tcp-rot6d \
	  --cuda-device 0 --transport http --host 0.0.0.0 --port 6000

然后向 ``rpent`` 传入 ``--vla-endpoint http://VLA_HOST:6000``。外部 endpoint
始终优先于本地自动启动。

连接外部环境服务
----------------

连接已经运行的双臂 Franka 环境服务：

.. code-block:: bash

	uv run --extra franka rpent --robot dual_franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

工具与状态产物
--------------

双臂 Franka 扩展提供 ``view_env_state``、``view_camera_meta``、
``move_delta``、``rotate_delta``、``open_gripper``、``close_gripper`` 和
``vla_right_grasp`` / ``vla_handoff`` / ``vla_left_place``。每次解析式运动只会作用于一条臂（``left`` 或 ``right``）。
所有会改变环境状态的工具都会在 RPent 统一的 ``EnvState`` 中保存每条臂的状态以及
同步的 left-wrist、base 和 right-wrist 图像。

安全要求
--------

两条臂都必须有操作员留在急停按钮旁。先使用极小的单臂动作验证任务 ``0``，
再尝试抓取。当相机与状态结果不一致、目标运动没有到位，或任何标定存在疑问时，
应立即停止。

手动技能测试
------------

当前任务运行中的人工评价/场景恢复使用独占终端输入。请在有 TTY 的普通终端
运行 runner；暂不支持 ``--interactive`` 或 Dashboard。启动时会在连接硬件前
拒绝这些组合。普通任务也会注册 ``request_operator_verdict``，收到人工评价后
才允许 finish；``request_scene_reset`` 仍只在探索模式注册。

部署脚本位于 ``robots/dual_franka/``。在仓库根目录运行：

.. code-block:: bash

   robots/dual_franka/run_manual_skill.sh --list-primitives
   robots/dual_franka/run_manual_skill.sh --schema vla_right_grasp

通过 ``--primitive NAME --params JSON`` 调用工具。``--task-id`` 为命名 VLA
技能选择任务配置中的 ``vla_instruction``，规划器的阶段 prompt 单独记入日志。
当前 clean-desk 任务继续使用 checkpoint 原来的训练指令。
``--robot-config`` 和 ``--calibration-path`` 分别选择机器参数与标定文件。
本地 SAM3 需要安装 ``sam3`` extra；远端服务可通过 ``--sam3-endpoint`` 接入。

机器人 Codex 运行配置隔离
-------------------------

上述启动脚本使用独立的 ``RPENT_CODEX_HOME``（默认仓库内
``.codex-rpent-live``），不继承编程终端的 ``CODEX_HOME``。
本地记忆默认放在该目录的 ``memory`` 中，Codex 状态数据库也使用独立目录。
按需在独立目录创建私人 ``config.toml``，不要覆盖已有私人配置。

API 部署需显式设置 ``RPENT_CODEX_API_KEY`` 和可选的
``RPENT_CODEX_BASE_URL``。脚本不再沿用普通 ``CODEX_API_KEY``、
``CODEX_BASE_URL``、``OPENAI_API_KEY`` 或 ``OPENAI_BASE_URL``。
不使用 API key 时，应在独立目录下单独登录；可配置文件保存凭据，
不要提交真实配置、凭据或日志。已有私人配置若使用系统钥匙串，须另行检查账号共享。

模型、推理强度、服务档位分别通过 ``RPENT_CODEX_MODEL``、
``RPENT_REASONING_EFFORT``、``RPENT_CODEX_SERVICE_TIER`` 设置，
默认保持 ``gpt-5.5``、``medium``、``fast``。
这些规则只适用于上述部署脚本，不改变直接调用 RPent CLI 的通用环境变量接口。
建议通过启动脚本运行；手动 source 会修改当前终端的环境变量。

目录隔离不是权限沙盒，也不隔离共享工作区文件。
当前 planner 显式使用不请求交互审批、完整文件访问的运行参数。
仅修改私人配置不能覆盖 planner 显式传入的权限；连接检测仍使用只读沙盒。
