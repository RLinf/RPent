Dual Franka
===========

RPent 可以通过 RLinf 的 ``RealWorldEnv`` 进程控制双臂 Franka 系统。
RPent 负责规划、工具、提示词与状态产物；RLinf 负责 Ray、ROS、两台机器人
控制器以及 RealSense/ZED/Lumos 相机。

前置条件
--------

请使用包含 ``DualFrankaTcpEnv-v1`` 和
``realworld_physical_agent_eval_dual_franka`` 配置的 RLinf checkout。在允许
智能体运动前，必须先独立验证两台 Franka 控制器、ROS 工作空间、相机驱动和
RLinf 环境。

请在运行 RPent 的同一个 Python 环境中安装 Franka extra。环境服务也会使用
这个解释器。如果 RLinf 不是通过 wheel 安装，请指定 RLinf checkout：

.. code-block:: bash

	pip install -e '.[franka]'
	export RPENT_RLINF_ROOT=/path/to/RLinf

请在 RLinf 配置中校准左右机器人 IP、base/left/right 相机序列号、每条臂的
reset 关节位姿和安全边界。不要直接复用其他工作空间的位姿或相机标定。

运行 smoke test
---------------

任务 ``0`` 是保守的单臂动作原语 smoke test。可以通过重复的
``--rlinf-override`` 参数传入临时 Hydra 配置：

.. code-block:: bash

	rpent --env dual_franka --task-id 0 \
	  --rlinf-override 'cluster.node_groups[0].hardware.configs[0].left_robot_ip=LEFT_ROBOT_IP' \
	  --rlinf-override 'cluster.node_groups[0].hardware.configs[0].right_robot_ip=RIGHT_ROBOT_IP' \
	  --planner api --model anthropic:claude-sonnet-4-5

Runner 会使用 RPent 解释器启动 ``robots/dual_franka/env_server.py``，等待其
``healthz`` 端点就绪，然后创建 planner toolkit。初始状态记录为 step ``0``。

连接已有服务
------------

也可以连接已有环境服务，而不在本地启动：

.. code-block:: bash

	rpent --env dual_franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

任务 ``1`` 会暴露 ``vla_grasp``。将 RPent 指向双臂 Franka SFT checkpoint，
并提供计算归一化统计时使用的数据集 repo ID：

.. code-block:: bash

	export PI05_CHECKPOINT_PATH=/path/to/checkpoints/global_step_N
	export DUAL_FRANKA_REPO_ID=org/dual-franka-tcp-rot6d

	rpent --env dual_franka --task-id 1 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --cuda-device 0 \
	  --planner api --model anthropic:claude-sonnet-4-5

任务 ``1`` 未设置 ``--vla-endpoint`` 时，RPent 会在本地启动
``robots/dual_franka/vla_server.py``。该服务只加载一次
``pi05_dualfranka_tcp_rot6d``，使用 20 维状态/动作布局以及 left-wrist、base、
right-wrist 三路相机。checkpoint 必须包含
``actor/model_state_dict/full_weights.pt`` 和
``<DUAL_FRANKA_REPO_ID>/norm_stats.json``。

也可以单独启动 VLA 服务：

.. code-block:: bash

	python -m robots.dual_franka.vla_server \
	  --model-path /path/to/checkpoints/global_step_N \
	  --repo-id org/dual-franka-tcp-rot6d \
	  --cuda-device 0 --transport http --host 0.0.0.0 --port 6000

然后向 ``rpent`` 传入 ``--vla-endpoint http://VLA_HOST:6000``。外部 endpoint
始终优先于本地自动启动。任务 ``0`` 只提供解析式 smoke-test primitives，
因此不会加载 VLA。

工具与状态产物
--------------

双臂 Franka 扩展提供 ``view_env_state``、``view_camera_meta``、``move_delta``、
``rotate_delta``、``open_gripper``、``close_gripper`` 和 ``vla_grasp``。
每次基于规则的运动都必须选择恰好一条臂（``left`` 或 ``right``），另一条臂保持
不下发指令；``move_delta`` 与 ``rotate_delta`` 使用固定的世界（right_base）
坐标系。每个会改变环境的工具都会通过 RPent 的中央 ``EnvState`` 自动保存每条臂
的机器人状态，以及同步的 left-wrist、base、right-wrist RGB 图像。

安全要求
--------

两条臂都必须有操作员留在急停按钮旁。先使用极小的单臂动作验证任务 ``0``，
再尝试抓取。当相机与状态结果不一致、目标运动未达到，或任何标定存在疑问时，
应立即停止。
