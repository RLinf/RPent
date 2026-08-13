Franka
======

RPent 可以通过 RLinf 的 ``RealWorldEnv`` 进程控制单台 Franka 机械臂。
RPent 负责规划、工具、提示词与状态产物；RLinf 负责 Ray、ROS、机器人控制器
和 RealSense 相机。

前置条件
--------

请使用包含 ``PhysicalAgentFrankaEnv-v1`` 和
``realworld_physical_agent_eval`` 配置的 RLinf checkout。在允许智能体运动前，
必须先独立验证 Franka 控制器、ROS 工作空间、相机驱动和 RLinf 环境。

请在运行 RPent 的同一个 Python 环境中安装 Franka extra。环境服务也会使用
这个解释器。如果 RLinf 不是通过 wheel 安装，请指定 RLinf checkout：

.. code-block:: bash

	pip install -e '.[franka]'
	export RPENT_RLINF_ROOT=/path/to/RLinf

请在 RLinf 配置中校准机器人 IP、相机序列号、TCP reset pose 和安全边界。
不要直接复用其他工作空间的位姿或相机标定。

运行 smoke test
---------------

任务 ``0`` 是保守的动作原语 smoke test。可以通过重复的
``--rlinf-override`` 参数传入临时 Hydra 配置：

.. code-block:: bash

	rpent --env franka --task-id 0 \
	  --rlinf-override 'cluster.node_groups[0].hardware.configs[0].robot_ip=ROBOT_IP' \
	  --planner api --model anthropic:claude-sonnet-4-5

Runner 会使用 RPent 解释器启动 ``robots/franka/env_server.py``，等待其
``healthz`` 端点就绪，然后创建 planner toolkit。初始状态记录为 step ``0``。

连接已有服务
------------

也可以连接已有环境服务，而不在本地启动：

.. code-block:: bash

	rpent --env franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

任务 ``1`` 会暴露 ``vla_grasp``。它需要外部 VLA 服务，并且该服务的观测键、
状态布局、动作维度、归一化和 checkpoint 必须与当前 Franka 配置一致：

.. code-block:: bash

	rpent --env franka --task-id 1 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --vla-endpoint http://VLA_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

RPent 不会为 Franka 自动启动 LIBERO VLA server。省略 ``--vla-endpoint`` 时，
解析式运动和夹爪工具仍然可用；调用 ``vla_grasp`` 会返回明确的运行时错误。

工具与状态产物
--------------

Franka 扩展提供 ``view_env_state``、``view_camera_meta``、``move_delta``、
``rotate_delta``、``open_gripper``、``close_gripper`` 和 ``vla_grasp``。
每个会改变环境的工具都会通过 RPent 的中央 ``EnvState`` 自动保存机器人状态、
腕部与外部 RGB 图像、可选的对齐深度数组以及相机元数据。

安全要求
--------

操作员必须留在急停按钮旁。先使用极小动作验证任务 ``0``，再尝试抓取。
当相机与状态结果不一致、目标运动未达到，或任何标定存在疑问时，应立即停止。
