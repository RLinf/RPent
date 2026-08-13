Franka
======

RPent 可以通过 RLinf ``RealWorldEnv`` worker 控制单台 Franka 机械臂。
这个开发分支将运行配置保存在 RPent 中，并通过 ``franka`` extra 安装所需的
RLinf/OpenPI 分支。不需要单独的 RLinf checkout、虚拟环境或安装步骤。

安装
----

在 RPent 仓库根目录运行：

.. code-block:: bash

	uv sync --extra franka

该命令按照 ``pyproject.toml`` 将自定义 RLinf Franka 分支和
``rlinf-openpi`` 安装到 ``.venv``。

开发配置
--------

仓库中的值是开发默认值，启用机械臂运动前必须逐项检查：

* ``robots/franka/config/realworld_physical_agent_eval.yaml`` 包含机器人 IP
  和 Ray placement。
* ``robots/franka/config/env/realworld_physical_agent_franka.yaml`` 包含相机
  序列号、相机名称、reset/target pose、action scale 和工作空间安全边界。
* ``robots/franka/controller_config.yaml`` 包含 primitive 超时与容差。
* ``robots/franka/calibration/hand_eye_calibration.json`` 包含感知工具使用的
  hand-eye calibration。

请替换这些文件中的 ``ROBOT_IP``、``CAMERA_SERIAL_WRIST`` 和
``CAMERA_SERIAL_EXTERNAL``。不要复用其他工作空间的位姿、边界、序列号或标定。

常规 RPent 命令会直接使用这些文件。``--rlinf-config-name``、
``--rlinf-override`` 和 ``--controller-config`` 仍作为开发调试入口保留，
但不再需要单独维护 RLinf 配置。

启动 Ray
--------

Ray 启动时会捕获环境变量，因此必须先设置 node rank：

.. code-block:: bash

	export RLINF_NODE_RANK=0
	ray stop --force
	ray start --head

运行 smoke test
---------------

任务 ``0`` 用于测试保守的解析式运动和夹爪 primitives：

.. code-block:: bash

	uv run --extra franka rpent --env franka --task-id 0 \
	  --planner api --model anthropic:claude-sonnet-4-5

RPent 使用当前解释器启动 ``robots/franka/env_server.py``，组合仓库中的本地
配置，连接 Ray，等待 ``healthz``，并将初始状态记录为 step ``0``。

VLA 任务
--------

任务 ``1`` 会暴露 ``vla_grasp``。单臂 Franka 当前需要兼容的外部 VLA 服务，
其观测布局、动作布局、checkpoint 和归一化统计必须与当前 Franka 训练配置一致：

.. code-block:: bash

	uv run --extra franka rpent --env franka --task-id 1 \
	  --vla-endpoint http://VLA_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

未设置 ``--vla-endpoint`` 时，解析式运动和夹爪工具仍然可用，但
``vla_grasp`` 会抛出运行时错误。LIBERO VLA server 与物理 Franka 不兼容。

连接外部环境服务
----------------

连接已经运行的 Franka 环境服务：

.. code-block:: bash

	uv run --extra franka rpent --env franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

工具与状态产物
--------------

Franka 扩展提供 ``view_env_state``、``view_camera_meta``、``move_delta``、
``rotate_delta``、``open_gripper``、``close_gripper`` 和 ``vla_grasp``。
每个会改变环境的工具都会在 RPent 中央 ``EnvState`` 中保存机器人状态、腕部与
外部 RGB 图像、可选的对齐深度数组和相机元数据。

安全要求
--------

操作员必须留在急停按钮旁。先使用极小动作验证任务 ``0``，再尝试抓取。
当相机与状态结果不一致、目标运动未达到，或任何标定存在疑问时，应立即停止。
