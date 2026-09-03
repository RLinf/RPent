Franka
======

RPent 可以通过 RLinf 的 ``RealWorldEnv`` worker 控制单台 Franka 机械臂。

安装
----

在 RPent 仓库根目录运行：

.. code-block:: bash

	uv sync --extra franka

该命令会把自定义的 RLinf Franka 分支和 ``rlinf-openpi`` 安装到 ``.venv``。

标定（Calibration）
----------------------

手眼标定使用 ROS 的 `easy_handeye
<https://github.com/IFL-CAMP/easy_handeye>`_ 完成。它为每台相机生成一个 YAML
（外部相机为 eye-on-base，腕部相机为 eye-on-hand），默认保存在
``~/.ros/easy_handeye/`` 下。

RPent 读取一个 JSON bundle（``hand_eye_calibration.json``），其中包含每台相机的
``source_name``、``parameters`` 和 ``transformation``；这些字段从各
``easy_handeye`` YAML 中拷贝而来。

bundle 的位置可通过 ``--calibration-path`` 配置，默认值为
``~/.ros/easy_handeye/hand_eye_calibration.json``。

开发配置
--------

仓库中给出的值是开发默认值，在启用机械臂运动前必须逐项核对：

* ``robots/franka/config/example.yaml``，包含机器身份（机器人 IP、相机序列号、
  夹爪）和工作空间几何（target/reset pose、安全边界）。

RPent 会把这份机器人配置转换成内部的 RLinf cluster 和环境对象。如需改用
其他文件，请传入 ``--robot-config /path/to/robot_config.yaml``。

启动 Ray
--------

Ray 在启动时会捕获环境变量，因此必须先设置 node rank 再启动：

.. code-block:: bash

	export RLINF_NODE_RANK=0
	ray stop --force
	ray start --head

运行冒烟测试
------------

冒烟测试用于验证基本的解析式运动和夹爪 primitives 是否正常工作。
如要进行冒烟测试，请在启动 RPent 时指定任务 ``0`` ：

.. code-block:: bash

	# replace --robot-config and --calibration-path with your own paths
	uv run --extra franka rpent --robot franka --task-id 0 \
	  --planner claude_code --model claude-opus-4-8      \
	  --robot-config robots/franka/config/example.yaml   \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

RPent 会使用当前解释器启动 ``robots/franka/env_server.py``：加载 RPent robot
config、生成内部的 RLinf adapter config、连接 Ray、等待
``healthz``，并把初始状态记录为 step ``0``。

VLA 抓取 DEMO
-------------

RPent 提供了一个使用 VLA 抓取物品的 DEMO。task-id ``1`` 会暴露
``vla_grasp`` 工具。当前单臂 Franka 需要兼容的外部 VLA 服务，
其观测布局、动作布局、checkpoint 和归一化统计必须与 Franka 训练配置一致：

.. code-block:: bash

	uv run --extra franka rpent --robot franka --task-id 1 \
	  --vla-endpoint http://VLA_HOST:PORT \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

目前 VLA server 需要单独部署。若未设置 ``--vla-endpoint``，
解析式运动和夹爪工具仍然可用，但 ``vla_grasp`` 会抛出运行时错误。

工具与状态产物
--------------

Franka 扩展提供 ``view_env_state``、``view_camera_meta``、``move_delta``、
``rotate_delta``、``open_gripper``、``close_gripper`` 和 ``vla_grasp``。
每个会改变环境的工具，都会把机器人状态、腕部与外部 RGB 图像、可选的对齐
深度数组和相机元数据保存到 RPent 中央的 ``EnvState`` 中。

安全要求
--------

操作员必须守在急停按钮旁。在尝试抓取之前，先用极小幅度动作验证任务 ``0``。
一旦相机与状态结果不一致、目标运动未达到，或任何标定存在疑问，应立即停止。
