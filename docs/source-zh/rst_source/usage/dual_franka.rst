Dual Franka
===========

RPent 可以通过 RLinf ``RealWorldEnv`` worker 控制双节点双臂 Franka 系统。
这个开发分支将运行配置保存在 RPent 中，并通过 ``franka`` extra 安装所需的
RLinf/OpenPI 分支。不需要单独的 RLinf checkout、虚拟环境或安装步骤。

安装
----

在 RPent 仓库根目录运行：

.. code-block:: bash

	pip install -e ".[franka]"

该命令将自定义 RLinf Franka 分支和 ``rlinf-openpi`` 安装到当前环境。

标定（Calibration）
----------------------

Hand-eye calibration 使用 ROS `easy_handeye
<https://github.com/IFL-CAMP/easy_handeye>`_ 完成。它为每个投影相机（
``base_camera`` 和 ``d455_camera``）生成一个 YAML，默认保存在
``~/.ros/easy_handeye/`` 下。

RPent 读取一个 JSON bundle（``hand_eye_calibration.json``），其中包含每个相机的
``source_name``、``parameters`` 和 ``transformation``。生成方式是从每个
``easy_handeye`` YAML 中拷贝这些字段。

bundle 位置可通过 ``--calibration-path`` 配置（默认
``~/.ros/easy_handeye/hand_eye_calibration.json``）。

开发配置
--------

启用机械臂运动前，请检查并修改仓库中的开发默认值：

* ``robots/dual_franka/config/example.yaml`` 包含机器身份（两台机器人 IP、相机
  序列号/类型、夹爪连接）、工作空间几何（target poses、安全边界）和感知定位
  边界 + base-frame 变换。

RPent 会将该机器人配置转换成内部双节点 RLinf cluster 和环境对象。如需使用
其他文件，请传入 ``--robot-config /path/to/robot_config.yaml``。

启动双节点 Ray 集群
--------------------

每个控制节点都必须在启动 Ray 前设置 ``RLINF_NODE_RANK``。仅在节点 ``0``
运行 RPent。

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

运行 smoke test
---------------

任务 ``0`` 用于测试保守的单臂解析式运动和夹爪 primitives：

.. code-block:: bash

	rpent --env dual_franka --task-id 0 \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

RPent 使用当前解释器启动 ``robots/dual_franka/env_server.py``，加载 RPent
robot config 并生成内部 RLinf adapter config，然后连接 Ray，等待 ``healthz``，
并将初始状态记录为 step ``0``。任务 ``0`` 不会加载 VLA。

VLA 抓取 DEMO
--------

RPent提供了一个使用 VLA 抓取物品的 DEMO。task-id ``1`` 会暴露 ``vla_grasp``。
并可在本地启动双臂 Franka VLA server。``PI05_CHECKPOINT_PATH`` 指向 
训练好的 Pi-05 checkpoint，``DUAL_FRANKA_REPO_ID`` 是用于查找对应归一化统计的数据集 ID：

.. code-block:: bash

	export PI05_CHECKPOINT_PATH=/path/to/checkpoints/global_step_N
	export DUAL_FRANKA_REPO_ID=org/dual-franka-tcp-rot6d

	rpent --env dual_franka --task-id 1 \
	  --cuda-device 0 \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

checkpoint 必须包含：

.. code-block:: text

	actor/model_state_dict/full_weights.pt
	<DUAL_FRANKA_REPO_ID>/norm_stats.json

未设置 ``--vla-endpoint`` 时，RPent 会启动
``robots/dual_franka/vla_server.py``，并只加载一次
``pi05_dualfranka_tcp_rot6d``。

也可以单独启动 VLA 服务：

.. code-block:: bash

	python -m robots.dual_franka.vla_server \
	  --model-path /path/to/checkpoints/global_step_N \
	  --repo-id org/dual-franka-tcp-rot6d \
	  --cuda-device 0 --transport http --host 0.0.0.0 --port 6000

然后向 ``rpent`` 传入 ``--vla-endpoint http://VLA_HOST:6000``。外部 endpoint
始终优先于本地自动启动。

连接外部环境服务
----------------

连接已经运行的双臂 Franka 环境服务：

.. code-block:: bash

	rpent --env dual_franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

工具与状态产物
--------------

双臂 Franka 扩展提供 ``view_env_state``、``view_camera_meta``、
``move_delta``、``rotate_delta``、``open_gripper``、``close_gripper`` 和
``vla_grasp``。每次解析式运动都只选择一条臂（``left`` 或 ``right``）。
每个会改变环境的工具都会在 RPent 中央 ``EnvState`` 中保存每条臂的状态以及
同步的 left-wrist、base 和 right-wrist 图像。

安全要求
--------

两条臂都必须有操作员留在急停按钮旁。先使用极小的单臂动作验证任务 ``0``，
再尝试抓取。当相机与状态结果不一致、目标运动未达到，或任何标定存在疑问时，
应立即停止。
