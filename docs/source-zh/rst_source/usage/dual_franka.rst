Dual Franka
===========

RPent 可以通过 RLinf ``RealWorldEnv`` worker 控制双节点双臂 Franka 系统。
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

启用机械臂运动前，请检查并修改仓库中的开发默认值：

* ``robots/dual_franka/config/realworld_physical_agent_eval_dual_franka.yaml``
  包含两台机器人 IP、相机序列号/类型、夹爪连接、controller node rank、Ray
  placement、reset joints 和 target poses。
* ``robots/dual_franka/config/env/realworld_dual_franka_tcp_rot6d.yaml`` 包含
  20 维 TCP-rot6d 环境、action scale 和工作空间边界。
* ``robots/dual_franka/controller_config.yaml`` 包含 primitive 限制、超时、
  容差和额外感知相机配置。
* ``robots/dual_franka/calibration/`` 包含相机内参和 hand-eye calibration。

请替换所有大写硬件占位符。不要复用其他工作空间的 IP、序列号、夹爪端口、
reset pose、边界或标定。

常规 RPent 命令会直接使用这些文件。``--rlinf-config-name``、
``--rlinf-override`` 和 ``--controller-config`` 仍作为开发调试入口保留，
但不再需要单独维护 RLinf 配置。

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

	uv run --extra franka rpent --env dual_franka --task-id 0 \
	  --planner api --model anthropic:claude-sonnet-4-5

RPent 使用当前解释器启动 ``robots/dual_franka/env_server.py``，组合仓库中的
本地配置，连接 Ray，等待 ``healthz``，并将初始状态记录为 step ``0``。

VLA 任务
--------

任务 ``1`` 会暴露 ``vla_grasp``，并可在本地启动双臂 Franka VLA server。
``PI05_CHECKPOINT_PATH`` 指向 SFT checkpoint，``DUAL_FRANKA_REPO_ID`` 是用于
查找对应归一化统计的数据集 ID：

.. code-block:: bash

	export PI05_CHECKPOINT_PATH=/path/to/checkpoints/global_step_N
	export DUAL_FRANKA_REPO_ID=org/dual-franka-tcp-rot6d

	uv run --extra franka rpent --env dual_franka --task-id 1 \
	  --cuda-device 0 \
	  --planner api --model anthropic:claude-sonnet-4-5

checkpoint 必须包含：

.. code-block:: text

	actor/model_state_dict/full_weights.pt
	<DUAL_FRANKA_REPO_ID>/norm_stats.json

未设置 ``--vla-endpoint`` 时，RPent 会启动
``robots/dual_franka/vla_server.py``，并只加载一次
``pi05_dualfranka_tcp_rot6d``。任务 ``0`` 不会加载 VLA。

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

	uv run --extra franka rpent --env dual_franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

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
