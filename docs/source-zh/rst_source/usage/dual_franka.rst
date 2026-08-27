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

* ``robots/dual_franka/robot_config.yaml`` 仅包含机器身份（两台机器人 IP、相机
  序列号/类型、夹爪连接）和工作空间几何（target poses、安全边界）。
* ``robots/dual_franka/config.py`` 存放开发者默认值（primitive 控制、感知调优、
  episode 长度、节点放置）。
* ``robots/dual_franka/calibration/`` 包含相机内参和 hand-eye calibration。

仓库中的 robot config 会有意保留当前实验室 IP、序列号和夹爪设备路径，方便
本地测试。其他系统必须检查并替换这些值。不要复用其他工作空间的 reset pose、
边界或标定。

RPent 会将该机器人配置转换成内部双节点 RLinf cluster 和环境对象。如需使用
其他文件，请传入 ``--robot-config /path/to/robot_config.yaml``。用户不再需要
接触 Hydra 或 RLinf 配置流程。

使用本地 RLinf checkout
------------------------

开发测试时，需要在启动 Ray 前让两个控制节点都能访问本地 checkout。两个节点
可以使用不同的绝对路径，但各自的 ``PYTHONPATH`` 必须指向该节点上的源码副本。

节点 ``0``：

.. code-block:: bash

	export RLINF_REPO_PATH=/path/to/RLinf
	export PYTHONPATH=$RLINF_REPO_PATH:${PYTHONPATH:-}
	export RLINF_NODE_RANK=0
	ray stop --force
	ray start --head --port=6379 --node-ip-address=HEAD_IP

节点 ``1``：

.. code-block:: bash

	export PYTHONPATH=$RLINF_REPO_PATH:${PYTHONPATH:-}
	export RLINF_NODE_RANK=1
	ray stop --force
	ray start --address=HEAD_IP:6379 --node-ip-address=WORKER_IP

在节点 ``0`` 使用本地 checkout 运行 RPent：

.. code-block:: bash

	uv run --extra franka rpent --env dual_franka --task-id 0 \
	  --planner api --model anthropic:claude-sonnet-4-5

自动启动的 ``env_server.py`` 和 ``vla_server.py`` 会在启动时读取
``RLINF_REPO_PATH`` 并把它放到 ``sys.path`` 最前面。在 ``ray start`` 前导出
``PYTHONPATH``，可让远程 Ray worker 加载同一 checkout。源码路径发生变化时，
必须在所有节点重启 Ray。

可以在每个节点运行以下命令确认实际加载位置：

.. code-block:: bash

	PYTHONPATH=/path/to/RLinf:$PYTHONPATH \
	  uv run --extra franka python -c \
	  'import rlinf; print(rlinf.__file__)'

若要让本地子进程恢复使用 ``.venv`` 中安装的版本，请取消设置
``RLINF_REPO_PATH``。

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

RPent 使用当前解释器启动 ``robots/dual_franka/env_server.py``，加载 RPent
robot config 并生成内部 RLinf adapter config，然后连接 Ray，等待 ``healthz``，
并将初始状态记录为 step ``0``。

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
